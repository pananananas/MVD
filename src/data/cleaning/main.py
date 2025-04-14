from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from typing import List, Dict, Any
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import tempfile
import textwrap
import zipfile
import sqlite3
import shutil
import random
import torch
import glob
import os
import io


# ===== CONFIGURATION =====
# DATASET_PATH = "/Users/ewojcik/Code/pwr/MVD/objaverse/renders"
DATASET_PATH = "/net/pr2/projects/plgrid/plggtattooai/MeshDatasets/objaverse/renders"
REJECTED_SAMPLES_PATH = "/net/pr2/projects/plgrid/plggtattooai/MeshDatasets/objaverse/rejected"
PROCESSING_QUEUE_PATH = "/net/pr2/projects/plgrid/plggtattooai/MeshDatasets/objaverse/queue"
os.makedirs(REJECTED_SAMPLES_PATH, exist_ok=True)
os.makedirs(DATASET_PATH, exist_ok=True)

DB_PATH = os.path.join(os.path.dirname(REJECTED_SAMPLES_PATH), "processing_status.db")

IMG_SIZE = (128, 128)
NUM_OBJECTS = 100000
MAX_NUM_VIEWS = 3

torch.set_float32_matmul_precision('high')
SCRATCH = os.getenv('SCRATCH', '/net/tscratch/people/plgewoj')
HUGGINGFACE_CACHE = os.path.join(SCRATCH, 'huggingface_cache')
os.makedirs(HUGGINGFACE_CACHE, exist_ok=True)
os.environ['HF_HOME'] = HUGGINGFACE_CACHE

# ===== PROMPTS =====
IMAGE_DESCRIPTION_PROMPT = "Provide a comprehensive and vivid description of this object. Emphasize the materials used, intricate textures, colors, and the overall shape. If the object appears mostly white or lacks distinct features, indicate that it is not clearly visible. DO NOT WRITE THAT THE OBJECT IS ABSTRACT."


DISTILLATION_PROMPT_TEMPLATE = """I have multiple descriptions of the same object from different angles:

{descriptions}

Based on these descriptions, write a concrete prompt that could be used to generate images of this object. Write about colors, and shape. The description must be concise and strictly limited to 2 sentences.
"""

FILTRATION_PROMPT_TEMPLATE = """I have multiple descriptions of the same object from different angles:

{descriptions}

Evaluate the provided descriptions to assess the usefulness of this data sample. A sample is considered useful if it is rich in detail and showcases vibrant textures. Conversely, a sample is deemed useless if it is partially obscured, predominantly white, or abstract with minimal detail. Additionally, if any view of the object is not visible, if the object appears very small, or if there is no clearly identifiable object, the sample should be classified as useless.
Write `True` or `False` as an output.
"""

# ===== MODEL CONFIG =====

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct",
    torch_dtype=torch.bfloat16,
    # attn_implementation= "eager",  # "flash_attention_2",
    device_map="auto",
    cache_dir=HUGGINGFACE_CACHE,
)

min_pixels = 64*28*28
max_pixels = 128*28*28
processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct",
    min_pixels=min_pixels,
    max_pixels=max_pixels,
    cache_dir=HUGGINGFACE_CACHE,
    use_fast=True,
)


def setup_database():
    """Setup SQLite database to track processing status."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
    CREATE TABLE IF NOT EXISTS samples (
        path TEXT PRIMARY KEY,
        processed BOOLEAN,
        is_useful BOOLEAN,
        prompt TEXT,
        error TEXT,
        processed_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    conn.commit()
    return conn


def wrap_text(text, width=50):
    if not text:
        return "No text available"
    return '\n'.join(textwrap.wrap(text, width))


def get_zip_files(data_path: str, limit: int = None) -> List[str]:
    zip_files = sorted(glob.glob(os.path.join(data_path, "*.zip")))
    
    if limit and len(zip_files) > limit:
        random.seed(42)
        zip_files = random.sample(zip_files, limit)
    
    return zip_files


def load_images_from_zip(zip_path: str, target_size: tuple = IMG_SIZE) -> Dict[str, Any]:
    object_data = {
        'object_uid': Path(zip_path).stem,
        'images': [],
        'image_paths': [],
        'prompt': None
    }
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        file_list = zip_ref.namelist()
        
        prompt_files = [f for f in file_list if f.endswith('prompt.txt')]
        if prompt_files:
            with zip_ref.open(prompt_files[0]) as f:
                object_data['prompt'] = f.read().decode('utf-8').strip()
        else:
            object_data['prompt'] = "Not provided"
        
        png_files = sorted([f for f in file_list if f.endswith('.png')])
        
        for png_file in png_files:

            with zip_ref.open(png_file) as f:
                img_data = f.read()
                img = Image.open(io.BytesIO(img_data))
                
                if img.mode == 'RGBA':
                    white_bg = Image.new('RGBA', img.size, (255, 255, 255, 255))
                    img = Image.alpha_composite(white_bg, img)
                
                img = img.convert('RGB')
                img = img.resize(target_size, Image.Resampling.LANCZOS)
                
                object_data['images'].append(img)
                object_data['image_paths'].append(png_file)
            continue
    
    return object_data


def generate_description(image: Image.Image) -> str:
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            temp_path = temp_file.name
            image.save(temp_path)
        
        file_path = f'file://{os.path.abspath(temp_path)}'
        
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": file_path},
                {"type": "text", "text": IMAGE_DESCRIPTION_PROMPT},
            ],
        }]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        inputs = inputs.to(model.device)
        
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=200)
            
            generated_ids_trimmed = [
                generated_ids[i][len(input_ids):] 
                for i, input_ids in enumerate(inputs.input_ids)
            ]
            
            output_text = processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=True
            )[0]
        
        if os.path.exists(temp_path):
            os.unlink(temp_path)
            
        return output_text.strip()
    
    except Exception as e:
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.unlink(temp_path)
        return "Error generating description"


def distill_descriptions(descriptions: List[str], original_prompt: str) -> str:
    try:
        descriptions_text = ' '.join([f'Description {i+1}: {desc}' for i, desc in enumerate(descriptions)])
        prompt = DISTILLATION_PROMPT_TEMPLATE.format(
            descriptions=descriptions_text,
            original_prompt=original_prompt
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = processor(
            text=[text],
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=300)
            
            generated_ids_trimmed = [
                generated_ids[i][len(input_ids):] 
                for i, input_ids in enumerate(inputs.input_ids)
            ]
            
            output_text = processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=True
            )[0]

        return output_text.strip()
    
    except Exception as e:
        return "Error creating distilled prompt"


def filter_sample(descriptions: List[str]) -> bool:
    try:
        descriptions_text = ' '.join([f'Description {i+1}: {desc}' for i, desc in enumerate(descriptions)])
        prompt = FILTRATION_PROMPT_TEMPLATE.format(
            descriptions=descriptions_text
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = processor(
            text=[text],
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=50)
            
            generated_ids_trimmed = [
                generated_ids[i][len(input_ids):] 
                for i, input_ids in enumerate(inputs.input_ids)
            ]
            
            output_text = processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=True
            )[0]

        output_text = output_text.strip().lower()
        is_useful = "true" in output_text
        
        return is_useful
    
    except Exception as e:
        return True


def add_prompt_to_zip(zip_path: str, prompt_text: str) -> bool:
    try:
        temp_dir = tempfile.mkdtemp()
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        extracted_dirs = [d for d in os.listdir(temp_dir) if os.path.isdir(os.path.join(temp_dir, d))]
        if extracted_dirs:
            content_dir = os.path.join(temp_dir, extracted_dirs[0])
            with open(os.path.join(content_dir, 'prompt.txt'), 'w') as f:
                f.write(prompt_text)
        else:
            with open(os.path.join(temp_dir, 'prompt.txt'), 'w') as f:
                f.write(prompt_text)
        
        temp_zip = os.path.join(temp_dir, "temp.zip")
        with zipfile.ZipFile(temp_zip, 'w') as new_zip:
            for root, _, files in os.walk(temp_dir):
                for file in files:
                    if file == "temp.zip":
                        continue
                    
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, temp_dir)
                    new_zip.write(file_path, arcname)
        
        shutil.move(temp_zip, zip_path)
        
        shutil.rmtree(temp_dir)
        return True
        
    except Exception as e:
        print(f"Error adding prompt to zip {zip_path}: {e}")
        if 'temp_dir' in locals() and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        return False


def main():
    conn = setup_database()
    cursor = conn.cursor()
    
    zip_files = get_zip_files(PROCESSING_QUEUE_PATH, limit=NUM_OBJECTS)
        
    for zip_path in zip_files:
        cursor.execute("INSERT OR IGNORE INTO samples (path, processed) VALUES (?, ?)", 
                      (zip_path, False))
    conn.commit()
    
    cursor.execute("SELECT path FROM samples WHERE processed = 0")
    unprocessed = [row[0] for row in cursor.fetchall()]
    
    print(f"Found {len(unprocessed)} unprocessed samples out of {len(zip_files)} total")
    
    for zip_path in tqdm(unprocessed, desc="Processing objects"):
        try:
            if not os.path.exists(zip_path):
                cursor.execute(
                    "UPDATE samples SET processed = 1, error = ? WHERE path = ?",
                    ("File not found", zip_path)
                )
                conn.commit()
                continue
                
            object_data = load_images_from_zip(zip_path)
            
            if len(object_data['images']) == 0:
                cursor.execute(
                    "UPDATE samples SET processed = 1, error = ? WHERE path = ?",
                    ("No images found", zip_path)
                )
                conn.commit()
                continue
            
            descriptions = []
            valid_descriptions = []
            valid_images = []

            images_to_process = object_data['images'][:MAX_NUM_VIEWS] if len(object_data['images']) > MAX_NUM_VIEWS else object_data['images']
            
            for img in tqdm(images_to_process, desc="Generating descriptions", leave=False):
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                desc = generate_description(img)
                descriptions.append(desc)
                
                if desc != "Error generating description":
                    valid_descriptions.append(desc)
                    valid_images.append(img)
            
            if not valid_descriptions:
                cursor.execute(
                    "UPDATE samples SET processed = 1, error = ? WHERE path = ?",
                    ("No valid descriptions generated", zip_path)
                )
                conn.commit()
                continue
            
            is_useful = filter_sample(valid_descriptions)
            filename = os.path.basename(zip_path)
            
            if is_useful:
                distilled_prompt = distill_descriptions(valid_descriptions, object_data['prompt'])
                add_prompt_to_zip(zip_path, distilled_prompt)
                
                dataset_file_path = os.path.join(DATASET_PATH, filename)
                shutil.move(zip_path, dataset_file_path)
                
                cursor.execute(
                    "UPDATE samples SET processed = 1, is_useful = ?, prompt = ?, path = ? WHERE path = ?",
                    (is_useful, distilled_prompt, dataset_file_path, zip_path)
                )
            else:
                distilled_prompt = 'useless sample'
                filtered_path = os.path.join(REJECTED_SAMPLES_PATH, filename)
                
                shutil.move(zip_path, filtered_path)
                
                cursor.execute(
                    "UPDATE samples SET processed = 1, is_useful = ?, prompt = ?, path = ? WHERE path = ?",
                    (is_useful, distilled_prompt, filtered_path, zip_path)
                )
            
            conn.commit()
            
        except Exception as e:
            print(f"Error processing {zip_path}: {e}")
            cursor.execute(
                "UPDATE samples SET processed = 1, error = ? WHERE path = ?",
                (str(e), zip_path)
            )
            conn.commit()
    
    cursor.execute("SELECT COUNT(*) FROM samples WHERE processed = 1 AND is_useful = 1")
    useful_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM samples WHERE processed = 1 AND is_useful = 0")
    not_useful_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM samples WHERE processed = 1 AND error IS NOT NULL")
    error_count = cursor.fetchone()[0]
    
    print(f"Processing complete. Summary:")
    print(f"  - Useful samples: {useful_count}")
    print(f"  - Non-useful samples: {not_useful_count}")
    print(f"  - Errors: {error_count}")
    
    conn.close()

if __name__ == "__main__":
    main()
