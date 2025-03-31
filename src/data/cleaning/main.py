# TODO:
# - filter out objects without textures
# - filter non-detailed objects


from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from typing import List, Dict, Any
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import numpy as np
import tempfile
import textwrap
import zipfile
import random
import torch
import glob
import os
import io


# ===== CONFIGURATION =====
DATASET_PATH = "/Users/ewojcik/Code/pwr/MVD/objaverse/renders"
IMG_SIZE = (512, 512)
OUTPUT_DIR = "output_visualizations"
NUM_OBJECTS = 300

# torch.set_float32_matmul_precision('high')
# SCRATCH = os.getenv('SCRATCH', '/net/tscratch/people/plgewoj')
# HUGGINGFACE_CACHE = os.path.join(SCRATCH, 'huggingface_cache')
# os.makedirs(HUGGINGFACE_CACHE, exist_ok=True)
# os.environ['HF_HOME'] = HUGGINGFACE_CACHE

# dataset_path = "/net/pr2/projects/plgrid/plggtattooai/MeshDatasets/objaverse/"
dataset_path = "/Users/ewojcik/Code/pwr/MVD/objaverse/renders"


# ===== PROMPTS =====
IMAGE_DESCRIPTION_PROMPT = "Describe this object in detail. Focus on materials, textures, colors, and shape."

DISTILLATION_PROMPT_TEMPLATE = """I have multiple descriptions of the same 3D object from different angles:

{descriptions}

Based on these descriptions, create detailed prompt that could be used to generate a 3D model of this object. Focus on materials, textures, colors, and shape. Make it 2 sentences max, so concise but detailed."""

FILTRATION_PROMPT_TEMPLATE = """I have multiple descriptions of the same 3D object from different angles:

{descriptions}

Based on these descriptions determine if this data sample is useful. Useful samples have a lot of detail and contain rich textires. Useless samples are partially not visible, mostly white. 
Write `True` of `False` as an output.
"""



# ===== SETUP =====

# Initialize model and processor
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct",
    torch_dtype=torch.bfloat16,
    attn_implementation="eager",  # "flash_attention_2",
    device_map="auto",
    # cache_dir=HUGGINGFACE_CACHE,
)

min_pixels = 128*28*28
max_pixels = 512*28*28
processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct",
    min_pixels=min_pixels,
    max_pixels=max_pixels,
    # cache_dir=HUGGINGFACE_CACHE,
)

# Configure matplotlib to handle CJK characters - use a font that supports them or fallback
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # Properly handle minus signs

def wrap_text(text, width=50):
    """Wrap text to fit within the specified width."""
    if not text:
        return "No text available"
    return '\n'.join(textwrap.wrap(text, width))

def get_zip_files(data_path: str, limit: int = NUM_OBJECTS) -> List[str]:
    """Get a limited number of zip files from the dataset."""
    zip_files = sorted(glob.glob(os.path.join(data_path, "*.zip")))
    
    if len(zip_files) > limit:
        random.seed(42)  # For reproducibility
        zip_files = random.sample(zip_files, limit)
    
    return zip_files

def load_images_from_zip(zip_path: str, target_size: tuple = IMG_SIZE) -> Dict[str, Any]:
    """Load all images and metadata from a zip file."""
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
    """Generate a description for an image using Qwen2.5-VL."""
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
    """Distill multiple descriptions into a single prompt using Qwen2.5-VL."""
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
    """Determine if the sample is useful based on its descriptions."""
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

        # Clean up the output to get a boolean
        output_text = output_text.strip().lower()
        is_useful = "true" in output_text
        
        return is_useful
    
    except Exception as e:
        # Default to True in case of error
        return True

def visualize_object_views(object_data: Dict[str, Any], descriptions: List[str], distilled_prompt: str, is_useful: bool, save_path: str = None):
    """Create a visualization of all views with descriptions and the distilled prompt."""
    num_views = len(object_data['images'])
    
    cols = min(4, num_views)
    rows = (num_views // cols) + (1 if num_views % cols != 0 else 0) + 1
    
    fig = plt.figure(figsize=(6 * cols, 6 * rows))
    
    # Add the filtration result at the top
    fig.suptitle(f"Sample is {'USEFUL' if is_useful else 'NOT USEFUL'}", 
                fontsize=14, 
                color='green' if is_useful else 'red',
                fontweight='bold')
    
    gs = fig.add_gridspec(rows, cols)
    
    # Plot each view with its description
    for i, (img, desc) in enumerate(zip(object_data['images'], descriptions)):
        row = i // cols
        col = i % cols
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(np.array(img))
        
        # Wrap text to prevent overlap
        wrapped_desc = wrap_text(desc, width=30)
        title = f"View {i+1}\n{wrapped_desc[:200]}..."
        ax.set_title(title, fontsize=9)
        ax.axis('off')
    
    # Add distilled prompt at the bottom, spanning all columns
    prompt_text = f"Distilled Prompt:\n{wrap_text(distilled_prompt, width=80)}"
    ax = fig.add_subplot(gs[rows-1, :])
    
    # Use text instead of title for better control over wrapping
    ax.text(0.5, 0.5, prompt_text, 
            ha='center', va='center', 
            fontsize=10, 
            wrap=True,
            transform=ax.transAxes,
            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        # Use a different approach to save with font fallback
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    
    plt.close()

def main():    
    zip_files = get_zip_files(DATASET_PATH, limit=NUM_OBJECTS)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    for i, zip_path in enumerate(tqdm(zip_files, desc="Processing objects")):

        object_data = load_images_from_zip(zip_path)
        
        if len(object_data['images']) == 0:
            continue
        
        descriptions = []
        valid_descriptions = []
        valid_images = []
        
        for j, img in enumerate(tqdm(object_data['images'], desc=f"Generating descriptions", leave=False)):
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            desc = generate_description(img)
            descriptions.append(desc)
            
            if desc != "Error generating description":
                valid_descriptions.append(desc)
                valid_images.append(img)
            
            desc_preview = desc[:50] + "..." if len(desc) > 50 else desc
        
        if not valid_descriptions:
            output_path = os.path.join(OUTPUT_DIR, f"{object_data['object_uid']}_error.png")
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, "Failed to generate any valid descriptions for this object", 
                        ha='center', va='center', fontsize=14)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(output_path, dpi=100, bbox_inches='tight')
            plt.close()
            continue
        
        distilled_prompt = distill_descriptions(valid_descriptions, object_data['prompt'])
        
        # Run filtration to determine if the sample is useful
        is_useful = filter_sample(valid_descriptions)
        
        output_path = os.path.join(OUTPUT_DIR, f"{object_data['object_uid']}_analysis.png")
        
        if len(valid_descriptions) < len(descriptions):
            valid_object_data = {
                'object_uid': object_data['object_uid'],
                'images': valid_images,
                'image_paths': [object_data['image_paths'][i] for i, desc in enumerate(descriptions) 
                                if desc != "Error generating description"],
                'prompt': object_data['prompt']
            }
            visualize_object_views(valid_object_data, valid_descriptions, distilled_prompt, is_useful, output_path)
        else:
            visualize_object_views(object_data, descriptions, distilled_prompt, is_useful, output_path)
        

    

if __name__ == "__main__":
    main()