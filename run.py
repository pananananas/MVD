from src.load_co3d import analyze_dataset, CO3DDatasetLoader
from src.vis_co3d import CO3DVisualizer
from icecream import ic

apple_path = "/Users/ewojcik/Code/datasets/3D/co3d/apple"

analyze_dataset(apple_path)

loader = CO3DDatasetLoader(apple_path)
visualizer = CO3DVisualizer(loader)


ic(loader.frame_annotations[0].keys())
ic(loader.sequence_annotations[0].keys())


sequence_id = "12_90_489"  # użyj sekwencji którą wcześniej analizowaliśmy

# 1. Zobacz kilka klatek z sekwencji
visualizer.visualize_sequence_frames(sequence_id)

# 2. Zobacz chmurę punktów i pozycje kamer
visualizer.visualize_cameras_and_pointcloud(sequence_id)

# 3. Zobacz projekcję punktów 3D na wybraną klatkę
visualizer.visualize_frame_with_projection(sequence_id, frame_number=0)


# from diffusers import StableDiffusionPipeline

# model_id = "runwayml/stable-diffusion-v1-5"
# pipe = StableDiffusionPipeline.from_pretrained(model_id)

# unet = pipe.unet
# ic(unet)  # Zobacz architekturę

