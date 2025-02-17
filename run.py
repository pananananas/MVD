from src.load_co3d import analyze_dataset, CO3DDatasetLoader
from diffusers import StableDiffusionPipeline
from src.vis_co3d import CO3DVisualizer
from icecream import ic
import wandb
import random

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="mvd",

    # track hyperparameters and run metadata
    config={
        "learning_rate": 0.02,
        "architecture": "CNN",
        "dataset": "CIFAR-100",
        "epochs": 10,
    }
)

data_path = "/Users/ewojcik/Code/datasets/co3d/laptop"
model_id = "runwayml/stable-diffusion-v1-5"
sequence_id = "62_4317_10724"

analyze_dataset(data_path)

loader = CO3DDatasetLoader(data_path)
visualizer = CO3DVisualizer(loader)


ic(loader.frame_annotations[0].keys())
ic(loader.sequence_annotations[0].keys())

visualizer.visualize_cameras_and_pointcloud(sequence_id)




# pipe = StableDiffusionPipeline.from_pretrained(model_id)
# unet = pipe.unet
# ic(unet)
 



# simulate training
epochs = wandb.config.epochs
offset = random.random() / 5
for epoch in range(2, epochs):
    acc = 1 - 2 ** -epoch - random.random() / epoch - offset
    loss = 2 ** -epoch + random.random() / epoch + offset

    # log metrics to wandb
    wandb.log({"acc": acc, "loss": loss})

# [optional] finish the wandb run, necessary in notebooks
wandb.finish()