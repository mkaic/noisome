import argparse
import shutil
from pathlib import Path

import torch
from PIL import Image
from torchvision.io import write_jpeg
from torchvision.transforms import ToTensor
from tqdm import tqdm

torch.manual_seed(42)
with torch.no_grad():

    output_dir = Path("outputs")
    timelapse_dir = output_dir / "timelapse"
    shutil.rmtree(timelapse_dir, ignore_errors=True)
    timelapse_dir.mkdir(parents=True, exist_ok=True)

    image = Image.open("branos.png").convert("RGB")
    image = image.resize((128, 128))
    image = ToTensor()(image).to("cuda")

    c, h, w = image.shape

    generated_image = torch.randn_like(image)
    loss = torch.square(generated_image - image).mean()

    pbar = tqdm(range(1_000_000))
    for i in pbar:
        candidate_noise = torch.randn_like(generated_image) * 0.01
        candidate_image = torch.sigmoid(generated_image + candidate_noise)
        candidate_loss = torch.square(candidate_image - image).mean()

        if candidate_loss < loss:
            generated_image += candidate_noise
            loss = candidate_loss

        pbar.set_description(f"loss: {loss.item():.4f}")

        if i % 1000 == 0:
            as_uint8 = (torch.sigmoid(generated_image) * 255).to(torch.uint8)
            write_jpeg(as_uint8.cpu(), (timelapse_dir / f"{i:04d}.jpg").as_posix())
            write_jpeg(as_uint8.cpu(), (output_dir / "latest.jpg").as_posix())