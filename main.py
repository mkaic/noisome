import torch
from torchvision.transforms.v2.functional import to_tensor
from torchvision.io import write_jpg
from PIL import Image
import argparse
from pathlib import Path
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("--degree", type=int, default=64)
args = parser.parse_args()

output_dir = Path("outputs")
timelapse_dir = output_dir / "timelapse"
shutil.rmtree(timelapse_dir, ignore_errors=True)
timelapse_dir.mkdir(parents=True, exist_ok=True)

image = Image.open("branos.png").convert("RGB")
image = to_tensor(image)
c, h, w = image.shape

torch.manual_seed(42)
basis_noise = torch.randn(args.degree, c, h, w)
blending_factors = torch.randn(args.degree)

optimizer = torch.optim.Adam([blending_factors], lr=0.01)

for i in range(1000):
    optimizer.zero_grad()
    generated_image = torch.sigmoid((basis_noise * blending_factors.view(-1, 1, 1, 1)).sum(0))
    loss = ((generated_image - image) ** 2).mean()
    loss.backward()
    optimizer.step()

    if i % 100 == 0:
        generated_image = (generated_image * 255).uint8()
        write_jpg(generated_image, timelapse_dir / f"{i:04d}.jpg")
        write_jpg(image * 255, output_dir / "latest.jpg")