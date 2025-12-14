# infer.py
from pathlib import Path

import hydra
import torch
from PIL import Image
from hydra.utils import instantiate
from omegaconf import DictConfig

from data.dvc_utils import pull_data
from data.transforms import val_test_transform


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # --- DVC ---
    pull_data()

    # --- Device ---
    device = torch.device(cfg.infer.device)

    # --- Model ---
    model = instantiate(cfg.model)
    model.load_state_dict(torch.load(cfg.infer.model_path, map_location=device))
    model.to(device)
    model.eval()

    # --- Image ---
    image_path = Path(cfg.infer.image_path)
    image = Image.open(image_path).convert("RGB")
    tensor = val_test_transform(image).unsqueeze(0).to(device)

    # --- Inference ---
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_idx].item()

    # --- Class name ---
    class_name = cfg.data.classes[pred_idx]

    print(
        f"Prediction: {class_name}\n"
        f"Confidence: {confidence:.4f}"
    )


if __name__ == "__main__":
    main()
