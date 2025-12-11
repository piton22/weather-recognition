# infer.py

import torch
from data.transforms import val_test_transform
from PIL import Image


def predict_single(model, image_path, class_names):
    img = Image.open(image_path).convert("RGB")
    tensor = val_test_transform(img).unsqueeze(0)

    with torch.no_grad():
        out = model(tensor)
        probs = torch.softmax(out, dim=1)
        idx = torch.argmax(probs).item()

    return class_names[idx], probs[0, idx].item()
