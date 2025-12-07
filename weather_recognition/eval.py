# eval.py
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report

def evaluate_model(model, dataloader, class_names):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in dataloader:
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.numpy())
            all_labels.extend(y.numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')

    print("Accuracy:", acc)
    print("Macro F1:", f1)
    print(classification_report(all_labels, all_preds, target_names=class_names))