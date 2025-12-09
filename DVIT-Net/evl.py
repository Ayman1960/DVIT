import torch
import torch.nn.functional as F
from tqdm import tqdm
import os
from tiff_data import TiffFolder
import numpy as np
import random
from network import *


class Evaluator:
    def __init__(self, model_name, model, dataloader, device):
        self.model_name = model_name
        self.model = model
        self.dataloader = dataloader
        self.device = device
        os.makedirs("evaluation_results", exist_ok=True)
        self.output_file = f"evaluation_results/{model_name}.txt"

    @staticmethod
    def calculate_metrics(y_true, y_pred):
        TP = ((y_pred == 1) & (y_true == 1)).sum().item()
        FP = ((y_pred == 1) & (y_true == 0)).sum().item()
        FN = ((y_pred == 0) & (y_true == 1)).sum().item()
        TN = ((y_pred == 0) & (y_true == 0)).sum().item()

        ua = TP / (TP + FP + 1e-10)
        pa = TP / (TP + FN + 1e-10)
        precision = TP / (TP + FP + 1e-10)
        recall = TP / (TP + FN + 1e-10)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)

        numerator = TP * TN - FP * FN
        denominator = (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN) ** 0.5 + 1e-7
        mcc = numerator / denominator

        intersection = (y_true * y_pred).sum().item()
        iou = intersection / (y_true.sum() + y_pred.sum() - intersection + 1e-7)

        return ua, pa, f1, mcc, iou

    @torch.inference_mode()
    def run(self, threshold=0.5):
        self.model.eval()
        metrics = []

        for images, masks_true in tqdm(self.dataloader, desc="Evaluating"):
            images, masks_true = images.to(self.device), masks_true.to(self.device)
            if images.dim() == 3:
                images = images.unsqueeze(0)

            masks_pred = (self.model(images) > threshold).float()
            metrics.append(self.calculate_metrics(masks_true, masks_pred))

        avg_metrics = [sum(m[i] for m in metrics) / len(metrics) for i in range(5)]

        with open(self.output_file, "w") as f:
            f.write(
                f"Threshold: {threshold}\nUA: {avg_metrics[0]}\nPA: {avg_metrics[1]}\n"
                f"F1: {avg_metrics[2]}\nMCC: {avg_metrics[3]}\nIoU: {avg_metrics[4]}\n"
                f"\nModel:\n{self.model}\n"
            )

        print(
            f"UA: {avg_metrics[0]:.4f}, PA: {avg_metrics[1]:.4f}, F1: {avg_metrics[2]:.4f}, "
            f"MCC: {avg_metrics[3]:.4f}, IoU: {avg_metrics[4]:.4f}"
        )
        return avg_metrics


def evaluate_model(checkpoint, model, data_loader, name):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.nn.DataParallel(model.to(device))
    model.load_state_dict(checkpoint["model_state_dict"])

    evaluator = Evaluator(name, model, data_loader, device)
    return evaluator.run()


