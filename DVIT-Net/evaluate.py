import torch
import torch.nn.functional as F
from tqdm import tqdm
import os
from network import *
from tiff_data import TiffFolder
import numpy as np
import random


class Evaluate:
    def __init__(
        self,
        modelName,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device,
    ):
        self.modelName = modelName
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.output_file = "evaluation_results/{}.txt".format(modelName)

        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)

    def recall_m(self, y_true, y_pred):
        true_positives = torch.sum(torch.round(torch.clip(y_true * y_pred, 0, 1)))
        possible_positives = torch.sum(torch.round(torch.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + 1e-7)
        return recall

    def precision_m(self, y_true, y_pred):
        true_positives = torch.sum(torch.round(torch.clip(y_true * y_pred, 0, 1)))
        predicted_positives = torch.sum(torch.round(torch.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + 1e-7)
        return precision

    def calculate_ua_pa(self, target, pred):
        pred = pred.view(-1)
        target = target.view(-1)

        TP = ((pred == 1) & (target == 1)).sum().item()
        FP = ((pred == 1) & (target == 0)).sum().item()
        FN = ((pred == 0) & (target == 1)).sum().item()

        UA = TP / (TP + FP + 1e-10)  
        PA = TP / (TP + FN + 1e-10)  

        return UA, PA

    def f1_m(self, y_true, y_pred):
        precision = self.precision_m(y_true, y_pred)
        recall = self.recall_m(y_true, y_pred)
        f1_score = 2 * ((precision * recall) / (precision + recall + 1e-7))
        return f1_score.item()

    def mcc_m(self, y_true, y_pred):
        TP = (y_true * y_pred).sum()
        TN = ((1 - y_true) * (1 - y_pred)).sum()
        FP = ((1 - y_true) * y_pred).sum()
        FN = (y_true * (1 - y_pred)).sum()

        numerator = TP * TN - FP * FN
        denominator = torch.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN) + 1e-7)
        mcc = numerator / denominator
        return mcc.item() 

    def iou_m(self, y_true, y_pred):
        intersection = torch.sum(y_true * y_pred)
        union = torch.sum(y_true) + torch.sum(y_pred) - intersection
        iou = intersection / (union + 1e-7)
        return iou.item()

    @torch.inference_mode()
    def run(self):
        self.model.eval()
        best_threshold = 0.5  

        all_ua = []
        all_pa = []
        all_f1s = []
        all_mcc = []
        all_iou = []

        with open(self.output_file, "w") as f:
            for batch in tqdm(
                self.dataloader,
                desc="Validation round with best threshold",
                unit="batch",
            ):
                images, masks_true = batch[0].to(self.device), batch[1].to(self.device)
                if images.dim() == 3:
                    images = images.unsqueeze(0)
                masks_pred = self.model(images)
                masks_pred[masks_pred > best_threshold] = 1
                masks_pred[masks_pred <= best_threshold] = 0
                masks_pred = masks_pred.float()

                ua, pa = self.calculate_ua_pa(masks_true, masks_pred)
                f1_score = self.f1_m(masks_true, masks_pred)
                mcc = self.mcc_m(masks_true, masks_pred)
                iou = self.iou_m(masks_true, masks_pred)

                all_ua.append(ua)
                all_pa.append(pa)
                all_f1s.append(f1_score)
                all_mcc.append(mcc)
                all_iou.append(iou)

            avg_ua = sum(all_ua) / len(all_ua)
            avg_pa = sum(all_pa) / len(all_pa)
            avg_f1 = sum(all_f1s) / len(all_f1s)
            avg_mcc = sum(all_mcc) / len(all_mcc)
            avg_iou = sum(all_iou) / len(all_iou)

            f.write(f"\nBest Threshold: {best_threshold}\n")
            f.write(f"Average UA: {avg_ua}\n")
            f.write(f"Average PA: {avg_pa}\n")
            f.write(f"Average F1 Score: {avg_f1}\n")
            f.write(f"Average MCC: {avg_mcc}\n")
            f.write(f"Average IoU: {avg_iou}\n")
            f.write(f"\nModel structure:\n {self.model}\n")

 
            print(f"ua: {avg_ua}")
            print(f"pa: {avg_pa}")
            print(f"F1 Score: {avg_f1}")
            print(f"mcc: {avg_mcc}")
            print(f"IoU: {avg_iou}")

        return avg_ua, avg_pa, avg_f1, avg_mcc, avg_iou


def Eva(checkpoint, model, data_loader, name):
    seed = 42
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    model.load_state_dict(checkpoint["model_state_dict"])
    evaluate = Evaluate(name, model, data_loader, device)
    evaluate.run()

