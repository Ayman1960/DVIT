import yaml
import torch
import os
import logging
from evaluate import Evaluate
from framework import MyFrame
from tiff_data import TiffFolder
from time import time
from network import *
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from datetime import datetime
import random
import numpy as np
from torch.cuda.amp import autocast, GradScaler


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logging(log_dir, name):
    # Make sure the log directory exists
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        print(f"Log directory created: {log_dir}")

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = os.path.join(log_dir, f"{current_time}_{name}__.log")

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),  # Also output to console
        ],
    )
    print(f"Log file created at: {log_file}")
    return current_time


def setup_directories(*dirs):
    for directory in dirs:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Directory created: {directory}")


def load_checkpoint(solver, checkpoint_dir, loadModel):
    try:
        checkpoint = torch.load(os.path.join(checkpoint_dir, loadModel))
        solver.net.load_state_dict(checkpoint["model_state_dict"])
        solver.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_loss = checkpoint["best_loss"]
        logging.info(f"Resuming from epoch {start_epoch}")
        return start_epoch, best_loss
    except FileNotFoundError:
        logging.warning("Checkpoint file not found. Starting from scratch.")
        return 1, float("inf")


def train_epoch(solver, data_loader, scaler, accumulation_steps=1):
    train_epoch_loss = 0
    solver.optimizer.zero_grad()

    for i, (img, mask) in enumerate(data_loader):
        solver.set_input(img, mask)

        # Use mixed precision training
        with autocast():
            loss = solver.optimize()
            loss = loss / accumulation_steps  # Scale loss

        # Use gradient scaler
        scaler.scale(loss).backward()

        if (i + 1) % accumulation_steps == 0:
            scaler.step(solver.optimizer)
            scaler.update()
            solver.optimizer.zero_grad()

        train_epoch_loss += loss.item() * accumulation_steps

    train_epoch_loss /= len(data_loader)
    return train_epoch_loss


def save_checkpoint(epoch, solver, best_loss, checkpoint_dir, saveModel):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": solver.net.state_dict(),
        "optimizer_state_dict": solver.optimizer.state_dict(),
        "best_loss": best_loss,
    }
    torch.save(checkpoint, os.path.join(checkpoint_dir, saveModel))
    logging.info(f"Model saved: {saveModel}")


def evaluate_model(solver, valdata_loader, netname, epoch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluate = Evaluate(netname, solver.net, valdata_loader, device)
    ua, pa, f1, mcc, iou = evaluate.run()
    logging.info(f"Epoch {epoch} evaluation results:")
    logging.info(
        f"UA: {ua:.4f}, PA: {pa:.4f}, F1: {f1:.4f}, MCC: {mcc:.4f}, IoU: {iou:.4f}"
    )
    return ua, pa, f1, mcc, iou


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True  # Enable cuDNN auto-tuner


def main():
    # Set random seed
    seed = 42
    set_seed(seed)

    config = load_config("config.yaml")

    # Make sure all necessary directories exist
    log_dir = config["train"]["log_dir"]
    weight_dir = config["train"]["weight_dir"]
    checkpoint_dir = config["train"]["checkpoint_dir"]
    setup_directories(log_dir, weight_dir, checkpoint_dir)

    # Set up logging
    root = config["data"]["root"]
    val = config["data"]["val"]
    netname = config["train"]["netname"]
    NAME = f"log__{netname}_"
    current_time = setup_logging(log_dir, NAME)

    logging.info(f"Random seed set: {seed}")
    logging.info(f"Dataset path: {root}")
    logging.info(f"Validation set path: {val}")
    logging.info(f"Model name: {netname}")

    BATCHSIZE_PER_CARD = config["train"]["batchsize_per_card"]
    total_epoch = config["train"]["total_epoch"]
    initial_lr = float(config["train"]["initial_lr"])
    lr_min = float(config["train"]["lr_min"])
    batchsize = torch.cuda.device_count() * BATCHSIZE_PER_CARD
    loadModel = config["train"]["loadModel"]
    saveModel = f"{current_time}_{NAME}_checkpoint.pth"
    resume = config["train"]["resume"]
    early_stop = config["train"]["early_stop"]
    net_model = DVit_net()
    SIZE = config["data"]["size"]

    # Set up data loaders
    num_workers = min(os.cpu_count() * 2, 16)  # Set number of workers
    dataset = TiffFolder(root, SIZE, False)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batchsize,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )
    valdata_loader = torch.utils.data.DataLoader(
        TiffFolder(val, SIZE, True),
        batch_size=16,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    solver = MyFrame(net_model, initial_lr, False, batchsize=batchsize)

    # Initialize gradient scaler
    scaler = GradScaler()

    # Set gradient accumulation steps
    accumulation_steps = 2  # Can be adjusted as needed

    start_epoch, train_epoch_best_loss = 1, float("inf")
    if resume:
        start_epoch, train_epoch_best_loss = load_checkpoint(
            solver, checkpoint_dir, loadModel
        )

    scheduler = ReduceLROnPlateau(
        solver.optimizer,
        "min",
        factor=0.5,
        patience=2,
        min_lr=float(lr_min),
        verbose=True,
    )

    no_optim = 0
    tic = time()
    logging.info(f"Starting training, configuration: {config}")
    logging.info(f"Using mixed precision training")
    logging.info(f"Gradient accumulation steps: {accumulation_steps}")
    logging.info(f"Data loader number of workers: {num_workers}")

    for epoch in range(start_epoch, total_epoch + 1):
        train_epoch_loss = train_epoch(solver, data_loader, scaler, accumulation_steps)

        logging.info(
            f"Epoch: {epoch}, Time: {int(time() - tic)}, Train Loss: {train_epoch_loss}"
        )

        scheduler.step(float(train_epoch_loss))

        if train_epoch_loss >= train_epoch_best_loss:
            no_optim += 1
        else:
            no_optim = 0
            train_epoch_best_loss = train_epoch_loss
            save_checkpoint(
                epoch, solver, train_epoch_best_loss, checkpoint_dir, saveModel
            )

        if no_optim > early_stop:
            logging.info(f"Early stopping at epoch {epoch}")
            break
        if no_optim > 3 and solver.old_lr < 5e-7:
            checkpoint = torch.load(os.path.join(checkpoint_dir, saveModel))
            solver.net.load_state_dict(checkpoint["model_state_dict"])

    evaluate_model(solver, valdata_loader, netname, epoch)
    logging.info(f"Final loss: {train_epoch_best_loss}")
    logging.info("Training completed!")


if __name__ == "__main__":
    main()
