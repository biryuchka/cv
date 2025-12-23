import os
import torch
import torch.nn as nn
import numpy as np
from torch.optim import AdamW
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        device: torch.device,
        epochs: int,
        log_freq: int = 1,
        save_freq: int = 5,
        lr: float = 1e-4,
        backbone_lr: float = 1e-5,
        weight_decay: float = 1e-4,
        checkpoint_dir: str = "ckpts",
        freeze_backbone: bool = False,
        log_dir: str = "runs",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = epochs
        self.log_freq = log_freq
        self.save_freq = save_freq
        self.checkpoint_dir = checkpoint_dir
        self.scaler = GradScaler()
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        if freeze_backbone:
            print("Freezing backbone...")
            for p in self.model.model.backbone.parameters():
                p.requires_grad = False

        backbone_params = []
        other_params = []

        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if "backbone" in name:
                backbone_params.append(p)
            else:
                other_params.append(p)

        self.optimizer = AdamW(
            [
                {"params": other_params, "lr": lr},
                {"params": backbone_params, "lr": backbone_lr},
            ],
            weight_decay=weight_decay,
        )

        nparams = sum(p.numel() for p in self.model.parameters() if p.requires_grad) / 1e6
        print(f"Trainable parameters: {nparams:.1f}M")

        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)

        self.hist = []
        self.hist_components = []

    def save_checkpoint(self, epoch):
        if (epoch + 1) % self.save_freq == 0:
            path = os.path.join(self.checkpoint_dir, f"detr_epoch_{epoch+1}.pt")
            torch.save(self.model.state_dict(), path)
            print(f"Checkpoint saved: {path}")

    def train(self):
        print(f"Training for {self.epochs} epochs on {self.device}")

        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0.0
            comp_losses = {}

            for batch in tqdm(self.train_loader, desc=f"Epoch {epoch+1}"):
                pixel_values = batch["pixel_values"].to(self.device)
                labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]

                self.optimizer.zero_grad(set_to_none=True)

                with autocast():
                    outputs = self.model(
                        pixel_values=pixel_values,
                        labels=labels,
                    )
                    loss = outputs.loss
                    loss_dict = outputs.loss_dict

                self.scaler.scale(loss).backward()

                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)

                self.scaler.step(self.optimizer)
                self.scaler.update()

                epoch_loss += loss.item()

                for k, v in loss_dict.items():
                    comp_losses[k] = comp_losses.get(k, 0.0) + v.item()

            epoch_loss /= len(self.train_loader)
            for k in comp_losses:
                comp_losses[k] /= len(self.train_loader)

            if (epoch + 1) % self.log_freq == 0:
                print(f"\nEpoch {epoch+1}/{self.epochs}")
                print(f"Total Loss: {epoch_loss:.4f}")
                for k, v in comp_losses.items():
                    print(f"  {k}: {v:.4f}")

                self.writer.add_scalar("Loss/Train", epoch_loss, epoch + 1)
                for k, v in comp_losses.items():
                    self.writer.add_scalar(f"Loss/Train_{k}", v, epoch + 1)
                
                if self.val_loader is not None:
                    val_loss = self.evaluate(epoch)

                self.hist.append(epoch_loss)
                self.hist_components.append(comp_losses)

            self.save_checkpoint(epoch)
        
        self.writer.close()


    def evaluate(self, epoch: int):
        self.model.eval()

        val_loss = 0.0
        comp_losses = {}

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation", leave=False):
                pixel_values = batch["pixel_values"].to(self.device)
                labels = [
                    {k: v.to(self.device) for k, v in t.items()}
                    for t in batch["labels"]
                ]

                outputs = self.model(
                    pixel_values=pixel_values,
                    labels=labels,
                )

                loss = outputs.loss
                loss_dict = outputs.loss_dict

                val_loss += loss.item()

                for k, v in loss_dict.items():
                    comp_losses[k] = comp_losses.get(k, 0.0) + v.item()

        val_loss /= len(self.val_loader)
        for k in comp_losses:
            comp_losses[k] /= len(self.val_loader)

        print("\nValidation")
        print(f"Total Loss: {val_loss:.4f}")
        for k, v in comp_losses.items():
            print(f"  {k}: {v:.4f}")

        # TensorBoard logging
        self.writer.add_scalar("Loss/Val", val_loss, epoch + 1)
        for k, v in comp_losses.items():
            self.writer.add_scalar(f"Loss/Val_{k}", v, epoch + 1)

        self.model.train()

        return val_loss
