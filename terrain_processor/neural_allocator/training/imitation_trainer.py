"""
Phase 1: Imitation Learning Trainer with regularization improvements.

- Label smoothing cross-entropy loss
- Early stopping on validation loss
- Dynamic batch padding via collate_fn
- OneCycleLR scheduler
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import time
from typing import Optional, Dict

from ..config import NeuralAllocatorConfig, TrainingConfig
from ..model import WaypointAllocationNet
from .dataset import AllocationDataset, dynamic_collate_fn


class ImitationTrainer:

    def __init__(self, model: WaypointAllocationNet, config: TrainingConfig,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.label_smoothing = model.config.label_smoothing

        self.optimizer = optim.AdamW(
            model.parameters(), lr=model.config.lr,
            weight_decay=model.config.weight_decay,
        )
        self.scheduler = None  # Created in train() when we know n_steps

    def _compute_loss(self, scores, labels, waypoint_mask):
        """Label-smoothed cross-entropy loss."""
        B = scores.shape[0]
        total_loss = torch.tensor(0.0, device=self.device)
        count = 0
        eps = self.label_smoothing

        for b in range(B):
            n_w = int(waypoint_mask[b].sum())
            if n_w == 0:
                continue

            # Only use valid robots (those with at least one label=1)
            l = labels[b, :, :n_w]
            n_r = int(l.sum(dim=1).gt(0).sum().clamp(min=1))
            s = scores[b, :n_r, :n_w]
            l = l[:n_r, :]

            target = l.argmax(dim=0)  # (n_w,)

            # Clamp scores to prevent -inf in log_softmax
            s = s.clamp(min=-50, max=50)
            log_probs = torch.log_softmax(s, dim=0)  # (n_r, n_w)

            nll = -log_probs.gather(0, target.unsqueeze(0)).squeeze(0)
            smooth = -log_probs.mean(dim=0)
            loss = (1 - eps) * nll.mean() + eps * smooth.mean()

            if torch.isfinite(loss):
                total_loss = total_loss + loss
                count += 1

        return total_loss / max(count, 1)

    def _compute_metrics(self, scores, labels, waypoint_mask):
        B = scores.shape[0]
        correct, total = 0, 0
        with torch.no_grad():
            for b in range(B):
                n_w = int(waypoint_mask[b].sum())
                if n_w == 0:
                    continue
                l = labels[b, :, :n_w]
                n_r = int(l.sum(dim=1).gt(0).sum().clamp(min=1))
                pred = scores[b, :n_r, :n_w].argmax(dim=0)
                target = l[:n_r, :].argmax(dim=0)
                correct += (pred == target).sum().item()
                total += n_w
        return {'accuracy': correct / max(total, 1)}

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss, total_acc, n_batches = 0.0, 0.0, 0

        for batch in dataloader:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            scores = self.model(
                batch['robot_feats'], batch['waypoint_feats'],
                batch['pairwise_feats'], batch['robot_mask'], batch['waypoint_mask'],
            )

            loss = self._compute_loss(scores, batch['label'], batch['waypoint_mask'])

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()

            metrics = self._compute_metrics(scores, batch['label'], batch['waypoint_mask'])
            total_loss += loss.item()
            total_acc += metrics['accuracy']
            n_batches += 1

        return {
            'loss': total_loss / max(n_batches, 1),
            'accuracy': total_acc / max(n_batches, 1),
            'lr': self.optimizer.param_groups[0]['lr'],
        }

    @torch.no_grad()
    def validate(self, dataloader):
        self.model.eval()
        total_loss, total_acc, n_batches = 0.0, 0.0, 0

        for batch in dataloader:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            scores = self.model(
                batch['robot_feats'], batch['waypoint_feats'],
                batch['pairwise_feats'], batch['robot_mask'], batch['waypoint_mask'],
            )
            loss = self._compute_loss(scores, batch['label'], batch['waypoint_mask'])
            metrics = self._compute_metrics(scores, batch['label'], batch['waypoint_mask'])
            total_loss += loss.item()
            total_acc += metrics['accuracy']
            n_batches += 1

        return {
            'val_loss': total_loss / max(n_batches, 1),
            'val_accuracy': total_acc / max(n_batches, 1),
        }

    def train(self, train_dataset, val_dataset=None,
              checkpoint_dir='checkpoints/', n_epochs=None):
        """Full training loop with early stopping."""
        os.makedirs(checkpoint_dir, exist_ok=True)
        n_epochs = n_epochs or self.config.n_epochs_imitation

        train_loader = DataLoader(
            train_dataset, batch_size=self.config.batch_size,
            shuffle=True, num_workers=0, collate_fn=dynamic_collate_fn,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.config.batch_size,
            shuffle=False, num_workers=0, collate_fn=dynamic_collate_fn,
        ) if val_dataset else None

        # OneCycleLR scheduler
        steps_per_epoch = len(train_loader)
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=self.model.config.lr,
            steps_per_epoch=steps_per_epoch, epochs=n_epochs,
            pct_start=0.1,
        )

        history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(n_epochs):
            t0 = time.time()
            train_metrics = self.train_epoch(train_loader)
            elapsed = time.time() - t0

            history['loss'].append(train_metrics['loss'])
            history['accuracy'].append(train_metrics['accuracy'])

            if (epoch + 1) % self.config.log_interval == 0:
                msg = (f"Epoch {epoch+1}/{n_epochs} "
                       f"loss={train_metrics['loss']:.4f} "
                       f"acc={train_metrics['accuracy']:.3f} "
                       f"lr={train_metrics['lr']:.6f} "
                       f"time={elapsed:.1f}s")

                if val_loader and (epoch + 1) % self.config.val_interval == 0:
                    val_metrics = self.validate(val_loader)
                    history['val_loss'].append(val_metrics['val_loss'])
                    history['val_accuracy'].append(val_metrics['val_accuracy'])
                    msg += (f" | val_loss={val_metrics['val_loss']:.4f} "
                            f"val_acc={val_metrics['val_accuracy']:.3f}")

                    # Early stopping check
                    if val_metrics['val_loss'] < best_val_loss:
                        best_val_loss = val_metrics['val_loss']
                        patience_counter = 0
                        torch.save(self.model.state_dict(),
                                   os.path.join(checkpoint_dir, 'best_imitation.pt'))
                    else:
                        patience_counter += 1

                    if patience_counter >= self.config.early_stopping_patience:
                        msg += f" | EARLY STOP (patience={self.config.early_stopping_patience})"
                        print(msg)
                        break

                print(msg)

        # Final save
        torch.save(self.model.state_dict(),
                   os.path.join(checkpoint_dir, 'imitation_final.pt'))
        best_val_acc = max(history['val_accuracy']) if history['val_accuracy'] else 0
        print(f"Training complete. Best val accuracy: {best_val_acc:.3f}")
        return history
