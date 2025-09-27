import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau

from tqdm.cli import tqdm
from typing import Optional
from pathlib import Path


class MaskRCNNTrainer:
    def __init__(
            self,
            model,
            dataloader_train: DataLoader,
            dataloader_valid: DataLoader,
            learning_rate: float = 1e-3,
            scheduler_factor: float = 0.1,
            scheduler_patience: int = 3,
            early_stop_patience: int = 5,
            log_batches_interval: int = 16,
            fpath_logs: str = "logs",
            fpath_models: str = "models",
            max_batches_train: Optional[int] = None,
            max_batches_valid: Optional[int] = None
        ):
        """Custom trainer for PyTorch's Mask R-CNN Model. Implements the Adam
        optimizer, early stopping, and a learning rate scheduler
        (ReduceLROnPlateau). Utilizes Tensorboard to log batch and epoch
        losses, learning rates, and a subset of the validation outputs.
        """
        # Select device automatically
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

        # Dataloaders & properties
        self.dl_train = dataloader_train
        self.dl_valid = dataloader_valid
        
        self.dl_train_size = len(self.dl_train)
        self.dl_valid_size = len(self.dl_valid)

        # Determine batch size by peeking at the first iteration
        self.train_batch_size = len(next(iter(self.dl_train)))
        self.valid_batch_size = len(next(iter(self.dl_valid)))

        # Model optimizer and learning rate scheduling
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(model.parameters(), learning_rate)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=scheduler_factor,
            patience=scheduler_patience,
        )

        # Early stopping
        self.early_stop_patience = early_stop_patience

        # File paths for tensorboard logs and model pickles
        self.fpath_logs = Path(fpath_logs)
        self.fpath_models = Path(fpath_models)

        # Allow a max batch size (Primarily for testing purposes)
        if max_batches_train is None:
            self.max_batches_train = np.inf
            self.final_size_train = self.dl_train_size
        else:
            self.max_batches_train = max_batches_train
            self.final_size_train = min(self.dl_train_size, max_batches_train)

        if max_batches_valid is None:
            self.max_batches_valid = np.inf
            self.final_size_valid = self.dl_valid_size
        else:
            self.max_batches_valid = max_batches_valid
            self.final_size_valid = min(self.dl_valid_size, max_batches_valid)

        # Training logging settings
        self.log_batches_interval = log_batches_interval

        # Store the state dict for our best and final model/optimizer/scheduler
        self.best_model_state_dicts = {
            "epoch": None,
            "valid_loss": None,
            "model_state_dict": None,
            "optimizer_state_dict": None,
            "scheduler_state_dict": None
        }

        self.final_model_state_dicts = {
            "epoch": None,
            "valid_loss": None,
            "model_state_dict": None,
            "optimizer_state_dict": None,
            "scheduler_state_dict": None
        }            

    def _train_loop(self, epoch: int, max_epochs: int, writer):
        epoch_loss = 0.0
        train_pbar = tqdm(
            enumerate(self.dl_train),
            total=self.final_size_train,
            desc=f"(Train) Epoch: {epoch + 1} / {max_epochs}"
        )

        self.model.train()
        for i, (images, targets) in train_pbar:
            # 🚀 Move data to same device as model
            images = [img.to(self.device) for img in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            # Forward pass
            loss_dict = self.model(images, targets)
            loss_dict["loss_overall"] = sum(loss for loss in loss_dict.values())

            # Backward pass
            self.optimizer.zero_grad()
            loss_dict["loss_overall"].backward()
            self.optimizer.step()

            # Track current epoch loss
            epoch_loss += loss_dict["loss_overall"].item()
            epoch_loss_avg = epoch_loss / (i + 1)
            train_pbar.set_postfix({"Avg Loss": f"{epoch_loss_avg:.4f}"})

            # Log Metrics
            if (i + 1) % self.log_batches_interval == 0 or (i + 1) == self.final_size_train:
                global_step = i + epoch * self.final_size_train
                writer.add_scalar("Loss/train_running", epoch_loss_avg, global_step)
                writer.add_scalar("Learning Rate", self.optimizer.param_groups[0]["lr"], global_step)

            # Leave if we are at the max batches
            if (i + 1) == self.final_size_train:
                break

        return epoch_loss_avg
            

    def _valid_loop(self, epoch: int, max_epochs: int, writer: Optional = None):
        epoch_loss = 0.0
        valid_pbar = tqdm(
            enumerate(self.dl_valid),
            total=self.final_size_valid,
            desc=f"(Valid) Epoch: {epoch + 1} / {max_epochs}"
        )

        # Even though we're validating, we'll stay in train mode (to get loss)
        self.model.train()
        for i, (images, targets) in valid_pbar:
            # 🚀 Move data to same device as model
            images = [img.to(self.device) for img in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            # Forward pass
            loss_dict = self.model(images, targets)
            loss_dict["loss_overall"] = sum(loss for loss in loss_dict.values())

            # Track current epoch loss
            epoch_loss += loss_dict["loss_overall"].item()
            epoch_loss_avg = epoch_loss / (i + 1)
            valid_pbar.set_postfix({"Avg Loss": f"{epoch_loss_avg:.4f}"})
    
            # Leave if we are at the max batches
            if (i + 1) == self.final_size_valid:
                break

        return epoch_loss_avg
        
    def train(self, max_epochs: int):
        writer = SummaryWriter(self.fpath_logs)
        epochs_since_improvement = 0
        best_valid_loss = np.inf

        for epoch in range(max_epochs):
            # Train loop
            epoch_train_loss = self._train_loop(epoch, max_epochs, writer)
            writer.add_scalar("Loss/train_epoch", epoch_train_loss, epoch)

            # Validation loop
            with torch.no_grad():
                epoch_loss_valid = self._valid_loop(epoch, max_epochs)
                writer.add_scalar("Loss/valid_epoch", epoch_loss_valid, epoch)

            # Update the learning rate scheduler
            self.scheduler.step(epoch_loss_valid)

            # Model checkpointing & Early stop prep
            if epoch_loss_valid < best_valid_loss:
                best_valid_loss = epoch_loss_valid
                epochs_since_improvement = 0

                checkpoint_path = self.fpath_models / f"model_checkpoint_epoch_{epoch}.pth"
                self.best_model_state_dicts["epoch"] = epoch
                self.best_model_state_dicts["valid_loss"] = epoch_loss_valid
                self.best_model_state_dicts["model_state_dict"] = self.model.state_dict()
                self.best_model_state_dicts["optimizer_state_dict"] = self.optimizer.state_dict()
                self.best_model_state_dicts["scheduler_state_dict"] = self.scheduler.state_dict()
                torch.save(self.best_model_state_dicts, checkpoint_path)
            else:
                epochs_since_improvement += 1

            # Early stopping & Final model saving
            if epochs_since_improvement == self.early_stop_patience or (epoch + 1) == max_epochs:
                checkpoint_path = self.fpath_models / f"final_model_checkpoint_epoch_{epoch}.pth"
                self.final_model_state_dicts["epoch"] = epoch
                self.final_model_state_dicts["valid_loss"] = epoch_loss_valid
                self.final_model_state_dicts["model_state_dict"] = self.model.state_dict()
                self.final_model_state_dicts["optimizer_state_dict"] = self.optimizer.state_dict()
                self.final_model_state_dicts["scheduler_state_dict"] = self.scheduler.state_dict()
                torch.save(self.final_model_state_dicts, checkpoint_path)

                return None
