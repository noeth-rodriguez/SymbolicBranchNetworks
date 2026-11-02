import torch
import numpy as np
import pandas as pd
from typing import Union
from tqdm import tqdm
from SymbolicBranchNet import SymbolicBranchNet
import matplotlib.pyplot as plt 
from torch.utils.data import Dataset,DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F

class TabularDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = X
        self.y = y

    def __getitem__(self, idx):
        x = self.X[idx, :]
        if self.y is not None:
            y = self.y[idx]
            return x, y
        else:
            return x

    def __len__(self):
        return self.X.shape[0]


class SBNModel(SymbolicBranchNet):
    """Inherit from SymbolicBranchNet to inherit forward and
    create fit and predict methods.
    """
    def __init__(
        self,
        teacher_model = None,
        training_w1=True,
        training_w2=False,
        device: str = None,
        dtype: torch.dtype = torch.float32
    ):
        super().__init__()
        self.teacher_model = teacher_model
        self.training_w1 = training_w1
        self.training_w2 = training_w2
        self.device = device
        self.dtype = dtype


    def train_step(
            self,
            X_train: torch.Tensor,
            y_train: torch.Tensor,
            optimizer: torch.optim.Optimizer
    ) -> float:
        self.train()
        device = self.device

        dataset = TabularDataset(X_train, y_train)
        loader = DataLoader(dataset, batch_size=min(128, len(dataset)),
                            shuffle=True, drop_last=False)

        loss_sum = 0.0
        for x, y in loader:
            x = x.to(device)
            y = y.view(-1)

            optimizer.zero_grad(set_to_none=True)

            logits = self.forward(x)  # [N,C]

            loss = F.cross_entropy(logits, y.to(device))

            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
        return loss_sum / max(1, len(loader))

    @torch.no_grad()
    def val_step(
            self,
            x_val: torch.Tensor,
            y_val: torch.Tensor
    ) -> float:
        self.eval()
        device = self.device

        dataset = TabularDataset(x_val, y_val)
        loader = DataLoader(dataset, batch_size=min(128, len(dataset)),
                            shuffle=False, drop_last=False)


        loss_sum = 0.0
        n_batches = 0
        for x, y in loader:
            x = x.to(device)
            y = y.view(-1)

            logits = self.forward(x)
            loss = F.cross_entropy(logits, y.long().to(device))
            loss_sum += loss.item()
            n_batches += 1

        self.train()
        return loss_sum / max(1, n_batches)

    def fit(
            self,
            x_train: np.ndarray,
            y_train: np.ndarray,
            x_val: np.ndarray,
            y_val: np.ndarray,
            learning_rate: float = 1e-1,
            epochs: int = 1500,
            loss_file: str | None = None,
    ):
        # tensors
        x_train = torch.from_numpy(x_train.copy()).float()
        x_val = torch.from_numpy(x_val.copy()).float()
        # keep labels as long for priors, cast to float later only if needed elsewhere
        y_train_t = torch.from_numpy(np.asarray(y_train).copy()).view(-1, 1).long()
        y_val_t = torch.from_numpy(np.asarray(y_val).copy()).view(-1, 1).long()

        # optimizer: AdamW with mild weight decay
        optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=1e-3, betas=(0.9, 0.98))

        #Sheduler
        plateau = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=35, cooldown=0,min_lr=1e-4)

        # simple linear warmup for first few epochs
        warmup_epochs = 5
        base_lr = learning_rate

        min_val_loss = float('inf')
        patience = 0
        max_patience = 100

        loss_history, val_loss_history = [], []
        progress_bar = tqdm(range(epochs))

        for epoch in progress_bar:
            # warmup
            if epoch < warmup_epochs:
                warmup_lr = base_lr * (epoch + 1) / warmup_epochs
                for pg in optimizer.param_groups:
                    pg['lr'] = warmup_lr

            # train
            train_loss = self.train_step(x_train, y_train_t, optimizer)
            loss_history.append(train_loss)

            # validate
            val_loss = self.val_step(x_val, y_val_t)
            val_loss_history.append(val_loss)

            progress_bar.set_description(
                f"loss {train_loss:.5f} | val {val_loss:.5f} | lr {optimizer.param_groups[0]['lr']:.2e}")

            # scheduler
            plateau.step(val_loss)

            # early stopping checkpoint
            if val_loss < min_val_loss - 1e-6:
                min_val_loss = val_loss
                torch.save(self.state_dict(), "temporal.pt")
                patience = 0
            else:
                patience += 1
                if patience >= max_patience:
                    break

        # restore best
        if patience < max_patience:
            self.load_state_dict(torch.load("temporal.pt", weights_only=True))

        #plot
        plt.figure(figsize=(12, 8))
        plt.plot(loss_history[5:], label="train")
        plt.plot(val_loss_history[5:], label="val")
        plt.legend()
        if loss_file is not None:
            plt.savefig(loss_file)
        plt.close()

        return self

    def predict(
            self, x_test: Union[pd.DataFrame, np.ndarray, torch.Tensor]
    ) -> torch.Tensor:
        x_test_copy = torch.from_numpy(x_test.values.copy()).float()
        dataset = TabularDataset(x_test_copy)
        dataloader = DataLoader(dataset, batch_size=min(200, len(dataset)), shuffle=False)

        preds = []
        self.eval()
        with torch.no_grad():
            for x in dataloader:
                x = x.to(self.device)
                logits = self.forward(x)
                probs = torch.softmax(logits, dim=1)  # [N,C]
                pred = torch.argmax(probs, dim=1)
                preds.append(pred.cpu())

        self.train()
        return torch.cat(preds, dim=0)

    def predict_proba(
            self, x_test: Union[pd.DataFrame, np.ndarray, torch.Tensor]
    ) -> torch.Tensor:
        x_test_copy = torch.from_numpy(x_test.values.copy()).float()
        dataset = TabularDataset(x_test_copy)
        dataloader = DataLoader(dataset, batch_size=min(200, len(dataset)), shuffle=False)

        prob_list = []
        self.eval()
        with torch.no_grad():
            for x in dataloader:
                x = x.to(self.device)
                logits = self.forward(x)
                probs = torch.softmax(logits, dim=1)
                prob_list.append(probs.cpu())
        self.train()
        return torch.cat(prob_list, dim=0)


