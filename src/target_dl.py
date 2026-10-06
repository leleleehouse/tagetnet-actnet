from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class TorchMLPClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, hidden_size=128, dropout_rate=0.5, learning_rate=0.001, 
                 weight_decay=0.001, epochs=20, batch_size=32, weight_for_class1=1.0,
                 random_state=42, verbose=0):
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_for_class1 = weight_for_class1
        self.random_state = random_state
        self.verbose = verbose
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_ = None

    def _build_model(self, input_dim):
        model = nn.Sequential(
            nn.Linear(input_dim, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(64, 1)
        )
        return model

    def fit(self, X, y, fold_idx=None):
        print(f"[Fold {fold_idx}] fit start")
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32).reshape(-1, 1)
        print(f"[Fold {fold_idx}] numpy conversion done: X={X.shape}, y={y.shape}")

        _, input_dim = X.shape
        self.classes_ = np.unique(y).flatten()

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        self.model_ = self._build_model(input_dim).to(self.device)

        pos_weight = torch.tensor([self.weight_for_class1], dtype=torch.float32).to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.Adam(
            self.model_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        loader = DataLoader(dataset, batch_size=int(self.batch_size), shuffle=True)

        self.model_.train()

        desc = f"Fold {fold_idx} epochs" if fold_idx is not None else "Training epochs"
        epoch_iter = tqdm(
            range(1, int(self.epochs) + 1),
            desc=desc,
            leave=True,
            disable=False
        )

        for epoch in epoch_iter:
            epoch_losses = []

            for batch_X, batch_y in loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = self.model_(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.item())

            mean_loss = float(np.mean(epoch_losses))
            epoch_iter.set_postfix({
                "epoch": f"{epoch}/{self.epochs}",
                "loss": f"{mean_loss:.4f}"
            })

            print(f"[Fold {fold_idx}] Epoch {epoch}/{self.epochs} loss={mean_loss:.4f}")

        print(f"[Fold {fold_idx}] fit end")
        return self
    
    def predict(self, X):
        self.model_.eval()
        X = np.array(X, dtype=np.float32)
        with torch.no_grad():
            inputs = torch.from_numpy(X).to(self.device)
            outputs = self.model_(inputs)
            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).cpu().numpy().astype(int).flatten()
        return preds

    def predict_proba(self, X):
        self.model_.eval()
        X = np.array(X, dtype=np.float32)
        with torch.no_grad():
            inputs = torch.from_numpy(X).to(self.device)
            outputs = self.model_(inputs)
            probs = torch.sigmoid(outputs).cpu().numpy()
            proba = np.hstack([1 - probs, probs])
        return proba

    def get_params(self, deep=True):
        return {
            'hidden_size': self.hidden_size,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'weight_for_class1': self.weight_for_class1,
            'random_state': self.random_state,
            'verbose': self.verbose
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self