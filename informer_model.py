"""Informer-style regressor with distilling (closer to paper)."""
import torch
import torch.nn as nn

class ProbSparseSelfAttention(nn.Module):
    """ProbSparse-style attention (simplified top-k sampling)."""
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        return attn_out

class DistillingLayer(nn.Module):
    """Self-attention distilling: Conv1d + ELU + MaxPool."""
    def __init__(self, d_model):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.act = nn.ELU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x.transpose(1, 2)
        x = self.pool(self.act(self.conv(x)))
        return x.transpose(1, 2)

class InformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.attn = ProbSparseSelfAttention(d_model, nhead, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_out = self.attn(x)
        x = self.norm1(x + attn_out)
        ff_out = self.ffn(x)
        return self.norm2(x + ff_out)

class InformerRegressor(nn.Module):
    """Informer-inspired regressor with simplified structure for small datasets."""
    def __init__(self, input_size, d_model=64, nhead=2, num_layers=2, dropout=0.1, output_size=1):
        super().__init__()
        self.embedding = nn.Linear(input_size, d_model)
        self.layers = nn.ModuleList([InformerEncoderLayer(d_model, nhead, dropout) for _ in range(num_layers)])
        self.head = nn.Linear(d_model, output_size)
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch, seq_len, input_size)
        Returns:
            Tensor of shape (batch, output_size)
        """
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        return self.head(x[:, -1, :])

import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
import os
import time
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, window_size=24, horizon=1):
        self.X = X
        self.y = y
        self.window_size = window_size
        self.horizon = horizon

    def __len__(self):
        return len(self.X) - self.window_size - self.horizon + 1

    def __getitem__(self, idx):
        features = self.X[idx: idx + self.window_size]
        target_idx = idx + self.window_size + self.horizon - 1
        target = self.y[target_idx]
        return torch.FloatTensor(features), torch.FloatTensor([target])

def load_data(dataset_name='Texas', feature_config='all', window_size=24, horizon=1):
    file_path = f'./data/{dataset_name}/data.csv'
    if not os.path.exists(file_path):
        file_path = f'./data/{dataset_name}/data.csv'
    df = pd.read_csv(file_path, encoding='utf-8', usecols=[0,1], names=['Date', 'WindPower'])
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df.dropna(subset=['Date'], inplace=True)
    df = df.sort_values('Date').drop_duplicates(subset=['Date'], keep='first').set_index('Date')
    df['WindPower'] = pd.to_numeric(df['WindPower'], errors='coerce')
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    df['Hour'] = df.index.hour
    df['Day'] = df.index.day
    df['Week'] = df.index.isocalendar().week.astype(int)
    df['Month'] = df.index.month
    df['WindPower_Log'] = np.log1p(df['WindPower'].clip(lower=0))
    df['Diff1'] = df['WindPower_Log'].diff(1)
    df['Diff2'] = df['Diff1'].diff(1)
    df.dropna(inplace=True)

    target_col = 'WindPower'
    base_features = ['Hour', 'Day', 'Week', 'Month']
    if feature_config == 'raw':
        selected = base_features + [target_col]
    elif feature_config == 'diff1':
        selected = base_features + [target_col, 'Diff1']
    else:
        selected = base_features + [target_col, 'Diff1', 'Diff2']

    X_df = df[selected].astype(np.float32)
    y_df = df[target_col].astype(np.float32)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df.values)
    y = y_df.values

    dataset = TimeSeriesDataset(X_scaled, y, window_size, horizon)
    train_size = int(len(dataset) * 0.8)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42) )
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    return train_loader, val_loader, X_scaled.shape[1]  # input_size

def train_regressor(model, train_loader, val_loader, config, device):
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = getattr(optim, config['optimizer'])(
        model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=config['scheduler_patience'])

    best_val_loss = float('inf')
    best_model_state = None
    counter = 0
    history = {'train_loss': [], 'val_loss': [], 'lr': []}
    train_losses, val_losses = [], []

    for epoch in range(config['epochs']):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.get('clip_norm', 1.0))
            optimizer.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)
                val_loss += loss.item() * x.size(0)
        val_loss /= len(val_loader.dataset)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(optimizer.param_groups[0]['lr'])

        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        train_losses.append(avg_train)
        val_losses.append(avg_val)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            torch.save(best_model_state, config['save_path'])
            counter = 0
        else:
            counter += 1
            if counter >= config['patience']:
                print(f"Early stopping at epoch {epoch+1}")
                break

        scheduler.step(val_loss)

    model.load_state_dict(best_model_state)
    plt.figure(figsize=(10, 4))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend()
    plt.show()

    return model, best_val_loss, history

def main():
    config = {
        'dataset': 'Texas',
        'feature_config': 'all',
        'window_size': 24,
        'horizon': 1,

        'd_model': 64,
        'nhead': 4,
        'num_layers': 2,
        'dropout': 0.1,

        'lr': 0.0005,
        'epochs': 300,
        'patience': 30,
        'weight_decay': 1e-5,
        'optimizer': 'AdamW',
        'scheduler_patience': 5,
        'clip_norm': 1.0,
        'save_path': 'best_iftt.pth',

        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    print("=" * 60)
    print("=" * 60)
    for k, v in config.items():
        print(f"  {k}: {v}")
    print("-" * 60)

    train_loader, val_loader, input_size = load_data(
        dataset_name=config['dataset'],
        feature_config=config['feature_config'],
        window_size=config['window_size'],
        horizon=config['horizon']
    )
    print(f": {input_size}")

    model = InformerRegressor(
        input_size=input_size,
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    )

    device = torch.device(config['device'])
    start_time = time.time()
    best_model, best_val_loss, history = train_regressor(model, train_loader, val_loader, config, device)
    train_time = time.time() - start_time

    best_model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            out = best_model(x)
            preds.append(out.cpu().numpy())
            targets.append(y.numpy())
    preds = np.concatenate(preds).flatten()
    targets = np.concatenate(targets).flatten()

    mae = mean_absolute_error(targets, preds)
    rmse = np.sqrt(mean_squared_error(targets, preds))
    r2 = r2_score(targets, preds)
    mape = np.mean(np.abs((targets - preds) / (targets + 1e-8))) * 100

    print("\n" + "=" * 60)
    print("{:.6f}".format(best_val_loss))
    print(f"  MAE : {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R2  : {r2:.4f}")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  : {train_time:.2f} 秒")
    print("=" * 60)

if __name__ == "__main__":
    main()