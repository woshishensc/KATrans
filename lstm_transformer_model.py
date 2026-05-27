import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class LSTMPositionalEncoding(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=3, dropout=0.2):
        super(LSTMPositionalEncoding, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.positional_encoding = PositionalEncoding(hidden_size, dropout)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # [batch, seq_len, hidden_size]
        enc_in = lstm_out.permute(1, 0, 2)  # [seq_len, batch, hidden]
        encoded = self.positional_encoding(enc_in)
        return lstm_out, encoded

class LSTMTransformer(nn.Module):
    def __init__(self, input_size, hidden_size=64, nhead=2, num_layers=3, dropout=0.2):
        super(LSTMTransformer, self).__init__()
        self.hidden_size = hidden_size
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.lstm_pe = LSTMPositionalEncoding(
            hidden_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout
        )
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=nhead, dim_feedforward=hidden_size, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.residual_proj = nn.Linear(hidden_size, hidden_size)
        self.reg_head = nn.Linear(hidden_size, 1)
        self.trans_gate_logit = nn.Parameter(torch.tensor(-8.0))

    def forward(self, x):
        x = self.input_layer(x)
        lstm_bf, transformer_input = self.lstm_pe(x)
        transformer_out = self.transformer_encoder(transformer_input)
        trans_bt = transformer_out.permute(1, 0, 2).contiguous()
        gate = torch.sigmoid(self.trans_gate_logit)
        fused = self.residual_proj(lstm_bf) + trans_bt * gate
        return self.reg_head(fused[:, -1, :])

def trend_aware_regression_loss(
    pred,
    target,
    huber_delta=1.0,
    trend_weight=0.25,
):
    p = pred.reshape(-1)
    t = target.reshape(-1)
    base = F.smooth_l1_loss(p, t, beta=huber_delta, reduction='mean')
    if p.numel() < 2 or trend_weight <= 0:
        return base
    dp = p[1:] - p[:-1]
    dt = t[1:] - t[:-1]
    eps = 1e-5
    mask = dt.abs() > eps
    if not mask.any():
        return base
    directional = F.relu(-(dp * dt)[mask]).mean()
    return base + trend_weight * directional

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

def load_data_1(dataset_name='Texas', feature_config='all', window_size=24, horizon=1):
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

def load_data_enhanced(dataset_name='Texas', window_size=48, shuffle_train=True):
    possible_paths = [
        f'./data/{dataset_name}/data.csv',
        f'data/{dataset_name}/data.csv',
        'data.csv',
        r'E:\22-SECP\code\data\Texas\data.csv'
    ]
    path = None
    for p in possible_paths:
        if os.path.exists(p):
            path = p
            break

    if path is None:
        raise FileNotFoundError(f"find none")
    print(f": {path}")

    try:
        df = pd.read_csv(path)
        if 'WindPower' not in df.columns:
            df = pd.read_csv(path, header=None)
            df = df.iloc[:, [0, 1]]
            df.columns = ['Date', 'WindPower']
    except Exception as e:
        raise ValueError(f"CSV 读取失败: {e}")

    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df.dropna(subset=['Date'], inplace=True)
    df = df.sort_values('Date').set_index('Date')
    df['WindPower'] = pd.to_numeric(df['WindPower'], errors='coerce')
    df['WindPower'] = df['WindPower'].interpolate(method='linear').fillna(method='bfill')

    df['Lag1'] = df['WindPower'].shift(1)
    df['Lag2'] = df['WindPower'].shift(2)
    df['Lag3'] = df['WindPower'].shift(3)

    df['Rolling_Mean_3'] = df['WindPower'].rolling(window=3).mean()

    df['Diff1'] = df['WindPower'].diff()

    df['Hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24.0)
    df['Hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24.0)
    df['Month_sin'] = np.sin(2 * np.pi * (df.index.month - 1) / 12.0)
    df['Month_cos'] = np.cos(2 * np.pi * (df.index.month - 1) / 12.0)

    df.dropna(inplace=True)

    feature_cols = [
        'WindPower',
        'Lag1', 'Lag2', 'Lag3', 'Rolling_Mean_3', 'Diff1',
        'Hour_sin', 'Hour_cos', 'Month_sin', 'Month_cos'
    ]

    print(f" {len(feature_cols)},  {len(df)}")

    X_raw = df[feature_cols].values.astype(np.float32)
    y_raw = df[['WindPower']].values.astype(np.float32)

    if len(X_raw) < window_size + 10:
        raise ValueError("lower")

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_scaled = scaler_X.fit_transform(X_raw)
    y_scaled = scaler_y.fit_transform(y_raw)

    full_dataset = TimeSeriesDataset(X_scaled, y_scaled, window_size=window_size)

    train_size = int(len(full_dataset) * 0.8)

    train_indices = list(range(train_size))
    val_indices = list(range(train_size, len(full_dataset)))

    train_db = torch.utils.data.Subset(full_dataset, train_indices)
    val_db = torch.utils.data.Subset(full_dataset, val_indices)

    train_loader = DataLoader(
        train_db, batch_size=64, shuffle=shuffle_train, drop_last=True
    )
    val_loader = DataLoader(val_db, batch_size=64, shuffle=False)

    return train_loader, val_loader, len(feature_cols), scaler_y

def train_regressor(model, train_loader, val_loader, config, device):
    model = model.to(device)
    use_trend = bool(config.get('use_trend_aware_loss', False))
    trend_w = float(config.get('trend_loss_weight', 0.25))
    hub_d = float(config.get('huber_delta', 1.0))
    criterion = nn.MSELoss()
    optimizer = getattr(optim, config['optimizer'])(
        model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=config['scheduler_patience'])

    def batch_loss(out, y_b):
        if use_trend:
            return trend_aware_regression_loss(
                out, y_b, huber_delta=hub_d, trend_weight=trend_w
            )
        return criterion(out, y_b)

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
            loss = batch_loss(out, y)
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
                loss = batch_loss(out, y)
                val_loss += loss.item() * x.size(0)
        val_loss /= len(val_loader.dataset)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(optimizer.param_groups[0]['lr'])

        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        train_losses.append(avg_train)
        val_losses.append(avg_val)

        #print(f"Epoch {epoch + 1:02d}: TrainLoss={avg_train:.6f}, ValLoss={avg_val:.6f}")

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

        'hidden_size': 128,
        'nhead': 4,
        'num_layers': 4,
        'dropout': 0.15,

        'lr': 0.0005,
        'epochs': 300,
        'patience': 40,
        'weight_decay': 1e-5,
        'optimizer': 'AdamW',
        'scheduler_patience': 5,
        'clip_norm': 1.0,
        'save_path': 'best_iftt.pth',
        'use_trend_aware_loss': True,
        'trend_loss_weight': 0.25,
        'huber_delta': 1.0,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    print("=" * 60)
    print("LSTM-TRANSOFRMER")
    print("=" * 60)
    for k, v in config.items():
        print(f"  {k}: {v}")
    print("-" * 60)

    train_loader, val_loader, input_size = load_data_enhanced(
        'Texas',
        config['window_size'],
        shuffle_train=not config.get('use_trend_aware_loss', False),
    )
    print(f": {input_size}")

    model = LSTMTransformer(
        input_size=input_size,
        hidden_size=config['hidden_size'],
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
    print(f": {train_time:.2f} 秒")
    print("=" * 60)

if __name__ == "__main__":
    main()