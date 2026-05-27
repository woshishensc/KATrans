import os
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from xgboost import XGBRegressor
from iftt_model import IFTTRegressor
from informer_model import InformerRegressor
from lstm_model import LSTMModel
from lstm_transformer_model import LSTMTransformer
from patchtst_model import PatchTSTRegressor
from transformer_model import TransformerModel

warnings.filterwarnings('ignore')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
COMPARE_MODELS = ('LSTM', 'Transformer', 'LSTMTransformer', 'IFTT', 'Informer', 'PatchTST')
EXP1_COMPARE_MODELS = COMPARE_MODELS + ('RF', 'XGBoost')
TREE_MODELS = frozenset({'RF', 'XGBoost'})
DEFAULT_DATASETS = ('Long', 'Texas', 'Wind')
EXP2_WINDOW_MODELS = ('LSTMTransformer', 'IFTT', 'Transformer')

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, ws, hs):
        self.X, self.y, self.ws, self.hs = X, y, ws, hs

    def __len__(self):
        return len(self.X) - self.ws - self.hs + 1

    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.X[idx: idx + self.ws]),
            torch.FloatTensor([self.y[idx + self.ws + self.hs - 1]]),
        )

def _patchtst_effective_window(cfg, dataset_name):
    path = os.path.join('data', dataset_name, 'data.csv')
    if not os.path.isfile(path):
        path = 'data.csv'
    if not os.path.isfile(path):
        return int(cfg.get('window_size', 24))
    n = len(pd.read_csv(path))
    want = int(cfg.get('patchtst_window_size', 336))
    reserve = int(cfg.get('patchtst_min_tail_rows', 200))
    cap = max(96, n - reserve)
    return max(24, min(want, cap))

def load_data_master(
    dataset_name='Texas',
    ws=24,
    hs=1,
    feat_mode='proposed',
    train_ratio=0.8,
    shuffle_train=True,
    pd_smooth_window=3,
):
    path = os.path.join('data', dataset_name, 'data.csv')
    if not os.path.exists(path):
        path = 'data.csv'
    df = pd.read_csv(path)
    df = df.iloc[:, [0, 1]].copy()
    df.columns = ['Date', 'Power']
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date']).sort_values('Date').set_index('Date')
    df['Power'] = pd.to_numeric(df['Power'], errors='coerce').interpolate().bfill()

    if pd_smooth_window and int(pd_smooth_window) > 1:
        pw = int(pd_smooth_window)
        p_smooth = df['Power'].rolling(window=pw, min_periods=1).mean()
    else:
        p_smooth = df['Power']

    df['V'] = p_smooth.diff().fillna(0)
    df['A'] = df['V'].diff().fillna(0)
    df['H_sin'] = np.sin(2 * np.pi * df.index.hour / 24.0)
    df['H_cos'] = np.cos(2 * np.pi * df.index.hour / 24.0)

    if feat_mode == 'raw':
        cols = ['Power']
    elif feat_mode == 'v':
        cols = ['Power', 'V']
    else:
        cols = ['Power', 'V', 'A', 'H_sin', 'H_cos']

    N = len(df)
    n_samples = N - ws - hs + 1
    if n_samples < 2:
        raise ValueError(f': N={N}, ws={ws}, hs={hs}')

    tr_sz = max(1, min(int(n_samples * train_ratio), n_samples - 1))
    fit_end = tr_sz + ws + hs - 1
    sc_x, sc_y = StandardScaler(), StandardScaler()
    sc_x.fit(df.iloc[:fit_end][cols].values)
    sc_y.fit(df.iloc[:fit_end][['Power']].values)
    X_s = sc_x.transform(df[cols].values)
    y_s = sc_y.transform(df[['Power']].values).flatten()

    ds = TimeSeriesDataset(X_s, y_s, ws, hs)
    tl = DataLoader(
        torch.utils.data.Subset(ds, range(tr_sz)),
        batch_size=64,
        shuffle=shuffle_train,
    )
    vl = DataLoader(torch.utils.data.Subset(ds, range(tr_sz, len(ds))), batch_size=64, shuffle=False)
    return tl, vl, len(cols), sc_y, X_s, y_s, tr_sz


def build_model(name, input_dim, cfg):
    h = cfg['hidden_size']
    nhead = cfg.get('nhead', 4)
    nlayers = cfg.get('num_layers', 3)
    dropout = cfg.get('dropout', 0.2)
    ws = cfg.get('window_size', 24)
    if h % nhead != 0:
        raise ValueError(f'hidden_size={h} must be nhead={nhead} ')

    if name == 'LSTM':
        return LSTMModel(input_dim, hidden_size=h, num_layers=nlayers, output_size=1)
    if name == 'Transformer':
        return TransformerModel(input_dim, nhead=nhead, nhid=h, nlayers=nlayers, output_size=1)
    if name == 'LSTMTransformer':
        return LSTMTransformer(
            input_dim, hidden_size=h, nhead=nhead, num_layers=nlayers, dropout=dropout
        )
    if name == 'IFTT':
        return IFTTRegressor(
            input_dim,
            d_model=h,
            nhead=min(nhead, 8),
            lstm_layers=1,
            attn_layers=max(1, nlayers),
            dropout=dropout,
        )
    if name == 'Informer':
        return InformerRegressor(
            input_dim, d_model=h, nhead=nhead, num_layers=nlayers, dropout=dropout
        )
    if name == 'PatchTST':
        pl = cfg.get('patchtst_patch_len')
        st = cfg.get('patchtst_stride')
        if pl is not None:
            patch_len = int(pl)
        else:
            patch_len = 16 if ws >= 64 else (12 if ws >= 48 else 8)
        stride = int(st) if st is not None else max(1, patch_len // 2)
        patch_len = min(patch_len, max(4, ws // 2))
        return PatchTSTRegressor(
            input_dim,
            patch_len=patch_len,
            stride=stride,
            d_model=h,
            nhead=nhead,
            num_layers=nlayers,
            dropout=dropout,
        )
    raise ValueError(f'Unknow: {name}')

def _trend_aware_alpha(cfg):
    if 'trend_aware_alpha' in cfg:
        return float(cfg['trend_aware_alpha'])
    if 'trend_loss_alpha' in cfg:
        return float(cfg['trend_loss_alpha'])
    return float(cfg.get('delta_loss_alpha', 1.0))

class TrendAwareLoss(nn.Module):
    def __init__(self, alpha=1.0, value_kind='mse', huber_delta=1.0):
        super().__init__()
        self.alpha = alpha
        vk = str(value_kind).lower()
        if vk == 'huber':
            self.value_loss = nn.HuberLoss(delta=float(huber_delta))
        else:
            self.value_loss = nn.MSELoss()
        self.trend_loss = nn.MSELoss()

    def forward(self, pred, target, last_input):
        l_val = self.value_loss(pred, target)
        pred_diff = pred - last_input
        target_diff = target - last_input
        l_trend = self.trend_loss(pred_diff, target_diff)
        return l_val + self.alpha * l_trend

def train_engine(model, tl, vl, config):
    device = torch.device(config['device'])
    opt = optim.AdamW(model.parameters(), lr=config['lr'])
    alpha = _trend_aware_alpha(config)
    value_kind = str(config.get('trend_aware_value_loss', 'mse')).lower()
    huber_delta = float(config.get('huber_delta', 1.0))
    criterion = TrendAwareLoss(
        alpha=alpha,
        value_kind=value_kind,
        huber_delta=huber_delta,
    ).to(device)
    best_v = float('inf')
    best_s = None
    patience = 0

    for _ in range(config['epochs']):
        model.train()
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            last_hist = x[:, -1, 0:1]
            pred = model(x)
            opt.zero_grad()
            loss = criterion(pred, y, last_hist)
            loss.backward()
            opt.step()

        model.eval()
        v_l = 0.0
        with torch.no_grad():
            for x, y in vl:
                x, y = x.to(device), y.to(device)
                last_hist = x[:, -1, 0:1]
                pv = model(x)
                v_l += criterion(pv, y, last_hist).item() * x.size(0)
        v_l /= len(vl.dataset)

        if v_l < best_v:
            best_v = v_l
            best_s = {k: v.cpu() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= config['patience']:
                break

    model.load_state_dict(best_s)
    return model

def mean_directional_accuracy(y_true, y_pred, eps=1e-6):
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if y_true.size < 2:
        return float('nan')
    d_true = y_true[1:] - y_true[:-1]
    d_pred_from_anchor = y_pred[1:] - y_true[:-1]
    mask = np.abs(d_true) > eps
    if not np.any(mask):
        return float('nan')
    gt = np.sign(d_true[mask])
    gp = np.sign(d_pred_from_anchor[mask])
    return float(np.mean(gt == gp))

VAL_METRIC_KEYS = ('MAE', 'RMSE', 'MSE', 'MAPE', 'R2', 'MDA')

def compute_val_metrics(y_true, y_pred, eps_mda=1e-6, mape_floor=1e-6):
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))
    denom = np.maximum(np.abs(y_true), mape_floor)
    mape = float(np.mean(np.abs(err) / denom) * 100.0)
    y_bar = float(np.mean(y_true))
    ss_tot = float(np.sum((y_true - y_bar) ** 2))
    r2 = float(1.0 - np.sum(err ** 2) / ss_tot) if ss_tot > 1e-12 else float('nan')
    mda = mean_directional_accuracy(y_true, y_pred, eps=eps_mda)
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MSE': mse,
        'MAPE': mape,
        'R2': r2,
        'MDA': mda,
    }

def run_eval(model, vl, sc_y, device):
    model.eval()
    ps, ts = [], []
    with torch.no_grad():
        for x, y in vl:
            ps.append(model(x.to(device)).cpu().numpy())
            ts.append(y.numpy())
    p_r = sc_y.inverse_transform(np.concatenate(ps).reshape(-1, 1))
    t_r = sc_y.inverse_transform(np.concatenate(ts).reshape(-1, 1))
    return compute_val_metrics(t_r.ravel(), p_r.ravel())

def _windows_to_flat_arrays(X_s, y_s, ws, hs, tr_sz):
    n = len(X_s) - ws - hs + 1
    n_feat = X_s.shape[1]
    X_flat = np.empty((n, ws * n_feat), dtype=np.float64)
    y_vec = np.empty(n, dtype=np.float64)
    for i in range(n):
        X_flat[i] = X_s[i : i + ws].ravel()
        y_vec[i] = y_s[i + ws + hs - 1]
    return X_flat[:tr_sz], y_vec[:tr_sz], X_flat[tr_sz:], y_vec[tr_sz:]

def run_eval_sklearn_regressor(reg, X_va, y_va_scaled, sc_y):
    pred_s = reg.predict(X_va).reshape(-1, 1)
    t_r = sc_y.inverse_transform(y_va_scaled.reshape(-1, 1))
    p_r = sc_y.inverse_transform(pred_s)
    return compute_val_metrics(t_r.ravel(), p_r.ravel())

def build_sklearn_regressor(name, seed, cfg):
    if name == 'RF':
        return RandomForestRegressor(
            n_estimators=int(cfg.get('rf_n_estimators', 200)),
            max_depth=cfg.get('rf_max_depth', None),
            random_state=seed,
            n_jobs=-1,
        )
    if name == 'XGBoost':
        return XGBRegressor(
            n_estimators=int(cfg.get('xgb_n_estimators', 300)),
            max_depth=int(cfg.get('xgb_max_depth', 6)),
            learning_rate=float(cfg.get('xgb_learning_rate', 0.05)),
            subsample=float(cfg.get('xgb_subsample', 0.8)),
            colsample_bytree=float(cfg.get('xgb_colsample_bytree', 0.8)),
            random_state=seed,
            n_jobs=-1,
            verbosity=0,
        )
    raise ValueError(f'Unknow: {name}')

def _set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _column_order(df):
    first = ['Dataset', 'Model', 'Seed', 'Run', 'WS', 'Step', 'Mode']
    base_metrics = list(VAL_METRIC_KEYS)
    summary_metrics = [f'{m}_mean' for m in VAL_METRIC_KEYS] + [
        f'{m}_std' for m in VAL_METRIC_KEYS
    ]
    all_metric_names = set(base_metrics + summary_metrics)
    ordered = [c for c in first if c in df.columns]
    mid = [
        c
        for c in df.columns
        if c not in first and c not in all_metric_names
    ]
    ordered.extend(sorted(mid))
    ordered.extend([c for c in base_metrics if c in df.columns])
    ordered.extend([c for c in summary_metrics if c in df.columns])
    return df[[c for c in ordered if c in df.columns]]

def save_experiment_csv(rows, filename):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, filename)
    df = pd.DataFrame(rows)
    df = _column_order(df)
    df.to_csv(path, index=False, encoding='utf-8-sig')
    print(f'Done: {path}')

def _train_ratio(cfg):
    return float(cfg.get('train_ratio', 0.8))

def _datasets_list(cfg):
    return tuple(cfg.get('datasets', DEFAULT_DATASETS))

def exp1_compare_all(cfg):
    print('\n[EXP1] + RF + XGBoost 多 Seed）...')
    res = []
    n_seeds = int(cfg.get('exp1_n_seeds', 5))
    base = int(cfg.get('base_seed', 42))
    tr = _train_ratio(cfg)
    ws_default = cfg['window_size']
    pd_sw = int(cfg.get('pd_smooth_window', 3))
    for ds in _datasets_list(cfg):
        path = os.path.join('data', ds, 'data.csv')
        if not os.path.isfile(path):
            print(f': {path}')
            continue
        print(f'  [Dataset] {ds}')
        for mname in EXP1_COMPARE_MODELS:
            for s in range(n_seeds):
                seed = base + s
                _set_seed(seed)
                ws_m = (
                    _patchtst_effective_window(cfg, ds)
                    if mname == 'PatchTST'
                    else ws_default
                )
                tl, vl, dim, sc_y, X_s, y_s, tr_sz = load_data_master(
                    ds,
                    ws_m,
                    1,
                    'proposed',
                    train_ratio=tr,
                    shuffle_train=True,
                    pd_smooth_window=pd_sw,
                )
                print(f'    --- {mname} | ws={ws_m} | Seed={seed} ---')
                if mname in TREE_MODELS:
                    X_tr, y_tr, X_va, y_va = _windows_to_flat_arrays(
                        X_s, y_s, ws_m, 1, tr_sz
                    )
                    reg = build_sklearn_regressor(mname, seed, cfg)
                    reg.fit(X_tr, y_tr)
                    metrics = run_eval_sklearn_regressor(reg, X_va, y_va, sc_y)
                else:
                    cfg_run = {**cfg, 'window_size': ws_m, 'dataset': ds}
                    model = build_model(mname, dim, cfg_run).to(cfg['device'])
                    model = train_engine(model, tl, vl, cfg)
                    metrics = run_eval(model, vl, sc_y, cfg['device'])
                res.append({
                    'Dataset': ds,
                    'Model': mname,
                    'Seed': seed,
                    **metrics,
                })
    save_experiment_csv(res, 'EXP1_six_models_comparison_runs.csv')
    if res:
        df = pd.DataFrame(res)
        grp = df.groupby(['Dataset', 'Model'], as_index=False)
        agg_kw = {}
        for k in VAL_METRIC_KEYS:
            if k in df.columns:
                agg_kw[f'{k}_mean'] = (k, 'mean')
                agg_kw[f'{k}_std'] = (k, 'std')
        summ = grp.agg(**agg_kw)
        for c in summ.columns:
            if c.endswith('_std'):
                summ[c] = summ[c].fillna(0.0)
        save_experiment_csv(summ.to_dict('records'), 'EXP1_six_models_comparison_summary.csv')

def exp1_baseline_multi_seed(cfg):
    print('\n[EXP1b] LSTMTransformer ...')
    res = []
    model_name = 'LSTMTransformer'
    for ds in _datasets_list(cfg):
        path = os.path.join('data', ds, 'data.csv')
        if not os.path.isfile(path):
            print(f': {path}')
            continue
        print(f'  [Dataset] {ds}')
        for i in range(cfg.get('n_baseline_runs', 5)):
            _set_seed(cfg.get('base_seed', 42) + i)
            tl, vl, dim, sc_y, _, _, _ = load_data_master(
                ds,
                cfg['window_size'],
                1,
                feat_mode='proposed',
                train_ratio=_train_ratio(cfg),
                shuffle_train=True,
                pd_smooth_window=int(cfg.get('pd_smooth_window', 3)),
            )
            c = {**cfg, 'dataset': ds}
            m = build_model(model_name, dim, c).to(cfg['device'])
            m = train_engine(m, tl, vl, cfg)
            metrics = run_eval(m, vl, sc_y, cfg['device'])
            res.append({
                'Dataset': ds,
                'Model': model_name,
                'Run': i + 1,
                **metrics,
            })
    save_experiment_csv(res, 'EXP1b_LSTMTransformer_multi_seed.csv')

def exp2_window(cfg):
    print('\n[EXP2] Window Size (LSTMTransformer / IFTT / Transformer)...')
    res = []
    tr = _train_ratio(cfg)
    seed = int(cfg.get('base_seed', 42))
    for ds in _datasets_list(cfg):
        path = os.path.join('data', ds, 'data.csv')
        if not os.path.isfile(path):
            print(f': {path}')
            continue
        print(f'  [Dataset] {ds}')
        for model_name in EXP2_WINDOW_MODELS:
            for ws in [12, 24, 48, 72, 96]:
                _set_seed(seed)
                tl, vl, dim, sc_y, _, _, _ = load_data_master(
                    ds,
                    ws,
                    1,
                    feat_mode='proposed',
                    train_ratio=tr,
                    shuffle_train=True,
                    pd_smooth_window=int(cfg.get('pd_smooth_window', 3)),
                )
                c = {**cfg, 'window_size': ws, 'dataset': ds}
                print(f'    --- {model_name} | WS={ws} ---')
                m = build_model(model_name, dim, c).to(cfg['device'])
                m = train_engine(m, tl, vl, cfg)
                metrics = run_eval(m, vl, sc_y, cfg['device'])
                res.append({
                    'Dataset': ds,
                    'Model': model_name,
                    'WS': ws,
                    **metrics,
                })
    save_experiment_csv(res, 'EXP2_window_sensitivity.csv')

def exp3_horizon(cfg):
    print('\n[EXP3]  (Horizon)...')
    res = []
    tr = _train_ratio(cfg)
    seed = int(cfg.get('base_seed', 42))
    horizons = cfg.get('exp3_horizons', [1, 2, 4, 6])
    for ds in _datasets_list(cfg):
        path = os.path.join('data', ds, 'data.csv')
        if not os.path.isfile(path):
            print(f' : {path}')
            continue
        print(f'  [Dataset] {ds}')
        for mname in COMPARE_MODELS:
            for hs in horizons:
                _set_seed(seed)
                ws_m = (
                    _patchtst_effective_window(cfg, ds)
                    if mname == 'PatchTST'
                    else cfg['window_size']
                )
                tl, vl, dim, sc_y, _, _, _ = load_data_master(
                    ds,
                    ws_m,
                    hs,
                    feat_mode='proposed',
                    train_ratio=tr,
                    shuffle_train=True,
                    pd_smooth_window=int(cfg.get('pd_smooth_window', 3)),
                )
                c = {**cfg, 'window_size': ws_m, 'dataset': ds}
                print(f'    --- {mname} | ws={ws_m} | Step={hs} ---')
                m = build_model(mname, dim, c).to(cfg['device'])
                m = train_engine(m, tl, vl, cfg)
                metrics = run_eval(m, vl, sc_y, cfg['device'])
                res.append({
                    'Dataset': ds,
                    'Model': mname,
                    'Step': hs,
                    **metrics,
                })
    save_experiment_csv(res, 'EXP3_horizon_all_models.csv')

def exp4_ablation_pd(cfg):
    print('\n[EXP4] A: (PD) (LSTMTransformer)...')
    res = []
    model_name = 'LSTMTransformer'
    for ds in _datasets_list(cfg):
        path = os.path.join('data', ds, 'data.csv')
        if not os.path.isfile(path):
            print(f' : {path}')
            continue
        print(f'  [Dataset] {ds}')
        for mode in ['raw', 'v', 'proposed']:
            tl, vl, dim, sc_y, _, _, _ = load_data_master(
                ds,
                cfg['window_size'],
                1,
                mode,
                train_ratio=_train_ratio(cfg),
                shuffle_train=True,
                pd_smooth_window=int(cfg.get('pd_smooth_window', 3)),
            )
            c = {**cfg, 'dataset': ds}
            m = build_model(model_name, dim, c).to(cfg['device'])
            m = train_engine(m, tl, vl, cfg)
            metrics = run_eval(m, vl, sc_y, cfg['device'])
            res.append({
                'Dataset': ds,
                'Model': model_name,
                'Mode': mode,
                **metrics,
            })
    save_experiment_csv(res, 'EXP4_ablation_physics_features.csv')

def exp5_ablation_arch(cfg):
    print('\n[EXP5] B: (LSTM / Transformer / LSTMTransformer)...')
    res = []
    for ds in _datasets_list(cfg):
        path = os.path.join('data', ds, 'data.csv')
        if not os.path.isfile(path):
            print(f': {path}')
            continue
        print(f'  [Dataset] {ds}')
        for name in ('LSTM', 'Transformer', 'LSTMTransformer'):
            tl, vl, dim, sc_y, _, _, _ = load_data_master(
                ds,
                cfg['window_size'],
                1,
                'proposed',
                train_ratio=_train_ratio(cfg),
                shuffle_train=True,
                pd_smooth_window=int(cfg.get('pd_smooth_window', 3)),
            )
            c = {**cfg, 'window_size': cfg['window_size'], 'dataset': ds}
            m = build_model(name, dim, c).to(cfg['device'])
            m = train_engine(m, tl, vl, cfg)
            metrics = run_eval(m, vl, sc_y, cfg['device'])
            res.append({
                'Dataset': ds,
                'Model': name,
                **metrics,
            })
    save_experiment_csv(res, 'EXP5_ablation_architecture.csv')

if __name__ == '__main__':
    config = {
        'mda_quick_preview': False,
        'datasets': list(DEFAULT_DATASETS),
        'window_size': 24,
        'hidden_size': 128,
        'nhead': 4,
        'num_layers': 3,
        'dropout': 0.2,
        'lr': 0.0005,
        'epochs': 100,
        'patience': 10,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'n_baseline_runs': 5,
        'base_seed': 42,
        'train_ratio': 0.8,
        'exp1_n_seeds': 5,
        'exp3_horizons': [1, 2, 4, 6],
        'rf_n_estimators': 200,
        'rf_max_depth': None,
        'xgb_n_estimators': 300,
        'xgb_max_depth': 6,
        'xgb_learning_rate': 0.05,
        'xgb_subsample': 0.8,
        'xgb_colsample_bytree': 0.8,
        'pd_smooth_window': 3,
        'patchtst_window_size': 336,
        'patchtst_min_tail_rows': 200,
        'patchtst_patch_len': None,
        'patchtst_stride': None,
        'trend_aware_alpha': 0.5,
        'trend_aware_value_loss': 'mse',
        'huber_delta': 1.0,
    }
    if config.get('mda_quick_preview', False):
        config = {
            **config,
            'datasets': ['Texas'],
            'exp1_n_seeds': 1,
            'epochs': 25,
            'patience': 5,
        }
        exp1_compare_all(config)
        print(
            f'\n[SUCCESS] （仅 EXP1 / Texas / 1 seed），结果目录: {RESULTS_DIR}'
        )
        print('mda_quick_preview=False')
    else:
        exp1_compare_all(config)
        exp1_baseline_multi_seed(config)
        exp2_window(config)
        exp3_horizon(config)
        exp4_ablation_pd(config)
        exp5_ablation_arch(config)
        print(f'\n[SUCCESS] done : {RESULTS_DIR}')
