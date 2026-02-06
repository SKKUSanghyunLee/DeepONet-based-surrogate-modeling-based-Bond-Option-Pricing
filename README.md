
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import norm
from scipy.interpolate import UnivariateSpline
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset, random_split
import copy
import random

SEED =42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED) 
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

df_full = pd.read_csv('Yield_curve(2010-2024).csv', index_col='Date', parse_dates=True)
df_full = df_full.interpolate(method='linear').dropna(axis=1, how='any').sort_index()
SPLINE_SMOOTHING = 2e-3

def B_func_np(t, T, a):
    if abs(a) < 1e-6: return T - t
    return (1 - np.exp(-a * (T - t))) / a

def hull_white_call_option_price(K, t, S, T, r_t, a, sigma, maturities_np, yields_np):
    spline = UnivariateSpline(maturities_np, yields_np, s=SPLINE_SMOOTHING, ext=1)
    def P_np(tau):
        y = spline(tau)
        return np.exp(-y * tau)
    P_t_S = P_np(S - t)
    P_t_T = P_np(T - t)
    if np.isnan(P_t_S) or np.isnan(P_t_T): return np.nan
    sigma_p_term = (sigma**2 / (2 * a)) * (1 - np.exp(-2 * a * (S - t))) if a > 1e-6 else (sigma**2) * (S - t)
    sigma_p = np.sqrt(np.maximum(0, sigma_p_term)) * B_func_np(S, T, a)
    if sigma_p < 1e-9: return max(0.0, P_t_T - K * P_t_S)
    with np.errstate(all='ignore'):
        d1 = (np.log(P_t_T / (K * P_t_S)) / sigma_p) + 0.5 * sigma_p
        d2 = d1 - sigma_p
    if not np.isfinite(d1) or not np.isfinite(d2): return np.nan
    return P_t_T * norm.cdf(d1) - K * P_t_S * norm.cdf(d2)

def generate_dataset_from_yields(df_yields, samples_per_day, device):
    maturities_np = np.array([float(m.split(' ')[0]) / 12 if 'Mo' in m else float(m.split(' ')[0]) for m in df_yields.columns])
    branch_data, trunk_data, labels = [], [], []
    for date in tqdm(df_yields.index, desc="Generating data from yield curves"):
        yields_np = df_yields.loc[date].values / 100.0
        if np.isnan(yields_np).any(): continue
        branch_input = torch.tensor(yields_np, dtype=torch.float32, device=device)
        spline = UnivariateSpline(maturities_np, yields_np, s=SPLINE_SMOOTHING, ext=1)
        r0_current = spline(1e-6)
        for _ in range(samples_per_day):
            a = np.random.uniform(0.05, 0.5)
            sigma = np.random.uniform(0.01, 0.15)
            S = np.random.uniform(0.5, 5.0)
            T = S + np.random.uniform(1.0, 10.0)
            y_T = spline(T)
            P0_T = np.exp(-y_T * T)
            if P0_T <= 1e-8: continue
            forward_price = P0_T / np.exp(-spline(S) * S)
            K = forward_price * np.random.uniform(0.9, 1.1)
            option_price = hull_white_call_option_price(K, 0.0, S, T, r0_current, a, sigma, maturities_np, yields_np)
            if np.isfinite(option_price) and 1e-7 < option_price < 1.0:
                tau = T - S
                moneyness = K / forward_price
                branch_data.append(branch_input)
                trunk_data.append(torch.tensor([S, T, K, tau, moneyness, a, sigma], device=device, dtype=torch.float32))
                labels.append(torch.tensor([option_price], device=device, dtype=torch.float32))
    if not branch_data: return None
    return TensorDataset(torch.stack(branch_data), torch.stack(trunk_data), torch.stack(labels))

def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None: nn.init.constant_(m.bias, 0.01)

class BranchNetCNN(nn.Module):
    def __init__(self, in_channels, p_dim):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32), nn.ReLU(), nn.AvgPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(), nn.AvgPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(), nn.AdaptiveAvgPool1d(1)
        )
        self.fc_layers = nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, p_dim))
        self.apply(init_weights)
    def forward(self, u):
        u = u.unsqueeze(1)
        x = self.conv_layers(u)
        x = torch.flatten(x, 1)
        return self.fc_layers(x)

class TrunkNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, p_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim * 2), nn.BatchNorm1d(hidden_dim * 2), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, p_dim))
        self.apply(init_weights)
    def forward(self, y): return self.network(y)

class DeepONet(nn.Module):
    def __init__(self, branch_net, trunk_net):
        super().__init__()
        self.branch_net = branch_net
        self.trunk_net = trunk_net
        self.bias = nn.Parameter(torch.full((1,), 0.1))
    def forward(self, u_y, y):
        branch_out = self.branch_net(u_y)
        trunk_out = self.trunk_net(y)
        x = torch.sum(branch_out * trunk_out, dim=1, keepdim=True) + self.bias
        return F.softplus(x)

SAMPLES_PER_DAY = 20
full_dataset = generate_dataset_from_yields(df_full, SAMPLES_PER_DAY, device)
print(f"\nTotal dataset size: {len(full_dataset)}")

dataset_size = len(full_dataset)
train_val_size = int(0.85 * dataset_size)
test_size = dataset_size - train_val_size
train_val_dataset, test_dataset = random_split(full_dataset, [train_val_size, test_size], generator=torch.Generator().manual_seed(42))

val_size = int(0.18 * train_val_size) 
train_size = train_val_size - val_size
train_dataset, val_dataset = random_split(train_val_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Test samples: {len(test_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=512)
test_loader = DataLoader(test_dataset, batch_size=512)

branch_input_dim = 1
trunk_input_dim = 7
p_dim = 256
branch_net = BranchNetCNN(in_channels=branch_input_dim, p_dim=p_dim).to(device)
trunk_net = TrunkNet(input_dim=trunk_input_dim, hidden_dim=256, p_dim=p_dim).to(device)
model = DeepONet(branch_net, trunk_net).to(device)
model.apply(init_weights)

optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=2, eta_min=1e-7)
loss_fn = nn.MSELoss()

num_epochs = 1000
best_val_loss = float('inf')
best_model_weights = None
patience = 100
patience_counter = 0

for epoch in tqdm(range(num_epochs), desc="Training DeepONet(Hull-White)"):
    model.train()
    for b_batch, t_batch, l_batch in train_loader:
        optimizer.zero_grad()
        preds = model(b_batch, t_batch).view(-1, 1)
        loss = loss_fn(preds, l_batch)
        if torch.isnan(loss): continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    
    model.eval()
    current_val_loss = 0.0
    with torch.no_grad():
        for b_batch, t_batch, l_batch in val_loader:
            preds = model(b_batch, t_batch).view(-1, 1)
            loss = loss_fn(preds, l_batch)
            current_val_loss += loss.item() * b_batch.size(0)
    
    if len(val_dataset) > 0:
        current_val_loss /= len(val_dataset)
    
    scheduler.step()

    if current_val_loss < best_val_loss:
        best_val_loss = current_val_loss
        best_model_weights = copy.deepcopy(model.state_dict())
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print(f"\nEarly stopping at epoch {epoch + 1}")
        break

print(f"Training finished. Best validation MSE: {best_val_loss:.8f}")

model.load_state_dict(best_model_weights)
model.eval()

test_loss = 0.0
all_preds = []
all_labels = []

with torch.no_grad():
    for b_batch, t_batch, l_batch in test_loader:
        preds = model(b_batch, t_batch).view(-1, 1)
        loss = loss_fn(preds, l_batch)
        test_loss += loss.item() * b_batch.size(0)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(l_batch.cpu().numpy())

test_loss /= len(test_dataset)

all_preds = np.concatenate(all_preds, axis=0)
all_labels = np.concatenate(all_labels, axis=0)

r2 = r2_score(all_labels, all_preds)

print("\n" + "="*50)'''
print("FINAL TEST RESULTS (Hull-White DeepONet)")
print("="*50)
print(f"Test Set MSE: {test_loss:.8f}")
print(f"Test Set R2 Score: {r2:.4f}")
