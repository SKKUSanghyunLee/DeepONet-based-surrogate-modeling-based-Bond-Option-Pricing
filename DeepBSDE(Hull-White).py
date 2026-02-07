import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import norm
from scipy.interpolate import UnivariateSpline
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset, random_split
import copy
import random

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using Device: {device}")

try:
    df_full = pd.read_csv('Yield_curve(2010-2024).csv', index_col='Date', parse_dates=True)
    df_full = df_full.interpolate(method='linear').dropna(axis=1, how='any').sort_index()
    print("Successfully loaded 'Yield_curve(2010-2024).csv'.")
except FileNotFoundError:
    print("Error: 'Yield_curve(2010-2024).csv' not found.")
    exit()

maturities_np = np.array([float(m.split(' ')[0]) / 12 if 'Mo' in m else float(m.split(' ')[0]) for m in df_full.columns])
n_maturities = len(maturities_np)

N_STEPS = 10
SPLINE_SMOOTHING = 2e-3

def B_func_np(t, T, a):
    if abs(a) < 1e-6: return T - t
    return (1 - np.exp(-a * (T - t))) / a

def hull_white_call_option_price(K, t, S, T, r_t, a, sigma, maturities_np, yields_np):
    spline = UnivariateSpline(maturities_np, yields_np, s=SPLINE_SMOOTHING, k=3, ext=1)
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


def get_f0t(t, spline):
    if t < 1e-6: t = 1e-6
    y_t = spline(t)
    y_prime_t = spline.derivative(1)(t)
    return y_t + t * y_prime_t

def get_theta_t(t, a, sigma, spline):
    f_0_t = get_f0t(t, spline)
    t = max(t, 1e-6)
    
    y_prime_t = spline.derivative(1)(t)
    y_double_prime_t = spline.derivative(2)(t)
    f_prime_0_t = 2 * y_prime_t + t * y_double_prime_t
    
    sigma_term = (sigma**2 / (2 * a)) * (1 - np.exp(-2 * a * t)) if a > 1e-6 else (sigma**2) * t
    
    return f_prime_0_t + a * f_0_t + sigma_term

def get_ln_A_tT(t, T, a, sigma, spline):
    P0_T_y = spline(T)
    P0_t_y = spline(t)

    P0_T = np.exp(-P0_T_y * T) if np.isfinite(P0_T_y) else np.nan
    P0_t = np.exp(-P0_t_y * t) if np.isfinite(P0_t_y) else np.nan
    
    if np.isnan(P0_T) or np.isnan(P0_t) or P0_t < 1e-9:
        return np.nan
    
    f0_t = get_f0t(t, spline)
    B_t_T = B_func_np(t, T, a)
    
    sigma_term = (sigma**2 / (4 * a)) * (1 - np.exp(-2 * a * t)) * (B_t_T**2) if a > 1e-6 else 0.0
    
    with np.errstate(divide='ignore'):
        log_term = np.log(P0_T / P0_t)
    if np.isinf(log_term): return np.nan

    return log_term + B_t_T * f0_t - sigma_term


class DeepBSDE_Y0_Model(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, n_layers=6, dropout_rate=0.2):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.GELU(), nn.Dropout(dropout_rate)]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.GELU(), nn.Dropout(dropout_rate)])
        layers.append(nn.Linear(hidden_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return F.softplus(self.network(x))

class DeepBSDE_Z_Model(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, n_layers=4, dropout_rate=0.2):
        super().__init__()
        z_input_dim = input_dim + 2 
        
        layers = [nn.Linear(z_input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.GELU(), nn.Dropout(dropout_rate)]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.GELU(), nn.Dropout(dropout_rate)])
        layers.append(nn.Linear(hidden_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, t, r, y_params):
        z_input = torch.cat((t, r, y_params), dim=1)
        return self.network(z_input)


def generate_dataset_from_yields(df_yields, samples_per_day, device):
    all_features, all_labels = [], []
    for date in tqdm(df_yields.index, desc="Generating data from yield curves (HW, s=SPLINE_SMOTHING)"):
        yields_np = df_full.loc[date].values / 100.0
        if np.isnan(yields_np).any(): continue
        
        spline = UnivariateSpline(maturities_np, yields_np, s=SPLINE_SMOOTHING, k=3, ext=1)
        
        for _ in range(samples_per_day):
            a = np.random.uniform(0.05, 0.5)
            sigma = np.random.uniform(0.01, 0.15)
            S = np.random.uniform(0.5, 5.0)
            T = S + np.random.uniform(1.0, 10.0)
            
            try:
                forward_price = np.exp(-spline(T) * T) / np.exp(-spline(S) * S)
            except (ValueError, ZeroDivisionError, OverflowError):
                continue
            
            if not np.isfinite(forward_price) or forward_price < 1e-9:
                continue

            r0_current = spline(1e-6)
            K = forward_price * np.random.uniform(0.9, 1.1)
            option_price = hull_white_call_option_price(K, 0.0, S, T, r0_current, a, sigma, maturities_np, yields_np)
            
            if np.isfinite(option_price) and 1e-7 < option_price < 1.0:
                tau = T - S
                moneyness = K / forward_price
                
                dt_np_vec = S / N_STEPS
                theta_path_np = np.zeros(N_STEPS)
                for j in range(N_STEPS):
                    t_j_np = j * dt_np_vec
                    theta_path_np[j] = get_theta_t(t_j_np, a, sigma, spline)
                
                ln_A_S_T = get_ln_A_tT(S, T, a, sigma, spline)
                B_S_T = B_func_np(S, T, a)

                feature_vector = [S, T, K, tau, moneyness, a, sigma] + list(yields_np) + \
                                 [r0_current, ln_A_S_T, B_S_T] + list(theta_path_np)
                
                if not np.isfinite(theta_path_np).all() or not np.isfinite(ln_A_S_T): continue

                all_features.append(torch.tensor(feature_vector, dtype=torch.float32))
                all_labels.append(torch.tensor([option_price], dtype=torch.float32))

    if not all_features:
        print("Warning: No data generated.")
        return None
    
    return TensorDataset(torch.stack(all_features), torch.stack(all_labels))

SAMPLES_PER_DAY = 20
full_dataset = generate_dataset_from_yields(df_full, SAMPLES_PER_DAY, 'cpu')
if full_dataset is None:
    exit()
    
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

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, drop_last=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=512, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=512, pin_memory=True)

IDX_S, IDX_T, IDX_K = 0, 1, 2
IDX_A, IDX_SIGMA = 5, 6
IDX_YIELDS_END = 7 + n_maturities
IDX_R0 = IDX_YIELDS_END
IDX_LN_A = IDX_R0 + 1
IDX_B_ST = IDX_LN_A + 1
IDX_THETA_PATH_START = IDX_B_ST + 1

pure_input_dim = IDX_YIELDS_END

torch.manual_seed(SEED)
model_Y0 = DeepBSDE_Y0_Model(input_dim=pure_input_dim).to(device)
torch.manual_seed(SEED)
model_Z = DeepBSDE_Z_Model(input_dim=pure_input_dim).to(device)

all_params = list(model_Y0.parameters()) + list(model_Z.parameters())
optimizer = torch.optim.Adam(all_params, lr=3e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=2, eta_min=1e-7)
loss_fn = nn.MSELoss()

num_epochs = 1000
best_val_loss = float('inf')
best_model_weights = None
patience = 100
patience_counter = 0

for epoch in tqdm(range(num_epochs), desc="Training DeepBSDE(Hull-White)"):
    
    model_Y0.train()
    model_Z.train()
    
    for features_batch_cpu, labels_batch_cpu in train_loader:
        features_batch = features_batch_cpu.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        batch_size = features_batch.shape[0]
        
        S_batch = features_batch[:, IDX_S].unsqueeze(1)
        K_batch = features_batch[:, IDX_K].unsqueeze(1)
        a_batch = features_batch[:, IDX_A].unsqueeze(1)
        sigma_batch = features_batch[:, IDX_SIGMA].unsqueeze(1)
        
        r0_batch = features_batch[:, IDX_R0].unsqueeze(1)
        ln_A_ST_batch = features_batch[:, IDX_LN_A].unsqueeze(1)
        B_ST_batch = features_batch[:, IDX_B_ST].unsqueeze(1)
        theta_path_batch = features_batch[:, IDX_THETA_PATH_START:IDX_THETA_PATH_START + N_STEPS]

        dt_batch = S_batch / N_STEPS
        
        Z_batch_shocks = torch.normal(mean=0.0, std=1.0, size=(batch_size, N_STEPS), device=device)
        dW_batch = Z_batch_shocks * torch.sqrt(dt_batch)
        
        pure_features_batch = features_batch[:, :IDX_YIELDS_END]
        
        Y_i = model_Y0(pure_features_batch)
        r_i = r0_batch
        
        for j in range(N_STEPS):
            t_j_val = (j * dt_batch)

            Z_i = model_Z(t_j_val, r_i, pure_features_batch)

            dW_j = dW_batch[:, j].unsqueeze(1)
            Z_j_shock = Z_batch_shocks[:, j].unsqueeze(1)

            Y_i = Y_i + (r_i * Y_i) * dt_batch + Z_i * dW_j
        
            theta_j = theta_path_batch[:, j].unsqueeze(1)

            exp_a_dt = torch.exp(-a_batch * dt_batch)
            exp_2a_dt = torch.exp(-2 * a_batch * dt_batch)
            
            one_minus_exp_a_dt = 1.0 - exp_a_dt
            mean_drift_term = torch.where(a_batch.abs() < 1e-6,
                                          theta_j * dt_batch,
                                          (theta_j / a_batch) * one_minus_exp_a_dt)
            
            var_term_inside_sqrt = torch.where(a_batch.abs() < 1e-6,
                                               dt_batch,
                                               (1.0 - exp_2a_dt) / (2.0 * a_batch))
            
            std_dev_term = sigma_batch * torch.sqrt(F.relu(var_term_inside_sqrt))
            
            r_i = r_i * exp_a_dt + mean_drift_term + std_dev_term * Z_j_shock
            
        Y_S_pred = Y_i

        P_S_T = torch.exp(ln_A_ST_batch) * torch.exp(-B_ST_batch * r_i)
        
        Y_S_target = F.relu(P_S_T - K_batch)

        valid_mask = torch.isfinite(Y_S_pred) & torch.isfinite(Y_S_target)
        
        num_valid = valid_mask.sum()
        
        if num_valid > 0:
            loss_total = loss_fn(Y_S_pred[valid_mask], Y_S_target[valid_mask])
            
            if not torch.isnan(loss_total):
                loss_total.backward()
                
                torch.nn.utils.clip_grad_norm_(model_Y0.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(model_Z.parameters(), 1.0)
                
                optimizer.step()
        
    model_Y0.eval()
    current_val_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for features_batch_cpu, labels_batch_cpu in val_loader:
            features_batch = features_batch_cpu.to(device, non_blocking=True)
            labels_batch = labels_batch_cpu.to(device, non_blocking=True)
            
            pure_features_batch = features_batch[:, :IDX_YIELDS_END]
            preds_val = model_Y0(pure_features_batch)
            
            loss_val = loss_fn(preds_val, labels_batch)
            
            if not torch.isnan(loss_val):
                current_val_loss += loss_val.item() * features_batch.size(0)
                total_samples += features_batch.size(0)
    
    if total_samples > 0:
        current_val_loss /= total_samples
    else:
        current_val_loss = 0.0
    
    scheduler.step()

    if current_val_loss < best_val_loss:
        best_val_loss = current_val_loss
        best_model_weights = copy.deepcopy(model_Y0.state_dict())
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print(f"\nEarly stopping at epoch {epoch + 1}")
        break

print(f"Training finished. Best validation MSE: {best_val_loss:.8f}")

model_Y0.load_state_dict(best_model_weights)
model_Y0.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for features_batch_cpu, labels_batch_cpu in test_loader:
        features_batch = features_batch_cpu.to(device, non_blocking=True)
        labels_batch = labels_batch_cpu.to(device, non_blocking=True)

        pure_features_batch = features_batch[:, :IDX_YIELDS_END]
        preds = model_Y0(pure_features_batch)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels_batch.cpu().numpy())

all_preds = np.concatenate(all_preds)
all_labels = np.concatenate(all_labels)

valid_indices = np.isfinite(all_preds.flatten()) & np.isfinite(all_labels.flatten())
all_preds = all_preds[valid_indices]
all_labels = all_labels[valid_indices]

r2 = r2_score(all_labels, all_preds)
final_mse = mean_squared_error(all_labels, all_preds)

print("\n" + "="*50)
print(f"FINAL DeepBSDE RESULTS (Hull-White)")
print("="*50)
print(f"Test Set MSE: {final_mse:.8f}")
print(f"Test Set R2 Score: {r2:.4f}")
print("="*50)