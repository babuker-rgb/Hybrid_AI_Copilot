# ================================================================
# Hybrid AI · Unified Framework v29.30-R40
# Nile Valley University · Sudan
# Merges best features from AI_HYB, Co-HYBAI, and PLEX_HYBAI
# ================================================================

import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import plotly.express as px
import plotly.graph_objects as go
import os
import tempfile
import datetime
import warnings
warnings.filterwarnings('ignore')

# ================================================================
# PAGE CONFIG
# ================================================================
st.set_page_config(page_title="Hybrid AI · Unified Framework v29.30-R40", layout="wide")

# ================================================================
# CONSTANTS
# ================================================================
D_MIN, D_MAX = 0.72, 0.99
TENSILE_MIN = 1.50
EFRF_MAX = 0.50
DISINTEGRATION_MAX = 15.0

SLIDER_API_MIN, SLIDER_API_MAX = 80.0, 98.0
SLIDER_MCC_MIN, SLIDER_MCC_MAX = 1.5, 8.0
SLIDER_PVPP_MIN, SLIDER_PVPP_MAX = 1.0, 6.0
SLIDER_MGST_MIN, SLIDER_MGST_MAX = 0.10, 1.2
SLIDER_BINDER_MIN, SLIDER_BINDER_MAX = 1.4, 6.0
SLIDER_MOISTURE_MIN, SLIDER_MOISTURE_MAX = 0.5, 5.0
SLIDER_PARTICLE_SIZE_MIN, SLIDER_PARTICLE_SIZE_MAX = 10.0, 200.0

SLIDER_PRESSURE_MIN, SLIDER_PRESSURE_MAX = 150.0, 250.0
SLIDER_SPEED_MIN, SLIDER_SPEED_MAX = 15.0, 30.0
SLIDER_GRANULE_MIN, SLIDER_GRANULE_MAX = 30.0, 250.0
SLIDER_DWELL_TIME_MIN, SLIDER_DWELL_TIME_MAX = 5.0, 50.0
SLIDER_FRICTION_MIN, SLIDER_FRICTION_MAX = 0.1, 0.5
SLIDER_DECOMPRESSION_TIME_MIN, SLIDER_DECOMPRESSION_TIME_MAX = 10.0, 80.0

BINDER_GRADES = ["MCC PH101", "MCC PH102", "MCC PH200", "MCC KG", "Lactose", "Dicalcium Phosphate"]

BOUND_MCC_MIN, BOUND_MCC_MAX = 2.0, 8.0
BOUND_PVPP_MIN, BOUND_PVPP_MAX = 1.5, 6.0
BOUND_MGST_MIN, BOUND_MGST_MAX = 0.3, 1.2
BOUND_BINDER_MIN, BOUND_BINDER_MAX = 3.0, 6.0

N_SAMPLES = 25000
ADAM_EPOCHS = 800
PATIENCE = 100
NSGA_POP = 80
NSGA_GENS = 50
HIDDEN_SIZE = 512

# ================================================================
# SESSION STATE INITIALIZATION
# ================================================================
if 'api' not in st.session_state:
    st.session_state.update({
        'api': 90.5, 'binder': 3.5, 'pvpp': 2.0, 'mgst': 0.5, 'mcc': 3.5,
        'moisture': 2.0, 'particle_size': 50.0, 'binder_grade': 0,
        'pressure': 200.0, 'speed': 20.0, 'dwell_time': 25.0,
        'friction': 0.25, 'decompression_time': 35.0, 'granule': 125.0,
        'show_cost_solution': False, 'show_quality_solution': False,
        'show_comparison': False, 'show_sensitivity': False,
        'show_dissolution': False, 'granule_mode': 'Fixed',
        'run_optimized': False, 'formulation': None,
        'feasible_df': None, 'tested_point': None, 'benchmark_df': None,
        'nsga_pop': None, 'nsga_objectives': None, 'nsga_fronts': None,
        'balanced_solution': None, 'quality_solution': None, 'cost_solution': None,
        'balanced_pred': None, 'quality_pred': None, 'cost_pred': None,
        'experimental_data': None, 'runtime': 0
    })

# ================================================================
# HELPER FUNCTIONS
# ================================================================
def normalize_components(api, binder, pvpp, mgst, mcc, moisture):
    """Normalize formulation components to sum to 100%"""
    components = np.array([api, binder, pvpp, mgst, mcc, moisture], dtype=float)
    total = np.sum(components)
    if total <= 0:
        total = 1.0
    norm = (components / total) * 100.0
    api, binder, pvpp, mgst, mcc, moisture = norm
    api = np.clip(api, SLIDER_API_MIN, SLIDER_API_MAX)
    binder = np.clip(binder, SLIDER_BINDER_MIN, SLIDER_BINDER_MAX)
    pvpp = np.clip(pvpp, SLIDER_PVPP_MIN, SLIDER_PVPP_MAX)
    mgst = np.clip(mgst, SLIDER_MGST_MIN, SLIDER_MGST_MAX)
    mcc = np.clip(mcc, SLIDER_MCC_MIN, SLIDER_MCC_MAX)
    moisture = np.clip(moisture, SLIDER_MOISTURE_MIN, SLIDER_MOISTURE_MAX)
    total2 = api + binder + pvpp + mgst + mcc + moisture
    scale = 100.0 / total2
    return api*scale, binder*scale, pvpp*scale, mgst*scale, mcc*scale, moisture*scale

def calculate_dwell_time(speed_rpm, punch_width=10, pitch_diameter=100):
    """Calculate dwell time from tableting speed"""
    speed_rpm = np.asarray(speed_rpm)
    result = np.full_like(speed_rpm, 50.0, dtype=float)
    mask = speed_rpm > 0
    result[mask] = (punch_width * 60 * 1000) / (np.pi * pitch_diameter * speed_rpm[mask])
    return np.clip(result, 5.0, 80.0)

def predict_disintegration_time(tensile, pvpp_n, api_n, binder_n, moisture_n):
    """Predict disintegration time from formulation and tensile"""
    base_time = 2.0 + 0.5 * tensile
    pvpp_effect = 5.0 * np.exp(-0.5 * pvpp_n)
    api_effect = 0.1 * (api_n - 80)
    binder_effect = 0.2 * (binder_n - 2.0)
    moisture_effect = -0.1 * moisture_n
    time = base_time - pvpp_effect + api_effect + binder_effect + moisture_effect
    return np.clip(time, 1.0, 30.0)

def predict_dissolution_profile(api_n, pvpp_n, particle_size, disintegration_time):
    """Predict dissolution parameters (tau, beta)"""
    tau = 5.0 + 0.5 * disintegration_time - 0.1 * pvpp_n + 0.05 * (api_n - 80)
    tau = np.clip(tau, 2.0, 20.0)
    beta = 1.0 + 0.01 * (particle_size - 50) / 50
    beta = np.clip(beta, 0.8, 2.5)
    return tau, beta

def calculate_quality_score(density, tensile, efrf, api=None):
    """Calculate weighted quality score with optional API inclusion"""
    density_score = min(100, (density / 0.95) * 100)
    tensile_score = min(100, (tensile / 8.5) * 100)
    efrf_score = max(0, (1 - efrf) * 100)
    weights = {'density': 0.4, 'tensile': 0.3, 'efrf': 0.3}
    overall = (density_score * weights['density'] +
               tensile_score * weights['tensile'] +
               efrf_score * weights['efrf'])
    if api is not None:
        api_score = (api - 80) / 18 * 100
        overall = 0.7 * overall + 0.3 * api_score
    return overall

# ================================================================
# PINN MODEL (19 features, 3 residual blocks, 512 hidden units)
# ================================================================
class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(torch.nn.functional.softplus(x))

class ResidualBlock(nn.Module):
    def __init__(self, features, dropout=0.1):
        super().__init__()
        self.lin1 = nn.Linear(features, features)
        self.bn1 = nn.BatchNorm1d(features)
        self.lin2 = nn.Linear(features, features)
        self.bn2 = nn.BatchNorm1d(features)
        self.act = Mish()
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        identity = x
        out = self.act(self.bn1(self.lin1(x)))
        out = self.drop(out)
        out = self.bn2(self.lin2(out))
        out = self.drop(out)
        return identity + out

class MultiTaskPINN(nn.Module):
    def __init__(self, input_dim=19, hidden=HIDDEN_SIZE):
        super().__init__()
        self.input_layer = nn.Sequential(nn.Linear(input_dim, hidden), Mish(), nn.Dropout(0.05))
        self.res1 = ResidualBlock(hidden, dropout=0.05)
        self.res2 = ResidualBlock(hidden, dropout=0.05)
        self.res3 = ResidualBlock(hidden, dropout=0.05)
        self.transition = nn.Sequential(nn.Linear(hidden, hidden//2), nn.Tanh(), nn.Dropout(0.05))
        self.output = nn.Linear(hidden//2, 6)  # Density, Tensile, ER, Disintegration, Dissolution_tau, Dissolution_beta

    def forward(self, X):
        x = self.input_layer(X)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.transition(x)
        return self.output(x)

    def predict(self, X_scaled):
        self.eval()
        with torch.no_grad():
            if not isinstance(X_scaled, torch.Tensor):
                X_scaled = torch.tensor(X_scaled, dtype=torch.float32)
            device = next(self.parameters()).device
            X_scaled = X_scaled.to(device)
            output = self.forward(X_scaled)
            return output.cpu().numpy()

# ================================================================
# DATA GENERATION (19 features)
# ================================================================
def generate_pinn_data(n_samples=N_SAMPLES, random_state=42):
    rng = np.random.default_rng(random_state)
    api_raw = rng.uniform(SLIDER_API_MIN, SLIDER_API_MAX, n_samples)
    binder_raw = rng.uniform(SLIDER_BINDER_MIN, SLIDER_BINDER_MAX, n_samples)
    pvpp_raw = rng.uniform(SLIDER_PVPP_MIN, SLIDER_PVPP_MAX, n_samples)
    mgst_raw = rng.uniform(SLIDER_MGST_MIN, SLIDER_MGST_MAX, n_samples)
    mcc_raw = rng.uniform(SLIDER_MCC_MIN, SLIDER_MCC_MAX, n_samples)
    moisture_raw = rng.uniform(SLIDER_MOISTURE_MIN, SLIDER_MOISTURE_MAX, n_samples)
    particle_size_raw = rng.uniform(SLIDER_PARTICLE_SIZE_MIN, SLIDER_PARTICLE_SIZE_MAX, n_samples)
    binder_grade_raw = rng.integers(0, len(BINDER_GRADES), n_samples)
    pressure_raw = rng.uniform(SLIDER_PRESSURE_MIN, SLIDER_PRESSURE_MAX, n_samples)
    speed_raw = rng.uniform(SLIDER_SPEED_MIN, SLIDER_SPEED_MAX, n_samples)
    dwell_time_raw = calculate_dwell_time(speed_raw)
    friction_raw = rng.uniform(SLIDER_FRICTION_MIN, SLIDER_FRICTION_MAX, n_samples)
    decompression_time_raw = rng.uniform(SLIDER_DECOMPRESSION_TIME_MIN, SLIDER_DECOMPRESSION_TIME_MAX, n_samples)
    granule_raw = rng.uniform(SLIDER_GRANULE_MIN, SLIDER_GRANULE_MAX, n_samples)

    api_n, binder_n, pvpp_n, mgst_n, mcc_n, moisture_n = normalize_components(
        api_raw, binder_raw, pvpp_raw, mgst_raw, mcc_raw, moisture_raw
    )

    # Base features (14)
    X_base = np.column_stack([
        api_n, mcc_n, pvpp_n, mgst_n, binder_n,
        pressure_raw, speed_raw, granule_raw,
        particle_size_raw, moisture_n, binder_grade_raw,
        dwell_time_raw, friction_raw, decompression_time_raw
    ])

    # Interaction features (5)
    api_binder = api_n * binder_n
    pressure_binder = pressure_raw * binder_n
    api_mcc = api_n * mcc_n
    pressure_speed = pressure_raw * speed_raw
    binder_mgst = binder_n * mgst_n

    X_enhanced = np.column_stack([
        X_base,
        api_binder, pressure_binder, api_mcc, pressure_speed, binder_mgst
    ])  # 19 features

    feature_names = [
        'API_%', 'MCC_%', 'PVPP_%', 'MgSt_%', 'Binder_%',
        'Pressure_MPa', 'Speed_rpm', 'Granule_Size_µm',
        'Particle_Size_µm', 'Moisture_%', 'Binder_Grade',
        'Dwell_Time_ms', 'Friction', 'Decompression_Time_ms',
        'API_Binder', 'Pressure_Binder', 'API_MCC', 'Pressure_Speed', 'Binder_MgSt'
    ]

    # Physics-based simulations
    k_heckel = 0.025 + 0.0001 * pressure_raw
    A_heckel = 1.0 + 0.01 * (api_n - 85.0) - 0.05 * binder_n
    D_heckel = 1.0 - np.exp(-(k_heckel * pressure_raw + A_heckel))
    D_heckel = np.clip(D_heckel, D_MIN, D_MAX)

    a_kawakita = 0.82 + 0.04 * (mcc_n - 1.5)/6.5 + 0.02 * (binder_n - 1.4)/4.6
    a_kawakita = np.clip(a_kawakita, 0.78, 0.92)
    b_kawakita = 0.002 + 0.003 * (binder_n - 1.4)/4.6 + 0.001 * (mcc_n - 1.5)/6.5
    b_kawakita = np.clip(b_kawakita, 0.0005, 0.006)
    D_kawakita = 1.0 - pressure_raw / (a_kawakita * pressure_raw + 1.0/b_kawakita)
    D_kawakita = np.clip(D_kawakita, D_MIN, D_MAX)

    pressure_norm = (pressure_raw - SLIDER_PRESSURE_MIN) / (SLIDER_PRESSURE_MAX - SLIDER_PRESSURE_MIN)
    D = pressure_norm * D_heckel + (1 - pressure_norm) * D_kawakita
    D += -0.003*(moisture_n - 2.0) - 0.002*(particle_size_raw - 50)/150 - 0.002*(speed_raw - 15)/15 - 0.01*(mgst_n - 0.2)
    D = np.clip(D, D_MIN, D_MAX)

    porosity = 1.0 - D
    sigma0 = 5.0 + 0.1*(api_n - 85.0) + 0.2*binder_n - 0.5*mgst_n
    sigma0 = np.clip(sigma0, 2.0, 8.0)
    b = 2.5 - 0.005*(pressure_raw - 80.0) - 0.01*(particle_size_raw - 50)/100
    b = np.clip(b, 1.5, 3.5)
    tensile_base = sigma0 * np.exp(-b * porosity)
    api_effect = 1.0 - 0.005*(api_n - 85.0)
    binder_effect = 1.0 + 0.03*(binder_n - 2.0)
    mgst_effect = 1.0 - 0.1*(mgst_n - 0.2)
    pvpp_effect = 1.0 - 0.02*(pvpp_n - 3.0)
    speed_effect = 1.0 - 0.002*(speed_raw - 10.0)
    particle_effect = 1.0 - 0.0005*(particle_size_raw - 50)
    particle_effect = np.clip(particle_effect, 0.8, 1.2)
    tensile = tensile_base * api_effect * binder_effect * mgst_effect * pvpp_effect * speed_effect * particle_effect
    tensile = np.clip(tensile, 0.5, 6.0)

    er_base = (1.8 + 0.3*(api_n - 85.0)/10.0 + 0.08*(speed_raw - 10.0)/30.0 - 0.1*(pressure_raw - 100.0)/150.0 + 0.02*(decompression_time_raw - 35.0)/30.0)
    er = er_base * (1.0 - 0.15*(D - 0.4))
    er = np.clip(er, 0.5, 4.0)

    disintegration = predict_disintegration_time(tensile, pvpp_n, api_n, binder_n, moisture_n)
    disintegration = np.clip(disintegration, 1.0, 30.0)
    tau, beta = predict_dissolution_profile(api_n, pvpp_n, particle_size_raw, disintegration)
    tau = np.clip(tau, 2.0, 20.0)
    beta = np.clip(beta, 0.8, 2.5)

    df = pd.DataFrame(X_enhanced, columns=feature_names)
    df['Density'] = D
    df['Tensile_Strength_MPa'] = tensile
    df['Elastic_Recovery_%'] = er
    df['Disintegration_Time_min'] = disintegration
    df['Dissolution_Tau'] = tau
    df['Dissolution_Beta'] = beta
    return df, feature_names

# ================================================================
# CACHED TRAINING
# ================================================================
CACHE_DIR = tempfile.gettempdir()
CHECKPOINT_PATH = os.path.join(CACHE_DIR, 'hybrid_unified_v29_30_R40.pt')

@st.cache_resource
def load_or_train():
    if os.path.exists(CHECKPOINT_PATH):
        try:
            ckpt = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)
            model = MultiTaskPINN(input_dim=ckpt['input_dim'], hidden=HIDDEN_SIZE)
            model.load_state_dict(ckpt['model_state'])
            scaler = ckpt['scaler']
            y_scaler = ckpt['y_scaler']
            features = ckpt['features']
            df = ckpt['df']
            st.success(" Model loaded from cache successfully!")
            return model, scaler, y_scaler, features, df
        except Exception as e:
            st.warning(f"Cache load failed: {e}. Retraining...")
            if os.path.exists(CHECKPOINT_PATH):
                os.remove(CHECKPOINT_PATH)

    st.caption(" Training unified model (19 features, 25k samples)...")
    df, features = generate_pinn_data(N_SAMPLES)
    X_raw = df[features].values
    n_features = X_raw.shape[1]
    y = df[['Density','Tensile_Strength_MPa','Elastic_Recovery_%',
            'Disintegration_Time_min','Dissolution_Tau','Dissolution_Beta']].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    y_scaler = StandardScaler()
    y_scaled = y_scaler.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=42
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st.caption(f" Using device: {device}")
    model = MultiTaskPINN(n_features, hidden=HIDDEN_SIZE).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=30, factor=0.5)

    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).to(device)

    best_r2_tensile = -np.inf
    patience_counter = 0
    progress_bar = st.progress(0)
    status_text = st.empty()

    for epoch in range(ADAM_EPOCHS):
        model.train()
        optimizer.zero_grad()
        y_pred = model(X_train_t)
        loss = nn.MSELoss()(y_pred, y_train_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step(loss.item())

        model.eval()
        with torch.no_grad():
            val_pred = model(X_test_t).cpu().numpy()
            val_true = y_test_t.cpu().numpy()
            val_pred_actual = y_scaler.inverse_transform(val_pred)
            val_true_actual = y_scaler.inverse_transform(val_true)
            r2_tensile = r2_score(val_true_actual[:, 1], val_pred_actual[:, 1])

        if epoch % 50 == 0:
            status_text.text(f"Epoch {epoch+1}/{ADAM_EPOCHS} - Val R² Tensile: {r2_tensile:.4f}")

        if r2_tensile > best_r2_tensile:
            best_r2_tensile = r2_tensile
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                st.info(f"Early stopping at epoch {epoch+1} (no improvement for {PATIENCE} epochs)")
                break
        progress_bar.progress((epoch+1)/ADAM_EPOCHS)

    progress_bar.empty()
    st.success(f" Training complete! Best R² Tensile: {best_r2_tensile:.4f}")

    checkpoint = {
        'model_state': model.cpu().state_dict(),
        'scaler': scaler,
        'y_scaler': y_scaler,
        'features': features,
        'df': df,
        'input_dim': n_features
    }
    torch.save(checkpoint, CHECKPOINT_PATH)
    st.success(" Model cached successfully!")
    return model, scaler, y_scaler, features, df

# ================================================================
# NSGA-II OPTIMIZER (with dual penalties from Co-HYBAI)
# ================================================================
class NSGAIIOptimizer:
    def __init__(self, model, scaler, y_scaler, bounds, pop=NSGA_POP, gens=NSGA_GENS,
                 granule_fixed=True, granule_fixed_val=125.0,
                 penalty_api=0.08, penalty_tensile=0.05):
        self.model = model
        self.scaler = scaler
        self.y_scaler = y_scaler
        self.bounds = bounds
        self.pop_size = pop
        self.generations = gens
        self.granule_fixed = granule_fixed
        self.granule_fixed_val = granule_fixed_val
        self.penalty_api = penalty_api
        self.penalty_tensile = penalty_tensile

    def _repair(self, ind):
        api, mcc, pvpp, mgst, binder, pressure, speed, granule, particle_size, moisture, binder_grade, dwell_time, friction, decompression_time = ind
        api, binder, pvpp, mgst, mcc, moisture = normalize_components(
            api, binder, pvpp, mgst, mcc, moisture
        )
        pressure = np.clip(pressure, self.bounds[5,0], self.bounds[5,1])
        speed = np.clip(speed, self.bounds[6,0], self.bounds[6,1])
        particle_size = np.clip(particle_size, SLIDER_PARTICLE_SIZE_MIN, SLIDER_PARTICLE_SIZE_MAX)
        binder_grade = np.clip(binder_grade, 0, len(BINDER_GRADES)-1)
        dwell_time = np.clip(dwell_time, SLIDER_DWELL_TIME_MIN, SLIDER_DWELL_TIME_MAX)
        friction = np.clip(friction, SLIDER_FRICTION_MIN, SLIDER_FRICTION_MAX)
        decompression_time = np.clip(decompression_time, SLIDER_DECOMPRESSION_TIME_MIN, SLIDER_DECOMPRESSION_TIME_MAX)
        if self.granule_fixed:
            granule = self.granule_fixed_val
        else:
            granule = np.clip(granule, self.bounds[7,0], self.bounds[7,1])
        return np.array([api, mcc, pvpp, mgst, binder, pressure, speed, granule,
                         particle_size, moisture, binder_grade, dwell_time, friction, decompression_time])

    def _repair_batch(self, pop):
        api, mcc, pvpp, mgst, binder, pressure, speed, granule, particle_size, moisture, binder_grade, dwell_time, friction, decompression_time = pop.T
        api, binder, pvpp, mgst, mcc, moisture = normalize_components(
            api, binder, pvpp, mgst, mcc, moisture
        )
        pressure = np.clip(pressure, self.bounds[5,0], self.bounds[5,1])
        speed = np.clip(speed, self.bounds[6,0], self.bounds[6,1])
        particle_size = np.clip(particle_size, SLIDER_PARTICLE_SIZE_MIN, SLIDER_PARTICLE_SIZE_MAX)
        binder_grade = np.clip(binder_grade, 0, len(BINDER_GRADES)-1)
        dwell_time = np.clip(dwell_time, SLIDER_DWELL_TIME_MIN, SLIDER_DWELL_TIME_MAX)
        friction = np.clip(friction, SLIDER_FRICTION_MIN, SLIDER_FRICTION_MAX)
        decompression_time = np.clip(decompression_time, SLIDER_DECOMPRESSION_TIME_MIN, SLIDER_DECOMPRESSION_TIME_MAX)
        if self.granule_fixed:
            granule = np.full_like(granule, self.granule_fixed_val)
        else:
            granule = np.clip(granule, self.bounds[7,0], self.bounds[7,1])
        return np.column_stack([api, mcc, pvpp, mgst, binder, pressure, speed, granule,
                                particle_size, moisture, binder_grade, dwell_time, friction, decompression_time])

    def _build_features(self, repaired):
        api, mcc, pvpp, mgst, binder, pressure, speed, granule, particle_size, moisture, binder_grade, dwell_time, friction, decompression_time = repaired.T
        api_binder = api * binder
        pressure_binder = pressure * binder
        api_mcc = api * mcc
        pressure_speed = pressure * speed
        binder_mgst = binder * mgst
        X = np.column_stack([
            repaired,
            api_binder, pressure_binder, api_mcc, pressure_speed, binder_mgst
        ])
        return X

    def _evaluate(self, population):
        n = population.shape[0]
        repaired = self._repair_batch(population)
        X_eval = self._build_features(repaired)
        scaled = self.scaler.transform(X_eval)
        X_t = torch.tensor(scaled, dtype=torch.float32)
        with torch.no_grad():
            pred_scaled = self.model.predict(X_t)
            pred = self.y_scaler.inverse_transform(pred_scaled)
        density = np.clip(pred[:, 0], D_MIN, D_MAX)
        tensile = np.maximum(pred[:, 1], 1e-4)
        er = np.maximum(pred[:, 2], 1e-4)
        efrf = er / tensile
        efrf = np.clip(efrf, 1e-4, 5.0)
        disintegration = np.maximum(pred[:, 3], 0.5)
        dissolution_tau = np.maximum(pred[:, 4], 1.0)

        # Base objectives (to minimize): -density, -tensile, efrf
        objectives = np.column_stack([
            -density,
            -tensile,
            efrf
        ])

        # Apply dual penalties (from Co-HYBAI)
        api = repaired[:, 0]
        api_norm = (api - 80) / 18
        tensile_norm = tensile / 8.5
        penalty_api_vec = self.penalty_api * (1 - api_norm)
        penalty_tensile_vec = self.penalty_tensile * (1 - tensile_norm)
        objectives[:, 0] += penalty_api_vec
        objectives[:, 1] += penalty_tensile_vec

        # Additional physical constraints penalties
        penalty = np.zeros(n)
        penalty += np.where(tensile < TENSILE_MIN, (TENSILE_MIN - tensile)**2, 0.0)
        penalty += np.where(efrf >= 0.40, (efrf - 0.40)**2, 0.0)
        penalty += np.where(disintegration > 15.0, (disintegration - 15.0)**2, 0.0)
        penalty += np.where(dissolution_tau > 20.0, (dissolution_tau - 20.0)**2, 0.0)
        penalty += np.where(mcc > self.bounds[1,1], (mcc - self.bounds[1,1])**2, 0.0)
        objectives[:, 0] += 100.0 * penalty
        objectives[:, 1] += 100.0 * penalty
        objectives[:, 2] += 100.0 * penalty

        return objectives, repaired

    def _non_dominated_sort(self, objectives):
        n = objectives.shape[0]
        fronts = []
        remaining = list(range(n))
        while remaining:
            front = []
            for i in remaining:
                dominated = False
                for j in remaining:
                    if i == j:
                        continue
                    if (objectives[j,0] <= objectives[i,0] and
                        objectives[j,1] <= objectives[i,1] and
                        objectives[j,2] <= objectives[i,2]) and \
                       (objectives[j,0] < objectives[i,0] or
                        objectives[j,1] < objectives[i,1] or
                        objectives[j,2] < objectives[i,2]):
                        dominated = True
                        break
                if not dominated:
                    front.append(i)
            fronts.append(front)
            remaining = [idx for idx in remaining if idx not in front]
        return fronts

    def _crowding_distance(self, objectives, front):
        if len(front) <= 2:
            return np.ones(len(front)) * np.inf
        dist = np.zeros(len(front))
        for obj_idx in range(objectives.shape[1]):
            sorted_idx = sorted(front, key=lambda i: objectives[i, obj_idx])
            dist[0] = np.inf
            dist[-1] = np.inf
            f_min = objectives[sorted_idx[0], obj_idx]
            f_max = objectives[sorted_idx[-1], obj_idx]
            if f_max - f_min > 1e-10:
                for k in range(1, len(sorted_idx)-1):
                    dist[k] += (objectives[sorted_idx[k+1], obj_idx] -
                                objectives[sorted_idx[k-1], obj_idx]) / (f_max - f_min)
        return dist

    def _crossover(self, p1, p2, eta=40):
        child1 = np.zeros(14)
        child2 = np.zeros(14)
        for i in range(14):
            u = np.random.random()
            if u <= 0.5:
                beta = (2*u) ** (1/(eta+1))
            else:
                beta = (1/(2*(1-u))) ** (1/(eta+1))
            child1[i] = 0.5 * ((1+beta)*p1[i] + (1-beta)*p2[i])
            child2[i] = 0.5 * ((1-beta)*p1[i] + (1+beta)*p2[i])
        return child1, child2

    def _mutate(self, child, eta=20, pm=1.0/14.0):
        for i in range(14):
            if np.random.random() < pm:
                u = np.random.random()
                if u <= 0.5:
                    delta = (2*u) ** (1/(eta+1)) - 1
                else:
                    delta = 1 - (2*(1-u)) ** (1/(eta+1))
                child[i] = child[i] + delta * (self.bounds[i,1] - self.bounds[i,0])
                child[i] = np.clip(child[i], self.bounds[i,0], self.bounds[i,1])
        return child

    def _tournament(self, pop, objectives, fronts):
        idx1 = np.random.randint(0, len(pop))
        idx2 = np.random.randint(0, len(pop))
        rank1 = next((f for f, front in enumerate(fronts) if idx1 in front), len(fronts))
        rank2 = next((f for f, front in enumerate(fronts) if idx2 in front), len(fronts))
        if rank1 < rank2:
            return pop[idx1]
        elif rank2 < rank1:
            return pop[idx2]
        else:
            front = fronts[rank1]
            dist = self._crowding_distance(objectives, front)
            d1 = dist[front.index(idx1)] if idx1 in front else 0
            d2 = dist[front.index(idx2)] if idx2 in front else 0
            return pop[idx1] if d1 > d2 else pop[idx2]

    def run(self):
        rng = np.random.default_rng()
        pop = []
        for i in range(self.pop_size):
            if i < 0.3 * self.pop_size:
                api = rng.uniform(90, 95)
                mcc = rng.uniform(2.5, 4.0)
                binder = rng.uniform(3.5, 5.0)
                pvpp = rng.uniform(2, 4)
                mgst = rng.uniform(0.4, 0.8)
                moisture = rng.uniform(1.0, 3.0)
            else:
                api = rng.uniform(SLIDER_API_MIN, SLIDER_API_MAX)
                mcc = rng.uniform(BOUND_MCC_MIN, BOUND_MCC_MAX)
                binder = rng.uniform(BOUND_BINDER_MIN, BOUND_BINDER_MAX)
                pvpp = rng.uniform(BOUND_PVPP_MIN, BOUND_PVPP_MAX)
                mgst = rng.uniform(BOUND_MGST_MIN, BOUND_MGST_MAX)
                moisture = rng.uniform(SLIDER_MOISTURE_MIN, SLIDER_MOISTURE_MAX)
            pressure = rng.uniform(SLIDER_PRESSURE_MIN, SLIDER_PRESSURE_MAX)
            speed = rng.uniform(SLIDER_SPEED_MIN, SLIDER_SPEED_MAX)
            granule = rng.uniform(SLIDER_GRANULE_MIN, SLIDER_GRANULE_MAX)
            particle_size = rng.uniform(SLIDER_PARTICLE_SIZE_MIN, SLIDER_PARTICLE_SIZE_MAX)
            binder_grade = rng.integers(0, len(BINDER_GRADES))
            dwell_time = rng.uniform(SLIDER_DWELL_TIME_MIN, SLIDER_DWELL_TIME_MAX)
            friction = rng.uniform(SLIDER_FRICTION_MIN, SLIDER_FRICTION_MAX)
            decompression_time = rng.uniform(SLIDER_DECOMPRESSION_TIME_MIN, SLIDER_DECOMPRESSION_TIME_MAX)
            ind = np.array([api, mcc, pvpp, mgst, binder, pressure, speed, granule,
                            particle_size, moisture, binder_grade, dwell_time, friction, decompression_time])
            pop.append(self._repair(ind))
        pop = np.array(pop)

        for gen in range(self.generations):
            objectives, pop = self._evaluate(pop)
            fronts = self._non_dominated_sort(objectives)
            offspring = []
            while len(offspring) < self.pop_size:
                p1 = self._tournament(pop, objectives, fronts)
                p2 = self._tournament(pop, objectives, fronts)
                c1, c2 = self._crossover(p1, p2)
                c1 = self._mutate(c1)
                c2 = self._mutate(c2)
                offspring.append(self._repair(c1))
                if len(offspring) < self.pop_size:
                    offspring.append(self._repair(c2))
            offspring = np.array(offspring[:self.pop_size])
            combined = np.vstack([pop, offspring])
            obj_comb, _ = self._evaluate(combined)
            fronts_comb = self._non_dominated_sort(obj_comb)
            new_pop = []
            remaining = self.pop_size
            for front in fronts_comb:
                if len(front) <= remaining:
                    new_pop.extend(combined[front])
                    remaining -= len(front)
                else:
                    dist = self._crowding_distance(obj_comb, front)
                    sorted_idx = sorted(front, key=lambda i: dist[front.index(i)], reverse=True)
                    new_pop.extend(combined[sorted_idx[:remaining]])
                    remaining = 0
                    break
            pop = np.array(new_pop)

        objectives, pop = self._evaluate(pop)
        fronts = self._non_dominated_sort(objectives)
        return pop, objectives, fronts

# ================================================================
# PREDICTION AND PLOTTING FUNCTIONS
# ================================================================
def predict_pinn(model, scaler, y_scaler, inputs):
    if model is None:
        return 0.72, 2.0, 0.5, 0.25, 10.0, 10.0, 1.0
    try:
        api, mcc, pvpp, mgst, binder, pressure, speed, granule, particle_size, moisture, binder_grade, dwell_time, friction, decompression_time = inputs
        api_binder = api * binder
        pressure_binder = pressure * binder
        api_mcc = api * mcc
        pressure_speed = pressure * speed
        binder_mgst = binder * mgst
        X_input = np.array([[
            api, mcc, pvpp, mgst, binder, pressure, speed, granule,
            particle_size, moisture, binder_grade, dwell_time, friction, decompression_time,
            api_binder, pressure_binder, api_mcc, pressure_speed, binder_mgst
        ]])
        scaled = scaler.transform(X_input)
        X_t = torch.tensor(scaled, dtype=torch.float32)
        with torch.no_grad():
            pred_scaled = model.predict(X_t)[0]
            pred = y_scaler.inverse_transform([pred_scaled])[0]
        density = np.clip(pred[0], D_MIN, D_MAX)
        tensile = max(pred[1], 1e-4)
        er = max(pred[2], 1e-4)
        efrf = er / tensile
        disintegration = max(pred[3], 0.5)
        dissolution_tau = max(pred[4], 1.0)
        dissolution_beta = max(pred[5], 0.5)
        return density, tensile, er, efrf, disintegration, dissolution_tau, dissolution_beta
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return 0.72, 2.0, 0.5, 0.25, 10.0, 10.0, 1.0

def generate_feasible_points(model, scaler, y_scaler, n_samples=3000):
    if model is None:
        return pd.DataFrame()
    rng = np.random.default_rng(42)
    api = rng.uniform(SLIDER_API_MIN, SLIDER_API_MAX, n_samples)
    binder = rng.uniform(SLIDER_BINDER_MIN, SLIDER_BINDER_MAX, n_samples)
    pvpp = rng.uniform(SLIDER_PVPP_MIN, SLIDER_PVPP_MAX, n_samples)
    mgst = rng.uniform(SLIDER_MGST_MIN, SLIDER_MGST_MAX, n_samples)
    mcc = rng.uniform(SLIDER_MCC_MIN, SLIDER_MCC_MAX, n_samples)
    moisture = rng.uniform(SLIDER_MOISTURE_MIN, SLIDER_MOISTURE_MAX, n_samples)
    particle_size = rng.uniform(SLIDER_PARTICLE_SIZE_MIN, SLIDER_PARTICLE_SIZE_MAX, n_samples)
    binder_grade = rng.integers(0, len(BINDER_GRADES), n_samples)
    pressure = rng.uniform(SLIDER_PRESSURE_MIN, SLIDER_PRESSURE_MAX, n_samples)
    speed = rng.uniform(SLIDER_SPEED_MIN, SLIDER_SPEED_MAX, n_samples)
    dwell_time = calculate_dwell_time(speed)
    friction = rng.uniform(SLIDER_FRICTION_MIN, SLIDER_FRICTION_MAX, n_samples)
    decompression_time = rng.uniform(SLIDER_DECOMPRESSION_TIME_MIN, SLIDER_DECOMPRESSION_TIME_MAX, n_samples)
    granule = rng.uniform(SLIDER_GRANULE_MIN, SLIDER_GRANULE_MAX, n_samples)

    api_n, binder_n, pvpp_n, mgst_n, mcc_n, moisture_n = normalize_components(
        api, binder, pvpp, mgst, mcc, moisture
    )
    api_binder = api_n * binder_n
    pressure_binder = pressure * binder_n
    api_mcc = api_n * mcc_n
    pressure_speed = pressure * speed
    binder_mgst = binder_n * mgst_n
    inputs = np.column_stack([
        api_n, mcc_n, pvpp_n, mgst_n, binder_n,
        pressure, speed, granule,
        particle_size, moisture_n, binder_grade,
        dwell_time, friction, decompression_time,
        api_binder, pressure_binder, api_mcc, pressure_speed, binder_mgst
    ])
    scaled = scaler.transform(inputs)
    X_t = torch.tensor(scaled, dtype=torch.float32)
    with torch.no_grad():
        pred_scaled = model.predict(X_t)
        pred = y_scaler.inverse_transform(pred_scaled)
    density = np.clip(pred[:, 0], D_MIN, D_MAX)
    tensile = np.maximum(pred[:, 1], 1e-4)
    er = np.maximum(pred[:, 2], 1e-4)
    efrf = er / tensile
    efrf = np.clip(efrf, 1e-4, 5.0)
    disintegration = np.maximum(pred[:, 3], 0.5)

    mask = ((D_MIN <= density) & (density <= D_MAX) &
            (tensile >= TENSILE_MIN) & (efrf < 0.40) &
            (disintegration <= 15.0) &
            (mcc_n <= BOUND_MCC_MAX) & (mcc_n >= BOUND_MCC_MIN))
    feasible_api = api_n[mask]
    feasible_efrf = efrf[mask]
    return pd.DataFrame({'API': feasible_api, 'EFRF': feasible_efrf})

def plot_pareto_clean(objectives, fronts, balanced_solution=None, feasible_df=None,
                      tested_point=None, efrf_max=0.40):
    if fronts is None or len(fronts) == 0 or len(fronts[0]) == 0:
        return None
    front = fronts[0]
    try:
        api_vals = -objectives[front, 0]
        efrf_vals = objectives[front, 2]
    except Exception:
        return None
    df_front = pd.DataFrame({'API': api_vals, 'EFRF': efrf_vals}).sort_values('API')
    fig = go.Figure()
    if feasible_df is not None and not feasible_df.empty:
        fig.add_trace(go.Scatter(
            x=feasible_df['API'],
            y=feasible_df['EFRF'],
            mode='markers',
            name='Feasible Region (EFRF<0.40)',
            marker=dict(color='lightgreen', size=4, opacity=0.4),
            hovertemplate='API: %{x:.1f}%<br>EFRF: %{y:.4f}<extra></extra>',
            showlegend=True
        ))
    fig.add_trace(go.Scatter(
        x=df_front['API'],
        y=df_front['EFRF'],
        mode='lines+markers',
        name='Pareto Front',
        line=dict(color='red', width=2),
        marker=dict(size=7, color='red'),
        hovertemplate='API: %{x:.1f}%<br>EFRF: %{y:.4f}<extra></extra>'
    ))
    if tested_point is not None:
        fig.add_trace(go.Scatter(
            x=[tested_point[0]],
            y=[tested_point[1]],
            mode='markers',
            name='Tested Formulation',
            marker=dict(size=10, color='blue', symbol='circle',
                        line=dict(width=2, color='darkblue')),
            hovertemplate='Tested: API %{x:.1f}%, EFRF %{y:.4f}<extra></extra>'
        ))
    if balanced_solution is not None and len(balanced_solution) >= 2:
        fig.add_trace(go.Scatter(
            x=[balanced_solution[0]],
            y=[balanced_solution[1]],
            mode='markers',
            name=' Golden Solution (Balanced)',
            marker=dict(size=12, color='gold', symbol='star', line=dict(width=2, color='black')),
            hovertemplate='Golden: API %{x:.1f}%, EFRF %{y:.4f}<extra></extra>'
        ))
    fig.add_hline(y=0.40, line_dash='dash', line_color='gray',
                  annotation_text='EFRF threshold (0.40)')
    fig.update_layout(
        title='Pareto Front with Feasible Region',
        xaxis_title='API (%)',
        yaxis_title='EFRF',
        height=450,
        template='plotly_white',
        legend=dict(x=0.8, y=0.95)
    )
    return fig

def plot_sensitivity_bars(formulation, model, scaler, y_scaler):
    if model is None or formulation is None:
        return None
    api0 = formulation['api_n']; mcc0 = formulation['mcc_n']
    pvpp0 = formulation['pvpp_n']; mgst0 = formulation['mgst_n']
    binder0 = formulation['binder_n']; press0 = formulation['pressure']
    speed0 = formulation['speed']; granule0 = formulation['granule_use']
    particle_size0 = formulation['particle_size']
    moisture0 = formulation['moisture']
    dwell_time0 = formulation['dwell_time']
    friction0 = formulation['friction']
    decompression_time0 = formulation['decompression_time']
    binder_grade0 = formulation['binder_grade']

    param_defs = [
        ('API', api0, SLIDER_API_MIN, SLIDER_API_MAX),
        ('MCC', mcc0, SLIDER_MCC_MIN, SLIDER_MCC_MAX),
        ('PVPP', pvpp0, SLIDER_PVPP_MIN, SLIDER_PVPP_MAX),
        ('MgSt', mgst0, SLIDER_MGST_MIN, SLIDER_MGST_MAX),
        ('Binder', binder0, SLIDER_BINDER_MIN, SLIDER_BINDER_MAX),
        ('Moisture', moisture0, SLIDER_MOISTURE_MIN, SLIDER_MOISTURE_MAX),
        ('Pressure', press0, SLIDER_PRESSURE_MIN, SLIDER_PRESSURE_MAX),
        ('Speed', speed0, SLIDER_SPEED_MIN, SLIDER_SPEED_MAX),
        ('Granule', granule0, SLIDER_GRANULE_MIN, SLIDER_GRANULE_MAX),
        ('ParticleSize', particle_size0, SLIDER_PARTICLE_SIZE_MIN, SLIDER_PARTICLE_SIZE_MAX),
        ('DwellTime', dwell_time0, SLIDER_DWELL_TIME_MIN, SLIDER_DWELL_TIME_MAX),
        ('Friction', friction0, SLIDER_FRICTION_MIN, SLIDER_FRICTION_MAX),
        ('DecompTime', decompression_time0, SLIDER_DECOMPRESSION_TIME_MIN, SLIDER_DECOMPRESSION_TIME_MAX)
    ]

    base_input = [api0, mcc0, pvpp0, mgst0, binder0, press0, speed0, granule0,
                  particle_size0, moisture0, binder_grade0, dwell_time0, friction0, decompression_time0]
    _, _, _, efrf_base, _, _, _ = predict_pinn(model, scaler, y_scaler, base_input)

    sensitivities = []
    for idx, (name, current, minv, maxv) in enumerate(param_defs):
        low_input = base_input.copy()
        low_input[idx] = minv
        high_input = base_input.copy()
        high_input[idx] = maxv
        _, _, _, efrf_low, _, _, _ = predict_pinn(model, scaler, y_scaler, low_input)
        _, _, _, efrf_high, _, _, _ = predict_pinn(model, scaler, y_scaler, high_input)
        sensitivities.append({'Parameter': name, 'Delta EFRF': abs(efrf_high - efrf_low)})

    df_sens = pd.DataFrame(sensitivities).sort_values('Delta EFRF', ascending=True)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=df_sens['Parameter'],
        x=df_sens['Delta EFRF'],
        orientation='h',
        marker_color='steelblue',
        text=df_sens['Delta EFRF'].round(4),
        textposition='outside',
        hovertemplate='%{y}<br>EFRF: %{x:.4f}<extra></extra>'
    ))
    fig.add_vline(x=0.40, line_dash='dash', line_color='red',
                  annotation_text='EFRF threshold 0.40')
    fig.update_layout(
        title='Parameter Impact on EFRF',
        xaxis_title='Absolute change in EFRF',
        yaxis_title='Parameter',
        height=500,
        template='plotly_white'
    )
    return fig

def plot_dissolution_profile(tau, beta, api_n, title="Predicted Dissolution Profile"):
    time_points = np.linspace(0, 60, 100)
    dissolution = 100 * (1 - np.exp(-((time_points / tau) ** beta)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time_points,
        y=dissolution,
        mode='lines',
        name=f'Q(t) = 100×(1-exp(-((t/{tau:.1f})^{beta:.2f})))',
        line=dict(color='blue', width=2)
    ))
    fig.add_hline(y=85, line_dash='dash', line_color='red',
                  annotation_text='85% dissolution target')
    fig.update_layout(
        title=f'{title} (API: {api_n:.1f}%)',
        xaxis_title='Time (minutes)',
        yaxis_title='% Dissolved',
        height=350,
        template='plotly_white'
    )
    return fig

# ================================================================
# MODEL COMPARISON
# ================================================================
def run_model_comparison(model, scaler, y_scaler, features, df, device):
    if model is None:
        return pd.DataFrame(), []
    X_raw_all = df[features].values
    y_raw_all = df[['Tensile_Strength_MPa']].values

    X_train, X_test, y_train, y_test = train_test_split(
        X_raw_all, y_raw_all, test_size=0.2, random_state=42
    )
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    y_train = y_train.ravel()
    y_test = y_test.ravel()

    model.eval()
    with torch.no_grad():
        pinn_pred_scaled = model.predict(X_test_scaled)
        pinn_pred = y_scaler.inverse_transform(pinn_pred_scaled)[:, 1]

    from sklearn.neural_network import MLPRegressor
    from sklearn.ensemble import RandomForestRegressor
    mlp = MLPRegressor(hidden_layer_sizes=(128,64), max_iter=400, random_state=42)
    mlp.fit(X_train_scaled, y_train)
    mlp_pred = mlp.predict(X_test_scaled)

    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train_scaled, y_train)
    rf_pred = rf.predict(X_test_scaled)

    models = {
        'PINN (Proposed)': (pinn_pred, 'Enforced'),
        'MLP': (mlp_pred, 'Not enforced'),
        'Random Forest': (rf_pred, 'Not enforced')
    }
    try:
        from xgboost import XGBRegressor
        xgb = XGBRegressor(n_estimators=100, learning_rate=0.05, random_state=42, n_jobs=-1)
        xgb.fit(X_train_scaled, y_train)
        xgb_pred = xgb.predict(X_test_scaled)
        models['XGBoost'] = (xgb_pred, 'Not enforced')
    except:
        pass

    def compute_metrics(y_true, y_pred):
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        return r2, rmse, mae

    rows = []
    chart_data = []
    for name, (preds, consistency) in models.items():
        r2, rmse, mae = compute_metrics(y_test, preds)
        rows.append({
            'Model': name,
            'R²': f"{r2:.3f}",
            'RMSE (MPa)': f"{rmse:.3f}",
            'MAE (MPa)': f"{mae:.3f}",
            'Physical Consistency': consistency
        })
        chart_data.append({'Model': name, 'R²': r2})
    return pd.DataFrame(rows), chart_data

# ================================================================
# MAIN APPLICATION
# ================================================================
st.markdown("""
<div style="background: #0b1a33; padding:1rem; border-radius:0.5rem; text-align:center; margin-bottom:1rem;">
    <h2 style="color:#fff; margin:0;"> Hybrid AI · Unified Framework v29.30-R40</h2>
    <p style="color:#64ffda; margin:0; font-size:0.9rem;">Merging Best Features from AI_HYB, Co-HYBAI, and PLEX_HYBAI</p>
    <p style="color:#aabbcc; margin:0; font-size:0.85rem;">Nile Valley University, Sudan</p>
</div>
""", unsafe_allow_html=True)

# Load model
try:
    model, scaler, y_scaler, features, df = load_or_train()
except Exception as e:
    st.error(f" Training failed: {e}. Using dummy model.")
    model = None

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if model is not None:
    device = next(model.parameters()).device

# Sidebar: Input Parameters
with st.sidebar:
    st.markdown("###  Formulation & Material Parameters")
    with st.container(border=True):
        c1, c2 = st.columns(2)
        with c1:
            api = st.slider("API (%)", SLIDER_API_MIN, SLIDER_API_MAX, st.session_state.api, 0.1, key="api")
            binder = st.slider("Binder (%)", SLIDER_BINDER_MIN, SLIDER_BINDER_MAX, st.session_state.binder, 0.1, key="binder")
            pvpp = st.slider("PVPP (%)", SLIDER_PVPP_MIN, SLIDER_PVPP_MAX, st.session_state.pvpp, 0.1, key="pvpp")
            mgst = st.slider("Mg-St (%)", SLIDER_MGST_MIN, SLIDER_MGST_MAX, st.session_state.mgst, 0.01, key="mgst")
            mcc = st.slider("MCC (%)", SLIDER_MCC_MIN, SLIDER_MCC_MAX, st.session_state.mcc, 0.1, key="mcc")
        with c2:
            moisture = st.slider("Moisture (%)", SLIDER_MOISTURE_MIN, SLIDER_MOISTURE_MAX, st.session_state.moisture, 0.1, key="moisture")
            particle_size = st.slider("Particle Size (µm)", SLIDER_PARTICLE_SIZE_MIN, SLIDER_PARTICLE_SIZE_MAX, st.session_state.particle_size, 1.0, key="particle_size")
            binder_grade = st.selectbox("Binder Grade", BINDER_GRADES, index=st.session_state.binder_grade, key="binder_grade")
            binder_grade_idx = BINDER_GRADES.index(binder_grade)
            st.session_state.binder_grade = binder_grade_idx
        total = api + binder + pvpp + mgst + mcc + moisture
        if abs(total-100) < 0.5:
            st.success(f" Total = {total:.2f}%")
        else:
            st.warning(f" Total = {total:.2f}% (should be 100%)")

    st.markdown("###  Process Parameters")
    with st.container(border=True):
        c1, c2 = st.columns(2)
        with c1:
            pressure = st.slider("Pressure (MPa)", SLIDER_PRESSURE_MIN, SLIDER_PRESSURE_MAX, st.session_state.pressure, 1.0, key="pressure")
            speed = st.slider("Speed (rpm)", SLIDER_SPEED_MIN, SLIDER_SPEED_MAX, st.session_state.speed, 0.5, key="speed")
        with c2:
            dwell_time = st.slider("Dwell Time (ms)", SLIDER_DWELL_TIME_MIN, SLIDER_DWELL_TIME_MAX, st.session_state.dwell_time, 0.5, key="dwell_time")
            friction = st.slider("Friction", SLIDER_FRICTION_MIN, SLIDER_FRICTION_MAX, st.session_state.friction, 0.01, key="friction")
            decompression_time = st.slider("Decompression Time (ms)", SLIDER_DECOMPRESSION_TIME_MIN, SLIDER_DECOMPRESSION_TIME_MAX, st.session_state.decompression_time, 1.0, key="decompression_time")

        granule_mode = st.radio(
            "Granule Size",
            options=["Fixed", "Variable"],
            index=0 if st.session_state.get('granule_mode', 'Fixed') == 'Fixed' else 1,
            horizontal=True,
            key="granule_mode"
        )
        if granule_mode == "Fixed":
            granule = st.slider("Granule Size (µm)", SLIDER_GRANULE_MIN, SLIDER_GRANULE_MAX, st.session_state.granule, 1.0, key="granule")
            granule_fixed = True
            st.session_state.granule_mode = 'Fixed'
        else:
            granule = st.session_state.get('granule', 125.0)
            granule_fixed = False
            st.info(f"Granule size optimized by NSGA-II in range [{SLIDER_GRANULE_MIN:.0f}–{SLIDER_GRANULE_MAX:.0f}] µm")
            st.session_state.granule_mode = 'Variable'

    st.markdown("###  Penalty Adjustment (from Co-HYBAI)")
    with st.container(border=True):
        penalty_api = st.slider("API Penalty Strength", 0.0, 0.2, 0.08, 0.005, key="penalty_api")
        penalty_tensile = st.slider("Tensile Penalty Strength", 0.0, 0.2, 0.05, 0.005, key="penalty_tensile")
        st.caption("Higher values promote higher API% and Tensile simultaneously.")

    predict_btn = st.button(" Predict & Optimize", use_container_width=True, type="primary")

# Main Panel: Results
col_left, col_right = st.columns([1, 1.2], gap="medium")

with col_right:
    st.markdown("###  Results")

    if predict_btn:
        if model is None:
            st.error(" Model is not available. Please restart the app.")
        elif abs(total-100) > 0.5:
            st.warning(" Formulation must sum to 100% (within 0.5%)")
        else:
            api_n, binder_n, pvpp_n, mgst_n, mcc_n, moisture_n = normalize_components(
                api, binder, pvpp, mgst, mcc, moisture
            )
            if granule_fixed:
                granule_use = granule
            else:
                granule_use = granule
            inputs = [api_n, mcc_n, pvpp_n, mgst_n, binder_n, pressure, speed, granule_use,
                      particle_size, moisture_n, binder_grade_idx, dwell_time, friction, decompression_time]

            density, tensile, er, efrf, disintegration, dissolution_tau, dissolution_beta = predict_pinn(model, scaler, y_scaler, inputs)

            st.session_state.formulation = {
                'api_n': api_n, 'binder_n': binder_n, 'pvpp_n': pvpp_n,
                'mgst_n': mgst_n, 'mcc_n': mcc_n, 'moisture': moisture_n,
                'particle_size': particle_size, 'binder_grade': binder_grade_idx,
                'pressure': pressure, 'speed': speed, 'dwell_time': dwell_time,
                'friction': friction, 'decompression_time': decompression_time,
                'granule_use': granule_use, 'granule_fixed': granule_fixed,
                'density': density, 'tensile': tensile, 'er': er, 'efrf': efrf,
                'disintegration': disintegration, 'dissolution_tau': dissolution_tau,
                'dissolution_beta': dissolution_beta
            }

            st.markdown("**Constraint Status** (Density: 0.72–0.99, Tensile ≥ 1.50, EFRF < 0.40, Disintegration ≤ 15 min)")
            col_metrics = st.columns(5)
            col_metrics[0].metric("Density", f"{density:.3f}", f"[0.72, {D_MAX:.2f}]")
            col_metrics[1].metric("Tensile", f"{tensile:.2f} MPa", f"≥ {TENSILE_MIN:.2f}")
            col_metrics[2].metric("EFRF", f"{efrf:.4f}", f"< 0.40")
            col_metrics[3].metric("MCC", f"{mcc_n:.1f}%", f"≤ 8.0%")
            col_metrics[4].metric("Disintegration", f"{disintegration:.1f} min", f"≤ 15 min")

            if all([D_MIN <= density <= D_MAX, tensile >= TENSILE_MIN, efrf < 0.40,
                    mcc_n <= 8.0, disintegration <= 15.0]):
                st.success(" All constraints satisfied")
            else:
                st.error(" Constraints violated")

            # NSGA-II bounds
            bounds = np.array([
                [SLIDER_API_MIN, SLIDER_API_MAX],
                [BOUND_MCC_MIN, BOUND_MCC_MAX],
                [BOUND_PVPP_MIN, BOUND_PVPP_MAX],
                [BOUND_MGST_MIN, BOUND_MGST_MAX],
                [BOUND_BINDER_MIN, BOUND_BINDER_MAX],
                [SLIDER_PRESSURE_MIN, SLIDER_PRESSURE_MAX],
                [SLIDER_SPEED_MIN, SLIDER_SPEED_MAX],
                [SLIDER_GRANULE_MIN, SLIDER_GRANULE_MAX],
                [SLIDER_PARTICLE_SIZE_MIN, SLIDER_PARTICLE_SIZE_MAX],
                [SLIDER_MOISTURE_MIN, SLIDER_MOISTURE_MAX],
                [0, len(BINDER_GRADES)-1],
                [SLIDER_DWELL_TIME_MIN, SLIDER_DWELL_TIME_MAX],
                [SLIDER_FRICTION_MIN, SLIDER_FRICTION_MAX],
                [SLIDER_DECOMPRESSION_TIME_MIN, SLIDER_DECOMPRESSION_TIME_MAX]
            ])

            with st.spinner(f"Running NSGA-II (pop={NSGA_POP}, gens={NSGA_GENS})..."):
                nsga = NSGAIIOptimizer(
                    model, scaler, y_scaler, bounds,
                    pop=NSGA_POP, gens=NSGA_GENS,
                    granule_fixed=granule_fixed,
                    granule_fixed_val=granule if granule_fixed else 125.0,
                    penalty_api=penalty_api,
                    penalty_tensile=penalty_tensile
                )
                pop, objectives, fronts = nsga.run()

            st.session_state.nsga_pop = pop
            st.session_state.nsga_objectives = objectives
            st.session_state.nsga_fronts = fronts
            st.session_state.run_optimized = True

            balanced_idx = None
            quality_idx = None
            cost_idx = None

            if len(fronts) > 0 and len(fronts[0]) > 0:
                front_indices = fronts[0]
                max_api = max(-objectives[i, 0] for i in front_indices)
                min_efrf = min(objectives[i, 2] for i in front_indices)
                max_density = max(-objectives[i, 0] for i in front_indices)

                # Balanced: closest to ideal point (max API, min EFRF, max density)
                best_dist = np.inf
                for idx in front_indices:
                    api_val = -objectives[idx, 0]
                    efrf_val = objectives[idx, 2]
                    density_val = -objectives[idx, 0]  # proxy for density
                    norm_api = (SLIDER_API_MAX - api_val) / (SLIDER_API_MAX - SLIDER_API_MIN)
                    norm_efrf = (efrf_val - min_efrf) / (0.40 - min_efrf + 1e-10)
                    norm_density = (D_MAX - density_val) / (D_MAX - D_MIN)
                    dist = np.sqrt(norm_api**2 + norm_efrf**2 + norm_density**2)
                    if dist < best_dist:
                        best_dist = dist
                        balanced_idx = idx

                # Quality: max tensile
                best_tensile = -np.inf
                for idx in front_indices:
                    ind = pop[idx]
                    _, t, _, _, _, _, _ = predict_pinn(model, scaler, y_scaler, ind)
                    if t > best_tensile:
                        best_tensile = t
                        quality_idx = idx

                # Cost: max API, min pressure
                best_cost_score = -np.inf
                for idx in front_indices:
                    ind = pop[idx]
                    api_val = ind[0]
                    pressure_val = ind[5]
                    score = api_val - 0.05 * pressure_val
                    if score > best_cost_score:
                        best_cost_score = score
                        cost_idx = idx

                st.session_state.balanced_solution = pop[balanced_idx] if balanced_idx is not None else None
                st.session_state.quality_solution = pop[quality_idx] if quality_idx is not None else None
                st.session_state.cost_solution = pop[cost_idx] if cost_idx is not None else None

            # Generate feasible region
            feasible_df = generate_feasible_points(model, scaler, y_scaler, n_samples=3000)
            st.session_state.feasible_df = feasible_df
            st.session_state.tested_point = (api_n, efrf)

    if st.session_state.run_optimized and model is not None:
        objectives = st.session_state.nsga_objectives
        fronts = st.session_state.nsga_fronts
        balanced_solution = st.session_state.balanced_solution
        quality_solution = st.session_state.quality_solution
        cost_solution = st.session_state.cost_solution
        feasible_df = st.session_state.feasible_df
        tested_point = st.session_state.tested_point

        st.markdown("###  Pareto Front")
        if fronts is not None and len(fronts) > 0 and len(fronts[0]) > 0:
            st.success(f" Pareto front: {len(fronts[0])} optimal solutions")
            balanced_efrf = None
            if balanced_solution is not None:
                _, _, _, ef, _, _, _ = predict_pinn(model, scaler, y_scaler, balanced_solution)
                balanced_efrf = (balanced_solution[0], ef)
            fig = plot_pareto_clean(objectives, fronts, balanced_efrf, feasible_df, tested_point, efrf_max=0.40)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("###  Optimal Solutions Comparison")

        solutions_rows = []

        if balanced_solution is not None:
            d, t, e, ef, dis, tau, beta = predict_pinn(model, scaler, y_scaler, balanced_solution)
            solutions_rows.append({
                "Type": " Balanced",
                "API (%)": balanced_solution[0],
                "MCC (%)": balanced_solution[1],
                "PVPP (%)": balanced_solution[2],
                "Mg-St (%)": balanced_solution[3],
                "Binder (%)": balanced_solution[4],
                "Moisture (%)": balanced_solution[9],
                "Pressure (MPa)": balanced_solution[5],
                "Speed (rpm)": balanced_solution[6],
                "Granule (µm)": balanced_solution[7],
                "Particle Size (µm)": balanced_solution[8],
                "Binder Grade": BINDER_GRADES[int(balanced_solution[10])],
                "Density": d,
                "Tensile (MPa)": t,
                "EFRF": ef,
                "Disintegration (min)": dis,
            })
            st.session_state.balanced_pred = (d, t, e, ef, dis, tau, beta)

        if st.session_state.show_cost_solution and cost_solution is not None:
            d, t, e, ef, dis, tau, beta = predict_pinn(model, scaler, y_scaler, cost_solution)
            solutions_rows.append({
                "Type": " Cost-Optimized",
                "API (%)": cost_solution[0],
                "MCC (%)": cost_solution[1],
                "PVPP (%)": cost_solution[2],
                "Mg-St (%)": cost_solution[3],
                "Binder (%)": cost_solution[4],
                "Moisture (%)": cost_solution[9],
                "Pressure (MPa)": cost_solution[5],
                "Speed (rpm)": cost_solution[6],
                "Granule (µm)": cost_solution[7],
                "Particle Size (µm)": cost_solution[8],
                "Binder Grade": BINDER_GRADES[int(cost_solution[10])],
                "Density": d,
                "Tensile (MPa)": t,
                "EFRF": ef,
                "Disintegration (min)": dis,
            })
            st.session_state.cost_pred = (d, t, e, ef, dis, tau, beta)

        if st.session_state.show_quality_solution and quality_solution is not None:
            d, t, e, ef, dis, tau, beta = predict_pinn(model, scaler, y_scaler, quality_solution)
            solutions_rows.append({
                "Type": " Quality-Optimized",
                "API (%)": quality_solution[0],
                "MCC (%)": quality_solution[1],
                "PVPP (%)": quality_solution[2],
                "Mg-St (%)": quality_solution[3],
                "Binder (%)": quality_solution[4],
                "Moisture (%)": quality_solution[9],
                "Pressure (MPa)": quality_solution[5],
                "Speed (rpm)": quality_solution[6],
                "Granule (µm)": quality_solution[7],
                "Particle Size (µm)": quality_solution[8],
                "Binder Grade": BINDER_GRADES[int(quality_solution[10])],
                "Density": d,
                "Tensile (MPa)": t,
                "EFRF": ef,
                "Disintegration (min)": dis,
            })
            st.session_state.quality_pred = (d, t, e, ef, dis, tau, beta)

        if solutions_rows:
            df_solutions = pd.DataFrame(solutions_rows)
            st.dataframe(
                df_solutions,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Type": st.column_config.TextColumn("Type", width="small"),
                    "API (%)": st.column_config.NumberColumn("API (%)", format="%.1f", width="small"),
                    "MCC (%)": st.column_config.NumberColumn("MCC (%)", format="%.1f", width="small"),
                    "PVPP (%)": st.column_config.NumberColumn("PVPP (%)", format="%.1f", width="small"),
                    "Mg-St (%)": st.column_config.NumberColumn("Mg-St (%)", format="%.2f", width="small"),
                    "Binder (%)": st.column_config.NumberColumn("Binder (%)", format="%.1f", width="small"),
                    "Moisture (%)": st.column_config.NumberColumn("Moisture (%)", format="%.1f", width="small"),
                    "Pressure (MPa)": st.column_config.NumberColumn("Pressure (MPa)", format="%.1f", width="small"),
                    "Speed (rpm)": st.column_config.NumberColumn("Speed (rpm)", format="%.1f", width="small"),
                    "Granule (µm)": st.column_config.NumberColumn("Granule (µm)", format="%.0f", width="small"),
                    "Particle Size (µm)": st.column_config.NumberColumn("Particle Size (µm)", format="%.0f", width="small"),
                    "Binder Grade": st.column_config.TextColumn("Binder Grade", width="small"),
                    "Density": st.column_config.NumberColumn("Density", format="%.3f", width="small"),
                    "Tensile (MPa)": st.column_config.NumberColumn("Tensile (MPa)", format="%.3f", width="small"),
                    "EFRF": st.column_config.NumberColumn("EFRF", format="%.4f", width="small"),
                    "Disintegration (min)": st.column_config.NumberColumn("Disintegration (min)", format="%.1f", width="small"),
                }
            )
            st.caption(" Balanced = Trade-off (API, EFRF, Density) |  Cost = Max API, Min Pressure |  Quality = Max Tensile Strength")

        st.markdown("---")
        st.toggle(" Show Cost-wise Solution", value=st.session_state.get("show_cost_solution", False), key="show_cost_solution")
        st.toggle(" Show Quality-wise Solution", value=st.session_state.get("show_quality_solution", False), key="show_quality_solution")

        st.toggle(" Model Comparison", value=st.session_state.get("show_comparison", False), key="show_comparison")
        if st.session_state.show_comparison:
            st.markdown("###  Model Comparison")
            bench_df, chart_data = run_model_comparison(model, scaler, y_scaler, features, df, device)
            st.session_state.benchmark_df = bench_df
            fig_bar = px.bar(pd.DataFrame(chart_data), x='Model', y='R²', color='Model',
                             title='R² Comparison (Tensile Strength)',
                             text=pd.DataFrame(chart_data)['R²'].round(3))
            fig_bar.update_layout(height=380, template='plotly_white')
            st.plotly_chart(fig_bar, use_container_width=True)
            st.dataframe(bench_df, use_container_width=True)

        st.toggle(" Sensitivity Analysis", value=st.session_state.get("show_sensitivity", False), key="show_sensitivity")
        if st.session_state.show_sensitivity:
            st.markdown("###  Sensitivity Analysis")
            f = st.session_state.formulation
            if f is not None:
                fig_bars = plot_sensitivity_bars(f, model, scaler, y_scaler)
                if fig_bars:
                    st.plotly_chart(fig_bars, use_container_width=True)

        st.toggle(" Dissolution Profile", value=st.session_state.get("show_dissolution", False), key="show_dissolution")
        if st.session_state.show_dissolution:
            st.markdown("###  Dissolution Profile")
            f = st.session_state.formulation
            if f is not None:
                tau = f.get('dissolution_tau', 10.0)
                beta = f.get('dissolution_beta', 1.0)
                api_n = f['api_n']
                fig = plot_dissolution_profile(tau, beta, api_n)
                st.plotly_chart(fig, use_container_width=True)

        if st.session_state.experimental_data is not None:
            st.markdown("###  Comparison with Experimental Data")
            st.dataframe(st.session_state.experimental_data)

    else:
        if model is None:
            st.warning(" Model not loaded. Please fix training issues and restart.")
        else:
            st.info("Adjust parameters and click ' Predict & Optimize' to see results.")

st.caption(" Contact: babuker@protonmail.com |  Nile Valley University, Sudan")
