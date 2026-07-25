# ================================================================
# Hybrid AI · Multi-Objective Tablet Optimization
# Nile Valley University · Sudan · v29.28‑R32
# COMPLETE PRODUCTION CODE – FIXED CONSTRAINTS & DIVERSE FRONT
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
st.set_page_config(page_title="Hybrid AI · Tablet Optimization v29.28‑R32", layout="wide")

# ================================================================
# CONSTANTS
# ================================================================
D_MIN, D_MAX = 0.72, 0.99
TENSILE_MIN = 1.50
EFRF_MAX = 0.40               # strict threshold for feasibility
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

# ---- NSGA‑II parameters ----
NSGA_POP = 60
NSGA_GENS = 40
HIDDEN_SIZE = 512

# ---- Fallback training parameters (used if full model missing) ----
FALLBACK_SAMPLES = 15000
FALLBACK_EPOCHS = 200

# ================================================================
# SESSION STATE
# ================================================================
if 'api' not in st.session_state:
    st.session_state.update({
        'api': 89.5, 'binder': 3.5, 'pvpp': 2.0, 'mgst': 0.5, 'mcc': 3.5,
        'moisture': 1.0, 'particle_size': 50.0, 'binder_grade_index': 0,
        'granule_mode_select': 'Fixed',
        'pressure': 200.0, 'speed': 20.0, 'dwell_time': 25.0,
        'friction': 0.25, 'decompression_time': 35.0, 'granule': 125.0,
        'show_cost_solution': True,
        'show_quality_solution': True,
        'show_comparison': False,
        'show_sensitivity': False,
        'show_dissolution': False,
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
    comps = np.array([api, binder, pvpp, mgst, mcc, moisture], dtype=float)
    total = np.sum(comps)
    if total <= 0:
        total = 1.0
    norm = (comps / total) * 100.0
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
    speed_rpm = np.asarray(speed_rpm)
    result = np.full_like(speed_rpm, 50.0, dtype=float)
    mask = speed_rpm > 0
    result[mask] = (punch_width * 60 * 1000) / (np.pi * pitch_diameter * speed_rpm[mask])
    return np.clip(result, 5.0, 80.0)

def predict_disintegration_time(tensile, pvpp_n, api_n, binder_n, moisture_n):
    base_time = 2.0 + 0.5 * tensile
    pvpp_effect = 5.0 * np.exp(-0.5 * pvpp_n)
    api_effect = 0.1 * (api_n - 80)
    binder_effect = 0.2 * (binder_n - 2.0)
    moisture_effect = -0.1 * moisture_n
    time = base_time - pvpp_effect + api_effect + binder_effect + moisture_effect
    return np.clip(time, 1.0, 30.0)

def predict_dissolution_profile(api_n, pvpp_n, particle_size, disintegration_time):
    tau = 5.0 + 0.5 * disintegration_time - 0.1 * pvpp_n + 0.05 * (api_n - 80)
    tau = np.clip(tau, 2.0, 20.0)
    beta = 1.0 + 0.01 * (particle_size - 50) / 50
    beta = np.clip(beta, 0.8, 2.5)
    return tau, beta

def calculate_quality_score(density, tensile, efrf, api=None):
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
# PINN MODEL
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
        self.output = nn.Linear(hidden//2, 6)

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
# DATA GENERATION (for fallback training)
# ================================================================
def generate_pinn_data(n_samples, random_state=42):
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

    X_base = np.column_stack([
        api_n, mcc_n, pvpp_n, mgst_n, binder_n,
        pressure_raw, speed_raw, granule_raw,
        particle_size_raw, moisture_n, binder_grade_raw,
        dwell_time_raw, friction_raw, decompression_time_raw
    ])

    api_binder = api_n * binder_n
    pressure_binder = pressure_raw * binder_n
    api_mcc = api_n * mcc_n
    pressure_speed = pressure_raw * speed_raw
    binder_mgst = binder_n * mgst_n

    X_enhanced = np.column_stack([
        X_base,
        api_binder, pressure_binder, api_mcc, pressure_speed, binder_mgst
    ])

    feature_names = [
        'API_%', 'MCC_%', 'PVPP_%', 'MgSt_%', 'Binder_%',
        'Pressure_MPa', 'Speed_rpm', 'Granule_Size_µm',
        'Particle_Size_µm', 'Moisture_%', 'Binder_Grade',
        'Dwell_Time_ms', 'Friction', 'Decompression_Time_ms',
        'API_Binder', 'Pressure_Binder', 'API_MCC', 'Pressure_Speed', 'Binder_MgSt'
    ]

    # Physics simulations (full implementation)
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
    tensile = np.clip(tensile, 0.5, 8.5)   # no longer clipping to max 8.5? keep for physics, but we'll penalise above max later

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
# MODEL LOADER (with fallback training)
# ================================================================
@st.cache_resource
def get_model():
    checkpoint_path = os.path.join(os.path.dirname(__file__), 'hybrid_unified_v29_30_R40.pt')
    if os.path.exists(checkpoint_path):
        try:
            ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            model = MultiTaskPINN(input_dim=ckpt['input_dim'], hidden=HIDDEN_SIZE)
            model.load_state_dict(ckpt['model_state'])
            scaler = ckpt['scaler']
            y_scaler = ckpt['y_scaler']
            features = ckpt['features']
            df = ckpt['df']
            st.success("✅ Pre-trained model loaded successfully!")
            return model, scaler, y_scaler, features, df
        except Exception as e:
            st.warning(f"⚠️ Failed to load pre-trained model: {e}. Training fallback model...")
    else:
        st.info("ℹ️ Pre-trained model not found. Training fallback model (this may take a few minutes)...")

    # ---- Fallback training ----
    N_SAMPLES = FALLBACK_SAMPLES
    ADAM_EPOCHS = FALLBACK_EPOCHS
    df, features = generate_pinn_data(N_SAMPLES)
    X_raw = df[features].values
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
    model = MultiTaskPINN(input_dim=X_raw.shape[1], hidden=HIDDEN_SIZE).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5)

    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).to(device)

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
        progress_bar.progress((epoch+1)/ADAM_EPOCHS)
        status_text.text(f"Training fallback model: epoch {epoch+1}/{ADAM_EPOCHS}")
    progress_bar.empty()
    status_text.empty()

    # Evaluate final model
    model.eval()
    with torch.no_grad():
        val_pred = model(X_test_t).cpu().numpy()
        val_true = y_test_t.cpu().numpy()
        val_pred_actual = y_scaler.inverse_transform(val_pred)
        val_true_actual = y_scaler.inverse_transform(val_true)
        r2_t = r2_score(val_true_actual[:, 1], val_pred_actual[:, 1])
        rmse_t = np.sqrt(mean_squared_error(val_true_actual[:, 1], val_pred_actual[:, 1]))
        st.success(f"✅ Fallback model trained: R² = {r2_t:.3f}, RMSE = {rmse_t:.3f} MPa")

    return model, scaler, y_scaler, features, df

# ================================================================
# NSGA-II OPTIMIZER (with proper constraint penalties)
# ================================================================
class NSGAIIOptimizer:
    def __init__(self, model, scaler, y_scaler, bounds, pop=NSGA_POP, gens=NSGA_GENS,
                 granule_fixed=True, granule_fixed_val=125.0,
                 penalty_api=0.08, penalty_tensile=0.05, penalty_efrf=0.08):
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
        self.penalty_efrf = penalty_efrf

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

        # Raw predictions (no clipping)
        density = pred[:, 0]
        tensile = pred[:, 1]
        er = pred[:, 2]
        disintegration = pred[:, 3]
        dissolution_tau = pred[:, 4]

        # Compute EFRF
        efrf = er / np.maximum(tensile, 1e-4)

        # ---- Penalties for constraint violations ----
        penalty = np.zeros(n)

        # Density bounds
        penalty += np.where(density < D_MIN, (D_MIN - density)**2, 0.0)
        penalty += np.where(density > D_MAX, (density - D_MAX)**2, 0.0)

        # Tensile minimum
        penalty += np.where(tensile < TENSILE_MIN, (TENSILE_MIN - tensile)**2, 0.0)

        # EFRF maximum
        penalty += np.where(efrf >= EFRF_MAX, (efrf - EFRF_MAX)**2, 0.0)

        # Disintegration time maximum
        penalty += np.where(disintegration > DISINTEGRATION_MAX, (disintegration - DISINTEGRATION_MAX)**2, 0.0)

        # MCC bounds
        mcc_val = repaired[:, 1]
        penalty += np.where(mcc_val < self.bounds[1,0], (mcc_val - self.bounds[1,0])**2, 0.0)
        penalty += np.where(mcc_val > self.bounds[1,1], (mcc_val - self.bounds[1,1])**2, 0.0)

        # ---- Objectives ----
        # 1: maximize density -> minimize -density
        # 2: maximize tensile -> minimize -tensile
        # 3: minimize efrf -> minimize efrf
        objectives = np.column_stack([
            -density,
            -tensile,
            efrf
        ])

        # ---- Additional penalty for low API and low tensile (from Co-HYBAI) ----
        api = repaired[:, 0]
        api_norm = (api - 80) / 18
        tensile_norm = tensile / 8.5
        penalty_api_vec = self.penalty_api * (1 - api_norm)
        penalty_tensile_vec = self.penalty_tensile * (1 - tensile_norm)
        objectives[:, 0] += penalty_api_vec
        objectives[:, 1] += penalty_tensile_vec

        # ---- EFRF penalty (extra push) ----
        efrf_penalty_vec = self.penalty_efrf * np.maximum(0, efrf - 0.40) ** 2
        objectives[:, 2] += efrf_penalty_vec

        # ---- Apply constraint penalties (large weight) ----
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
        density = pred[0]
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
    density = pred[:, 0]
    tensile = np.maximum(pred[:, 1], 1e-4)
    er = np.maximum(pred[:, 2], 1e-4)
    efrf = er / tensile
    efrf = np.clip(efrf, 1e-4, 5.0)
    disintegration = np.maximum(pred[:, 3], 0.5)

    mask = ((D_MIN <= density) & (density <= D_MAX) &
            (tensile >= TENSILE_MIN) & (efrf < EFRF_MAX) &
            (disintegration <= DISINTEGRATION_MAX) &
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
            name='⭐ Golden Solution (Balanced)',
            marker=dict(size=12, color='gold', symbol='star', line=dict(width=2, color='black')),
            hovertemplate='Golden: API %{x:.1f}%, EFRF %{y:.4f}<extra></extra>'
        ))
    fig.add_hline(y=EFRF_MAX, line_dash='dash', line_color='gray',
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

# ================================================================
# MAIN APPLICATION
# ================================================================
def main():
    st.markdown("""
    <div style="background: #0b1a33; padding:1rem; border-radius:0.5rem; text-align:center; margin-bottom:1rem;">
        <h2 style="color:#fff; margin:0;">🧬 Hybrid AI · Multi-Objective Tablet Optimization</h2>
        <p style="color:#64ffda; margin:0; font-size:0.9rem;">Nile Valley University · Sudan · v29.28‑R32</p>
    </div>
    """, unsafe_allow_html=True)

    # ---- Load model (with fallback) ----
    model, scaler, y_scaler, features, df = get_model()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if model is not None:
        device = next(model.parameters()).device

    # ---- Sidebar ----
    with st.sidebar:
        st.markdown("### 📊 Formulation & Material Parameters")
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
                binder_grade = st.selectbox("Binder Grade", BINDER_GRADES, index=st.session_state.binder_grade_index, key="binder_grade_select")
                st.session_state.binder_grade_index = BINDER_GRADES.index(binder_grade)
            total = api + binder + pvpp + mgst + mcc + moisture
            if abs(total-100) < 0.5:
                st.success(f"✅ Total = {total:.2f}%")
            else:
                st.warning(f"⚠️ Total = {total:.2f}% (should be 100%)")

        st.markdown("### ⚙️ Process Parameters")
        with st.container(border=True):
            c1, c2 = st.columns(2)
            with c1:
                pressure = st.slider("Pressure (MPa)", SLIDER_PRESSURE_MIN, SLIDER_PRESSURE_MAX, st.session_state.pressure, 1.0, key="pressure")
                speed = st.slider("Speed (rpm)", SLIDER_SPEED_MIN, SLIDER_SPEED_MAX, st.session_state.speed, 0.5, key="speed")
                granule_mode = st.radio("Granule Size", options=["Fixed", "Variable"], horizontal=True, key="granule_mode_select")
                if granule_mode == "Fixed":
                    granule = st.slider("Granule Size (µm)", SLIDER_GRANULE_MIN, SLIDER_GRANULE_MAX, st.session_state.granule, 1.0, key="granule")
                    granule_fixed = True
                else:
                    granule = st.session_state.get('granule', 125.0)
                    granule_fixed = False
                    st.info(f"Granule size optimised by NSGA-II")
            with c2:
                dwell_time = st.slider("Dwell Time (ms)", SLIDER_DWELL_TIME_MIN, SLIDER_DWELL_TIME_MAX, st.session_state.dwell_time, 0.5, key="dwell_time")
                friction = st.slider("Friction", SLIDER_FRICTION_MIN, SLIDER_FRICTION_MAX, st.session_state.friction, 0.01, key="friction")
                decompression_time = st.slider("Decompression Time (ms)", SLIDER_DECOMPRESSION_TIME_MIN, SLIDER_DECOMPRESSION_TIME_MAX, st.session_state.decompression_time, 1.0, key="decompression_time")

        st.markdown("### ⚙️ Penalty Adjustment")
        with st.container(border=True):
            penalty_api = st.slider("API Penalty", 0.0, 0.2, 0.08, 0.005, key="penalty_api")
            penalty_tensile = st.slider("Tensile Penalty", 0.0, 0.2, 0.05, 0.005, key="penalty_tensile")
            penalty_efrf = st.slider("EFRF Penalty", 0.0, 0.2, 0.08, 0.005, key="penalty_efrf")

        predict_btn = st.button("🚀 Predict & Optimize", use_container_width=True, type="primary")

    # ---- Main Panel ----
    col_left, col_right = st.columns([1, 1.2], gap="medium")

    with col_right:
        st.markdown("### 📈 Results")

        if predict_btn:
            if model is None:
                st.error("❌ Model not loaded. Please check the checkpoint file.")
            elif abs(total-100) > 0.5:
                st.warning("⚠️ Formulation must sum to 100% (within 0.5%)")
            else:
                api_n, binder_n, pvpp_n, mgst_n, mcc_n, moisture_n = normalize_components(
                    api, binder, pvpp, mgst, mcc, moisture
                )
                if granule_fixed:
                    granule_use = granule
                else:
                    granule_use = granule
                inputs = [api_n, mcc_n, pvpp_n, mgst_n, binder_n, pressure, speed, granule_use,
                          particle_size, moisture_n, st.session_state.binder_grade_index, dwell_time, friction, decompression_time]

                density, tensile, er, efrf, disintegration, dissolution_tau, dissolution_beta = predict_pinn(model, scaler, y_scaler, inputs)

                st.session_state.formulation = {
                    'api_n': api_n, 'binder_n': binder_n, 'pvpp_n': pvpp_n,
                    'mgst_n': mgst_n, 'mcc_n': mcc_n, 'moisture': moisture_n,
                    'particle_size': particle_size, 'binder_grade': st.session_state.binder_grade_index,
                    'pressure': pressure, 'speed': speed, 'dwell_time': dwell_time,
                    'friction': friction, 'decompression_time': decompression_time,
                    'granule_use': granule_use, 'granule_fixed': granule_fixed,
                    'density': density, 'tensile': tensile, 'er': er, 'efrf': efrf,
                    'disintegration': disintegration, 'dissolution_tau': dissolution_tau,
                    'dissolution_beta': dissolution_beta
                }

                # ---- Constraint Status (real, no clipping) ----
                st.markdown("**Constraint Status** (Density: 0.72–0.99, Tensile ≥ 1.50, EFRF < 0.40, Disintegration ≤ 15 min)")
                col_metrics = st.columns(5)
                col_metrics[0].metric("Density", f"{density:.3f}", f"[0.72, {D_MAX:.2f}]")
                col_metrics[1].metric("Tensile", f"{tensile:.2f} MPa", f"≥ {TENSILE_MIN:.2f}")
                col_metrics[2].metric("EFRF", f"{efrf:.4f}", f"< 0.40")
                col_metrics[3].metric("MCC", f"{mcc_n:.1f}%", f"≤ 8.0%")
                col_metrics[4].metric("Disintegration", f"{disintegration:.1f} min", f"≤ 15 min")

                constraints_ok = (D_MIN <= density <= D_MAX and tensile >= TENSILE_MIN and
                                   efrf < 0.40 and mcc_n <= 8.0 and disintegration <= 15.0)
                if constraints_ok:
                    st.success("✅ All constraints satisfied")
                else:
                    st.error("❌ Constraints violated")

                # ---- NSGA-II Optimisation ----
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

                with st.spinner(f"Running NSGA‑II (pop={NSGA_POP}, gens={NSGA_GENS})..."):
                    nsga = NSGAIIOptimizer(
                        model, scaler, y_scaler, bounds,
                        pop=NSGA_POP, gens=NSGA_GENS,
                        granule_fixed=granule_fixed,
                        granule_fixed_val=granule if granule_fixed else 125.0,
                        penalty_api=penalty_api,
                        penalty_tensile=penalty_tensile,
                        penalty_efrf=penalty_efrf
                    )
                    pop, objectives, fronts = nsga.run()

                st.session_state.nsga_pop = pop
                st.session_state.nsga_objectives = objectives
                st.session_state.nsga_fronts = fronts
                st.session_state.run_optimized = True

                # ---- Extract 3 distinct solutions ----
                balanced_solution = None
                quality_solution = None
                cost_solution = None

                if len(fronts) > 0 and len(fronts[0]) > 0:
                    front_indices = fronts[0]
                    n_front = len(front_indices)
                    if n_front >= 1:
                        candidates = []
                        for idx in front_indices:
                            ind = pop[idx]
                            api_val = -objectives[idx, 0]
                            efrf_val = objectives[idx, 2]
                            _, t, _, _, _, _, _ = predict_pinn(model, scaler, y_scaler, ind)
                            pressure_val = ind[5]
                            candidates.append({
                                'idx': idx,
                                'ind': ind,
                                'api': api_val,
                                'efrf': efrf_val,
                                'tensile': t,
                                'pressure': pressure_val,
                                'balanced_score': api_val - efrf_val * 20,
                                'quality_score': t,
                                'cost_score': api_val - 0.05 * pressure_val
                            })

                        # Distance function (full 14‑D)
                        ranges = np.array([
                            SLIDER_API_MAX - SLIDER_API_MIN,
                            SLIDER_MCC_MAX - SLIDER_MCC_MIN,
                            SLIDER_PVPP_MAX - SLIDER_PVPP_MIN,
                            SLIDER_MGST_MAX - SLIDER_MGST_MIN,
                            SLIDER_BINDER_MAX - SLIDER_BINDER_MIN,
                            SLIDER_PRESSURE_MAX - SLIDER_PRESSURE_MIN,
                            SLIDER_SPEED_MAX - SLIDER_SPEED_MIN,
                            SLIDER_GRANULE_MAX - SLIDER_GRANULE_MIN,
                            SLIDER_PARTICLE_SIZE_MAX - SLIDER_PARTICLE_SIZE_MIN,
                            SLIDER_MOISTURE_MAX - SLIDER_MOISTURE_MIN,
                            len(BINDER_GRADES) - 1,
                            SLIDER_DWELL_TIME_MAX - SLIDER_DWELL_TIME_MIN,
                            SLIDER_FRICTION_MAX - SLIDER_FRICTION_MIN,
                            SLIDER_DECOMPRESSION_TIME_MAX - SLIDER_DECOMPRESSION_TIME_MIN
                        ])
                        ranges[ranges == 0] = 1.0

                        def full_distance(sol1, sol2):
                            diff = (sol1 - sol2) / ranges
                            return np.sqrt(np.sum(diff ** 2))

                        # Balanced: best balanced score
                        balanced_sorted = sorted(candidates, key=lambda x: x['balanced_score'], reverse=True)
                        balanced_candidate = balanced_sorted[0]
                        balanced_solution = balanced_candidate['ind']
                        balanced_idx = balanced_candidate['idx']
                        balanced_vec = balanced_solution.copy()

                        # Quality: farthest from balanced
                        quality_candidate = None
                        max_dist = -1
                        for c in candidates:
                            if c['idx'] != balanced_idx:
                                dist = full_distance(balanced_vec, c['ind'])
                                if dist > max_dist or (dist == max_dist and c['tensile'] > (quality_candidate['tensile'] if quality_candidate else -1)):
                                    max_dist = dist
                                    quality_candidate = c
                        if quality_candidate is None:
                            quality_candidate = balanced_candidate
                        quality_solution = quality_candidate['ind']
                        quality_idx = quality_candidate['idx']
                        quality_vec = quality_solution.copy()

                        # Cost: farthest from both balanced and quality
                        cost_candidate = None
                        max_min_dist = -1
                        for c in candidates:
                            if c['idx'] != balanced_idx and c['idx'] != quality_idx:
                                d1 = full_distance(balanced_vec, c['ind'])
                                d2 = full_distance(quality_vec, c['ind'])
                                min_dist = min(d1, d2)
                                if min_dist > max_min_dist or (min_dist == max_min_dist and c['cost_score'] > (cost_candidate['cost_score'] if cost_candidate else -1)):
                                    max_min_dist = min_dist
                                    cost_candidate = c
                        if cost_candidate is None:
                            for c in candidates:
                                if c['idx'] != balanced_idx and c['idx'] != quality_idx:
                                    cost_candidate = c
                                    break
                        if cost_candidate is None:
                            cost_candidate = balanced_candidate
                        cost_solution = cost_candidate['ind']

                        # Store
                        st.session_state.balanced_solution = balanced_solution
                        st.session_state.quality_solution = quality_solution
                        st.session_state.cost_solution = cost_solution

                # Store predictions
                if balanced_solution is not None:
                    d, t, e, ef, dis, tau, beta = predict_pinn(model, scaler, y_scaler, balanced_solution)
                    st.session_state.balanced_pred = (d, t, e, ef, dis, tau, beta)
                if quality_solution is not None:
                    d, t, e, ef, dis, tau, beta = predict_pinn(model, scaler, y_scaler, quality_solution)
                    st.session_state.quality_pred = (d, t, e, ef, dis, tau, beta)
                if cost_solution is not None:
                    d, t, e, ef, dis, tau, beta = predict_pinn(model, scaler, y_scaler, cost_solution)
                    st.session_state.cost_pred = (d, t, e, ef, dis, tau, beta)

                # ---- Show Pareto Front ----
                st.markdown("### 📉 Pareto Front")
                if fronts is not None and len(fronts) > 0 and len(fronts[0]) > 0:
                    st.success(f"✅ Pareto front: {len(fronts[0])} optimal solutions")
                    balanced_efrf = None
                    if balanced_solution is not None:
                        _, _, _, ef, _, _, _ = predict_pinn(model, scaler, y_scaler, balanced_solution)
                        balanced_efrf = (balanced_solution[0], ef)
                    feasible_df = generate_feasible_points(model, scaler, y_scaler, n_samples=3000)
                    fig = plot_pareto_clean(objectives, fronts, balanced_efrf, feasible_df, tested_point=(api_n, efrf), efrf_max=0.40)
                    if fig is not None:
                        st.plotly_chart(fig, use_container_width=True)

                # ---- Show 3 Solutions ----
                st.markdown("### 📊 Optimal Solutions Comparison")
                solutions_rows = []

                if balanced_solution is not None and st.session_state.balanced_pred is not None:
                    d, t, e, ef, dis, tau, beta = st.session_state.balanced_pred
                    solutions_rows.append({
                        "Type": "⚖️ Balanced",
                        "API (%)": round(balanced_solution[0], 1),
                        "MCC (%)": round(balanced_solution[1], 1),
                        "PVPP (%)": round(balanced_solution[2], 1),
                        "Mg-St (%)": round(balanced_solution[3], 2),
                        "Binder (%)": round(balanced_solution[4], 1),
                        "Moisture (%)": round(balanced_solution[9], 1),
                        "Pressure (MPa)": round(balanced_solution[5], 1),
                        "Speed (rpm)": round(balanced_solution[6], 1),
                        "Granule (µm)": round(balanced_solution[7], 0),
                        "Particle Size (µm)": round(balanced_solution[8], 0),
                        "Binder Grade": BINDER_GRADES[int(balanced_solution[10])],
                        "Density": round(d, 3),
                        "Tensile (MPa)": round(t, 3),
                        "EFRF": round(ef, 4),
                        "Disintegration (min)": round(dis, 1),
                    })

                if st.session_state.show_cost_solution and cost_solution is not None and st.session_state.cost_pred is not None:
                    d, t, e, ef, dis, tau, beta = st.session_state.cost_pred
                    solutions_rows.append({
                        "Type": "💰 Cost-Optimized",
                        "API (%)": round(cost_solution[0], 1),
                        "MCC (%)": round(cost_solution[1], 1),
                        "PVPP (%)": round(cost_solution[2], 1),
                        "Mg-St (%)": round(cost_solution[3], 2),
                        "Binder (%)": round(cost_solution[4], 1),
                        "Moisture (%)": round(cost_solution[9], 1),
                        "Pressure (MPa)": round(cost_solution[5], 1),
                        "Speed (rpm)": round(cost_solution[6], 1),
                        "Granule (µm)": round(cost_solution[7], 0),
                        "Particle Size (µm)": round(cost_solution[8], 0),
                        "Binder Grade": BINDER_GRADES[int(cost_solution[10])],
                        "Density": round(d, 3),
                        "Tensile (MPa)": round(t, 3),
                        "EFRF": round(ef, 4),
                        "Disintegration (min)": round(dis, 1),
                    })

                if st.session_state.show_quality_solution and quality_solution is not None and st.session_state.quality_pred is not None:
                    d, t, e, ef, dis, tau, beta = st.session_state.quality_pred
                    solutions_rows.append({
                        "Type": "🏆 Quality-Optimized",
                        "API (%)": round(quality_solution[0], 1),
                        "MCC (%)": round(quality_solution[1], 1),
                        "PVPP (%)": round(quality_solution[2], 1),
                        "Mg-St (%)": round(quality_solution[3], 2),
                        "Binder (%)": round(quality_solution[4], 1),
                        "Moisture (%)": round(quality_solution[9], 1),
                        "Pressure (MPa)": round(quality_solution[5], 1),
                        "Speed (rpm)": round(quality_solution[6], 1),
                        "Granule (µm)": round(quality_solution[7], 0),
                        "Particle Size (µm)": round(quality_solution[8], 0),
                        "Binder Grade": BINDER_GRADES[int(quality_solution[10])],
                        "Density": round(d, 3),
                        "Tensile (MPa)": round(t, 3),
                        "EFRF": round(ef, 4),
                        "Disintegration (min)": round(dis, 1),
                    })

                if solutions_rows:
                    df_solutions = pd.DataFrame(solutions_rows)
                    st.dataframe(df_solutions, use_container_width=True, hide_index=True)

                st.markdown("---")
                st.toggle("💰 Show Cost-wise Solution", value=st.session_state.get("show_cost_solution", True), key="show_cost_solution")
                st.toggle("🏆 Show Quality-wise Solution", value=st.session_state.get("show_quality_solution", True), key="show_quality_solution")

        else:
            st.info("👆 Adjust parameters and click 'Predict & Optimize' to see results.")

    st.caption("📧 Contact: babuker@protonmail.com | 🏛️ Nile Valley University, Sudan")

if __name__ == "__main__":
    main()
