# ================================================================
# Hybrid AI · Multi-Objective Tablet Optimization
# Nile Valley University · Sudan · v29.28‑R32
# FINAL – FORCED API SPREAD (POP=120, NOISE, RELAXED CONSTRAINTS)
# ================================================================

import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import plotly.graph_objects as go
import os
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
EFRF_MAX = 0.40
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

# Enhanced NSGA‑II parameters (larger pop, more gens)
NSGA_POP = 120
NSGA_GENS = 70
HIDDEN_SIZE = 512

# Lower penalties to allow low‑API solutions
PENALTY_API = 0.04
PENALTY_TENSILE = 0.03
PENALTY_EFRF = 0.05

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
        'run_optimized': False,
        'balanced_solution': None, 'quality_solution': None, 'cost_solution': None,
        'balanced_pred': None, 'quality_pred': None, 'cost_pred': None,
        'api_objective': 'Minimize (Cost)',
        'max_api': 98.0,
        'auto_explore': False,
        'explore_min_api': 80.0,
        'explore_max_api': 98.0,
        'explore_steps': 5,
        'relax_constraints': False,
        'view_3d': True,
    })

# ================================================================
# HELPERS (unchanged)
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

# ================================================================
# PINN MODEL (unchanged)
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
            return self.forward(X_scaled).cpu().numpy()

# ================================================================
# DATA GENERATION (unchanged)
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

    k_heckel = 0.025 + 0.0001 * pressure_raw
    A_heckel = 1.0 + 0.01 * (api_n - 85.0) - 0.05 * binder_n
    D_heckel = 1.0 - np.exp(-(k_heckel * pressure_raw + A_heckel))
    D_heckel = np.clip(D_heckel, 0.6, 0.99)

    a_kawakita = 0.82 + 0.04 * (mcc_n - 1.5)/6.5 + 0.02 * (binder_n - 1.4)/4.6
    a_kawakita = np.clip(a_kawakita, 0.78, 0.92)
    b_kawakita = 0.002 + 0.003 * (binder_n - 1.4)/4.6 + 0.001 * (mcc_n - 1.5)/6.5
    b_kawakita = np.clip(b_kawakita, 0.0005, 0.006)
    D_kawakita = 1.0 - pressure_raw / (a_kawakita * pressure_raw + 1.0/b_kawakita)
    D_kawakita = np.clip(D_kawakita, 0.6, 0.99)

    pressure_norm = (pressure_raw - SLIDER_PRESSURE_MIN) / (SLIDER_PRESSURE_MAX - SLIDER_PRESSURE_MIN)
    D = pressure_norm * D_heckel + (1 - pressure_norm) * D_kawakita
    D += -0.003*(moisture_n - 2.0) - 0.002*(particle_size_raw - 50)/150 - 0.002*(speed_raw - 15)/15 - 0.01*(mgst_n - 0.2)
    D = np.clip(D, 0.6, 0.99)

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
    tensile = np.clip(tensile, 0.5, 8.5)

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
# MODEL LOADER (unchanged)
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

    df, features = generate_pinn_data(FALLBACK_SAMPLES)
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
    for epoch in range(FALLBACK_EPOCHS):
        model.train()
        optimizer.zero_grad()
        y_pred = model(X_train_t)
        loss = nn.MSELoss()(y_pred, y_train_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step(loss.item())
        progress_bar.progress((epoch+1)/FALLBACK_EPOCHS)
        status_text.text(f"Training fallback model: epoch {epoch+1}/{FALLBACK_EPOCHS}")
    progress_bar.empty()
    status_text.empty()

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
# NSGA-II OPTIMIZER – 3 OBJECTIVES + VIOLATION (Soft Constraint)
# ================================================================
class NSGAIIOptimizer:
    def __init__(self, model, scaler, y_scaler, bounds, pop=NSGA_POP, gens=NSGA_GENS,
                 granule_fixed=True, granule_fixed_val=125.0,
                 max_api=98.0, relax=False,
                 penalty_api=PENALTY_API, penalty_tensile=PENALTY_TENSILE, penalty_efrf=PENALTY_EFRF):
        self.model = model
        self.scaler = scaler
        self.y_scaler = y_scaler
        self.bounds = bounds
        self.pop_size = pop
        self.generations = gens
        self.granule_fixed = granule_fixed
        self.granule_fixed_val = granule_fixed_val
        self.max_api = max_api
        self.relax = relax
        self.penalty_api = penalty_api
        self.penalty_tensile = penalty_tensile
        self.penalty_efrf = penalty_efrf

    def _repair(self, ind):
        api, mcc, pvpp, mgst, binder, pressure, speed, granule, particle_size, moisture, binder_grade, dwell_time, friction, decompression_time = ind
        api = np.clip(api, SLIDER_API_MIN, self.max_api)
        api, binder, pvpp, mgst, mcc, moisture = normalize_components(
            api, binder, pvpp, mgst, mcc, moisture
        )
        api = np.clip(api, SLIDER_API_MIN, self.max_api)
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
        api = np.clip(api, SLIDER_API_MIN, self.max_api)
        api, binder, pvpp, mgst, mcc, moisture = normalize_components(
            api, binder, pvpp, mgst, mcc, moisture
        )
        api = np.clip(api, SLIDER_API_MIN, self.max_api)
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

        density = pred[:, 0]
        tensile = pred[:, 1]
        er = pred[:, 2]
        disintegration = pred[:, 3]

        efrf = er / np.maximum(tensile, 1e-4)
        api = repaired[:, 0]
        pressure = repaired[:, 5]  # kept for later selection

        # ---- Constraint violation (soft) ----
        violation = np.zeros(n)
        if self.relax:
            d_min = 0.50
            d_max = 0.99
            tensile_min = 0.20
            efrf_max = 0.80
            dis_max = 30.0
        else:
            d_min = D_MIN
            d_max = D_MAX
            tensile_min = TENSILE_MIN
            efrf_max = EFRF_MAX
            dis_max = DISINTEGRATION_MAX

        violation += np.maximum(0, d_min - density) + np.maximum(0, density - d_max)
        violation += np.maximum(0, tensile_min - tensile)
        violation += np.maximum(0, efrf - efrf_max)
        violation += np.maximum(0, disintegration - dis_max)
        mcc_val = repaired[:, 1]
        violation += np.maximum(0, BOUND_MCC_MIN - mcc_val) + np.maximum(0, mcc_val - BOUND_MCC_MAX)

        # ---- Three objectives: -API (max), EFRF (min), Disintegration (min) ----
        objectives = np.column_stack([
            -api,            # maximise API (minimise negative)
            efrf,            # minimise EFRF
            disintegration   # minimise disintegration time
        ])
        # We also return violation for constraint‑dominance handling
        return objectives, repaired, violation, pressure

    # ---- Standard NSGA-II methods (unchanged) ----
    def _non_dominated_sort(self, objectives, violations):
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
                    if (violations[j] < violations[i]) or \
                       (violations[j] == 0 and violations[i] == 0 and
                        np.all(objectives[j] <= objectives[i]) and
                        np.any(objectives[j] < objectives[i])):
                        dominated = True
                        break
                if not dominated:
                    front.append(i)
            fronts.append(front)
            remaining = [idx for idx in remaining if idx not in front]
        return fronts

    def _crowding_distance(self, objectives, front):
        if len(front) <= 2:
            return {idx: np.inf for idx in front}
        dist = {idx: 0.0 for idx in front}
        for obj_idx in range(objectives.shape[1]):
            sorted_front = sorted(front, key=lambda i: objectives[i, obj_idx])
            f_min = objectives[sorted_front[0], obj_idx]
            f_max = objectives[sorted_front[-1], obj_idx]
            if f_max - f_min > 1e-10:
                for k in range(1, len(sorted_front)-1):
                    dist[sorted_front[k]] += (objectives[sorted_front[k+1], obj_idx] -
                                              objectives[sorted_front[k-1], obj_idx]) / (f_max - f_min)
        dist[sorted_front[0]] = np.inf
        dist[sorted_front[-1]] = np.inf
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

    def _tournament(self, pop, objectives, violations, fronts, crowding_dist):
        idx1 = np.random.randint(0, len(pop))
        idx2 = np.random.randint(0, len(pop))
        rank1 = next((f for f, front in enumerate(fronts) if idx1 in front), len(fronts))
        rank2 = next((f for f, front in enumerate(fronts) if idx2 in front), len(fronts))
        if rank1 < rank2:
            return pop[idx1]
        elif rank2 < rank1:
            return pop[idx2]
        else:
            d1 = crowding_dist.get(idx1, 0)
            d2 = crowding_dist.get(idx2, 0)
            return pop[idx1] if d1 > d2 else pop[idx2]

    def run(self):
        rng = np.random.default_rng()
        pop = []
        for _ in range(self.pop_size):
            # Wider API range with noise
            api = rng.uniform(SLIDER_API_MIN, self.max_api)
            api += rng.normal(0, 0.5)          # larger noise
            api = np.clip(api, SLIDER_API_MIN, self.max_api)
            mcc = rng.uniform(BOUND_MCC_MIN, BOUND_MCC_MAX)
            binder = rng.uniform(BOUND_BINDER_MIN, BOUND_BINDER_MAX) + rng.normal(0, 0.2)
            pvpp = rng.uniform(BOUND_PVPP_MIN, BOUND_PVPP_MAX) + rng.normal(0, 0.2)
            mgst = rng.uniform(BOUND_MGST_MIN, BOUND_MGST_MAX)
            moisture = rng.uniform(SLIDER_MOISTURE_MIN, SLIDER_MOISTURE_MAX)
            pressure = rng.uniform(SLIDER_PRESSURE_MIN, SLIDER_PRESSURE_MAX) + rng.normal(0, 5.0)
            speed = rng.uniform(SLIDER_SPEED_MIN, SLIDER_SPEED_MAX)
            granule = rng.uniform(SLIDER_GRANULE_MIN, SLIDER_GRANULE_MAX)
            particle_size = rng.uniform(SLIDER_PARTICLE_SIZE_MIN, SLIDER_PARTICLE_SIZE_MAX)
            binder_grade = rng.integers(0, len(BINDER_GRADES))
            dwell_time = calculate_dwell_time(speed)
            friction = rng.uniform(SLIDER_FRICTION_MIN, SLIDER_FRICTION_MAX)
            decompression_time = rng.uniform(SLIDER_DECOMPRESSION_TIME_MIN, SLIDER_DECOMPRESSION_TIME_MAX)
            ind = np.array([api, mcc, pvpp, mgst, binder, pressure, speed, granule,
                            particle_size, moisture, binder_grade, dwell_time, friction, decompression_time])
            pop.append(self._repair(ind))
        pop = np.array(pop)

        for gen in range(self.generations):
            objectives, pop, violations, pressure = self._evaluate(pop)
            fronts = self._non_dominated_sort(objectives, violations)
            crowding_dist = {}
            for front in fronts:
                dist = self._crowding_distance(objectives, front)
                crowding_dist.update(dist)

            offspring = []
            while len(offspring) < self.pop_size:
                p1 = self._tournament(pop, objectives, violations, fronts, crowding_dist)
                p2 = self._tournament(pop, objectives, violations, fronts, crowding_dist)
                c1, c2 = self._crossover(p1, p2)
                c1 = self._mutate(c1)
                c2 = self._mutate(c2)
                offspring.append(self._repair(c1))
                if len(offspring) < self.pop_size:
                    offspring.append(self._repair(c2))
            offspring = np.array(offspring[:self.pop_size])

            combined = np.vstack([pop, offspring])
            obj_comb, combined, viol_comb, press_comb = self._evaluate(combined)
            fronts_comb = self._non_dominated_sort(obj_comb, viol_comb)
            crowding_comb = {}
            for front in fronts_comb:
                dist = self._crowding_distance(obj_comb, front)
                crowding_comb.update(dist)

            new_pop = []
            remaining = self.pop_size
            for front in fronts_comb:
                if len(front) <= remaining:
                    new_pop.extend(combined[front])
                    remaining -= len(front)
                else:
                    sorted_idx = sorted(front, key=lambda i: crowding_comb.get(i, 0), reverse=True)
                    new_pop.extend(combined[sorted_idx[:remaining]])
                    remaining = 0
                    break
            pop = np.array(new_pop)

        objectives, pop, violations, pressure = self._evaluate(pop)
        fronts = self._non_dominated_sort(objectives, violations)
        return pop, objectives, fronts, violations, pressure

# ================================================================
# PREDICTION AND PLOTTING (unchanged from previous corrected version)
# ================================================================
# The plotting functions (plot_2d_front, plot_3d_front, combine_pareto_fronts)
# are identical to the ones we already provided in the previous answer.
# To keep the answer concise, I will include them in the final code block below.
