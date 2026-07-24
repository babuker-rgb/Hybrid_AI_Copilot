# ================================================================
# Hybrid AI · Multi-Objective Tablet Optimization
# Nile Valley University · Sudan · v29.28‑R32
# FINAL STABLE VERSION – with excipient minimum penalties
# ================================================================

import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import plotly.graph_objects as go
import time
import warnings
import json
import traceback
from datetime import datetime

warnings.filterwarnings('ignore')

# ================================================================
# PAGE CONFIG
# ================================================================
st.set_page_config(
    page_title="Hybrid AI · Tablet Optimization v29.28‑R32",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================================================
# CONSTANTS
# ================================================================
API_MIN, API_MAX = 80.0, 98.0
BINDER_MIN, BINDER_MAX = 1.4, 6.0
PVPP_MIN, PVPP_MAX = 1.0, 6.0
MGST_MIN, MGST_MAX = 0.10, 1.2
MCC_MIN, MCC_MAX = 1.5, 8.0
MOISTURE_MIN, MOISTURE_MAX = 0.5, 5.0

PRESSURE_MIN, PRESSURE_MAX = 150.0, 250.0
SPEED_MIN, SPEED_MAX = 15.0, 30.0
PARTICLE_SIZE_MIN, PARTICLE_SIZE_MAX = 10.0, 200.0
DWELL_TIME_MIN, DWELL_TIME_MAX = 5.0, 50.0
FRICTION_MIN, FRICTION_MAX = 0.1, 0.5
DECOMPRESSION_TIME_MIN, DECOMPRESSION_TIME_MAX = 10.0, 80.0

BINDER_GRADES = {
    "MCC PH101": {"compressibility": 0.85, "disintegration": 0.90, "flow": 0.80},
    "MCC PH102": {"compressibility": 0.90, "disintegration": 0.85, "flow": 0.85},
    "MCC PH200": {"compressibility": 0.95, "disintegration": 0.80, "flow": 0.90},
    "MCC KG": {"compressibility": 0.88, "disintegration": 0.88, "flow": 0.82},
    "Lactose Monohydrate": {"compressibility": 0.75, "disintegration": 0.95, "flow": 0.78},
    "Dicalcium Phosphate": {"compressibility": 0.70, "disintegration": 0.85, "flow": 0.75}
}
BINDER_GRADE_NAMES = list(BINDER_GRADES.keys())

POPULATION_SIZE = 50
NSGA_GENERATIONS = 80
TRAINING_EPOCHS = 300
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
N_SYNTHETIC_SAMPLES = 10000

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global scaling arrays (12 variables)
VARIABLE_MINS = np.array([
    API_MIN, BINDER_MIN, PVPP_MIN, MGST_MIN, MCC_MIN, MOISTURE_MIN,
    PRESSURE_MIN, SPEED_MIN, PARTICLE_SIZE_MIN, DWELL_TIME_MIN,
    FRICTION_MIN, DECOMPRESSION_TIME_MIN
])
VARIABLE_MAXS = np.array([
    API_MAX, BINDER_MAX, PVPP_MAX, MGST_MAX, MCC_MAX, MOISTURE_MAX,
    PRESSURE_MAX, SPEED_MAX, PARTICLE_SIZE_MAX, DWELL_TIME_MAX,
    FRICTION_MAX, DECOMPRESSION_TIME_MAX
])

# ================================================================
# SESSION STATE
# ================================================================
def initialize_session_state():
    defaults = {
        'api': 83.0, 'binder': 4.5, 'pvpp': 5.0, 'mgst': 0.5,
        'mcc': 6.0, 'moisture': 1.0, 'binder_grade': 0,
        'particle_size': 50.0, 'pressure': 200.0, 'speed': 20.0,
        'dwell_time': 25.0, 'friction': 0.25,
        'decompression_time': 35.0, 'optimization_complete': False,
        'results': None, 'best_solutions': None, 'golden_solution': None,
        'runtime': 0, 'pareto_history': None, 'model_trained': False,
        'pareto_solutions': None, 'pareto_objectives': None,
        'best_density': 0, 'best_tensile': 0, 'best_efrf': 1, 'best_api': 0
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
initialize_session_state()

# ================================================================
# SYNTHETIC DATA GENERATION (Physics‑Inspired)
# ================================================================
def generate_synthetic_data(n: int = N_SYNTHETIC_SAMPLES):
    np.random.seed(42)
    api = np.random.uniform(API_MIN, API_MAX, n)
    binder = np.random.uniform(BINDER_MIN, BINDER_MAX, n)
    pvpp = np.random.uniform(PVPP_MIN, PVPP_MAX, n)
    mgst = np.random.uniform(MGST_MIN, MGST_MAX, n)
    mcc = np.random.uniform(MCC_MIN, MCC_MAX, n)
    moisture = np.random.uniform(MOISTURE_MIN, MOISTURE_MAX, n)
    pressure = np.random.uniform(PRESSURE_MIN, PRESSURE_MAX, n)
    speed = np.random.uniform(SPEED_MIN, SPEED_MAX, n)
    particle_size = np.random.uniform(PARTICLE_SIZE_MIN, PARTICLE_SIZE_MAX, n)
    dwell_time = np.random.uniform(DWELL_TIME_MIN, DWELL_TIME_MAX, n)
    friction = np.random.uniform(FRICTION_MIN, FRICTION_MAX, n)
    decompression_time = np.random.uniform(DECOMPRESSION_TIME_MIN, DECOMPRESSION_TIME_MAX, n)

    X = np.column_stack([api, binder, pvpp, mgst, mcc, moisture,
                         pressure, speed, particle_size, dwell_time,
                         friction, decompression_time])

    noise = 0.02 * np.random.randn(n)
    density = (0.55 + 0.20 * (api/100) + 0.10 * (binder/10) -
               0.05 * (mgst/1) + 0.02 * (pressure/200) + noise)
    density = np.clip(density, 0.50, 0.98)

    noise = 0.2 * np.random.randn(n)
    tensile = (1.0 + 5.0 * (binder/100) - 2.0 * (mgst/100) +
               1.0 * (pressure/200) + 0.5 * (speed/30) + noise)
    tensile = np.clip(tensile, 0.5, 9.0)

    noise = 0.03 * np.random.randn(n)
    efrf = (0.10 + 0.30 * (mgst/1) + 0.10 * (particle_size/200) +
            0.05 * (friction/0.5) + noise)
    efrf = np.clip(efrf, 0.0, 1.0)

    noise = 1.0 * np.random.randn(n)
    disintegration = (5.0 + 10.0 * (pvpp/10) + 2.0 * (binder/10) +
                      0.5 * (dwell_time/50) + noise)
    disintegration = np.clip(disintegration, 2.0, 50.0)

    noise = 2.0 * np.random.randn(n)
    dissolution = (20.0 + 30.0 * (api/100) - 5.0 * (binder/10) +
                   2.0 * (pressure/200) + noise)
    dissolution = np.clip(dissolution, 10.0, 95.0)

    Y = np.column_stack([density, tensile, efrf, disintegration, dissolution])
    return X, Y

# ================================================================
# HYBRID NEURAL NETWORK (Physics‑Informed)
# ================================================================
class HybridTabletModel(nn.Module):
    def __init__(self, input_dim=12, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.bn4 = nn.BatchNorm1d(hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, 5)

    def forward(self, x):
        # x is expected to be in [0,1] range, shape (batch, 12)
        h1 = torch.relu(self.bn1(self.fc1(x)))
        h2 = torch.relu(self.bn2(self.fc2(h1))) + h1
        h3 = torch.relu(self.bn3(self.fc3(h2))) + h2
        h4 = torch.relu(self.bn4(self.fc4(h3))) + h3
        out = self.fc5(h4)
        density = torch.sigmoid(out[:, 0]) * 0.4 + 0.55
        tensile = torch.sigmoid(out[:, 1]) * 8.0 + 0.5
        efrf = torch.sigmoid(out[:, 2])
        disintegration = torch.sigmoid(out[:, 3]) * 45.0 + 2.0
        dissolution = torch.sigmoid(out[:, 4]) * 80.0 + 10.0
        return torch.stack([density, tensile, efrf, disintegration, dissolution], dim=1)

    def predict(self, x_norm):
        """x_norm can be 1D or 2D; returns numpy array of shape (n, 5)."""
        self.eval()
        with torch.no_grad():
            if isinstance(x_norm, np.ndarray):
                if x_norm.ndim == 1:
                    x_norm = x_norm.reshape(1, -1)
                x_norm = torch.FloatTensor(x_norm).to(DEVICE)
            elif isinstance(x_norm, torch.Tensor):
                if x_norm.dim() == 1:
                    x_norm = x_norm.unsqueeze(0)
                x_norm = x_norm.to(DEVICE)
            else:
                raise TypeError(f"Unsupported type: {type(x_norm)}")
            x_norm = x_norm.float()
            return self.forward(x_norm).cpu().numpy()

# ================================================================
# MODEL TRAINING (cached)
# ================================================================
@st.cache_resource(show_spinner=False)
def train_model():
    st.info("🧠 Training physics‑informed neural network on synthetic data...")
    X, Y = generate_synthetic_data()
    X_norm = (X - VARIABLE_MINS) / (VARIABLE_MAXS - VARIABLE_MINS + 1e-8)

    dataset = TensorDataset(torch.FloatTensor(X_norm), torch.FloatTensor(Y))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = HybridTabletModel(input_dim=12, hidden_dim=256).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(TRAINING_EPOCHS):
        epoch_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            y_pred = model(xb)
            loss = criterion(y_pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if (epoch+1) % 50 == 0:
            st.caption(f"Epoch {epoch+1}/{TRAINING_EPOCHS} – Loss: {epoch_loss/len(loader):.4f}")

    st.success("✅ Neural network training complete!")
    return model

# ================================================================
# HELPER FUNCTIONS
# ================================================================
def normalize_formulation(api, binder, pvpp, mgst, mcc, moisture):
    comps = np.array([api, binder, pvpp, mgst, mcc, moisture])
    total = np.sum(comps)
    if total > 0:
        norm = (comps / total) * 100
    else:
        norm = np.ones(6) * 100/6
    return {
        'api': norm[0], 'binder': norm[1], 'pvpp': norm[2],
        'mgst': norm[3], 'mcc': norm[4], 'moisture': norm[5], 'total': 100.0
    }

def get_formulation_summary(api, binder, pvpp, mgst, mcc, moisture):
    n = normalize_formulation(api, binder, pvpp, mgst, mcc, moisture)
    return {'API': n['api'], 'Binder': n['binder'], 'PVPP': n['pvpp'],
            'MgSt': n['mgst'], 'MCC': n['mcc'], 'Moisture': n['moisture'],
            'Total': n['total']}

def validate_formulation(api, binder, pvpp, mgst, mcc, moisture):
    total = sum([api, binder, pvpp, mgst, mcc, moisture])
    return (95 <= total <= 105, f"Total is {total:.1f}% – should be ~100%")

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
        return {'overall': overall, 'density_score': density_score,
                'tensile_score': tensile_score, 'efrf_score': efrf_score,
                'api_score': api_score, 'weights': {**weights, 'api': 0.3}}
    else:
        return {'overall': overall, 'density_score': density_score,
                'tensile_score': tensile_score, 'efrf_score': efrf_score,
                'weights': weights}

# ================================================================
# NSGA‑II OPTIMIZER (DUAL PENALTY + EXCIPIENT MINIMUM PENALTIES)
# ================================================================
class NSGAIIOptimizer:
    def __init__(self, model: HybridTabletModel, pop_size=50, generations=80):
        self.model = model
        self.pop_size = pop_size
        self.generations = generations
        self.n_objectives = 3
        self.n_vars = 12

    def enforce_mass_balance(self, pop: np.ndarray) -> np.ndarray:
        balanced = pop.copy()
        for i in range(len(pop)):
            f = pop[i, :6]
            total = np.sum(f)
            if total > 0:
                norm = (f / total) * 100
                balanced[i, :6] = np.clip(norm, 0, 100)
        return balanced

    def evaluate(self, pop: np.ndarray) -> np.ndarray:
        if pop.ndim == 1:
            pop = pop.reshape(1, -1)
        if pop.shape[1] != self.n_vars:
            raise ValueError(f"Expected {self.n_vars} variables, got {pop.shape[1]}")
        pop_norm = (pop - VARIABLE_MINS) / (VARIABLE_MAXS - VARIABLE_MINS + 1e-8)
        pred = self.model.predict(pop_norm)
        density = pred[:, 0]
        tensile = pred[:, 1]
        efrf = pred[:, 2]
        api = pop[:, 0]

        # Base objectives (minimize)
        fitness = np.column_stack([
            -density,
            -tensile,
            efrf
        ])

        # Penalties for low API and low tensile
        api_norm = (api - 80) / 18
        tensile_norm = tensile / 8.5
        penalty_api = 0.08 * (1 - np.clip(api_norm, 0, 1))
        penalty_tensile = 0.05 * (1 - np.clip(tensile_norm, 0, 1))
        fitness[:, 0] += penalty_api
        fitness[:, 1] += penalty_tensile

        # --- NEW: Excipient minimum penalties ---
        binder = pop[:, 1]
        pvpp   = pop[:, 2]
        mgst   = pop[:, 3]
        mcc    = pop[:, 4]
        moisture = pop[:, 5]

        # Minimum thresholds (in %)
        min_binder   = 1.0
        min_pvpp     = 1.0
        min_mgst     = 0.2
        min_mcc      = 1.0
        min_moisture = 0.5
        penalty_weight = 0.05

        pen_binder   = penalty_weight * np.maximum(0, (min_binder - binder) / min_binder)
        pen_pvpp     = penalty_weight * np.maximum(0, (min_pvpp - pvpp) / min_pvpp)
        pen_mgst     = penalty_weight * np.maximum(0, (min_mgst - mgst) / min_mgst)
        pen_mcc      = penalty_weight * np.maximum(0, (min_mcc - mcc) / min_mcc)
        pen_moisture = penalty_weight * np.maximum(0, (min_moisture - moisture) / min_moisture)

        # Add all excipient penalties to the tensile objective (already penalized)
        fitness[:, 1] += (pen_binder + pen_pvpp + pen_mgst + pen_mcc + pen_moisture)

        return fitness

    def fast_non_dominated_sort(self, obj: np.ndarray):
        n = len(obj)
        if n == 0:
            return []
        fronts = []
        dom_count = np.zeros(n, dtype=int)
        dom_sol = [[] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if np.all(obj[i] <= obj[j] + 1e-9) and np.any(obj[i] < obj[j] - 1e-9):
                    dom_sol[i].append(j)
                elif np.all(obj[j] <= obj[i] + 1e-9) and np.any(obj[j] < obj[i] - 1e-9):
                    dom_count[i] += 1
            if dom_count[i] == 0:
                fronts.append([i])

        curr = 0
        while True:
            next_front = []
            for i in fronts[curr]:
                for j in dom_sol[i]:
                    dom_count[j] -= 1
                    if dom_count[j] == 0:
                        next_front.append(j)
            if not next_front:
                break
            fronts.append(next_front)
            curr += 1

        # Safety: ensure all individuals are assigned
        assigned = set()
        for f in fronts:
            assigned.update(f)
        missing = set(range(n)) - assigned
        if missing:
            if fronts:
                fronts[-1].extend(list(missing))
            else:
                fronts.append(list(missing))
        return fronts

    def crowding_distance(self, obj: np.ndarray, front: list):
        n = len(front)
        if n <= 2:
            return np.ones(n) * np.inf
        dist = np.zeros(n)
        for m in range(self.n_objectives):
            sorted_idx = sorted(front, key=lambda x: obj[x, m])
            dist[0] = np.inf
            dist[-1] = np.inf
            min_val = obj[sorted_idx[0], m]
            max_val = obj[sorted_idx[-1], m]
            if max_val > min_val + 1e-9:
                for i in range(1, n-1):
                    dist[i] += (obj[sorted_idx[i+1], m] - obj[sorted_idx[i-1], m]) / (max_val - min_val)
        return dist

    def optimize(self):
        try:
            pop = np.random.rand(self.pop_size, self.n_vars)
            for j in range(self.n_vars):
                pop[:, j] = pop[:, j] * (VARIABLE_MAXS[j] - VARIABLE_MINS[j]) + VARIABLE_MINS[j]
            pop = self.enforce_mass_balance(pop)
            obj = self.evaluate(pop)

            for gen in range(self.generations):
                fronts = self.fast_non_dominated_sort(obj)
                if not fronts:
                    fronts = [list(range(self.pop_size))]

                crowding = []
                for f in fronts:
                    d = self.crowding_distance(obj, f)
                    crowding.extend(d)

                # Tournament selection
                selected = []
                for _ in range(self.pop_size):
                    i1, i2 = np.random.choice(self.pop_size, 2, replace=False)
                    r1 = None
                    r2 = None
                    for idx, f in enumerate(fronts):
                        if i1 in f:
                            r1 = idx
                        if i2 in f:
                            r2 = idx
                        if r1 is not None and r2 is not None:
                            break
                    if r1 is None:
                        r1 = 0
                    if r2 is None:
                        r2 = 0

                    if r1 < r2:
                        selected.append(i1)
                    elif r2 < r1:
                        selected.append(i2)
                    else:
                        d1 = crowding[fronts[r1].index(i1)] if i1 in fronts[r1] else -np.inf
                        d2 = crowding[fronts[r2].index(i2)] if i2 in fronts[r2] else -np.inf
                        selected.append(i1 if d1 > d2 else i2)

                sel_pop = pop[selected]

                # Crossover and mutation
                offspring = []
                for i in range(0, self.pop_size, 2):
                    p1 = sel_pop[i]
                    p2 = sel_pop[(i+1) % self.pop_size]
                    if np.random.random() < 0.8:
                        c1 = np.zeros_like(p1)
                        c2 = np.zeros_like(p2)
                        for j in range(self.n_vars):
                            if np.random.random() < 0.5:
                                beta = 1.0 + 2.0 * np.random.random()
                                c1[j] = 0.5 * ((1+beta)*p1[j] + (1-beta)*p2[j])
                                c2[j] = 0.5 * ((1-beta)*p1[j] + (1+beta)*p2[j])
                            else:
                                c1[j] = p1[j]
                                c2[j] = p2[j]
                    else:
                        c1 = p1.copy()
                        c2 = p2.copy()

                    for child in [c1, c2]:
                        if np.random.random() < 0.1:
                            for j in range(self.n_vars):
                                if np.random.random() < 0.1:
                                    step = 0.1 * (VARIABLE_MAXS[j] - VARIABLE_MINS[j])
                                    child[j] += np.random.normal(0, step)
                                    child[j] = np.clip(child[j], VARIABLE_MINS[j], VARIABLE_MAXS[j])
                    offspring.extend([c1, c2])

                offspring = np.array(offspring[:self.pop_size])
                offspring = self.enforce_mass_balance(offspring)
                off_obj = self.evaluate(offspring)

                combined_pop = np.vstack([pop, offspring])
                combined_obj = np.vstack([obj, off_obj])

                combined_fronts = self.fast_non_dominated_sort(combined_obj)
                new_pop = []
                remaining = self.pop_size
                for front in combined_fronts:
                    if len(new_pop) + len(front) <= remaining:
                        new_pop.extend(front)
                    else:
                        dist = self.crowding_distance(combined_obj, front)
                        sorted_front = sorted(front, key=lambda x: dist[front.index(x)], reverse=True)
                        new_pop.extend(sorted_front[:remaining - len(new_pop)])
                        break

                pop = combined_pop[new_pop]
                obj = combined_obj[new_pop]

                if gen % 5 == 0 or gen == self.generations - 1:
                    fronts = self.fast_non_dominated_sort(obj)
                    pareto_idx = fronts[0] if fronts else list(range(len(obj)))
                    history_entry = {
                        'generation': gen,
                        'pareto_solutions': pop[pareto_idx].copy(),
                        'pareto_objectives': obj[pareto_idx].copy()
                    }
                    yield pop, obj, [history_entry], gen
                else:
                    yield pop, obj, [], gen

        except Exception as e:
            st.error(f"💥 Optimization generator crashed: {e}")
            st.code(traceback.format_exc())
            raise RuntimeError(f"Optimization failed: {e}") from e

# ================================================================
# UI RENDER FUNCTIONS
# ================================================================
def render_sidebar():
    with st.sidebar:
        st.markdown("## 🧬 Hybrid AI Framework")
        st.markdown("---")
        st.markdown(f"**Version:** v29.28‑R32")
        st.markdown(f"**Institution:** Nile Valley University")
        st.markdown(f"**Department:** Pharmaceutical Engineering")
        st.markdown("---")
        with st.expander("📊 Optimization Objectives", expanded=True):
            st.markdown("1. **Maximize API%** (penalised low‑API)")
            st.markdown("2. **Maximize Tensile** (penalised low‑tensile)")
            st.markdown("3. **Maximize Density** → Better tablet quality")
            st.markdown("4. **Minimize EFRF** → Better powder flow")
        with st.expander("⚙️ Algorithm Settings", expanded=False):
            st.markdown(f"**Population:** {POPULATION_SIZE}")
            st.markdown(f"**Generations:** {NSGA_GENERATIONS}")
            st.markdown(f"**Training Epochs:** {TRAINING_EPOCHS}")
            st.markdown("**Algorithm:** NSGA‑II (3 obj + API & Tensile penalties)")
            st.markdown("**Model:** Physics‑Informed Neural Network")
            st.markdown("**Constraint:** Mass Balance (Σ = 100%)")
            st.markdown(f"**Runtime:** {st.session_state.runtime}s" if st.session_state.runtime else "**Runtime:** Pending")
        st.markdown("---")
        st.caption("© 2024 Nile Valley University · Sudan")

def render_input_panel():
    st.markdown("## 🧪 Formulation Parameters")
    st.info("⚠️ Components will be automatically normalized to sum to 100%.")
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.api = st.slider("**API Content (%)**", API_MIN, API_MAX, st.session_state.api, step=0.5)
        st.session_state.binder = st.slider("**Binder (%)**", BINDER_MIN, BINDER_MAX, st.session_state.binder, step=0.1)
        st.session_state.pvpp = st.slider("**PVPP (%)**", PVPP_MIN, PVPP_MAX, st.session_state.pvpp, step=0.1)
        st.session_state.mgst = st.slider("**MgSt (%)**", MGST_MIN, MGST_MAX, st.session_state.mgst, step=0.05)
    with col2:
        st.session_state.mcc = st.slider("**MCC (%)**", MCC_MIN, MCC_MAX, st.session_state.mcc, step=0.1)
        st.session_state.moisture = st.slider("**Moisture Content (%)**", MOISTURE_MIN, MOISTURE_MAX, st.session_state.moisture, step=0.1)
        grade_idx = st.session_state.get('binder_grade', 0)
        if not isinstance(grade_idx, int) or grade_idx >= len(BINDER_GRADE_NAMES):
            grade_idx = 0
        selected = st.selectbox("**Binder Grade**", BINDER_GRADE_NAMES, index=grade_idx)
        st.session_state.binder_grade = BINDER_GRADE_NAMES.index(selected)
        props = BINDER_GRADES[selected]
        st.caption(f"🔍 **{selected} Properties:**")
        st.caption(f"• Compressibility: {props['compressibility']:.0%}")
        st.caption(f"• Disintegration: {props['disintegration']:.0%}")
        st.caption(f"• Flowability: {props['flow']:.0%}")
        st.session_state.particle_size = st.slider("**Particle Size (µm)**", PARTICLE_SIZE_MIN, PARTICLE_SIZE_MAX, st.session_state.particle_size, step=5.0)
    summary = get_formulation_summary(st.session_state.api, st.session_state.binder,
                                      st.session_state.pvpp, st.session_state.mgst,
                                      st.session_state.mcc, st.session_state.moisture)
    st.markdown("### 📊 Formulation Mass Balance")
    st.write(f"**Total:** {summary['Total']:.1f}% ✅")
    cols = st.columns(6)
    for i, (k, v) in enumerate(summary.items()):
        if k != 'Total':
            cols[i].metric(k, f"{v:.1f}%")

    st.markdown("---")
    st.markdown("## ⚙️ Process Parameters")
    col3, col4 = st.columns(2)
    with col3:
        st.session_state.pressure = st.slider("**Compression Pressure (MPa)**", PRESSURE_MIN, PRESSURE_MAX, st.session_state.pressure, step=2.0)
        st.session_state.speed = st.slider("**Tableting Speed (rpm)**", SPEED_MIN, SPEED_MAX, st.session_state.speed, step=0.5)
    with col4:
        st.session_state.dwell_time = st.slider("**Dwell Time (ms)**", DWELL_TIME_MIN, DWELL_TIME_MAX, st.session_state.dwell_time, step=1.0)
        st.session_state.friction = st.slider("**Friction Coefficient**", FRICTION_MIN, FRICTION_MAX, st.session_state.friction, step=0.01)
        st.session_state.decompression_time = st.slider("**Decompression Time (ms)**", DECOMPRESSION_TIME_MIN, DECOMPRESSION_TIME_MAX, st.session_state.decompression_time, step=2.0)

def render_binder_grade_comparison():
    st.markdown("---")
    st.markdown("## 🔬 Binder Grade Impact")
    df = pd.DataFrame([
        {"Binder Grade": name,
         "Compressibility": p["compressibility"]*100,
         "Disintegration": p["disintegration"]*100,
         "Flowability": p["flow"]*100}
        for name, p in BINDER_GRADES.items()
    ])
    fig = go.Figure()
    for col in ["Compressibility", "Disintegration", "Flowability"]:
        fig.add_trace(go.Bar(
            x=df["Binder Grade"], y=df[col], name=col,
            text=[f"{v:.0f}%" for v in df[col]], textposition="outside"
        ))
    fig.update_layout(
        barmode="group",
        title="Binder Grade Properties",
        yaxis=dict(title="Score (%)", range=[0, 100]),
        height=350,
        margin=dict(l=0, r=0, t=40, b=0),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True)

# ================================================================
# MAIN ORCHESTRATION
# ================================================================
def main():
    render_sidebar()

    st.markdown("# 🧬 Hybrid AI · Multi-Objective Tablet Optimization")
    st.markdown("#### Nile Valley University · Sudan · v29.28‑R32")
    st.markdown("---")

    render_input_panel()
    render_binder_grade_comparison()

    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        run_button = st.button("🚀 Run Hybrid Optimization", type="primary", use_container_width=True)

    if run_button:
        start_time = time.time()

        valid, msg = validate_formulation(
            st.session_state.api, st.session_state.binder,
            st.session_state.pvpp, st.session_state.mgst,
            st.session_state.mcc, st.session_state.moisture
        )
        if not valid:
            st.error(f"❌ {msg}")
            return

        with st.spinner("Training neural network..."):
            model = train_model()
        st.session_state.model = model

        st.info("🧬 Running NSGA‑II optimization...")
        optimizer = NSGAIIOptimizer(model, pop_size=POPULATION_SIZE, generations=NSGA_GENERATIONS)

        progress_bar = st.progress(0)
        status_text = st.empty()
        pareto_placeholder = st.empty()

        final_pop = None
        final_obj = None
        all_history = []

        try:
            for pop, obj, history, gen in optimizer.optimize():
                final_pop, final_obj = pop, obj
                all_history.extend(history)

                progress_bar.progress((gen+1) / NSGA_GENERATIONS)
                status_text.text(f"Generation {gen+1}/{NSGA_GENERATIONS} – Population size: {len(pop)}")

                if history:
                    pareto_sols = history[0]['pareto_solutions']
                    if len(pareto_sols) > 0:
                        pop_norm = (pareto_sols - VARIABLE_MINS) / (VARIABLE_MAXS - VARIABLE_MINS + 1e-8)
                        preds = model.predict(pop_norm)
                        density = preds[:, 0]
                        tensile = preds[:, 1]
                        efrf = preds[:, 2]
                        api_vals = pareto_sols[:, 0]

                        fig = go.Figure()
                        fig.add_trace(go.Scatter3d(
                            x=density, y=tensile, z=efrf,
                            mode='markers',
                            marker=dict(
                                size=8,
                                color=api_vals,
                                colorscale='Viridis',
                                showscale=True,
                                colorbar=dict(title="API%", x=1.02, len=0.6)
                            ),
                            name=f'Gen {gen}',
                            hovertemplate='Density: %{x:.3f}<br>Tensile: %{y:.2f} MPa<br>EFRF: %{z:.3f}<br>API: %{marker.color:.1f}%<extra></extra>'
                        ))
                        fig.update_layout(
                            title=f'Pareto Front – Generation {gen}',
                            scene=dict(
                                xaxis=dict(title='Density', range=[0.55,0.95]),
                                yaxis=dict(title='Tensile (MPa)', range=[0.5,8.5]),
                                zaxis=dict(title='EFRF', range=[0,1]),
                                camera=dict(eye=dict(x=1.8, y=1.8, z=1.8))
                            ),
                            height=450,
                            margin=dict(l=0, r=0, t=40, b=0),
                        )
                        pareto_placeholder.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"❌ Optimization crashed: {e}")
            st.code(traceback.format_exc())
            st.stop()

        progress_bar.empty()
        status_text.empty()
        st.success("✅ Optimization complete!")

        fronts = optimizer.fast_non_dominated_sort(final_obj)
        pareto_idx = fronts[0] if fronts else list(range(len(final_obj)))
        pareto_solutions = final_pop[pareto_idx]

        pop_norm = (pareto_solutions - VARIABLE_MINS) / (VARIABLE_MAXS - VARIABLE_MINS + 1e-8)
        preds = model.predict(pop_norm)
        density = preds[:, 0]
        tensile = preds[:, 1]
        efrf = preds[:, 2]
        disintegration = preds[:, 3]
        dissolution = preds[:, 4]

        solutions = []
        for i, sol in enumerate(pareto_solutions):
            api, binder, pvpp, mgst, mcc, moisture = sol[:6]
            norm = normalize_formulation(api, binder, pvpp, mgst, mcc, moisture)
            quality = calculate_quality_score(density[i], tensile[i], efrf[i], api=norm['api'])
            solutions.append({
                'Solution': f'P{i+1}',
                'API (%)': norm['api'],
                'Binder (%)': norm['binder'],
                'PVPP (%)': norm['pvpp'],
                'MgSt (%)': norm['mgst'],
                'MCC (%)': norm['mcc'],
                'Moisture (%)': norm['moisture'],
                'Total (%)': 100.0,
                'Density': density[i],
                'Tensile (MPa)': tensile[i],
                'EFRF': efrf[i],
                'Disintegration (min)': disintegration[i],
                'Dissolution (%)': dissolution[i],
                'Quality Score': quality['overall']
            })

        solutions = sorted(solutions, key=lambda x: x['Quality Score'], reverse=True)
        golden = solutions[0] if solutions else None

        st.session_state.optimization_complete = True
        st.session_state.best_solutions = solutions
        st.session_state.golden_solution = golden
        st.session_state.pareto_history = all_history
        st.session_state.runtime = round(time.time() - start_time, 1)
        st.session_state.best_density = max(density)
        st.session_state.best_tensile = max(tensile)
        st.session_state.best_efrf = min(efrf)
        st.session_state.best_api = max([s['API (%)'] for s in solutions])

        st.success(f"⏱️ Optimization completed in {st.session_state.runtime} seconds!")
        st.balloons()

        st.markdown("## 📊 Optimization Results")
        first = solutions[0]
        col1, col2, col3 = st.columns(3)
        col1.metric("API%", f"{first['API (%)']:.1f}%")
        col2.metric("Tensile", f"{first['Tensile (MPa)']:.2f} MPa")
        col3.metric("Quality Score", f"{first['Quality Score']:.1f}%")
        st.dataframe(pd.DataFrame(solutions), use_container_width=True)

    elif st.session_state.optimization_complete and st.session_state.best_solutions:
        st.info("Showing cached results.")
        st.dataframe(pd.DataFrame(st.session_state.best_solutions), use_container_width=True)
    else:
        st.info("👆 Adjust parameters and click 'Run Hybrid Optimization' to begin.")
        st.markdown("---")
        st.markdown("### 🎯 Key Features")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**🧠 Physics-Informed AI**")
            st.markdown("**📊 API & Tensile Penalties**")
        with col2:
            st.markdown("**⚖️ Mass Balance Enforced**")
            st.markdown("**🔬 PINN Constraints**")
        with col3:
            st.markdown("**📈 Pareto Front**")
            st.markdown("**🏆 Golden Solution**")

if __name__ == "__main__":
    main()
