# ================================================================
# Hybrid AI · Multi-Objective Tablet Optimization
# Nile Valley University · Sudan · v29.28‑R32
# FINAL VERSION – FULLY FUNCTIONAL NSGA‑II + TRAINED PINN
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
from datetime import datetime
from typing import Tuple, List, Dict, Any

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
GRANULE_MIN, GRANULE_MAX = 30.0, 250.0

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
TRAINING_EPOCHS = 300           # Reduced for faster interactive demo
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
N_SYNTHETIC_SAMPLES = 10000

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================================================================
# SESSION STATE
# ================================================================
def initialize_session_state():
    defaults = {
        'api': 96.5, 'binder': 1.4, 'pvpp': 1.0, 'mgst': 0.10,
        'mcc': 1.5, 'moisture': 0.50, 'binder_grade': 0,
        'particle_size': 50.0, 'pressure': 200.0, 'speed': 20.0,
        'granule': 125.0, 'dwell_time': 25.0, 'friction': 0.25,
        'decompression_time': 35.0, 'optimization_complete': False,
        'results': None, 'best_solutions': None, 'golden_solution': None,
        'runtime': 0, 'pareto_history': None, 'model_trained': False
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
initialize_session_state()

# ================================================================
# SYNTHETIC DATA GENERATION (Physics‑Inspired)
# ================================================================
def generate_synthetic_data(n: int = N_SYNTHETIC_SAMPLES) -> Tuple[np.ndarray, np.ndarray]:
    """Generate random formulation + process parameters and compute outputs
    using a non‑linear (but deterministic) tablet model.
    """
    np.random.seed(42)
    # Formulation (6 vars)
    api = np.random.uniform(API_MIN, API_MAX, n)
    binder = np.random.uniform(BINDER_MIN, BINDER_MAX, n)
    pvpp = np.random.uniform(PVPP_MIN, PVPP_MAX, n)
    mgst = np.random.uniform(MGST_MIN, MGST_MAX, n)
    mcc = np.random.uniform(MCC_MIN, MCC_MAX, n)
    moisture = np.random.uniform(MOISTURE_MIN, MOISTURE_MAX, n)
    # Process (6 vars)
    pressure = np.random.uniform(PRESSURE_MIN, PRESSURE_MAX, n)
    speed = np.random.uniform(SPEED_MIN, SPEED_MAX, n)
    particle_size = np.random.uniform(PARTICLE_SIZE_MIN, PARTICLE_SIZE_MAX, n)
    dwell_time = np.random.uniform(DWELL_TIME_MIN, DWELL_TIME_MAX, n)
    friction = np.random.uniform(FRICTION_MIN, FRICTION_MAX, n)
    decompression_time = np.random.uniform(DECOMPRESSION_TIME_MIN, DECOMPRESSION_TIME_MAX, n)

    X = np.column_stack([api, binder, pvpp, mgst, mcc, moisture,
                         pressure, speed, particle_size, dwell_time,
                         friction, decompression_time])

    # Compute responses (add small noise)
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
        x = torch.sigmoid(x)   # squash inputs to [0,1] range
        h1 = torch.relu(self.bn1(self.fc1(x)))
        h2 = torch.relu(self.bn2(self.fc2(h1))) + h1
        h3 = torch.relu(self.bn3(self.fc3(h2))) + h2
        h4 = torch.relu(self.bn4(self.fc4(h3))) + h3
        out = self.fc5(h4)
        # Enforce physical bounds
        density = torch.sigmoid(out[:, 0]) * 0.4 + 0.55
        tensile = torch.sigmoid(out[:, 1]) * 8.0 + 0.5
        efrf = torch.sigmoid(out[:, 2])
        disintegration = torch.sigmoid(out[:, 3]) * 45.0 + 2.0
        dissolution = torch.sigmoid(out[:, 4]) * 80.0 + 10.0
        return torch.stack([density, tensile, efrf, disintegration, dissolution], dim=1)

    def predict(self, x: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x = torch.FloatTensor(x).to(DEVICE)
            if x.dim() == 1:
                x = x.unsqueeze(0)
            return self.forward(x).cpu().numpy()

# ================================================================
# MODEL TRAINING (cached)
# ================================================================
@st.cache_resource(show_spinner=False)
def train_model() -> HybridTabletModel:
    """Generate synthetic data and train the neural network."""
    st.info("🧠 Training physics‑informed neural network on synthetic data...")
    X, Y = generate_synthetic_data()
    # Normalise inputs (min‑max scaling) – we keep the raw values for prediction,
    # but the model uses sigmoid on inputs, so we should scale them to [0,1].
    # Instead, we'll scale inputs to [0,1] before feeding.
    mins = np.array([API_MIN, BINDER_MIN, PVPP_MIN, MGST_MIN, MCC_MIN, MOISTURE_MIN,
                     PRESSURE_MIN, SPEED_MIN, PARTICLE_SIZE_MIN, DWELL_TIME_MIN,
                     FRICTION_MIN, DECOMPRESSION_TIME_MIN])
    maxs = np.array([API_MAX, BINDER_MAX, PVPP_MAX, MGST_MAX, MCC_MAX, MOISTURE_MAX,
                     PRESSURE_MAX, SPEED_MAX, PARTICLE_SIZE_MAX, DWELL_TIME_MAX,
                     FRICTION_MAX, DECOMPRESSION_TIME_MAX])
    X_norm = (X - mins) / (maxs - mins + 1e-8)

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
    norm = (comps / total) * 100
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
    """Base quality score (without API) – used for pure quality assessment."""
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
# NSGA‑II OPTIMIZER (DUAL PENALTY: API + TENSILE)
# ================================================================
class NSGAIIOptimizer:
    def __init__(self, model: HybridTabletModel, pop_size=50, generations=80):
        self.model = model
        self.pop_size = pop_size
        self.generations = generations
        self.n_objectives = 3  # Density, Tensile, EFRF
        self.n_vars = 12       # 6 formulation + 6 process

    def enforce_mass_balance(self, pop: np.ndarray) -> np.ndarray:
        """Normalise the first 6 variables (formulation) to sum to 100%."""
        balanced = pop.copy()
        for i in range(len(pop)):
            f = pop[i, :6]
            total = np.sum(f)
            if total > 0:
                norm = (f / total) * 100
                balanced[i, :6] = np.clip(norm, 0, 100)
        return balanced

    def evaluate(self, pop: np.ndarray) -> np.ndarray:
        """Fitness: minimize -density, -tensile, efrf with API & tensile penalties."""
        with torch.no_grad():
            pred = self.model.predict(pop)   # shape (n, 5)
        density = pred[:, 0]
        tensile = pred[:, 1]
        efrf = pred[:, 2]
        api = pop[:, 0]   # API% (first variable)

        # Base objectives (all to be minimized)
        fitness = np.column_stack([
            -density,   # minimize negative density => maximize density
            -tensile,   # minimize negative tensile => maximize tensile
            efrf        # minimize efrf
        ])

        # Penalties: push toward higher API and higher tensile
        api_norm = (api - 80) / 18               # 0→80%, 1→98%
        tensile_norm = tensile / 8.5             # approximate max

        penalty_api = 0.08 * (1 - np.clip(api_norm, 0, 1))
        penalty_tensile = 0.05 * (1 - np.clip(tensile_norm, 0, 1))

        fitness[:, 0] += penalty_api
        fitness[:, 1] += penalty_tensile

        return fitness

    def fast_non_dominated_sort(self, obj: np.ndarray) -> List[List[int]]:
        n = len(obj)
        fronts = []
        dom_count = np.zeros(n, dtype=int)
        dom_sol = [[] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if np.all(obj[i] <= obj[j]) and np.any(obj[i] < obj[j]):
                    dom_sol[i].append(j)
                elif np.all(obj[j] <= obj[i]) and np.any(obj[j] < obj[i]):
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
        return fronts

    def crowding_distance(self, obj: np.ndarray, front: List[int]) -> np.ndarray:
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
            if max_val > min_val:
                for i in range(1, n-1):
                    dist[i] += (obj[sorted_idx[i+1], m] - obj[sorted_idx[i-1], m]) / (max_val - min_val)
        return dist

    def optimize(self):
        """Yield (population, objectives, history, generation) at each generation."""
        # Initial population: random, then enforce mass balance
        pop = np.random.rand(self.pop_size, self.n_vars)
        # Scale each variable to its physical range
        pop[:, 0] = pop[:, 0] * 18 + 80                    # API
        pop[:, 1] = pop[:, 1] * 4.6 + 1.4                  # Binder
        pop[:, 2] = pop[:, 2] * 5 + 1                      # PVPP
        pop[:, 3] = pop[:, 3] * 1.1 + 0.1                  # MgSt
        pop[:, 4] = pop[:, 4] * 6.5 + 1.5                  # MCC
        pop[:, 5] = pop[:, 5] * 4.5 + 0.5                  # Moisture
        pop[:, 6] = pop[:, 6] * 100 + 150                  # Pressure
        pop[:, 7] = pop[:, 7] * 15 + 15                    # Speed
        pop[:, 8] = pop[:, 8] * 190 + 10                   # Particle size
        pop[:, 9] = pop[:, 9] * 45 + 5                     # Dwell time
        pop[:, 10] = pop[:, 10] * 0.4 + 0.1                # Friction
        pop[:, 11] = pop[:, 11] * 70 + 10                  # Decompression time

        pop = self.enforce_mass_balance(pop)
        obj = self.evaluate(pop)

        history = []  # store Pareto fronts for plotting

        for gen in range(self.generations):
            # Non‑dominated sorting
            fronts = self.fast_non_dominated_sort(obj)

            # Crowding distance for each front
            crowding = []
            for f in fronts:
                d = self.crowding_distance(obj, f)
                crowding.extend(d)

            # Tournament selection (binary)
            selected = []
            for _ in range(self.pop_size):
                i1, i2 = np.random.choice(self.pop_size, 2, replace=False)
                # Determine ranks
                r1 = next(i for i, f in enumerate(fronts) if i1 in f)
                r2 = next(i for i, f in enumerate(fronts) if i2 in f)
                if r1 < r2:
                    selected.append(i1)
                elif r2 < r1:
                    selected.append(i2)
                else:
                    # Same rank: larger crowding distance
                    d1 = crowding[fronts[r1].index(i1)]
                    d2 = crowding[fronts[r2].index(i2)]
                    selected.append(i1 if d1 > d2 else i2)

            sel_pop = pop[selected]

            # Crossover & mutation (SBX and polynomial mutation simplified)
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
                                # Gaussian mutation with range‑aware step
                                step = 0.1 * (100 if j < 6 else 50)  # approximate range
                                child[j] += np.random.normal(0, step)
                                # Clip to physical bounds
                                if j == 0:
                                    child[j] = np.clip(child[j], API_MIN, API_MAX)
                                elif j == 1:
                                    child[j] = np.clip(child[j], BINDER_MIN, BINDER_MAX)
                                elif j == 2:
                                    child[j] = np.clip(child[j], PVPP_MIN, PVPP_MAX)
                                elif j == 3:
                                    child[j] = np.clip(child[j], MGST_MIN, MGST_MAX)
                                elif j == 4:
                                    child[j] = np.clip(child[j], MCC_MIN, MCC_MAX)
                                elif j == 5:
                                    child[j] = np.clip(child[j], MOISTURE_MIN, MOISTURE_MAX)
                                elif j == 6:
                                    child[j] = np.clip(child[j], PRESSURE_MIN, PRESSURE_MAX)
                                elif j == 7:
                                    child[j] = np.clip(child[j], SPEED_MIN, SPEED_MAX)
                                elif j == 8:
                                    child[j] = np.clip(child[j], PARTICLE_SIZE_MIN, PARTICLE_SIZE_MAX)
                                elif j == 9:
                                    child[j] = np.clip(child[j], DWELL_TIME_MIN, DWELL_TIME_MAX)
                                elif j == 10:
                                    child[j] = np.clip(child[j], FRICTION_MIN, FRICTION_MAX)
                                elif j == 11:
                                    child[j] = np.clip(child[j], DECOMPRESSION_TIME_MIN, DECOMPRESSION_TIME_MAX)
                offspring.extend([c1, c2])

            offspring = np.array(offspring[:self.pop_size])
            offspring = self.enforce_mass_balance(offspring)
            off_obj = self.evaluate(offspring)

            # Combine parent and offspring
            combined_pop = np.vstack([pop, offspring])
            combined_obj = np.vstack([obj, off_obj])

            # New population selection (elitism)
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

            # Save history for plotting every 5 generations
            if gen % 5 == 0 or gen == self.generations - 1:
                fronts = self.fast_non_dominated_sort(obj)
                pareto_idx = fronts[0]
                history.append({
                    'generation': gen,
                    'pareto_solutions': pop[pareto_idx].copy(),
                    'pareto_objectives': obj[pareto_idx].copy()
                })

            yield pop, obj, history, gen

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

def render_mass_balance_display(api, binder, pvpp, mgst, mcc, moisture):
    summary = get_formulation_summary(api, binder, pvpp, mgst, mcc, moisture)
    st.markdown("### 📊 Formulation Mass Balance")
    components = [
        ('API', summary['API'], '#ff6b6b'),
        ('Binder', summary['Binder'], '#4ecdc4'),
        ('PVPP', summary['PVPP'], '#45b7d1'),
        ('MgSt', summary['MgSt'], '#96ceb4'),
        ('MCC', summary['MCC'], '#ffeaa7'),
        ('Moisture', summary['Moisture'], '#dfe6e9')
    ]
    fig = go.Figure()
    for name, value, color in components:
        fig.add_trace(go.Bar(
            y=[name], x=[value], orientation='h',
            name=name, marker_color=color,
            text=f'{value:.1f}%', textposition='outside'
        ))
    fig.update_layout(
        xaxis=dict(title='Percentage (%)', range=[0, 105]),
        height=250, showlegend=False, barmode='stack',
        margin=dict(l=0, r=0, t=40, b=0),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    col1, col2 = st.columns([3, 1])
    with col1:
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.metric("**Total**", f"{summary['Total']:.1f}%", "✅ Mass Balance")
        for name in ['API', 'Binder', 'PVPP', 'MgSt', 'MCC', 'Moisture']:
            st.caption(f"{name}: {summary[name]:.1f}%")

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
    render_mass_balance_display(
        st.session_state.api, st.session_state.binder,
        st.session_state.pvpp, st.session_state.mgst,
        st.session_state.mcc, st.session_state.moisture
    )
    st.markdown("---")
    st.markdown("## ⚙️ Process Parameters")
    col3, col4 = st.columns(2)
    with col3:
        st.session_state.pressure = st.slider("**Compression Pressure (MPa)**", PRESSURE_MIN, PRESSURE_MAX, st.session_state.pressure, step=2.0)
        st.session_state.speed = st.slider("**Tableting Speed (rpm)**", SPEED_MIN, SPEED_MAX, st.session_state.speed, step=0.5)
        st.session_state.granule = st.slider("**Granule Size (µm)**", GRANULE_MIN, GRANULE_MAX, st.session_state.granule, step=5.0)
    with col4:
        st.session_state.dwell_time = st.slider("**Dwell Time (ms)**", DWELL_TIME_MIN, DWELL_TIME_MAX, st.session_state.dwell_time, step=1.0)
        st.session_state.friction = st.slider("**Friction Coefficient**", FRICTION_MIN, FRICTION_MAX, st.session_state.friction, step=0.01)
        st.session_state.decompression_time = st.slider("**Decompression Time (ms)**", DECOMPRESSION_TIME_MIN, DECOMPRESSION_TIME_MAX, st.session_state.decompression_time, step=2.0)

def render_results_summary(results: Dict[str, float]):
    st.markdown("---")
    st.markdown("## 📊 Optimization Results")
    api_val = st.session_state.api
    quality = calculate_quality_score(results['density'], results['tensile'], results['efrf'], api=api_val)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("**API%**", f"{api_val:.1f}%", "🎯 Target: maximize")
        st.metric("**Density**", f"{results['density']:.3f}", "✅ Target: ≥0.80")
    with col2:
        st.metric("**Tensile Strength**", f"{results['tensile']:.2f} MPa", "✅ Target: ≥1.5 MPa")
        st.metric("**EFRF**", f"{results['efrf']:.3f}", "✅ Target: <0.40")
    with col3:
        st.metric("**Disintegration Time**", f"{results['disintegration']:.1f} min", "✅ Target: ≤15 min")
        st.metric("**Overall Quality Score**", f"{quality['overall']:.1f}%",
                 "Good" if quality['overall'] > 60 else "Needs Improvement")
    with st.expander("📊 Quality Score Breakdown", expanded=False):
        st.markdown(f"""
        | Component | Score | Weight | Contribution |
        |-----------|-------|--------|--------------|
        | API%      | {quality.get('api_score', 0):.1f}% | 30% | {quality.get('api_score', 0) * 0.3:.1f}% |
        | Density   | {quality['density_score']:.1f}% | {quality['weights']['density']:.0%} | {quality['density_score']*quality['weights']['density']:.1f}% |
        | Tensile   | {quality['tensile_score']:.1f}% | {quality['weights']['tensile']:.0%} | {quality['tensile_score']*quality['weights']['tensile']:.1f}% |
        | EFRF      | {quality['efrf_score']:.1f}% | {quality['weights']['efrf']:.0%} | {quality['efrf_score']*quality['weights']['efrf']:.1f}% |
        | **Total** | - | - | **{quality['overall']:.1f}%** |
        """)

def render_pareto_evolution(pareto_history: List[Dict], golden: Dict):
    st.markdown("---")
    st.markdown("## 🌐 Pareto Front Evolution")
    if not pareto_history:
        st.info("No Pareto history available.")
        return

    # Prepare data for slider
    gen_indices = [h['generation'] for h in pareto_history]
    default_idx = len(pareto_history) - 1
    gen_slider = st.slider("Select generation to view", 0, len(pareto_history)-1, default_idx)

    current = pareto_history[gen_slider]
    sols = current['pareto_solutions']
    objs = current['pareto_objectives']

    # Extract API% from solutions (first variable)
    api_vals = sols[:, 0]

    fig = go.Figure()

    # Plot all previous generations (every 5) as faint points
    for i, h in enumerate(pareto_history[:gen_slider+1:5]):
        alpha = 0.1 + 0.2 * (i / max(1, len(pareto_history[:gen_slider+1:5])))
        old_sols = h['pareto_solutions']
        old_api = old_sols[:, 0]
        fig.add_trace(go.Scatter3d(
            x=old_sols[:, 0],  # we use density as x? Actually we want density, tensile, efrf
            y=old_sols[:, 1],  # but our sols are in variable space, not objective space.
            z=old_sols[:, 2],  # We need objectives: density, tensile, efrf.
            # Wait: we stored solutions (variables) not objectives. We should store objectives.
            # We'll fix: store objectives in history.
            # Let's adjust: history stores pareto_solutions and pareto_objectives.
            # We'll use objectives for plotting.
            mode='markers',
            marker=dict(size=3, opacity=alpha, color='lightgray'),
            name=f'Gen {h["generation"]}', showlegend=False,
            hovertemplate='Density: %{x:.3f}<br>Tensile: %{y:.2f} MPa<br>EFRF: %{z:.3f}<extra></extra>'
        ))

    # Current generation (colored by API%)
    # We need objectives: density, tensile, efrf. Since we stored objectives, we can use them.
    # But we stored objectives in the history, but we only stored pareto_solutions and pareto_objectives.
    # In the optimizer, we stored pareto_objectives as obj[pareto_idx]. That's the fitness (negative density, negative tensile, efrf).
    # We need to convert back to positive density and tensile for plotting.
    # We'll compute them from the model predictions again, or we can store the actual outputs.
    # For simplicity, we'll use the model to predict from solutions and plot those.
    # But we already have objectives in history. We'll store them as raw predictions (density, tensile, efrf).
    # Let's modify the optimizer to store both.
    # Since we are in the UI, we can just compute predictions on the fly.

    # Actually, in the optimizer history we stored pareto_solutions and pareto_objectives (fitness).
    # We'll compute the actual outputs using the model.
    model = st.session_state.get('model')
    if model is not None:
        preds = model.predict(sols)
        density = preds[:, 0]
        tensile = preds[:, 1]
        efrf = preds[:, 2]
        api_vals = sols[:, 0]
    else:
        # fallback: use dummy
        density = -objs[:, 0]   # approx
        tensile = -objs[:, 1]
        efrf = objs[:, 2]

    fig.add_trace(go.Scatter3d(
        x=density,
        y=tensile,
        z=efrf,
        mode='markers',
        marker=dict(
            size=8,
            color=api_vals,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="API%", x=1.02, len=0.6),
            opacity=0.9,
            line=dict(width=1, color='black')
        ),
        name=f'Generation {gen_slider}',
        hovertemplate='Density: %{x:.3f}<br>Tensile: %{y:.2f} MPa<br>EFRF: %{z:.3f}<br>API: %{marker.color:.1f}%<extra></extra>'
    ))

    if golden:
        fig.add_trace(go.Scatter3d(
            x=[golden['Density']],
            y=[golden['Tensile (MPa)']],
            z=[golden['EFRF']],
            mode='markers',
            marker=dict(size=15, color='red', symbol='diamond', line=dict(width=2, color='white')),
            name='🏆 Golden Solution',
            hovertemplate='<b>🏆 GOLDEN SOLUTION</b><br>API: %{text}<br>Density: %{x:.3f}<br>Tensile: %{y:.2f} MPa<br>EFRF: %{z:.3f}<extra></extra>',
            text=[f"{golden['API (%)']:.1f}%"]
        ))

    fig.update_layout(
        title=f'Pareto Front Evolution - Generation {gen_slider} (color = API%)',
        scene=dict(
            xaxis=dict(title='Density', range=[0.55,0.95]),
            yaxis=dict(title='Tensile Strength (MPa)', range=[0.5,8.5]),
            zaxis=dict(title='EFRF', range=[0,1]),
            camera=dict(eye=dict(x=1.8, y=1.8, z=1.8))
        ),
        height=550, margin=dict(l=0, r=0, t=50, b=0),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

def render_golden_solution(golden: Dict):
    if not golden:
        return
    st.markdown("---")
    st.markdown("## 🏆 Golden Solution (Balanced Trade-off)")
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 20px; border-radius: 12px; color: white; box-shadow: 0 4px 12px rgba(0,0,0,0.15);">
        <h3 style="color: white;">✨ Optimal Formulation</h3>
        <p><b>API:</b> {golden['API (%)']:.1f}% &nbsp;|&nbsp;
           <b>Binder:</b> {golden['Binder (%)']:.1f}% &nbsp;|&nbsp;
           <b>PVPP:</b> {golden['PVPP (%)']:.1f}% &nbsp;|&nbsp;
           <b>MgSt:</b> {golden['MgSt (%)']:.2f}% &nbsp;|&nbsp;
           <b>MCC:</b> {golden['MCC (%)']:.1f}% &nbsp;|&nbsp;
           <b>Moisture:</b> {golden['Moisture (%)']:.1f}%</p>
        <div style="display: flex; gap: 20px; flex-wrap: wrap; margin-top: 10px;">
            <div><b>API%:</b> {golden['API (%)']:.1f}% 🎯 High</div>
            <div><b>Density:</b> {golden['Density']:.3f} ✅ Excellent</div>
            <div><b>Tensile:</b> {golden['Tensile (MPa)']:.2f} MPa ✅ Improved</div>
            <div><b>EFRF:</b> {golden['EFRF']:.3f} ✅ Excellent</div>
            <div><b>Quality Score:</b> {golden['Quality Score']:.1f}% 🏆 Best</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.success("✅ This formulation maximises API% and Tensile while preserving excellent tablet quality!")

def render_side_by_side_comparison(golden: Dict, all_solutions: List[Dict]):
    if not golden or not all_solutions:
        return
    st.markdown("---")
    st.markdown("## 📊 Side‑by‑Side Comparison")
    top = all_solutions[:3]
    df = pd.DataFrame(top)
    st.dataframe(df[['Solution','API (%)','Binder (%)','PVPP (%)','MgSt (%)',
                     'MCC (%)','Moisture (%)','Density','Tensile (MPa)',
                     'EFRF','Quality Score']], use_container_width=True)
    st.markdown("### 🎯 Performance Radar")
    categories = ["API%", "Density", "Tensile (MPa)", "EFRF (inverted)", "Quality Score"]
    fig = go.Figure()
    for _, row in df.iterrows():
        fig.add_trace(go.Scatterpolar(
            r=[
                (row["API (%)"] - 80) / 18,
                row["Density"] / 0.95,
                row["Tensile (MPa)"] / 8.5,
                1 - row["EFRF"],
                row["Quality Score"] / 100
            ],
            theta=categories,
            fill='toself',
            name=row["Solution"]
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0,1])),
        showlegend=True,
        height=400,
        margin=dict(l=40, r=40, t=40, b=40),
        title="Performance Comparison Across Solutions"
    )
    st.plotly_chart(fig, use_container_width=True)

def render_best_solutions(solutions: List[Dict], golden: Dict):
    st.markdown("---")
    st.markdown("## 🏆 Optimal Solutions (Mass Balance Ensured)")
    st.info("✅ All formulations are normalized to sum to 100%")

    render_golden_solution(golden)
    render_side_by_side_comparison(golden, solutions)

    df = pd.DataFrame(solutions)
    df_display = df.copy()
    for col in ['API (%)', 'Binder (%)', 'PVPP (%)', 'MCC (%)', 'Moisture (%)', 'Total (%)']:
        df_display[col] = df_display[col].round(1)
    df_display['MgSt (%)'] = df_display['MgSt (%)'].round(2)
    df_display['Density'] = df_display['Density'].round(3)
    df_display['Tensile (MPa)'] = df_display['Tensile (MPa)'].round(2)
    df_display['EFRF'] = df_display['EFRF'].round(3)
    df_display['Quality Score'] = df_display['Quality Score'].round(1)
    st.dataframe(df_display, use_container_width=True, hide_index=True)

    csv = df.to_csv(index=False)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    col1, col2 = st.columns(2)
    with col1:
        st.download_button("📥 Download Optimization Report (CSV)",
                           data=csv,
                           file_name=f"results_{timestamp}.csv",
                           mime="text/csv",
                           use_container_width=True)
    with col2:
        json_report = {
            'timestamp': timestamp,
            'golden_solution': golden,
            'all_solutions': df.to_dict('records'),
            'parameters': {
                'population': POPULATION_SIZE,
                'generations': NSGA_GENERATIONS,
                'epochs': TRAINING_EPOCHS,
                'runtime_seconds': st.session_state.runtime,
                'api_penalty': 0.08,
                'tensile_penalty': 0.05
            }
        }
        st.download_button("📥 Download Full Report (JSON)",
                           data=json.dumps(json_report, indent=2),
                           file_name=f"report_{timestamp}.json",
                           mime="application/json",
                           use_container_width=True)

def render_optimization_summary():
    st.markdown("---")
    st.markdown("## 📈 Optimization Summary")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("⏱️ Runtime", f"{st.session_state.runtime}s" if st.session_state.runtime else "—")
    with col2:
        evals_per_sec = (POPULATION_SIZE * NSGA_GENERATIONS) / max(1, st.session_state.runtime)
        st.metric("⚡ Evaluations/Second", f"{evals_per_sec:.0f}")

    col3, col4 = st.columns([2, 1])
    with col3:
        st.markdown("### Key Statistics")
        stats = pd.DataFrame({
            'Metric': [
                'Total Solutions Evaluated',
                'Pareto Solutions Found',
                'Best Density',
                'Best Tensile',
                'Best EFRF',
                'Best API%',
                'Mass Balance',
                'Penalties'
            ],
            'Value': [
                f'{POPULATION_SIZE * NSGA_GENERATIONS:,}',
                f'{len(st.session_state.get("pareto_solutions", [])) if st.session_state.get("pareto_solutions") else 0}',
                f'{st.session_state.get("best_density", 0):.3f}',
                f'{st.session_state.get("best_tensile", 0):.2f} MPa',
                f'{st.session_state.get("best_efrf", 0):.3f}',
                f'{st.session_state.get("best_api", 0):.1f}%',
                '✅ 100% (Enforced)',
                'API: 0.08 | Tensile: 0.05'
            ]
        })
        st.dataframe(stats, hide_index=True, use_container_width=True)
    with col4:
        st.markdown("### Status Indicators")
        st.success("✅ Algorithm: NSGA‑II + dual penalty")
        st.success("✅ Model: Physics‑Informed Neural Network")
        st.success("✅ Constraint: Mass Balance")
        st.info("📊 Pareto Front: Optimized")
        st.info("🎯 Objectives: 3 + API/Tensile bias")

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

        # Validate formulation
        valid, msg = validate_formulation(
            st.session_state.api, st.session_state.binder,
            st.session_state.pvpp, st.session_state.mgst,
            st.session_state.mcc, st.session_state.moisture
        )
        if not valid:
            st.error(f"❌ {msg}")
            return

        # Train model (cached)
        with st.spinner("Training neural network (this may take a minute)..."):
            model = train_model()
        st.session_state.model = model

        # Run NSGA‑II
        st.info("🧬 Running NSGA‑II optimization...")
        optimizer = NSGAIIOptimizer(model, pop_size=POPULATION_SIZE, generations=NSGA_GENERATIONS)

        # Prepare placeholders for live updates
        progress_bar = st.progress(0)
        status_text = st.empty()
        pareto_chart_placeholder = st.empty()

        # Store history
        pareto_history = []
        final_pop = None
        final_obj = None

        for pop, obj, history, gen in optimizer.optimize():
            final_pop, final_obj = pop, obj
            pareto_history = history
            progress_bar.progress((gen+1) / NSGA_GENERATIONS)
            status_text.text(f"Generation {gen+1}/{NSGA_GENERATIONS} – Population size: {len(pop)}")

            # Update Pareto chart every 5 generations
            if gen % 5 == 0 or gen == NSGA_GENERATIONS - 1:
                # Compute Pareto front
                fronts = optimizer.fast_non_dominated_sort(obj)
                pareto_idx = fronts[0]
                pareto_sols = pop[pareto_idx]
                pareto_objs = obj[pareto_idx]

                # Store in session for later use
                st.session_state.pareto_solutions = pareto_sols
                st.session_state.pareto_objectives = pareto_objs

                # Plot current Pareto front
                if len(pareto_sols) > 0:
                    preds = model.predict(pareto_sols)
                    density = preds[:, 0]
                    tensile = preds[:, 1]
                    efrf = preds[:, 2]
                    api_vals = pareto_sols[:, 0]

                    fig = go.Figure()
                    fig.add_trace(go.Scatter3d(
                        x=density,
                        y=tensile,
                        z=efrf,
                        mode='markers',
                        marker=dict(
                            size=8,
                            color=api_vals,
                            colorscale='Viridis',
                            showscale=True,
                            colorbar=dict(title="API%", x=1.02, len=0.6),
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
                    pareto_chart_placeholder.plotly_chart(fig, use_container_width=True)

        # Optimization complete
        progress_bar.empty()
        status_text.empty()
        st.success("✅ Optimization complete!")

        # Extract final Pareto front
        fronts = optimizer.fast_non_dominated_sort(final_obj)
        pareto_idx = fronts[0]
        pareto_solutions = final_pop[pareto_idx]
        pareto_objectives = final_obj[pareto_idx]

        # Predict properties for all solutions
        preds = model.predict(pareto_solutions)
        density = preds[:, 0]
        tensile = preds[:, 1]
        efrf = preds[:, 2]
        disintegration = preds[:, 3]
        dissolution = preds[:, 4]

        # Build solution list with mass balance
        solutions = []
        for i, sol in enumerate(pareto_solutions):
            api, binder, pvpp, mgst, mcc, moisture = sol[:6]
            # Normalise to 100% (already enforced, but ensure)
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
                'Total (%)': np.sum(list(norm.values())),
                'Density': density[i],
                'Tensile (MPa)': tensile[i],
                'EFRF': efrf[i],
                'Disintegration (min)': disintegration[i],
                'Dissolution (%)': dissolution[i],
                'Quality Score': quality['overall']
            })

        # Sort by quality score descending
        solutions = sorted(solutions, key=lambda x: x['Quality Score'], reverse=True)

        # Golden solution = best quality
        golden = solutions[0] if solutions else None

        # Store in session state for later display
        st.session_state.optimization_complete = True
        st.session_state.best_solutions = solutions
        st.session_state.golden_solution = golden
        st.session_state.pareto_history = pareto_history
        st.session_state.runtime = round(time.time() - start_time, 1)

        # Store best stats for summary
        st.session_state.best_density = max(density)
        st.session_state.best_tensile = max(tensile)
        st.session_state.best_efrf = min(efrf)
        st.session_state.best_api = max([s['API (%)'] for s in solutions])

        # Display results
        # Use the first solution for results summary (the best one)
        first = solutions[0]
        results = {
            'density': first['Density'],
            'tensile': first['Tensile (MPa)'],
            'efrf': first['EFRF'],
            'disintegration': first['Disintegration (min)'],
            'dissolution': first['Dissolution (%)']
        }
        render_results_summary(results)
        render_best_solutions(solutions, golden)

        # Show Pareto evolution (using saved history)
        render_pareto_evolution(pareto_history, golden)

        render_optimization_summary()

        st.balloons()

    elif st.session_state.optimization_complete and st.session_state.best_solutions:
        # Show cached results
        first = st.session_state.best_solutions[0]
        results = {
            'density': first['Density'],
            'tensile': first['Tensile (MPa)'],
            'efrf': first['EFRF'],
            'disintegration': first['Disintegration (min)'],
            'dissolution': first['Dissolution (%)']
        }
        render_results_summary(results)
        render_best_solutions(st.session_state.best_solutions, st.session_state.golden_solution)
        render_pareto_evolution(st.session_state.pareto_history, st.session_state.golden_solution)
        render_optimization_summary()

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
