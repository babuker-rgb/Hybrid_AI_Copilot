# ================================================================
# Hybrid AI v32.0-Ultimate · Unified Release
# (NSGA-III Adaptive + PINN + Physics + 2D/3D/Radar)
# ================================================================

import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import plotly.graph_objects as go
import time
import warnings
import json
import os
import tempfile
from datetime import datetime
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings('ignore')

# ================================================================
# PAGE CONFIG
# ================================================================
st.set_page_config(
    page_title="Hybrid AI v32.0-Ultimate", page_icon="🧬", layout="wide"
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
EFRF_THRESHOLD = 0.40

POPULATION_SIZE = 80
NSGA_GENERATIONS = 80
TRAINING_EPOCHS = 1200
EARLY_STOPPING_PATIENCE = 100
HIDDEN_SIZE = 512
N_SAMPLES = 10000

BINDER_GRADES = {
    "MCC PH101": {"compressibility": 0.85, "disintegration": 0.90, "flow": 0.80},
    "MCC PH102": {"compressibility": 0.90, "disintegration": 0.85, "flow": 0.85},
    "MCC PH200": {"compressibility": 0.95, "disintegration": 0.80, "flow": 0.90},
    "MCC KG": {"compressibility": 0.88, "disintegration": 0.88, "flow": 0.82},
    "Lactose Monohydrate": {"compressibility": 0.75, "disintegration": 0.95, "flow": 0.78},
    "Dicalcium Phosphate": {"compressibility": 0.70, "disintegration": 0.85, "flow": 0.75}
}
BINDER_GRADE_NAMES = list(BINDER_GRADES.keys())

# ================================================================
# HELPER FUNCTIONS
# ================================================================
def normalize_formulation(api, binder, pvpp, mgst, mcc, moisture):
    comps = np.array([api, binder, pvpp, mgst, mcc, moisture])
    total = np.sum(comps)
    if total <= 0: total = 1.0
    norm = (comps / total) * 100
    return {'api': norm[0], 'binder': norm[1], 'pvpp': norm[2],
            'mgst': norm[3], 'mcc': norm[4], 'moisture': norm[5]}

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
                'tensile_score': tensile_score, 'efrf_score': efrf_score, 'weights': weights}

# ================================================================
# SCALER & HYBRID PINN MODEL
# ================================================================
class InputScaler:
    def fit(self, X): 
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ < 1e-8] = 1.0
        return self
    def transform(self, X): 
        return (X - self.mean_) / self.std_

class HybridTabletModel(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=HIDDEN_SIZE):
        super().__init__()
        self.fc1, self.bn1 = nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim)
        self.fc2, self.bn2 = nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim)
        self.fc3, self.bn3 = nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim)
        self.fc4, self.bn4 = nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim)
        self.fc5, self.dropout = nn.Linear(hidden_dim, 5), nn.Dropout(0.1)
        for m in self.modules():
            if isinstance(m, nn.Linear): nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x):
        h1 = torch.relu(self.bn1(self.fc1(x))); h1 = self.dropout(h1)
        h2 = torch.relu(self.bn2(self.fc2(h1)))+h1; h2 = self.dropout(h2)
        h3 = torch.relu(self.bn3(self.fc3(h2)))+h2; h3 = self.dropout(h3)
        out = self.fc5(h3)
        density = torch.sigmoid(out[:,0])*0.4+0.55
        tensile = torch.sigmoid(out[:,1])*8.0+0.5
        efrf = torch.sigmoid(out[:,2])
        disintegration = torch.sigmoid(out[:,3])*45.0+2.0
        dissolution = torch.sigmoid(out[:,4])*80.0+10.0
        return torch.stack([density, tensile, efrf, disintegration, dissolution], 1)

    def predict_with_uncertainty(self, x, n_samples=20):
        self.train()
        with torch.no_grad():
            if not torch.is_tensor(x): x = torch.tensor(x, dtype=torch.float32)
            x_repeat = x.repeat(n_samples, 1)
            preds = self.forward(x_repeat).numpy().reshape(n_samples, -1, 5)
        self.eval()
        return np.mean(preds, 0), np.std(preds, 0)

# ================================================================
# DATA GENERATION & PHYSICS
# ================================================================
def generate_synthetic_data(n_samples=N_SAMPLES, seed=42):
    rng = np.random.default_rng(seed)
    api = rng.uniform(API_MIN, API_MAX, n_samples)
    binder = rng.uniform(BINDER_MIN, BINDER_MAX, n_samples)
    pvpp = rng.uniform(PVPP_MIN, PVPP_MAX, n_samples)
    mgst = rng.uniform(MGST_MIN, MGST_MAX, n_samples)
    mcc = rng.uniform(MCC_MIN, MCC_MAX, n_samples)
    moisture = rng.uniform(MOISTURE_MIN, MOISTURE_MAX, n_samples)
    pressure = rng.uniform(PRESSURE_MIN, PRESSURE_MAX, n_samples)
    speed = rng.uniform(SPEED_MIN, SPEED_MAX, n_samples)

    X = np.column_stack([api, binder, pvpp, mgst, mcc, moisture, pressure, speed]).astype(np.float32)
    density = np.clip(0.55 + 0.3 * (pressure-150)/100 - 0.01*(binder-3.0) + rng.normal(0,0.01,n_samples), 0.55, 0.95)
    tensile = np.clip(1.0 + 6.0*(density-0.55) + 0.2*(api-80)/18 - 0.5*(mgst-0.1) + rng.normal(0,0.2,n_samples), 0.5, 8.5)
    efrf = np.clip(0.6 - 0.5*(density-0.55) + 0.2*(mgst-0.1) + rng.normal(0,0.05,n_samples), 0.02, 0.98)
    disintegration = np.clip(10.0 - 2.0*(pvpp-1.0)/5.0 + 3.0*(binder-1.4)/4.6 + rng.normal(0,1.0,n_samples), 2.0, 45.0)
    dissolution = np.clip(2.0*disintegration + 10.0 + rng.normal(0,2.0,n_samples), 10.0, 90.0)
    y = np.column_stack([density, tensile, efrf, disintegration, dissolution]).astype(np.float32)
    return X, y

def calculate_heckel_density(pressure, binder):
    return 0.55 + 0.3 * (pressure - 150) / 100 - 0.01 * (binder - 3.0)

# ================================================================
# TRAINING LOOP (Stable Cache, Progress bar, Physics Loss)
# ================================================================
CHECKPOINT_PATH = os.path.join(tempfile.gettempdir(), 'hybrid_ai_v32_ultimate.pt')

@st.cache_resource(show_spinner=False)
def train_model():
    if os.path.exists(CHECKPOINT_PATH):
        try:
            ckpt = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)
            model = HybridTabletModel(input_dim=8, hidden_dim=HIDDEN_SIZE)
            model.load_state_dict(ckpt['model_state'])
            model.eval()
            return model, ckpt['scaler']
        except: pass

    X, y = generate_synthetic_data(n_samples=N_SAMPLES)
    scaler = InputScaler().fit(X)
    X_scaled = scaler.transform(X)
    X_t, y_t = torch.tensor(X_scaled, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    
    model = HybridTabletModel(input_dim=8, hidden_dim=HIDDEN_SIZE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    target_var = torch.clamp(y_t.var(0, unbiased=False), min=1e-6)
    def mse(pred, true): return (((pred - true) ** 2) / target_var).mean()
    
    pressure_input, binder_input = X[:, 6], X[:, 1]
    best_loss = np.inf; patience = 0
    
    for epoch in range(TRAINING_EPOCHS):
        model.train(); opt.zero_grad(); pred = model(X_t)
        loss = mse(pred, y_t)
        physical = torch.tensor(calculate_heckel_density(pressure_input, binder_input), dtype=torch.float32)
        physics_loss = torch.mean((pred[:, 0] - physical) ** 2) * 0.1
        loss += physics_loss
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        
        if epoch % 10 == 0:
            if '_train_pb' in st.session_state and st.session_state['_train_pb'] is not None:
                st.session_state['_train_pb'].progress(epoch / TRAINING_EPOCHS)
            time.sleep(0.01)
            
        if epoch % 100 == 0:
            val = mse(model(X_t), y_t).item()
            if val < best_loss: best_loss = val; patience = 0
            else: patience += 1
            if patience >= EARLY_STOPPING_PATIENCE: break
            
    model.eval()
    torch.save({'model_state': model.state_dict(), 'scaler': scaler}, CHECKPOINT_PATH)
    return model, scaler

# ================================================================
# ADVANCED OPTIMIZER (NSGA-II + Adaptive Mutation)
# ================================================================
class AdvancedOptimizer:
    def __init__(self, model, scaler, pop_size=POPULATION_SIZE, generations=NSGA_GENERATIONS):
        self.model, self.scaler = model, scaler
        self.pop_size, self.generations = pop_size, generations
        self.mutation_rate = 0.1
        self.GENE_BOUNDS = [
            (API_MIN, API_MAX), (BINDER_MIN, BINDER_MAX), (PVPP_MIN, PVPP_MAX),
            (MGST_MIN, MGST_MAX), (MCC_MIN, MCC_MAX), (MOISTURE_MIN, MOISTURE_MAX),
            (PRESSURE_MIN, PRESSURE_MAX), (SPEED_MIN, SPEED_MAX)
        ]

    def enforce_mass_balance(self, pop):
        balanced = pop.copy()
        lo = np.array([b[0] for b in self.GENE_BOUNDS[:6]])
        hi = np.array([b[1] for b in self.GENE_BOUNDS[:6]])
        comps = np.clip(pop[:, :6], lo, hi)
        total = comps.sum(axis=1, keepdims=True)
        balanced[:, :6] = np.clip(comps / (total if (total > 0).all() else 1.0) * 100.0, lo, hi)
        return balanced

    def evaluate(self, pop):
        pop_scaled = self.scaler.transform(pop)
        if isinstance(pop_scaled, pd.DataFrame): pop_scaled = pop_scaled.values
        self.model.eval()
        with torch.no_grad():
            pred = self.model(torch.tensor(pop_scaled, dtype=torch.float32)).numpy()
        density, tensile, efrf = pred[:, 0], pred[:, 1], pred[:, 2]
        penalty = 1.0 / (1.0 + np.clip(np.abs(pop_scaled) - 2.5, 0, None).sum(axis=1))
        fitness = np.column_stack([-density*penalty, -tensile*penalty, -pop[:,0]*penalty, efrf*penalty])
        fitness[:, 3] += np.maximum(0, efrf - EFRF_THRESHOLD) * 20.0
        return fitness

    def adaptive_mutation(self, pop):
        diversity = np.std(pop, axis=0).mean()
        if diversity < 0.05: self.mutation_rate = min(0.2, self.mutation_rate + 0.02)
        elif diversity > 0.2: self.mutation_rate = max(0.02, self.mutation_rate - 0.01)
        return diversity

    def optimize(self):
        pop = np.random.rand(self.pop_size, 8)
        for i, (lo, hi) in enumerate(self.GENE_BOUNDS): pop[:, i] = pop[:, i] * (hi - lo) + lo
        pop = self.enforce_mass_balance(pop)
        obj = self.evaluate(pop)
        history = []
        for gen in range(self.generations):
            self.adaptive_mutation(pop)
            selected = []
            for _ in range(self.pop_size):
                idx = np.random.choice(self.pop_size, 2, replace=False)
                selected.append(idx[np.argmin(obj[idx].sum(axis=1))])
            sel_pop = pop[selected]
            offspring = []
            for i in range(0, self.pop_size, 2):
                p1, p2 = sel_pop[i], sel_pop[(i+1)%self.pop_size]
                c1, c2 = p1.copy(), p2.copy()
                for j in range(8):
                    if np.random.rand() < 0.8:
                        beta = 1.0 + 2.0 * np.random.rand()
                        c1[j] = 0.5 * ((1+beta)*p1[j] + (1-beta)*p2[j])
                        c2[j] = 0.5 * ((1-beta)*p1[j] + (1+beta)*p2[j])
                    if np.random.rand() < self.mutation_rate:
                        lo, hi = self.GENE_BOUNDS[j]
                        c1[j] = np.clip(c1[j] + np.random.normal(0, 0.1) * (hi-lo), lo, hi)
                offspring.extend([c1, c2])
            offspring = np.array(offspring[:self.pop_size])
            offspring = self.enforce_mass_balance(offspring)
            combined = np.vstack([pop, offspring])
            combined_obj = np.vstack([obj, self.evaluate(offspring)])
            pareto_idx = np.argsort(combined_obj.sum(axis=1))[:self.pop_size]
            pop, obj = combined[pareto_idx], combined_obj[pareto_idx]
            if gen % 10 == 0 or gen == self.generations - 1:
                history.append({'generation': gen, 'population': pop.copy(), 'objectives': obj.copy()})
            yield pop, obj, history, gen

# ================================================================
# ANALYSIS & PLOTTING FUNCTIONS
# ================================================================
def perform_sensitivity_analysis(model, scaler, ref_solution):
    try:
        rf = RandomForestRegressor(n_estimators=50)
        X_local = np.random.normal(loc=ref_solution, scale=0.05*np.abs(ref_solution), size=(500, 8))
        bounds_min = np.array([API_MIN, BINDER_MIN, PVPP_MIN, MGST_MIN, MCC_MIN, MOISTURE_MIN, PRESSURE_MIN, SPEED_MIN])
        bounds_max = np.array([API_MAX, BINDER_MAX, PVPP_MAX, MGST_MAX, MCC_MAX, MOISTURE_MAX, PRESSURE_MAX, SPEED_MAX])
        X_local = np.clip(X_local, bounds_min, bounds_max)
        X_scaled = scaler.transform(X_local)
        y_local = model(torch.tensor(X_scaled, dtype=torch.float32)).numpy()[:, 0]
        rf.fit(X_scaled, y_local)
        perm_importance = permutation_importance(rf, X_scaled, y_local)
        feature_names = ['API', 'Binder', 'PVPP', 'MgSt', 'MCC', 'Moisture', 'Pressure', 'Speed']
        return dict(zip(feature_names, perm_importance.importances_mean))
    except: return None

def render_2d_pareto_evolution(pareto_history, golden, tested=None):
    if not pareto_history: return
    generations_recorded = [h['generation'] for h in pareto_history]
    gen_slider = st.select_slider("Select generation to view", options=generations_recorded, value=generations_recorded[-1])
    current_entry = next(h for h in pareto_history if h['generation'] == gen_slider)
    obj = current_entry['objectives']; pop = current_entry['population']
    api_vals, efrf_vals = pop[:, 0], obj[:, 3]
    feasible_mask = efrf_vals < 0.40
    api_feas, efrf_feas = api_vals[feasible_mask], efrf_vals[feasible_mask]
    sort_idx = np.argsort(api_feas)
    api_sorted, efrf_sorted = api_feas[sort_idx], efrf_feas[sort_idx]
    if len(efrf_sorted) > 0: cummax_efrf = np.maximum.accumulate(efrf_sorted)
    else: cummax_efrf = efrf_sorted

    fig = go.Figure()
    fig.add_hrect(y0=0, y1=0.40, x0=API_MIN, x1=API_MAX, fillcolor='rgba(144, 238, 144, 0.25)', line_width=0, layer='below')
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(size=12, symbol='square', color='rgba(144, 238, 144, 0.5)'), name='Feasible region (EFRF < 0.40)'))
    
    fig.add_trace(go.Scatter(x=api_sorted, y=cummax_efrf, mode='lines+markers', name='Pareto Front', line=dict(color='red', width=2), marker=dict(size=8, color='#a3c4f3', line=dict(width=1, color='#4a6fa5'))))
    
    if golden:
        fig.add_trace(go.Scatter(x=[golden['API (%)']], y=[golden['EFRF']], mode='markers', name='🏆 Golden Solution', marker=dict(size=22, color='gold', symbol='star', line=dict(width=1.5, color='#8a6d00'))))
    if tested:
        fig.add_trace(go.Scatter(x=[tested['api']], y=[tested['efrf']], mode='markers', name='🔵 Tested Formulation', marker=dict(size=14, color='blue', symbol='circle', line=dict(width=1.5, color='white'))))

    fig.add_hline(y=0.40, line_dash='dash', line_color='gray', annotation_text='EFRF limit (0.40)', annotation_position='top left')
    fig.add_vline(x=API_MIN, line_dash='dash', line_color='gray', annotation_text=f'API min ({API_MIN}%)', annotation_position='bottom left')
    fig.add_vline(x=API_MAX, line_dash='dash', line_color='gray', annotation_text=f'API max ({API_MAX}%)', annotation_position='bottom right')
    
    fig.update_layout(title=f'2D Pareto Front - Generation {gen_slider}', xaxis_title='API (%)', yaxis_title='EFRF', height=500, template='plotly_white', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig, use_container_width=True)

def render_3d_pareto(pop, obj, golden_idx, tested_data=None):
    fig = go.Figure(data=[go.Scatter3d(x=pop[:, 0], y=obj[:, 3], z=-obj[:, 1], mode='markers', marker=dict(size=4, color=pop[:, 0], colorscale='Viridis'), name='Pareto')])
    if golden_idx is not None:
        fig.add_trace(go.Scatter3d(x=[pop[golden_idx, 0]], y=[obj[golden_idx, 3]], z=[-obj[golden_idx, 1]], mode='markers', marker=dict(size=15, color='gold', symbol='diamond'), name='Golden'))
    if tested_data is not None:
        fig.add_trace(go.Scatter3d(x=[tested_data['api']], y=[tested_data['efrf']], z=[tested_data['tensile']], mode='markers', marker=dict(size=12, color='blue', symbol='circle', line=dict(color='white', width=1)), name='Tested Formulation'))
    fig.update_layout(scene=dict(xaxis_title='API (%)', yaxis_title='EFRF', zaxis_title='Tensile (MPa)'), height=450)
    st.plotly_chart(fig, use_container_width=True)

def render_dynamic_radar(solutions_df, selected_solutions):
    if not selected_solutions: return
    fig = go.Figure()
    for i, row in solutions_df.iterrows():
        if row['Solution'] in selected_solutions:
            fig.add_trace(go.Scatterpolar(r=[(row['API (%)']-80)/18, row['Density']/0.95, row['Tensile (MPa)']/8.5, 1-row['EFRF'], row['Quality Score']/100], theta=['API%', 'Density', 'Tensile', 'EFRF (Inv)', 'Quality'], fill='toself', name=row['Solution']))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])), showlegend=True, height=380)
    st.plotly_chart(fig, use_container_width=True)

def target_status(value, threshold, mode='min', comfortable=None):
    if mode == 'min':
        if value < threshold: return "🔴 Below target"
        if comfortable is not None and value >= comfortable: return "✅ Excellent"
        return "✅ Passes (near limit)"
    else:
        if value > threshold: return "🔴 Exceeds limit"
        if comfortable is not None and value <= comfortable: return "✅ Excellent"
        return "⚠️ Passes (near limit)"

# ================================================================
# UI RENDER FUNCTIONS
# ================================================================
def render_sidebar():
    with st.sidebar:
        st.markdown("## 🧬 Hybrid AI Framework")
        st.markdown("---"); st.markdown(f"**Version:** v32.0-Ultimate")
        st.markdown(f"**Institution:** Nile Valley University")
        st.markdown("---")
        st.sidebar.header("⚖️ Custom Recommender")
        w_api = st.sidebar.slider("Weight for API", 0.0, 1.0, 0.4)
        w_quality = st.sidebar.slider("Weight for Quality", 0.0, 1.0, 0.6)
        st.sidebar.info("Prioritizes solutions matching your weights.")
        st.markdown("---")
        st.caption("© 2024 Nile Valley University · Sudan")
        return w_api, w_quality

def render_input_panel():
    st.markdown("## ⚙️ Formulation & Process")
    col1, col2 = st.columns(2)
    with col1:
        api = st.slider("API (%)", API_MIN, API_MAX, 85.0)
        binder = st.slider("Binder (%)", BINDER_MIN, BINDER_MAX, 5.0)
        pvpp = st.slider("PVPP (%)", PVPP_MIN, PVPP_MAX, 2.0)
        mgst = st.slider("MgSt (%)", MGST_MIN, MGST_MAX, 0.5)
    with col2:
        mcc = st.slider("MCC (%)", MCC_MIN, MCC_MAX, 4.0)
        moisture = st.slider("Moisture (%)", MOISTURE_MIN, MOISTURE_MAX, 1.5)
        pressure = st.slider("Pressure (MPa)", PRESSURE_MIN, PRESSURE_MAX, 200.0)
        speed = st.slider("Speed (rpm)", SPEED_MIN, SPEED_MAX, 20.0)
    return api, binder, pvpp, mgst, mcc, moisture, pressure, speed

# ================================================================
# MAIN APPLICATION
# ================================================================
def main():
    st.title("🧬 Hybrid AI v32.0-Ultimate · Unified 2D/3D/Radar")
    w_api, w_quality = render_sidebar()
    api, binder, pvpp, mgst, mcc, moisture, pressure, speed = render_input_panel()

    if st.button("🚀 Run Ultimate Optimization"):
        # 1. Prepare Progress Bar & Train Model
        progress_bar = st.progress(0)
        st.session_state['_train_pb'] = progress_bar
        with st.spinner("Training Physics Model (1st time takes ~15s)..."):
            model, scaler = train_model()
            st.session_state['_train_pb'].progress(1.0)

        # 2. Run Optimization
        start_time = time.time()
        with st.status("Running NSGA-II + Adaptive Mutation...", expanded=True) as status:
            opt_bar = st.progress(0)
            optimizer = AdvancedOptimizer(model, scaler)
            final_pop, final_obj, gen_history = None, None, None
            for i, (pop, obj, history, gen) in enumerate(optimizer.optimize()):
                final_pop, final_obj, gen_history = pop, obj, history
                opt_bar.progress((gen+1)/NSGA_GENERATIONS)
                if gen % 10 == 0: status.update(label=f"Generation {gen+1}/{NSGA_GENERATIONS}")
            status.update(label="Optimization Complete ✅", state="complete")

        # 3. Process Solutions & Golden Star
        weights = np.array([w_api, w_quality])
        scores = []
        for i in range(len(final_pop)):
            s = (final_pop[i,0]/100 * weights[0]) + ((1 - final_obj[i].sum()/4) * weights[1])
            scores.append(s)
        golden_idx = np.argmax(scores)
        best_sol = final_pop[golden_idx]
        
        pop_scaled = scaler.transform([best_sol])
        preds, unc = model.predict_with_uncertainty(torch.tensor(pop_scaled, dtype=torch.float32))
        preds, unc = preds[0], unc[0]
        st.success(f"🏆 Golden Solution Found!\nAPI: {best_sol[0]:.2f}% | EFRF: {preds[2]:.3f} ± {unc[2]:.3f}")
        st.caption(f"Optimization took {time.time() - start_time:.2f} seconds.")

        # 4. Compute Tested Formulation Data
        slider_form = np.array([[api, binder, pvpp, mgst, mcc, moisture, pressure, speed]], dtype=np.float32)
        slider_preds, _ = model.predict_with_uncertainty(torch.tensor(scaler.transform(slider_form), dtype=torch.float32))
        tested_data = {'api': float(api), 'efrf': float(slider_preds[0][2]), 'tensile': float(slider_preds[0][1])}

        # 5. Build Solutions DataFrame for Radar / Exports
        sol_list = []
        sorted_indices = np.argsort([-scores[i] for i in range(len(final_pop))])
        for idx in sorted_indices[:10]:
            sol_list.append({
                'Solution': f'S{idx+1}',
                'API (%)': float(final_pop[idx, 0]),
                'Density': float(-final_obj[idx, 0]),
                'Tensile (MPa)': float(-final_obj[idx, 1]),
                'EFRF': float(final_obj[idx, 3]),
                'Quality Score': float(100 - (final_obj[idx].sum() * 20))
            })
        sol_df = pd.DataFrame(sol_list)
        golden = {'API (%)': best_sol[0], 'EFRF': preds[2]} # Simplified golden dict for plots

        # 6. Render All Visualizations
        st.subheader("🌐 2D Pareto Front (API vs EFRF)")
        render_2d_pareto_evolution(gen_history, golden, tested_data)
        
        st.subheader("🌐 3D Pareto Front (API - EFRF - Tensile)")
        render_3d_pareto(final_pop, final_obj, golden_idx, tested_data)
        
        col_a, col_b = st.columns([1, 2])
        with col_a:
            st.subheader("🏆 Top Solutions")
            selected = st.multiselect("Select for Radar", sol_df['Solution'], default=[f'S{str(sorted_indices[0]+1)}', f'S{str(sorted_indices[1]+1)}'])
        with col_b:
            st.subheader("📊 Dynamic Radar Comparison")
            render_dynamic_radar(sol_df, selected)

        # 7. Sensitivity & Export
        with st.expander("🔬 Sensitivity Analysis (Local)"):
            sens_data = perform_sensitivity_analysis(model, scaler, best_sol)
            if sens_data: st.bar_chart(pd.Series(sens_data))
            else: st.warning("Could not compute local sensitivity.")
        
        report = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'golden_api': float(best_sol[0]), 'golden_efrf': float(preds[2]), 'golden_tensile': float(preds[1]),
            'top_solutions': sol_df.to_dict('records'), 'tested_formulation': tested_data
        }
        st.download_button("📥 Download Report (JSON)", data=json.dumps(report, indent=2, default=str), file_name="ultimate_report.json")

if __name__ == "__main__":
    main()
