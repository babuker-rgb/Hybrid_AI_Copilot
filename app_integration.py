# ================================================================
# Streamlit App Integration - Complete Application
# Nile Valley University · Sudan · v29.28-R32
# ================================================================

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import yaml

from model import HybridTabletModel, HybridLoss
from optimizer import NSGAIIOptimizer
from data_processor import DataProcessor
from utils import Visualizer, ReportGenerator, validate_inputs

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

class HybridApp:
    """Main application class for Streamlit integration"""
    
    def __init__(self):
        self.model = HybridTabletModel(
            input_dim=config['model']['input_dim'],
            hidden_dim=config['model']['hidden_dim']
        )
        self.processor = DataProcessor()
        self.visualizer = Visualizer()
        self.report_generator = ReportGenerator()
        self.optimizer = None
        
    def run_optimization(self, params: np.ndarray) -> Dict:
        """Run the hybrid optimization pipeline"""
        # Initialize optimizer
        self.optimizer = NSGAIIOptimizer(
            model=self.model,
            n_objectives=3,
            population_size=config['optimization']['population_size'],
            generations=config['optimization']['generations']
        )
        
        # Run optimization
        results = []
        for pop, fitness, pareto_history, gen in self.optimizer.optimize(len(params)):
            if gen % 10 == 0:
                results.append({
                    'generation': gen,
                    'population': pop,
                    'fitness': fitness,
                    'pareto': pareto_history[-1]
                })
        
        # Get best solutions
        best_solutions = self.optimizer.get_best_solutions(5)
        
        return {
            'best_solutions': best_solutions,
            'history': results,
            'final_population': pop,
            'final_fitness': fitness
        }
    
    def display_results(self, results: Dict):
        """Display optimization results in Streamlit"""
        # Best solutions table
        solutions_data = []
        for i, sol in enumerate(results['best_solutions']):
            # Convert normalized parameters back to original scale
            scaled_solution = self._scale_to_original(sol.solution)
            solutions_data.append({
                'Solution': f'S{i+1}',
                'API': f'{scaled_solution[0]:.1f}%',
                'Binder': f'{scaled_solution[1]:.1f}%',
                'PVPP': f'{scaled_solution[2]:.1f}%',
                'MgSt': f'{scaled_solution[3]:.2f}%',
                'Density': f'{self.model.predict_single(sol.solution)["density"]:.3f}',
                'Tensile': f'{self.model.predict_single(sol.solution)["tensile"]:.1f} MPa',
                'EFRF': f'{self.model.predict_single(sol.solution)["efrf"]:.3f}'
            })
        
        df = pd.DataFrame(solutions_data)
        st.dataframe(df, use_container_width=True)
        
        # Pareto front visualization
        st.markdown("### 🌐 Pareto Front")
        pareto_solutions = np.array([s.solution for s in results['best_solutions']])
        pareto_fitness = np.array([s.objectives for s in results['best_solutions']])
        
        fig = self.visualizer.plot_pareto_front(pareto_solutions, pareto_fitness)
        st.plotly_chart(fig, use_container_width=True)
        
        # Download report button
        if st.button("📥 Download Optimization Report"):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.report_generator.generate_csv_report(df, timestamp)
            self.report_generator.generate_json_report(solutions_data, {}, timestamp)
            st.success(f"Report saved to ./results/")
    
    def _scale_to_original(self, normalized: np.ndarray) -> np.ndarray:
        """Scale normalized parameters back to original ranges"""
        ranges = [
            (80.0, 98.0),     # API
            (1.4, 6.0),       # Binder
            (1.0, 6.0),       # PVPP
            (0.10, 1.2),      # MgSt
            (1.5, 8.0),       # MCC
            (0.5, 5.0),       # Moisture
            (150.0,
