# ================================================================
# Utility Functions for Visualization and Reporting
# Nile Valley University · Sudan · v29.28-R32
# ================================================================

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Tuple
import json
from datetime import datetime
import os

class Visualizer:
    """Visualization utilities for optimization results"""
    
    @staticmethod
    def plot_pareto_front(solutions: np.ndarray, 
                         objectives: np.ndarray,
                         title: str = "Pareto Front") -> go.Figure:
        """Plot 3D Pareto front"""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter3d(
            x=solutions[:, 0],
            y=solutions[:, 1],
            z=solutions[:, 2],
            mode='markers',
            marker=dict(
                size=6,
                color=objectives[:, 2],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="EFRF")
            ),
            name='Pareto Solutions'
        ))
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='Density',
                yaxis_title='Tensile Strength (MPa)',
                zaxis_title='EFRF',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            height=500
        )
        
        return fig
    
    @staticmethod
    def plot_training_metrics(loss_history: List[float], 
                              r2_history: List[float]) -> go.Figure:
        """Plot training metrics over epochs"""
        fig = make_subplots(rows=2, cols=1, 
                           subplot_titles=('Loss', 'R² Score'))
        
        fig.add_trace(go.Scatter(
            y=loss_history,
            mode='lines',
            name='Loss',
            line=dict(color='red', width=2)
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            y=r2_history,
            mode='lines',
            name='R²',
            line=dict(color='green', width=2)
        ), row=2, col=1)
        
        fig.update_layout(height=400, showlegend=False)
        return fig
    
    @staticmethod
    def plot_sensitivity_analysis(variable_names: List[str],
                                  effects: np.ndarray) -> go.Figure:
        """Plot sensitivity analysis results"""
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=variable_names,
            y=effects,
            marker_color='steelblue'
        ))
        
        fig.update_layout(
            title='Variable Importance',
            xaxis_title='Variables',
            yaxis_title='Effect Size',
            height=400
        )
        
        return fig
    
    @staticmethod
    def plot_solution_comparison(solutions: List[Dict]) -> go.Figure:
        """Compare multiple solutions"""
        df = pd.DataFrame(solutions)
        fig = go.Figure()
        
        for col in ['density', 'tensile', 'efrf']:
            fig.add_trace(go.Bar(
                x=df['solution_id'],
                y=df[col],
                name=col
            ))
        
        fig.update_layout(
            title='Solution Comparison',
            barmode='group',
            height=400
        )
        
        return fig

class ReportGenerator:
    """Generate optimization reports"""
    
    def __init__(self, output_dir: str = "./results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def generate_json_report(self, 
                            best_solutions: List[Dict],
                            metrics: Dict,
                            timestamp: str = None) -> str:
        """Generate JSON report"""
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
        report = {
            'timestamp': timestamp,
            'best_solutions': best_solutions,
            'metrics': metrics
        }
        
        filepath = os.path.join(self.output_dir, f'report_{timestamp}.json')
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
            
        return filepath
    
    def generate_csv_report(self, 
                           solutions: pd.DataFrame,
                           timestamp: str = None) -> str:
        """Generate CSV report"""
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
        filepath = os.path.join(self.output_dir, f'solutions_{timestamp}.csv')
        solutions.to_csv(filepath, index=False)
        return filepath

class ParallelProcessor:
    """Parallel processing utilities"""
    
    @staticmethod
    def chunk_data(data: np.ndarray, n_chunks: int) -> List[np.ndarray]:
        """Split data into chunks for parallel processing"""
        chunk_size = len(data) // n_chunks
        chunks = []
        for i in range(n_chunks):
            start = i * chunk_size
            end = (i + 1) * chunk_size if i < n_chunks - 1 else len(data)
            chunks.append(data[start:end])
        return chunks
    
    @staticmethod
    def merge_results(results: List[np.ndarray]) -> np.ndarray:
        """Merge results from parallel processing"""
        return np.vstack(results)

def validate_inputs(params: Dict[str, float]) -> Tuple[bool, str]:
    """Validate input parameters"""
    validation_rules = {
        'api': (80.0, 98.0),
        'binder': (1.4, 6.0),
        'pvpp': (1.0, 6.0),
        'mgst': (0.10, 1.2),
        'mcc': (1.5, 8.0),
        'moisture': (0.5, 5.0),
        'pressure': (150.0, 250.0),
        'speed': (15.0, 30.0)
    }
    
    for param, (min_val, max_val) in validation_rules.items():
        if param in params:
            value = params[param]
            if not (min_val <= value <= max_val):
                return False, f"{param} must be between {min_val} and {max_val}"
    
    return True, "All inputs valid"

def calculate_statistics(values: np.ndarray) -> Dict[str, float]:
    """Calculate descriptive statistics"""
    return {
        'mean': float(np.mean(values)),
        'std': float(np.std(values)),
        'min': float(np.min(values)),
        'max': float(np.max(values)),
        'q25': float(np.percentile(values, 25)),
        'median': float(np.median(values)),
        'q75': float(np.percentile(values, 75))
    }
