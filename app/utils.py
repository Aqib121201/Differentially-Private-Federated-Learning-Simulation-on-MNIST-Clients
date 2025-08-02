"""
Utility functions for the Streamlit web application
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Any
import json
from pathlib import Path

def load_results(filepath: str) -> Dict[str, Any]:
    """Load experiment results from JSON file"""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"Results file not found: {filepath}")
        return {}
    except json.JSONDecodeError:
        st.error(f"Invalid JSON in results file: {filepath}")
        return {}

def create_metrics_card(title: str, value: float, unit: str = "", delta: float = None) -> None:
    """Create a metrics card for displaying key statistics"""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.metric(
            label=title,
            value=f"{value:.2f}{unit}",
            delta=f"{delta:.2f}{unit}" if delta is not None else None
        )

def plot_training_curves_plotly(training_history: Dict, title: str = "Training Curves") -> go.Figure:
    """Create Plotly training curves visualization"""
    fig = go.Figure()
    
    if 'round' in training_history:
        rounds = training_history['round']
        
        # Add accuracy curves
        if 'val_accuracy' in training_history:
            fig.add_trace(go.Scatter(
                x=rounds,
                y=training_history['val_accuracy'],
                mode='lines+markers',
                name='Validation Accuracy',
                line=dict(color='blue', width=2),
                marker=dict(size=6)
            ))
        
        if 'test_accuracy' in training_history:
            fig.add_trace(go.Scatter(
                x=rounds,
                y=training_history['test_accuracy'],
                mode='lines+markers',
                name='Test Accuracy',
                line=dict(color='red', width=2),
                marker=dict(size=6)
            ))
        
        # Add loss curves on secondary y-axis
        if 'val_loss' in training_history or 'test_loss' in training_history:
            fig.add_trace(go.Scatter(
                x=rounds,
                y=training_history.get('val_loss', []),
                mode='lines+markers',
                name='Validation Loss',
                yaxis='y2',
                line=dict(color='green', width=2, dash='dash'),
                marker=dict(size=6)
            ))
            
            fig.add_trace(go.Scatter(
                x=rounds,
                y=training_history.get('test_loss', []),
                mode='lines+markers',
                name='Test Loss',
                yaxis='y2',
                line=dict(color='orange', width=2, dash='dash'),
                marker=dict(size=6)
            ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Communication Rounds',
        yaxis_title='Accuracy (%)',
        yaxis2=dict(
            title='Loss',
            overlaying='y',
            side='right'
        ),
        height=500,
        showlegend=True,
        hovermode='x unified'
    )
    
    return fig

def plot_privacy_budget_plotly(privacy_budget: List[Dict]) -> go.Figure:
    """Create Plotly privacy budget visualization"""
    if not privacy_budget:
        return go.Figure()
    
    rounds = list(range(1, len(privacy_budget) + 1))
    epsilons = [budget['epsilon'] for budget in privacy_budget]
    deltas = [budget['delta'] for budget in privacy_budget]
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Privacy Budget (ε)', 'Privacy Budget (δ)'),
        specs=[[{"type": "scatter"}, {"type": "scatter", "yaxis_type": "log"}]]
    )
    
    # Epsilon plot
    fig.add_trace(
        go.Scatter(
            x=rounds,
            y=epsilons,
            mode='lines+markers',
            name='Epsilon (ε)',
            line=dict(color='red', width=2),
            marker=dict(size=6)
        ),
        row=1, col=1
    )
    
    # Delta plot (log scale)
    fig.add_trace(
        go.Scatter(
            x=rounds,
            y=deltas,
            mode='lines+markers',
            name='Delta (δ)',
            line=dict(color='blue', width=2),
            marker=dict(size=6)
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title='Differential Privacy Budget Consumption',
        height=400,
        showlegend=False
    )
    
    fig.update_xaxes(title_text='Communication Rounds', row=1, col=1)
    fig.update_xaxes(title_text='Communication Rounds', row=1, col=2)
    fig.update_yaxes(title_text='Epsilon (ε)', row=1, col=1)
    fig.update_yaxes(title_text='Delta (δ)', row=1, col=2)
    
    return fig

def create_comparison_table(results: Dict[str, Dict]) -> pd.DataFrame:
    """Create a comparison table from results"""
    comparison_data = []
    
    for method, result in results.items():
        final_metrics = result['final_metrics']
        comparison_data.append({
            'Method': method,
            'Test Accuracy (%)': f"{final_metrics['test_accuracy']:.2f}",
            'Test Loss': f"{final_metrics['test_loss']:.4f}",
            'Training Time (s)': f"{result['total_training_time']:.1f}",
            'Privacy ε': f"{result.get('privacy_report', {}).get('total_epsilon', 0.0):.2f}"
        })
    
    return pd.DataFrame(comparison_data)

def display_experiment_summary(results: Dict[str, Dict]) -> None:
    """Display a comprehensive experiment summary"""
    st.subheader("📊 Experiment Summary")
    
    # Create comparison table
    comparison_df = create_comparison_table(results)
    st.dataframe(comparison_df, use_container_width=True)
    
    # Key metrics cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        best_accuracy = max(
            results[method]['final_metrics']['test_accuracy'] 
            for method in results.keys()
        )
        st.metric(
            "Best Accuracy",
            f"{best_accuracy:.2f}%"
        )
    
    with col2:
        total_time = sum(
            results[method]['total_training_time'] 
            for method in results.keys()
        )
        st.metric(
            "Total Training Time",
            f"{total_time:.1f}s"
        )
    
    with col3:
        privacy_epsilon = max(
            results[method].get('privacy_report', {}).get('total_epsilon', 0.0)
            for method in results.keys()
        )
        st.metric(
            "Max Privacy ε",
            f"{privacy_epsilon:.2f}"
        )

def save_experiment_results(results: Dict[str, Dict], filename: str = "experiment_results.json") -> None:
    """Save experiment results to file"""
    results_path = Path("logs") / filename
    results_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    st.success(f"Results saved to {results_path}")

def load_experiment_results(filename: str = "experiment_results.json") -> Dict[str, Dict]:
    """Load experiment results from file"""
    results_path = Path("logs") / filename
    
    if not results_path.exists():
        st.warning(f"No results file found: {results_path}")
        return {}
    
    return load_results(str(results_path))

# Import plotly subplots for privacy budget visualization
from plotly.subplots import make_subplots 