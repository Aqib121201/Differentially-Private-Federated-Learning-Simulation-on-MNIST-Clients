"""
Streamlit Web Application for Differentially Private Federated Learning Simulation

This app provides an interactive interface for running federated learning experiments
and visualizing results.
"""

import streamlit as st
import sys
from pathlib import Path
import json
import time
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.config import federated_config, privacy_config, training_config
from src.federated_training import FederatedLearningSimulator, run_centralized_baseline
from src.visualization import create_comprehensive_report

# Page configuration
st.set_page_config(
    page_title="DP Federated Learning Simulator",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    """Main application function"""
    
    # Header
    st.title("🔒 Differentially Private Federated Learning Simulator")
    st.markdown("---")
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Federated learning parameters
    st.sidebar.subheader("Federated Learning")
    num_clients = st.sidebar.slider("Number of Clients", 2, 10, 5)
    num_rounds = st.sidebar.slider("Number of Rounds", 10, 200, 50)
    local_epochs = st.sidebar.slider("Local Epochs", 1, 10, 3)
    batch_size = st.sidebar.selectbox("Batch Size", [16, 32, 64], index=1)
    learning_rate = st.sidebar.selectbox("Learning Rate", [0.001, 0.01, 0.1], index=1)
    
    # Privacy parameters
    st.sidebar.subheader("Differential Privacy")
    enable_dp = st.sidebar.checkbox("Enable Differential Privacy", value=True)
    noise_multiplier = st.sidebar.slider("Noise Multiplier", 0.1, 5.0, 1.0, 0.1)
    max_grad_norm = st.sidebar.slider("Max Gradient Norm", 0.1, 5.0, 1.0, 0.1)
    target_epsilon = st.sidebar.slider("Target Epsilon", 1.0, 20.0, 10.0, 0.5)
    
    # Model parameters
    st.sidebar.subheader("Model")
    hidden_size_1 = st.sidebar.slider("Hidden Layer 1", 64, 1024, 512, 64)
    hidden_size_2 = st.sidebar.slider("Hidden Layer 2", 32, 512, 256, 32)
    hidden_size_3 = st.sidebar.slider("Hidden Layer 3", 16, 256, 128, 16)
    dropout_rate = st.sidebar.slider("Dropout Rate", 0.0, 0.5, 0.2, 0.1)
    
    # Data distribution
    st.sidebar.subheader("Data Distribution")
    data_distribution = st.sidebar.selectbox("Distribution Type", ["iid", "non_iid"])
    if data_distribution == "non_iid":
        non_iid_alpha = st.sidebar.slider("Non-IID Alpha", 0.1, 2.0, 0.5, 0.1)
    
    # Update configuration
    federated_config.num_clients = num_clients
    federated_config.num_rounds = num_rounds
    federated_config.local_epochs = local_epochs
    federated_config.batch_size = batch_size
    federated_config.learning_rate = learning_rate
    federated_config.data_distribution = data_distribution
    
    if data_distribution == "non_iid":
        federated_config.non_iid_alpha = non_iid_alpha
    
    privacy_config.enable_dp = enable_dp
    privacy_config.noise_multiplier = noise_multiplier
    privacy_config.max_grad_norm = max_grad_norm
    privacy_config.target_epsilon = target_epsilon
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["🏃‍♂️ Run Experiments", "📊 Results", "📈 Visualizations", "📋 About"])
    
    with tab1:
        run_experiments_tab()
    
    with tab2:
        results_tab()
    
    with tab3:
        visualizations_tab()
    
    with tab4:
        about_tab()

def run_experiments_tab():
    """Tab for running experiments"""
    st.header("Run Federated Learning Experiments")
    
    # Experiment selection
    col1, col2, col3 = st.columns(3)
    
    with col1:
        run_centralized = st.checkbox("Centralized Training", value=True)
    
    with col2:
        run_federated_no_dp = st.checkbox("Federated (No DP)", value=True)
    
    with col3:
        run_federated_with_dp = st.checkbox("Federated (With DP)", value=True)
    
    # Run button
    if st.button("🚀 Start Experiments", type="primary"):
        if not any([run_centralized, run_federated_no_dp, run_federated_with_dp]):
            st.error("Please select at least one experiment to run.")
            return
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results = {}
        
        # Run experiments
        experiment_count = sum([run_centralized, run_federated_no_dp, run_federated_with_dp])
        current_experiment = 0
        
        # Centralized training
        if run_centralized:
            status_text.text("Running centralized training...")
            try:
                results['centralized'] = run_centralized_baseline()
                current_experiment += 1
                progress_bar.progress(current_experiment / experiment_count)
                st.success("✅ Centralized training completed!")
            except Exception as e:
                st.error(f"❌ Error in centralized training: {e}")
        
        # Federated learning without DP
        if run_federated_no_dp:
            status_text.text("Running federated learning without DP...")
            try:
                # Temporarily disable DP
                original_enable_dp = privacy_config.enable_dp
                privacy_config.enable_dp = False
                
                simulator_no_dp = FederatedLearningSimulator(enable_dp=False)
                results['federated_no_dp'] = simulator_no_dp.run_federated_training()
                
                # Restore DP setting
                privacy_config.enable_dp = original_enable_dp
                
                current_experiment += 1
                progress_bar.progress(current_experiment / experiment_count)
                st.success("✅ Federated learning (no DP) completed!")
            except Exception as e:
                st.error(f"❌ Error in federated learning (no DP): {e}")
        
        # Federated learning with DP
        if run_federated_with_dp and privacy_config.enable_dp:
            status_text.text("Running federated learning with DP...")
            try:
                simulator_with_dp = FederatedLearningSimulator(enable_dp=True)
                results['federated_with_dp'] = simulator_with_dp.run_federated_training()
                
                current_experiment += 1
                progress_bar.progress(current_experiment / experiment_count)
                st.success("✅ Federated learning (with DP) completed!")
            except Exception as e:
                st.error(f"❌ Error in federated learning (with DP): {e}")
        
        # Save results
        if results:
            # Save to session state
            st.session_state['experiment_results'] = results
            
            # Save to file
            results_path = Path("logs/app_results.json")
            results_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            st.success("🎉 All experiments completed! Results saved.")
            
            # Display quick summary
            display_quick_summary(results)

def display_quick_summary(results):
    """Display a quick summary of results"""
    st.subheader("📊 Quick Summary")
    
    summary_data = []
    for method, result in results.items():
        final_metrics = result['final_metrics']
        summary_data.append({
            'Method': method,
            'Test Accuracy (%)': f"{final_metrics['test_accuracy']:.2f}",
            'Test Loss': f"{final_metrics['test_loss']:.4f}",
            'Training Time (s)': f"{result['total_training_time']:.1f}",
            'Privacy ε': f"{result.get('privacy_report', {}).get('total_epsilon', 0.0):.2f}"
        })
    
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True)

def results_tab():
    """Tab for displaying results"""
    st.header("📊 Experiment Results")
    
    # Load results
    if 'experiment_results' not in st.session_state:
        st.info("No results available. Please run experiments first.")
        return
    
    results = st.session_state['experiment_results']
    
    # Results overview
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Performance Comparison")
        
        # Create performance comparison chart
        methods = list(results.keys())
        accuracies = [results[method]['final_metrics']['test_accuracy'] for method in methods]
        losses = [results[method]['final_metrics']['test_loss'] for method in methods]
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Test Accuracy (%)', 'Test Loss'),
            specs=[[{"type": "bar"}, {"type": "bar"}]]
        )
        
        fig.add_trace(
            go.Bar(x=methods, y=accuracies, name='Accuracy', marker_color='lightblue'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(x=methods, y=losses, name='Loss', marker_color='lightcoral'),
            row=1, col=2
        )
        
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Training Time Comparison")
        
        times = [results[method]['total_training_time'] for method in methods]
        
        fig = px.bar(
            x=methods, 
            y=times,
            title="Training Time (seconds)",
            color=methods,
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed results
    st.subheader("Detailed Results")
    
    for method, result in results.items():
        with st.expander(f"📋 {method} - Detailed Results"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Final Metrics:**")
                final_metrics = result['final_metrics']
                st.write(f"- Test Accuracy: {final_metrics['test_accuracy']:.2f}%")
                st.write(f"- Test Loss: {final_metrics['test_loss']:.4f}")
                st.write(f"- Training Time: {result['total_training_time']:.1f} seconds")
                
                if 'privacy_report' in result:
                    privacy_report = result['privacy_report']
                    st.write("**Privacy Report:**")
                    st.write(f"- Total ε: {privacy_report['total_epsilon']:.4f}")
                    st.write(f"- Total δ: {privacy_report['total_delta']:.2e}")
                    st.write(f"- Noise Multiplier: {privacy_report['noise_multiplier']:.4f}")
            
            with col2:
                st.write("**Model Summary:**")
                model_summary = result['model_summary']
                st.write(f"- Total Parameters: {model_summary['total_parameters']:,}")
                st.write(f"- Model Size: {model_summary['model_size_mb']:.2f} MB")
                st.write(f"- Hidden Layers: {model_summary['hidden_sizes']}")

def visualizations_tab():
    """Tab for visualizations"""
    st.header("📈 Training Visualizations")
    
    if 'experiment_results' not in st.session_state:
        st.info("No results available. Please run experiments first.")
        return
    
    results = st.session_state['experiment_results']
    
    # Select experiment for visualization
    selected_method = st.selectbox("Select experiment for detailed visualization:", list(results.keys()))
    
    if selected_method:
        result = results[selected_method]
        training_history = result['training_history']
        
        # Training curves
        st.subheader(f"Training Curves - {selected_method}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Accuracy plot
            if 'val_accuracy' in training_history and 'test_accuracy' in training_history:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=training_history['round'],
                    y=training_history['val_accuracy'],
                    mode='lines+markers',
                    name='Validation Accuracy',
                    line=dict(color='blue')
                ))
                fig.add_trace(go.Scatter(
                    x=training_history['round'],
                    y=training_history['test_accuracy'],
                    mode='lines+markers',
                    name='Test Accuracy',
                    line=dict(color='red')
                ))
                fig.update_layout(
                    title='Accuracy vs Communication Rounds',
                    xaxis_title='Rounds',
                    yaxis_title='Accuracy (%)',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Loss plot
            if 'val_loss' in training_history and 'test_loss' in training_history:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=training_history['round'],
                    y=training_history['val_loss'],
                    mode='lines+markers',
                    name='Validation Loss',
                    line=dict(color='blue')
                ))
                fig.add_trace(go.Scatter(
                    x=training_history['round'],
                    y=training_history['test_loss'],
                    mode='lines+markers',
                    name='Test Loss',
                    line=dict(color='red')
                ))
                fig.update_layout(
                    title='Loss vs Communication Rounds',
                    xaxis_title='Rounds',
                    yaxis_title='Loss',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Privacy budget (if available)
        if 'privacy_budget' in training_history and training_history['privacy_budget']:
            st.subheader("Privacy Budget Consumption")
            
            privacy_budget = training_history['privacy_budget']
            rounds = list(range(1, len(privacy_budget) + 1))
            epsilons = [budget['epsilon'] for budget in privacy_budget]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=rounds,
                y=epsilons,
                mode='lines+markers',
                name='Epsilon (ε)',
                line=dict(color='purple')
            ))
            fig.update_layout(
                title='Privacy Budget (ε) vs Communication Rounds',
                xaxis_title='Rounds',
                yaxis_title='Epsilon (ε)',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

def about_tab():
    """Tab for project information"""
    st.header("📋 About This Project")
    
    st.markdown("""
    ## Differentially Private Federated Learning Simulation on MNIST
    
    This application simulates federated learning with differential privacy on the MNIST dataset.
    
    ### Key Features:
    - **Federated Learning**: Train models across multiple clients without sharing raw data
    - **Differential Privacy**: Protect individual privacy with noise addition
    - **MNIST Dataset**: Handwritten digit recognition (0-9)
    - **Interactive Interface**: Easy-to-use web interface for experimentation
    
    ### Architecture:
    - **Model**: Multi-layer perceptron with dropout
    - **Optimizer**: Stochastic Gradient Descent (SGD)
    - **Loss Function**: Cross-entropy loss
    - **Privacy**: Opacus-based differential privacy
    
    ### Privacy Guarantees:
    - **ε-Differential Privacy**: Formal privacy guarantee
    - **Configurable Parameters**: Adjustable noise and gradient clipping
    - **Privacy Budget Tracking**: Monitor privacy consumption
    
    ### Use Cases:
    - Healthcare data analysis
    - Financial fraud detection
    - Mobile device learning
    - Collaborative AI research
    
    ### Technical Details:
    - **Framework**: PyTorch + Opacus
    - **Federated Learning**: Flower framework
    - **Visualization**: Plotly + Streamlit
    - **Privacy**: RDP accountant
    
    ### Configuration Options:
    - Number of clients (2-10)
    - Communication rounds (10-200)
    - Local epochs (1-10)
    - Privacy parameters (noise, epsilon)
    - Model architecture (hidden layers, dropout)
    - Data distribution (IID/non-IID)
    
    ### How to Use:
    1. Configure parameters in the sidebar
    2. Select experiments to run
    3. Click "Start Experiments"
    4. View results and visualizations
    5. Analyze privacy-utility tradeoffs
    
    ### Privacy-Utility Tradeoff:
    - Higher noise → Better privacy, lower accuracy
    - Lower noise → Lower privacy, better accuracy
    - Target epsilon controls the balance
    
    ### References:
    - [Federated Learning Paper](https://arxiv.org/abs/1602.05629)
    - [Differential Privacy Paper](https://arxiv.org/abs/1607.00133)
    - [Opacus Documentation](https://opacus.ai/)
    - [Flower Framework](https://flower.dev/)
    """)

if __name__ == "__main__":
    main() 