#!/bin/bash

# Entrypoint script for Differentially Private Federated Learning Simulation

set -e

echo "🚀 Starting Differentially Private Federated Learning Simulation..."

# Function to display help
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --pipeline     Run the complete federated learning pipeline"
    echo "  --app          Start the Streamlit web application"
    echo "  --notebook     Start Jupyter notebook server"
    echo "  --test         Run unit tests"
    echo "  --help         Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  NUM_CLIENTS    Number of federated learning clients (default: 5)"
    echo "  NUM_ROUNDS     Number of communication rounds (default: 100)"
    echo "  ENABLE_DP      Enable differential privacy (default: true)"
    echo "  NOISE_MULT     Noise multiplier for DP (default: 1.0)"
    echo ""
}

# Function to run pipeline
run_pipeline() {
    echo "📊 Running federated learning pipeline..."
    python run_pipeline.py \
        --num-clients ${NUM_CLIENTS:-5} \
        --num-rounds ${NUM_ROUNDS:-100} \
        --noise-multiplier ${NOISE_MULT:-1.0} \
        ${ENABLE_DP:+--enable-dp} \
        ${DISABLE_DP:+--disable-dp}
}

# Function to start app
start_app() {
    echo "🌐 Starting Streamlit web application..."
    streamlit run app/app.py \
        --server.port=8501 \
        --server.address=0.0.0.0 \
        --server.headless=true \
        --browser.gatherUsageStats=false
}

# Function to start notebook
start_notebook() {
    echo "📓 Starting Jupyter notebook server..."
    jupyter notebook \
        --ip=0.0.0.0 \
        --port=8888 \
        --no-browser \
        --allow-root \
        --NotebookApp.token='' \
        --NotebookApp.password=''
}

# Function to run tests
run_tests() {
    echo "🧪 Running unit tests..."
    pytest tests/ -v --cov=src --cov-report=html
}

# Main execution
case "${1:-app}" in
    --pipeline)
        run_pipeline
        ;;
    --app)
        start_app
        ;;
    --notebook)
        start_notebook
        ;;
    --test)
        run_tests
        ;;
    --help|-h)
        show_help
        ;;
    *)
        echo "Unknown option: $1"
        show_help
        exit 1
        ;;
esac 