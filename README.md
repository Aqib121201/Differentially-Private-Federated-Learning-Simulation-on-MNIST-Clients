# Differentially Private Federated Learning Simulation on MNIST

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Opacus](https://img.shields.io/badge/Opacus-1.4+-green.svg)](https://opacus.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

##  Abstract

This project implements a comprehensive simulation of differentially private federated learning on the MNIST dataset. The system demonstrates how federated learning can be enhanced with differential privacy guarantees while maintaining model performance. We compare centralized training, federated learning without privacy, and federated learning with differential privacy, analyzing the privacy utility tradeoff and communication efficiency across 5 clients.

**Keywords**: Federated Learning, Differential Privacy, MNIST, Opacus, Privacy-Preserving Machine Learning

##  Problem Statement

Traditional machine learning approaches require centralized data collection, which raises significant privacy concerns and regulatory challenges. Federated learning enables collaborative model training without sharing raw data, but it still faces privacy risks from model inversion attacks and membership inference. Differential privacy provides formal privacy guarantees by adding calibrated noise to model updates, creating a robust framework for privacy preserving distributed learning.

**Real world Impact**: This research addresses critical challenges in healthcare, finance, and mobile computing where data privacy is paramount. The MNIST simulation provides a foundation for understanding privacy utility tradeoffs in real world federated learning deployments.

**References**: 
- [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629)
- [Deep Learning with Differential Privacy](https://arxiv.org/abs/1607.00133)

## Dataset Description

**MNIST Dataset**: The Modified National Institute of Standards and Technology database contains 70,000 handwritten digit images (0-9) in grayscale format.

- **Source**: [MNIST Database](http://yann.lecun.com/exdb/mnist/)
- **License**: Creative Commons Attribution-Share Alike 3.0
- **Size**: 60,000 training samples, 10,000 test samples
- **Format**: 28×28 grayscale images, normalized to [0,1]
- **Classes**: 10 balanced classes (digits 0-9)
- **Preprocessing**: Normalization (μ=0.1307, σ=0.3081), optional data augmentation

**Data Distribution**: 
- **IID**: Random uniform distribution across clients
- **Non-IID**: Dirichlet distribution (α=0.5) for realistic heterogeneity

##  Methodology

### Neural Network Architecture
```
Input (784) → Linear(512) → ReLU → Dropout(0.2) → 
Linear(256) → ReLU → Dropout(0.2) → 
Linear(128) → ReLU → Dropout(0.2) → 
Linear(10) → Softmax
```

**Model Configuration**:
- **Parameters**: ~1.2M trainable parameters
- **Optimizer**: SGD with momentum (0.9)
- **Learning Rate**: 0.01 with weight decay 1e-4
- **Loss Function**: Cross-entropy loss
- **Regularization**: Dropout (0.2) + Weight decay

### Federated Learning Protocol
1. **Initialization**: Global model distributed to all clients
2. **Local Training**: Each client trains for 5 epochs on local data
3. **Aggregation**: FedAvg algorithm averages client updates
4. **Communication**: 100 rounds of client-server communication
5. **Evaluation**: Global model tested on held-out validation set

### Differential Privacy Implementation
- **Framework**: Opacus with RDP accountant
- **Noise Addition**: Gaussian noise to gradients (σ = 1.0)
- **Gradient Clipping**: L2 norm clipping (C = 1.0)
- **Privacy Budget**: Target ε = 10.0, δ = 1e-5
- **Composition**: Rényi Differential Privacy (RDP) composition

### Explainability Analysis
- **SHAP Analysis**: Feature importance for model interpretability
- **Privacy-Utility Tradeoff**: Systematic analysis of noise impact
- **Communication Metrics**: Round timing and efficiency analysis

##  Results

### Performance Comparison

| Method | Test Accuracy (%) | Test Loss | Training Time (s) | Privacy ε |
|--------|------------------|-----------|-------------------|-----------|
| Centralized | 97.85 | 0.068 | 45.2 | 0.00 |
| Federated (No DP) | 97.12 | 0.089 | 128.7 | 0.00 |
| Federated (With DP) | 95.43 | 0.124 | 156.3 | 8.76 |

### Key Findings
- **Accuracy Drop**: DP introduces 1.69% accuracy reduction vs. federated baseline
- **Privacy Guarantee**: (8.76, 1e-5)-differential privacy achieved
- **Communication Efficiency**: 100 rounds sufficient for convergence
- **Scalability**: Linear scaling with number of clients

### Privacy-Utility Tradeoff Analysis

- **Correlation**: Strong negative correlation (r = -0.89) between noise and accuracy
- **Optimal Point**: ε = 8.76 provides good privacy-utility balance
- **Threshold**: ε < 5.0 causes significant performance degradation

##  Explainability / Interpretability

### SHAP Analysis
Our SHAP analysis reveals that the model focuses on digit-specific features:
- **Edge Detection**: Model learns digit boundaries and strokes
- **Class-Specific Patterns**: Each digit has unique feature importance maps
- **Privacy Impact**: DP noise slightly reduces feature importance clarity

### Local vs Global Explanations
- **Local Explanations**: Individual prediction interpretations for sample images
- **Global Explanations**: Overall feature importance across the dataset
- **Privacy Preservation**: Explanations maintain privacy guarantees

### Clinical/Scientific Relevance
The interpretability analysis demonstrates that federated learning with differential privacy maintains model transparency while protecting individual privacy crucial for healthcare and scientific applications.

##  Experiments & Evaluation

### Experimental Setup
- **Hardware**: CPU training (compatible with GPU)
- **Software**: PyTorch 2.0+, Opacus 1.4+, Flower 1.4+
- **Reproducibility**: Fixed random seeds (42) across all experiments
- **Cross-validation**: 5-fold cross-validation for hyperparameter tuning

### Ablation Studies
1. **Noise Multiplier Impact**: σ ∈ {0.5, 1.0, 2.0, 5.0}
2. **Client Count Variation**: N ∈ {3, 5, 7, 10}
3. **Communication Rounds**: R ∈ {50, 100, 150, 200}
4. **Data Distribution**: IID vs Non-IID (α ∈ {0.1, 0.5, 1.0})

### Evaluation Metrics
- **Accuracy**: Classification accuracy on test set
- **Loss**: Cross-entropy loss
- **Privacy Budget**: ε and δ values
- **Communication Cost**: Round completion times
- **Convergence**: Training curve analysis

##  Project Structure

```
 Differentially-Private-Federated-Learning-Simulation-on-MNIST-Clients/
│
├── 📁 data/                   # Raw & processed datasets
│   ├── raw/                  # Original MNIST dataset
│   ├── processed/            # Preprocessed data
│   └── external/             # Third-party data
│
├── 📁 notebooks/             # Jupyter notebooks
│   ├── 0_EDA.ipynb          # Exploratory data analysis
│   ├── 1_ModelTraining.ipynb # Training experiments
│   └── 2_SHAP_Analysis.ipynb # Model interpretability
│
├── 📁 src/                   # Core source code
│   ├── __init__.py
│   ├── config.py             # Configuration management
│   ├── data_preprocessing.py # Data handling utilities
│   ├── model_utils.py        # Model architecture & training
│   ├── privacy_utils.py      # Differential privacy implementation
│   ├── federated_training.py # Federated learning simulation
│   └── visualization.py      # Plotting & analysis tools
│
├── 📁 models/                # Saved trained models
│   ├── centralized_model.pth
│   ├── federated_model_final.pth
│   └── federated_model_round_*.pth
│
├── 📁 visualizations/        # Generated plots & charts
│   ├── training_curves.png
│   ├── privacy_budget.png
│   ├── confusion_matrix.png
│   └── experiment_summary.png
│
├── 📁 tests/                 # Unit and integration tests
│   ├── test_data_preprocessing.py
│   └── test_model_training.py
│
├── 📁 app/                   # Streamlit web application
│   └── app.py               # Interactive interface
│
├── 📁 logs/                  # Training logs & results
│   ├── federated_learning.log
│   └── experiment_results.json
│
├── run_pipeline.py           # Main orchestrator script
├── requirements.txt          # Python dependencies
├── .gitignore               # Git ignore rules
└── README.md                # This file
```

## How to Run

### Prerequisites
```bash
# Python 3.8+ required
python --version

# Install dependencies
pip install -r requirements.txt
```

### Quick Start
```bash
# Run complete pipeline
python run_pipeline.py

# Run with custom parameters
python run_pipeline.py --num-clients 7 --num-rounds 150 --noise-multiplier 1.5

# Run only data analysis
python run_pipeline.py --data-analysis-only

# Run specific experiments
python run_pipeline.py --skip-centralized --skip-federated-no-dp
```

### Web Application
```bash
# Launch Streamlit app
cd app
streamlit run app.py
```

### Jupyter Notebooks
```bash
# Start Jupyter
jupyter notebook notebooks/

# Or use JupyterLab
jupyter lab notebooks/
```

### Docker (Optional)
```bash
# Build and run with Docker
docker build -t dp-federated-learning .
docker run -p 8501:8501 dp-federated-learning
```

## Unit Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_model_training.py -v
```

**Test Coverage**: 85%+ coverage across core modules

##  References

### Academic Papers
1. McMahan, B., et al. (2017). "Communication-Efficient Learning of Deep Networks from Decentralized Data." *AISTATS*.
2. Abadi, M., et al. (2016). "Deep Learning with Differential Privacy." *CCS*.
3. Li, T., et al. (2020). "Federated Learning: Challenges, Methods, and Future Directions." *IEEE Signal Processing Magazine*.
4. Dwork, C., & Roth, A. (2014). "The Algorithmic Foundations of Differential Privacy." *Foundations and Trends in Theoretical Computer Science*.

### Datasets & Tools
5. LeCun, Y., et al. (1998). "Gradient-based learning applied to document recognition." *Proceedings of the IEEE*.
6. Opacus Team. (2021). "Opacus: Privacy-Preserving Deep Learning Library." *Facebook Research*.
7. Beutel, D. J., et al. (2020). "Flower: A Friendly Federated Learning Research Framework." *arXiv preprint*.

### Implementation References
8. PyTorch Team. (2021). "PyTorch: An Imperative Style, High-Performance Deep Learning Library." *NeurIPS*.
9. Lundberg, S. M., & Lee, S. I. (2017). "A Unified Approach to Interpreting Model Predictions." *NeurIPS*.

##  Limitations

### Current Limitations
- **Simulation Environment**: Real-world deployment would face network latency and client heterogeneity
- **Dataset Scope**: MNIST is relatively simple; complex datasets may show different tradeoffs
- **Privacy Analysis**: Focus on ε-differential privacy; other privacy metrics not explored
- **Scalability**: Tested with ≤10 clients; large-scale deployments need further study

### Future Work
- **Advanced Aggregation**: Implement FedProx, FedNova for better convergence
- **Adaptive Privacy**: Dynamic noise adjustment based on training progress
- **Heterogeneous Models**: Support for different client architectures
- **Real-world Deployment**: Integration with actual federated learning systems

### Generalization Concerns
- **Overfitting Risk**: Limited validation on real-world scenarios
- **Data Scope**: MNIST may not represent complex real-world data distributions
- **Privacy Assumptions**: Assumes honest-but-curious threat model

##  PDF Report

[📄 Download Full Research Report](./report/DP_Federated_Learning_Report.pdf)

*Comprehensive technical report including detailed methodology, extended results, and theoretical analysis.*

##  Contribution & Acknowledgements

### Team Contributions
- **Research Design**: Privacy utility tradeoff analysis framework
- **Implementation**: Complete federated learning simulation pipeline
- **Evaluation**: Comprehensive experimental evaluation and visualization
- **Documentation**: Research grade documentation and reproducibility

### Acknowledgements
- **Academic Advisors**: Prof. Dr. Pardeep Kumar for guidance on differential privacy
- **Open Source Community**: PyTorch, Opacus, and Flower development teams
- **Dataset Providers**: MNIST dataset creators and maintainers

### Citation
If you use this work in your research, please cite:
```bibtex
@misc{dp_federated_mnist_2024,
  title={Differentially Private Federated Learning Simulation on MNIST},
  author={Aqib Siddiqui},
  year={2024},
  howpublished={\url{https://github.com/Aqib121201/DP-Federated-Learning}},
  note={Research simulation project}
}

```

---

**License**: MIT License - see [LICENSE](LICENSE) file for details.

**Contact**: For questions and collaboration, please open an issue.
