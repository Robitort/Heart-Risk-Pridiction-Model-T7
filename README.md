# Heart-Risk-Pridiction-Model-T7
One of my first Iterations of this model, utilizing analysis of Raw ECG(12 lead) and clinical parameters

# Metrics:
Multi-modal learning: ECG time-series + clinical data
Multi-label classification: Detects multiple cardiac conditions in one pass
Risk scoring: Cardiovascular risk prediction (0–100 scale)
Signal quality control: Flatline & artifact detection
Interpretable AI: Generates reasoning reports and confidence metrics

# Architecture(inshort):
1. Inputs
    ECG Signals (multi-lead---12 lead priority, 500 Hz)
    Clinical Features (demographics, lifestyle factors)
    Multi-label Targets (cardiac conditions)

2. Signal Processing
    WFDB format compatibility
    Multi-lead spatial-temporal analysis
    Signal quality assessment + flatline detection
   
4. Neural Network
    Convolutional Backbone: 2D Conv layers, Swish activation, multi-scale branches
    Feature Fusion: Entropy-based ECG + clinical feature integration

       
