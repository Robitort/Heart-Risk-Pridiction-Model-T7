# Heart Risk Prediction Model T7

This project represents one of my initial iterations of a multi-modal AI model for cardiovascular risk prediction. It utilizes both raw 12-lead ECG signals and clinical parameters to provide comprehensive risk analysis and cardiac condition detection.

---

## Key Features

- **Multi-Modal Learning**: Integrates ECG time-series data with clinical information (demographics, lifestyle factors).
- **Multi-Label Classification**: Simultaneously detects multiple cardiac conditions in a single pass.
- **Risk Scoring**: Predicts cardiovascular risk on a 0–100 scale.
- **Signal Quality Control**: Detects flatline and artifacts to ensure reliable input.
- **Interpretable AI**: Generates reasoning reports and confidence metrics for transparent decision-making.

---

## Model Architecture (Overview)

### 1. Inputs
- **ECG Signals**: 12-lead priority, sampled at 500 Hz.
- **Clinical Features**: Demographic data, lifestyle factors.
- **Targets**: Multiple cardiac condition labels (multi-label).

### 2. Signal Processing
- **Format**: Compatible with WFDB (WaveForm DataBase) format.
- **Analysis**: Multi-lead spatial-temporal analysis of ECG signals.
- **Quality Assessment**: Flatline and artifact detection for signal validation.

### 3. Neural Network Backbone
- **Convolutional Layers**: 2D Conv layers with Swish activation functions; includes multi-scale branches for feature extraction.
- **Feature Fusion**: Entropy-based integration of ECG and clinical features for improved predictive power.

---

## Project Status

This is an early-stage iteration and will evolve with further research and feedback. Contributions and suggestions are welcome!

---
