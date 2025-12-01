# Flight Delay Prediction using Neural ODEs  
**Course:** EN.560.652.01.FA25 — Scientific Machine Learning for Dynamics and Control  
**Institution:** Johns Hopkins University  
**Author:** Handi Wang  
**Repository:** https://github.com/Co1ddog/SciML

---

## Overview
This repository implements a complete continuous-time forecasting framework using **Neural Ordinary Differential Equations (Neural ODEs)** to model and predict **airport arrival delays**.  
The project integrates multi-year operational data from **Baltimore/Washington International Airport (BWI)**, including flight-level arrival records, METAR weather observations, and temporal descriptors.

The project evaluates whether controlled Neural ODEs can capture airport delay dynamics and compares them with LSTM, MLP, and naive persistence baselines.

---

## Key Features
- Continuous-time modeling of airport arrival delay using **Neural ODEs**
- Controlled NODE with stability terms (damping, clipping, vector-field scaling)
- 5-minute resolution time series constructed from real-world data
- Feature set includes **weather**, **temporal encodings**, **holiday effects**, and **demand pressure**
- Latent dynamics analysis  
- Baseline comparison: **LSTM**, **MLP**, **Naive**
- Operational scenario evaluation: severe weather / holidays / peak hours

---

## Key scripts and artifacts

- **`node_delay_only.py`** – end-to-end training pipeline for the Neural ODE that models smoothed arrival delays. It ingests the 5-minute aggregated flight + weather dataset, applies normalization, trains the controlled NODE with truncated backpropagation through time, evaluates it across operating conditions (peak/off-peak, severe weather, holidays), and writes plots/metrics alongside the best checkpoint. The script saves the weight file `node_delay_smooth_best.pt` when validation loss improves.
- **`node_delay_smooth_best.pt`** – the serialized PyTorch checkpoint produced by `node_delay_only.py` containing the trained model parameters, normalization statistics, and feature column order required for inference or analysis.
- **`analyze_node_dynamics.py`** – post-training analysis utility that loads `node_delay_smooth_best.pt`, replays the model on the normalized input sequence, and reports latent-state dynamics (state velocity, vector-field magnitude) across operational segments. Optional plotting hooks (PCA trajectories, vector-field norms, prediction traces) are included but commented for quick command-line runs.

Dataset can be downloaded from Google Drive: https://drive.google.com/file/d/1PDbUZRKDsuhGtLhWEaZaf1fl0e6TXLSt/view?usp=sharing


