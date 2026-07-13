# TSPeakNet — District-Level Crop Pest and Disease Time-Series Forecasting and Peak-Event Detection

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.7.7-blue.svg" alt="Python 3.7.7">
  <img src="https://img.shields.io/badge/PyTorch-1.10.2-orange.svg" alt="PyTorch 1.10.2">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="MIT License">
  <img src="https://img.shields.io/badge/Status-Research%20Code-success.svg" alt="Research code">
</p>

<p align="center">
  <strong>Dual-scale neural forecasting and peak-event detection for district-level crop pest and disease warning support</strong>
  <br>
  <em>TSPeakNet: A dual-scale neural architecture for district-level crop pest and disease time-series forecasting and peak-event detection</em>
  <br>
  <em>Yuanze Qin, Zonghuan Han, Lingxian Zhang, and Yiding Zhang</em>
</p>

---

## Overview

This repository contains the complete implementation of **TSPeakNet**, a dual-branch neural architecture for district-level crop pest and disease prescription-count forecasting and peak-event detection. It also includes the associated warning-support platform, model evaluation scripts, interpretability utilities, and system performance-testing tools.

The Beijing experiments use district-level time series derived from **Plant Electronic Medical Records (PEMRs)**. PEMR-derived prescription counts are treated as **operational proxies for pest and disease service demand**, rather than direct measurements of biological incidence or severity. Accordingly, model outputs should be interpreted as prescription-count forecasts, peak-event indicators, and demand-oriented risk signals for inspection planning and resource allocation.

---

## Main Components

- **TSPeakNet forecasting model**  
  A short-window branch models local fluctuations, while a long-window branch represents broader temporal patterns.

- **Period-KAN module**  
  Dual-period two-dimensional convolutions capture multi-scale seasonality, and spline-based Kolmogorov–Arnold Network layers model flexible nonlinear mappings.

- **AdaptiveMix fusion**  
  Sample-wise branch weights combine short- and long-window predictions and provide model-internal diagnostics of temporal-scale allocation.

- **Event-level peak detection**  
  The evaluation protocol uses training-derived district-specific thresholds, contiguous-event merging, one-to-one temporal matching, signed timing-error assessment, and peak-height bias.

- **Model interpretation and diagnostics**  
  Utilities are provided for AdaptiveMix-weight analysis, KAN activation visualization, and SHAP-based post-hoc attribution.

- **Benchmarking and sensitivity analysis**  
  The repository includes forecasting baselines, ablation experiments, learning-rate scheduler analysis, sparse-count metrics, and runtime evaluation.

- **Warning-support platform**  
  A web-based dashboard visualizes district-level prescription-count forecasts, demand-oriented risk patterns, peak-event indicators, and model comparisons.

- **External validation**  
  The model is also evaluated on a public weekly aphid-trapping dataset from Coxilha and Passo Fundo.

---

## System Screenshots

### Home Page

![Home page](README.assets/image-20251118105536358.png)

### Data Analysis

![Data analysis 1](README.assets/image-20251118105613532.png)

![Data analysis 2](README.assets/image-20251118105632452.png)

![Data analysis 3](README.assets/image-20251118105714655.png)

![Data analysis 4](README.assets/image-20251118105746599.png)

![Data analysis 5](README.assets/image-20251118105805674.png)

![Data analysis 6](README.assets/image-20251118105822093.png)

*Interactive visualizations for annual, monthly, regional, and multidimensional exploratory analysis.*

### Model Prediction

![Model prediction](README.assets/image-20251118132308853.png)

*Comparison of TSPeakNet with multiple forecasting baselines.*

### Regional Warning-Support Dashboard

![Regional warning](README.assets/image-20251118132550848.png)

*District-level visualization of demand-oriented risk patterns and peak-event indicators.*

### English Interface

![English interface](README.assets/image-20251118132538433.png)

### Pest and Disease Knowledge Panel

![Disease details](README.assets/image-20251118132633146.png)

*Structured information on symptoms, occurrence patterns, and management measures.*

### PEMR Data Entry

![Data collection](README.assets/image-20251118132732173.png)

*Data-entry interface for standardized plant-clinic records and related contextual information.*

---

## Technical Architecture

```text
┌───────────────────────────────────────────────────────────┐
│                  Presentation Layer                       │
│       ECharts 5.x | Plotly.js | HTML5 | CSS3 | ES6+      │
└──────────────────────────┬────────────────────────────────┘
                           │
┌──────────────────────────┴────────────────────────────────┐
│                 Application Layer                         │
│       Flask | HTTP services | Data processing             │
└──────────────────────────┬────────────────────────────────┘
                           │
┌──────────────────────────┴────────────────────────────────┐
│                   Modeling Layer                          │
│ TSPeakNet | Statistical, ML, and deep-learning baselines  │
└──────────────────────────┬────────────────────────────────┘
                           │
┌──────────────────────────┴────────────────────────────────┐
│                     Data Layer                            │
│        PEMR time series | Excel | GeoJSON                 │
└───────────────────────────────────────────────────────────┘
```

### Tested Research Environment

- **Operating system**: Windows 10
- **Python**: 3.7.7
- **PyTorch**: 1.10.2
- **CUDA**: 11.1
- **GPU**: NVIDIA GeForce RTX 4070 Ti
- **Web framework**: Flask
- **Visualization**: ECharts and Plotly
- **Data processing**: NumPy, pandas, SciPy, scikit-learn, statsmodels, and openpyxl

The versions in `requirements.txt` are pinned to remain compatible with the reported Python 3.7.7 environment.

---

## Data

### Beijing PEMR Dataset

- **Source**: Anonymized Plant Electronic Medical Records from plant clinics in ten Beijing districts
- **Observation period**: 25 September 2018 to 18 June 2021
- **Temporal coverage**: 998 daily observations per district
- **Record count**: 144,845 PEMR records
- **Districts**: Daxing, Miyun, Pinggu, Yanqing, Huairou, Fangshan, Changping, Haidian, Tongzhou, and Shunyi
- **Availability**: The district-level PEMR data are available from the corresponding authors upon reasonable request and subject to institutional authorization

### Public Aphid Dataset

The external-validation dataset contains weekly aphid-trapping observations from Coxilha and Passo Fundo:

https://github.com/GabrielRPalma/TimeSeriesReconstruction

---

## Installation

### 1. Create a Python Environment

```bash
conda create -n tspeaknet python=3.7.7
conda activate tspeaknet
```

### 2. Install PyTorch

For the reported CUDA 11.1 environment:

```bash
pip install torch==1.10.2+cu111 -f https://download.pytorch.org/whl/cu111/torch_stable.html
```

For CPU-only execution:

```bash
pip install torch==1.10.2+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html
```

### 3. Install the Remaining Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify the Environment

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

---

## Platform Quick Start

```bash
python prediction_server.py
```

Open the platform in a browser:

```text
http://localhost:8003
```

### Main Pages

| Page | URL | Function |
|---|---|---|
| Home | `/` | Platform navigation and overview |
| Data Collection | `/data-collection` | Entry of PEMR and contextual records |
| Data Analysis | `/data-analysis` | Exploratory analysis and visualization |
| Model Prediction | `/model-prediction` | Forecast comparison across models |
| Regional Warning Support | `/regional-warning` | District-level Chinese dashboard |
| English Dashboard | `/regional-warning-en` | District-level English dashboard |

---

## Data Format

### Observed District-Level Series

```text
Date        | Node_DaXing | Node_MiYun | Node_PingGu | ...
2018-09-25  | 4           | 16          | 16          | ...
2018-09-26  | 5           | 15          | 17          | ...
```

### Forecast Output

```text
Date        | Node_DaXing | Node_MiYun | Node_PingGu | ...
2021-01-01  | 2.34        | 12.45       | 15.67       | ...
2021-01-02  | 2.56        | 13.21       | 16.23       | ...
```

All chronological train-validation-test splits, preprocessing operations, and normalization statistics should be generated causally to prevent information leakage.

---

## Forecasting and Event-Evaluation Workflow

```text
Historical district-level prescription-count series
                         ↓
             Causal preprocessing
                         ↓
        Short-window and long-window inputs
                         ↓
                   TSPeakNet
                         ↓
          One-step-ahead count forecast
                         ↓
 Training-derived district-specific threshold
                         ↓
       Contiguous peak-event construction
                         ↓
 One-to-one matching within the temporal tolerance
                         ↓
 Precision, recall, F1, signed timing error, and PHB
```

### Interpretation of Timing Error

For a matched observed and predicted event:

```text
signed timing error = predicted peak day − observed peak day
```

- A negative value indicates early detection.
- Zero indicates on-time detection.
- A positive value indicates delayed detection.

The event-level results should therefore be interpreted as **peak-event detection and timing diagnostics**, not as uniform advance-warning capability in every district.

---

## Optional Online Recalibration

The main comparative experiments are conducted **without test-time adaptation**. The repository also contains an optional strictly causal online recalibration procedure for deployment. A forecast is generated first; after the true observation becomes available, a small corrective update may be applied before the next prediction.

This mechanism is intended for gradual recalibration under changing reporting conditions and is not the source of the main offline performance claim.

---

## Reproducibility Notes

- Use chronological data splits; do not randomly shuffle time-series samples across train, validation, and test periods.
- Estimate normalization statistics from the training split only.
- Derive district-specific event thresholds from training data only.
- Use identical processed inputs and forecast horizons when comparing models.
- Record random seeds, selected hyperparameters, and software versions for each experiment.
- Keep the main benchmark separate from optional online recalibration experiments.
- Report sparse-count diagnostics together with MAE, RMSE, MASE, RMSSE, and event-level metrics.

---

## Representative Project Structure

```text
TSPeakNet/
├── prediction_server.py          # Warning-support platform server
├── simple_data_reader.py         # Data-reading utilities
├── data_analyzer.py              # Analysis utilities
├── data_collector.py             # Data-entry utilities
├── model_prediction_page.html    # Model-comparison page
├── requirements.txt              # Reproducible Python environment
├── README.md                     # Project documentation
├── README.assets/                # README screenshots
├── 时序数据/                     # Local time-series data directory
│   ├── 原始数据.xlsx
│   ├── LSTM-预测数据.xlsx
│   ├── GRU-预测数据.xlsx
│   ├── TSPeakNet-预测模型.xlsx
│   ├── ...                       # Other model outputs
│   └── 北京.json                 # Map data
└── static/                       # Static platform assets
```

Private or institutionally restricted PEMR files should not be committed to the public repository.

---

## Citation

The manuscript associated with this repository is:

```bibtex
@article{qin2026tspeaknet,
  title   = {TSPeakNet: A dual-scale neural architecture for district-level crop pest and disease time-series forecasting and peak-event detection},
  author  = {Qin, Yuanze and Han, Zonghuan and Zhang, Lingxian and Zhang, Yiding},
  year    = {2026},
  note    = {Manuscript under review}
}
```

Please update the citation with the journal, volume, pages, and DOI after publication.

---

## License

This project is released under the **MIT License**. See the `LICENSE` file for details.

---

## Contact

- *Yuanze Qin**: qinyuanze@cau.edu.cn
- **Repository**: https://github.com/qyz1998453X/TSPeakNet
