# 🧬 Synaptica Q — Quantum Coherence Index (QCI)

**Author:** Luca Melli  
**License:** Creative Commons Attribution–NonCommercial–ShareAlike 4.0 International (CC BY-NC-SA 4.0)  
**Repository:** [github.com/lucamelli/Synaptica_Q](https://github.com/lucamelli/Synaptica_Q)

---

## 🧠 Overview

*Synaptica Q* is a research framework that models and quantifies **quantum-biological coherence in neural systems**.  
It introduces the **Quantum Coherence Index (QCI)**, a metric derived from EEG signals that estimates the dynamic balance between coherence, entropy, and energetic input within cortical networks.

The QCI is governed by the differential equation:

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?\frac{dQCI(t)}{dt}=\alpha\,E_{in}(t)-\beta\,S(t)+\gamma\,C_{ent}(t)-\kappa\,QCI(t)" alt="QCI differential equation">
</p>

where:

| Symbol | Meaning |
|:-------:|----------|
| **E<sub>in</sub>(t)** | Normalized cortical energetic input (α-band power) |
| **S(t)** | Spectral entropy (measure of disorder) |
| **C<sub>ent</sub>(t)** | Magnitude-squared coherence between EEG channels |
| **QCI(t)** | Damping term representing information dissipation |

---

## ⚙️ Algorithms

| File | Description |
|------|--------------|
| **`synaptica_qci.py`** | Main implementation of the QCI model — computes α, β, γ, κ and fits them to EEG data. |
| **`synaptica_qci_lagsboot.py`** | Lag optimization and bootstrap validation of parameters; generates diagnostic plots and summary CSV. |

---

## 📂 Output Structure

Each subject folder (e.g. `Subject 0`, `Subject 1`, …) contains:

| File | Description |
|------|--------------|
| `qci_timeseries.png` | Normalized features and QCI curve |
| `qci_corr.png` | Feature correlation matrix |
| `qci_lagsboot_fit_train.png` | Model fit on training data |
| `qci_lagsboot_params_summary.csv` | Parameter table with α, β, γ, κ, corr_test, rmse |

---

## 🧪 EEG Dataset

Validated using **PhysioNet — EEG During Mental Arithmetic Tasks**  
[https://physionet.org/content/eegmat/1.0.0/](https://physionet.org/content/eegmat/1.0.0/)  

Channels analyzed: *C3, Cz, C4, Pz, P3, P4* (central-parietal cortex).

---

## 📦 Dependencies

Install the required Python packages:

```bash
pip install numpy scipy mne matplotlib pandas scikit-learn seaborn
