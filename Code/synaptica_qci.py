# © 2025 Luca Melli — Synaptica Q
# Licensed under CC-BY-NC-SA 4.0 International
# https://creativecommons.org/licenses/by-nc-sa/4.0/
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
synaptica_qci.py
================
Pipeline QCI (dati EEG reali in EDF -> feature -> dQCI/dt -> QCI(t))

Formula implementata:
    dQCI/dt = α * E_in(t) - β * S(t) + γ * C_ent(t) - κ * QCI(t)

Cosa fa:
- Carica 'Subject1.0.edf' e 'Subject1.1.edf' da ~/Downloads (se esistono).
- Seleziona canali centrali/parietali (fallback: tutti gli EEG).
- Preprocess: notch (50Hz), bandpass (1–45 Hz), resampling (250 Hz).
- Finestre scorrevoli: estrae
    E_in  = potenza banda(i) (default: alpha 8–12 Hz) media sui canali.
    S     = entropia spettrale normalizzata.
    C_ent = coerenza media (magnitude-squared) in banda alpha tra coppie di canali.
- Normalizza robustamente le feature, calcola dQCI/dt e integra QCI (Eulero).
- Salva:
    ~/Downloads/qci_timeseries.csv
    ~/Downloads/qci_timeseries.png
    ~/Downloads/qci_corr.csv (+ heatmap qci_corr.png)
    ~/Downloads/qci_conditions.csv (se ci sono annotazioni REST/TASK/EC/EO)

Modifica in testa i parametri (bande, finestre, α β γ κ, ecc.) secondo necessità.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
from scipy.signal import welch, coherence

# -----------------------------
# Parametri principali
# -----------------------------
DOWNLOADS = Path.home() / "Downloads"
EDF_FILES = [DOWNLOADS / "Subject00_1.edf", DOWNLOADS / "Subject00_2.edf"]

# Canali target (fallback automatico se non trovati)
PREFERRED_CHANNELS = [
    "EEG C3", "EEG Cz", "EEG C4", "EEG P3", "EEG P4", "EEG Pz",
    "C3", "Cz", "C4", "P3", "P4", "Pz"
]

FS_TARGET = 250          # Hz
NOTCH_FREQ = 50.0        # Hz (Italia)
HPF, LPF = 1.0, 45.0     # filtraggio banda stretta

# Finestre (secondi)
WIN_LEN = 2.0
WIN_STEP = 0.5

# Bande Hz per E_in (puoi combinare pesi)
BANDS = {
    "alpha": (8.0, 12.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 45.0),
}
EIN_WEIGHTS = {"alpha": 1.0, "beta": 0.0, "gamma": 0.0}  # default: solo alpha

# Parametri formula (α, β, γ, κ)
ALPHA_PARAM = 1.0
BETA_PARAM  = 1.0
GAMMA_PARAM = 1.0
KAPPA_PARAM = 0.10
QCI0        = 0.0

# Smussamento opzionale di QCI (secondi; 0=off)
QCI_SMOOTH_SEC = 5.0

# Output
OUT_CSV       = DOWNLOADS / "qci_timeseries.csv"
OUT_PNG       = DOWNLOADS / "qci_timeseries.png"
OUT_CORR_CSV  = DOWNLOADS / "qci_corr.csv"
OUT_CORR_PNG  = DOWNLOADS / "qci_corr.png"
OUT_COND_CSV  = DOWNLOADS / "qci_conditions.csv"  # creato solo se ci sono annotazioni utili


# -----------------------------
# Utility numeriche
# -----------------------------
def trapz_safe(y, x):
    """Compat per NumPy >=1.26 (trapezoid) e precedenti (trapz)."""
    return np.trapezoid(y, x) if hasattr(np, "trapezoid") else np.trapz(y, x)

def pick_available_channels(raw, preferred_list):
    """Tenta il match case-insensitive; fallback a tutti gli EEG."""
    lower_map = {ch.lower(): ch for ch in raw.ch_names}
    sel = [lower_map[c.lower()] for c in preferred_list if c.lower() in lower_map]
    if not sel:
        picks = mne.pick_types(raw.info, eeg=True, meg=False, eog=False, ecg=False, stim=False)
        sel = [raw.ch_names[i] for i in picks]
    return sel

def welch_params(n_win):
    nper = max(32, n_win // 2)
    return nper, nper // 2

def bandpower_ts(x, fs, fmin, fmax):
    """Potenza integrata nella banda su finestra 1D."""
    nper, nov = welch_params(len(x))
    f, pxx = welch(x, fs=fs, nperseg=nper, noverlap=nov)
    m = (f >= fmin) & (f <= fmax)
    if not np.any(m): return 0.0
    return float(trapz_safe(pxx[m], f[m]))

def spectral_entropy_ts(x, fs, fmin=HPF, fmax=LPF):
    """Entropia spettrale normalizzata (0..1)."""
    nper, nov = welch_params(len(x))
    f, pxx = welch(x, fs=fs, nperseg=nper, noverlap=nov)
    m = (f >= fmin) & (f <= fmax)
    p = pxx[m]
    p = p / (np.sum(p) + 1e-12)
    H = -np.sum(p * np.log2(p + 1e-12))
    Hmax = np.log2(np.sum(m) + 1e-12)
    return float(H / (Hmax + 1e-12))

def mean_coherence_band(W, fs, fband):
    """Coerenza media (magnitude-squared) in banda tra tutte le coppie di canali."""
    nC, _ = W.shape
    if nC < 2: return 0.0
    pairs = [(i, j) for i in range(nC) for j in range(i+1, nC)]
    nper, nov = welch_params(W.shape[1])
    vals = []
    for i, j in pairs:
        f, Cxy = coherence(W[i], W[j], fs=fs, nperseg=nper, noverlap=nov)
        m = (f >= fband[0]) & (f <= fband[1])
        if np.any(m): vals.append(np.mean(Cxy[m]))
    return float(np.mean(vals)) if vals else 0.0

def sliding_indices(n_samples, fs, win_len_s, win_step_s):
    win = int(round(win_len_s * fs))
    step = int(round(win_step_s * fs))
    starts = np.arange(0, n_samples - win + 1, step)
    return [(s, s + win) for s in starts]

def robust_unit(x):
    """Scala robusta ~[0,1] usando 5°–95° percentile (evita outlier)."""
    q1, q3 = np.quantile(x, [0.05, 0.95])
    denom = (q3 - q1) if (q3 - q1) > 1e-12 else (np.std(x) + 1e-12)
    return np.clip((x - q1) / denom, 0, 1)

def moving_average(x, k):
    if k <= 1: return x.copy()
    ker = np.ones(int(k)) / int(k)
    return np.convolve(x, ker, mode="same")


# -----------------------------
# Pipeline principale
# -----------------------------
def main():
    # Caricamento EDF
    raws = []
    for f in EDF_FILES:
        if f.exists():
            print(f"[INFO] Carico: {f}")
            raws.append(mne.io.read_raw_edf(f.as_posix(), preload=True, verbose=False))
        else:
            print(f"[WARN] File non trovato: {f}")
    if not raws:
        raise FileNotFoundError("Nessun EDF trovato in ~/Downloads (Subject1.0.edf, Subject1.1.edf).")

    raw = mne.concatenate_raws(raws, verbose=False)

    # Picks canali
    sel = pick_available_channels(raw, PREFERRED_CHANNELS)
    print(f"[INFO] Canali selezionati: {sel}")
    raw.pick(sel)

    # Preprocess
    raw.notch_filter([NOTCH_FREQ], picks="eeg", verbose=False)
    raw.filter(HPF, LPF, picks="eeg", method="fir", verbose=False)
    raw.resample(FS_TARGET, npad="auto", verbose=False)

    data, fs = raw.get_data(picks="eeg"), raw.info["sfreq"]
    nC, nS = data.shape
    print(f"[INFO] Dati: {nC} canali, {nS} campioni, fs={fs:.1f} Hz")

    # Finestre
    idx = sliding_indices(nS, fs, WIN_LEN, WIN_STEP)
    times = np.array([raw.times[s:e].mean() for (s, e) in idx])

    # Estrazione feature per finestra
    E_raw, S_raw, C_raw = [], [], []
    for (s, e) in idx:
        W = data[:, s:e]

        # E_in: somma pesata delle bandpower (media sui canali)
        band_powers = {}
        for bname, (f1, f2) in BANDS.items():
            bp = [bandpower_ts(W[ch], fs, f1, f2) for ch in range(nC)]
            band_powers[bname] = float(np.mean(bp))
        E_in = sum(EIN_WEIGHTS.get(b, 0.0) * band_powers[b] for b in BANDS.keys())

        # S: entropia spettrale media sui canali
        ents = [spectral_entropy_ts(W[ch], fs, HPF, LPF) for ch in range(nC)]
        S = float(np.mean(ents))

        # C_ent: coerenza media in banda alpha
        C_ent = mean_coherence_band(W, fs, BANDS["alpha"])

        E_raw.append(E_in); S_raw.append(S); C_raw.append(C_ent)

    E_raw = np.array(E_raw)
    S_raw = np.array(S_raw)
    C_raw = np.array(C_raw)

    # Normalizzazioni per stabilità numerica
    E = robust_unit(E_raw)
    S = np.clip(S_raw, 0, 1)
    C = np.clip(C_raw, 0, 1)

    # Integrazione QCI (Eulero)
    dt = WIN_STEP
    QCI = np.zeros_like(times)
    dQ = np.zeros_like(times)
    QCI[0] = QCI0
    for k in range(len(times)):
        d = ALPHA_PARAM * E[k] - BETA_PARAM * S[k] + GAMMA_PARAM * C[k] - KAPPA_PARAM * QCI[k]
        dQ[k] = d
        if k < len(times) - 1:
            QCI[k+1] = QCI[k] + d * dt

    # Smoothing opzionale (solo per visualizzazione)
    ksm = int(round(QCI_SMOOTH_SEC / dt)) if QCI_SMOOTH_SEC > 0 else 1
    QCI_sm = moving_average(QCI, ksm)

    # Salvataggio CSV principale
    df = pd.DataFrame({
        "time_s": times,
        "E_in_raw": E_raw, "E_in_norm": E,
        "S_entropy": S_raw, "S_entropy_norm": S,
        "C_ent_alpha": C_raw, "C_ent_alpha_norm": C,
        "dQCI_dt": dQ,
        "QCI": QCI, "QCI_smooth": QCI_sm
    })
    df.to_csv(OUT_CSV, index=False, float_format="%.6f")
    print(f"[OK] Salvato: {OUT_CSV}")

    # Figura principale
    plt.figure(figsize=(12, 7))
    plt.title("QCI pipeline (feature normalizzate e QCI)")
    plt.plot(times, E, label="E_in (norm)")
    plt.plot(times, 1.0 - S, label="1 - S (entropy)")
    plt.plot(times, C, label="C_ent α (norm)")
    qscaled = (QCI_sm - np.min(QCI_sm)) / (np.max(QCI_sm) - np.min(QCI_sm) + 1e-12)
    plt.plot(times, qscaled, label="QCI (smooth, scaled)")
    plt.xlabel("Time (s)"); plt.ylabel("Normalized units"); plt.legend(loc="best")
    plt.tight_layout(); plt.savefig(OUT_PNG, dpi=150); plt.close()
    print(f"[OK] Figura salvata: {OUT_PNG}")

    # Matrice di correlazione (facoltativa ma utile)
    corr = df[["E_in_norm","S_entropy_norm","C_ent_alpha_norm","dQCI_dt","QCI_smooth"]].corr()
    corr.to_csv(OUT_CORR_CSV, float_format="%.4f")
    plt.figure(figsize=(6,5))
    plt.imshow(corr.values, aspect="auto", origin="lower")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    plt.yticks(range(len(corr.index)), corr.index)
    plt.colorbar(label="Pearson r")
    plt.title("Correlation matrix")
    plt.tight_layout(); plt.savefig(OUT_CORR_PNG, dpi=150); plt.close()
    print(f"[OK] Correlazioni: {OUT_CORR_CSV}, {OUT_CORR_PNG}")

    # Statistiche per condizioni dagli annotation (se presenti)
    cond_rows = extract_condition_stats(raw, df)
    if cond_rows:
        pd.DataFrame(cond_rows).to_csv(OUT_COND_CSV, index=False, float_format="%.6f")
        print(f"[OK] Statistiche per condizione: {OUT_COND_CSV}")
    else:
        print("[INFO] Nessuna annotazione REST/TASK/EC/EO trovata: skip condizioni.")

def extract_condition_stats(raw, df_ts):
    """Cerca annotazioni (REST/TASK/EC/EO) e calcola medie delle feature/QCI per intervallo."""
    anns = getattr(raw, "annotations", None)
    if anns is None or len(anns) == 0: return []
    rows = []
    for ann in anns:
        desc = str(ann["description"]).upper().strip()
        onset = float(ann["onset"]); dur = float(ann["duration"]) if "duration" in ann else 0.0
        if any(k in desc for k in ["REST","BASELINE"]): tag = "REST"
        elif "TASK" in desc: tag = "TASK"
        elif "EC" in desc or "EYES CLOSED" in desc: tag = "EC"
        elif "EO" in desc or "EYES OPEN" in desc: tag = "EO"
        else: tag = None
        if not tag: continue
        t0, t1 = onset, onset + max(dur, 0.0)
        mask = (df_ts["time_s"] >= t0) & (df_ts["time_s"] <= t1)
        if not mask.any(): continue
        sub = df_ts.loc[mask]
        rows.append({
            "condition": tag, "t_start": t0, "t_end": t1, "n_windows": int(mask.sum()),
            "E_in_norm_mean": sub["E_in_norm"].mean(),
            "S_entropy_norm_mean": sub["S_entropy_norm"].mean(),
            "C_ent_alpha_norm_mean": sub["C_ent_alpha_norm"].mean(),
            "dQCI_dt_mean": sub["dQCI_dt"].mean(),
            "QCI_smooth_mean": sub["QCI_smooth"].mean(),
        })
    return rows


if __name__ == "__main__":
    main()
