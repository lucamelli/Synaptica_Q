# © 2025 Luca Melli — Synaptica Q
# Licensed under CC-BY-NC-SA 4.0 International
# https://creativecommons.org/licenses/by-nc-sa/4.0/
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Synaptica QCI – Companion per LAG causali + Bootstrap (NO overwrite)
- Canali A -> ingressi (E,S,C); Canali B -> osservabile y (indipendente)
- Train/Test split temporale
- LAG multipli su E,S,C (es. 0, 0.5, 1.0 s)
- MLE vincolata (softplus) con a = exp(-kappa*dt)
- Bootstrap a blocchi (IC 95%) sui parametri e sulle metriche TEST

Output in ~/Downloads con suffisso *_lagsboot.*
"""

from pathlib import Path
import numpy as np
import pandas as pd
import mne
from scipy.signal import welch, coherence
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# -------------------- Config --------------------
DOWNLOADS   = Path.home() / "Downloads"
EDF_FILES   = [DOWNLOADS / "Subject00_1.edf", DOWNLOADS / "Subject00_2.edf"]

# Set A (input) e B (osservabile indipendente)
CHAN_A_PREF = ["EEG C3","EEG Cz","EEG C4","C3","Cz","C4"]
CHAN_B_PREF = ["EEG P3","EEG P4","EEG Pz","P3","P4","Pz"]

FS_TARGET = 250
NOTCH_FREQ = 50.0
HPF, LPF   = 1.0, 45.0

WIN_LEN, WIN_STEP = 2.0, 0.5
DT = WIN_STEP
ALPHA_BAND = (8.0, 12.0)

# LAG causali (secondi) — puoi cambiare liberamente
LAG_SEC = [0.0, 0.5, 1.0]          # ⇒ [0, 1, 2] step
TRAIN_FRAC = 0.7                   # split temporale

# Limiti “fisici” morbidi (evitano tracker aggressivi)
KAPPA_MAX = 10.0                   # s^-1
R_MIN     = 1e-3
Q_MIN     = 1e-5

# Bootstrap
BOOTSTRAP_B = 100
BOOT_BLOCK_SEC = 5.0               # lunghezza blocchi (≈ 10 finestre da 0.5 s)

# Output (non sovrascrive i file esistenti)
OUTDIR = DOWNLOADS
TAG = "lagsboot"
OUT_SERIES = OUTDIR / f"qci_{TAG}_series.csv"
OUT_PARAMS = OUTDIR / f"qci_{TAG}_params.csv"
OUT_PARAMS_BOOT = OUTDIR / f"qci_{TAG}_params_boot.csv"
OUT_PARAMS_SUMM = OUTDIR / f"qci_{TAG}_params_summary.csv"
OUT_FIT_TRAIN = OUTDIR / f"qci_{TAG}_fit_train.png"
OUT_FIT_TEST  = OUTDIR / f"qci_{TAG}_fit_test.png"

# -------------------- Utils --------------------
def trapz_safe(y, x): return np.trapezoid(y, x) if hasattr(np, "trapezoid") else np.trapz(y, x)
def zscore(x): x=np.asarray(x,float); m=x.mean(); s=x.std()+1e-12; return (x-m)/s
def sliding_idx(n,fs,L,S):
    w=int(round(L*fs)); step=int(round(S*fs)); starts=np.arange(0,n-w+1,step); return [(s,s+w) for s in starts]
def welch_params(n): nper=max(32,n//2); return nper,nper//2

def bandpower(x,fs,f1,f2):
    nper,nov=welch_params(len(x)); f,pxx=welch(x,fs=fs,nperseg=nper,noverlap=nov)
    m=(f>=f1)&(f<=f2);  return float(trapz_safe(pxx[m],f[m])) if np.any(m) else 0.0

def total_power(x,fs,f1=HPF,f2=LPF):
    nper,nov=welch_params(len(x)); f,pxx=welch(x,fs=fs,nperseg=nper,noverlap=nov)
    m=(f>=f1)&(f<=f2);  return float(trapz_safe(pxx[m],f[m])) if np.any(m) else 1e-12

def spectral_entropy(x,fs,fmin=HPF,fmax=LPF):
    nper,nov=welch_params(len(x)); f,pxx=welch(x,fs=fs,nperseg=nper,noverlap=nov)
    m=(f>=fmin)&(f<=fmax); p=pxx[m]; p=p/(p.sum()+1e-12)
    H=-np.sum(p*np.log2(p+1e-12)); Hmax=np.log2(m.sum()+1e-12); return float(H/(Hmax+1e-12))

def mean_coh_alpha(W,fs,band):
    nC=W.shape[0]
    if nC<2: return 0.0
    pairs=[(i,j) for i in range(nC) for j in range(i+1,nC)]
    nper,nov=welch_params(W.shape[1]); vals=[]
    for i,j in pairs:
        f,Cxy=coherence(W[i],W[j],fs=fs,nperseg=nper,noverlap=nov)
        m=(f>=band[0])&(f<=band[1])
        if np.any(m): vals.append(Cxy[m].mean())
    return float(np.mean(vals)) if vals else 0.0

def pick(raw, pref):
    lower={c.lower():c for c in raw.ch_names}
    sel=[lower[c.lower()] for c in pref if c.lower() in lower]
    if not sel:
        picks=mne.pick_types(raw.info,eeg=True,meg=False)
        sel=[raw.ch_names[i] for i in picks]
    return sel

# -------------------- Features A/B + lag --------------------
def extract_AB_features():
    # carica e preprocess
    raws=[]
    for f in EDF_FILES:
        if f.exists(): raws.append(mne.io.read_raw_edf(f.as_posix(), preload=True, verbose=False))
    if not raws: raise FileNotFoundError("EDF non trovati.")
    raw=mne.concatenate_raws(raws, verbose=False)

    chA=pick(raw, CHAN_A_PREF); chB=pick(raw, CHAN_B_PREF)
    rawA=raw.copy().pick(chA); rawB=raw.copy().pick(chB)

    for r in (rawA, rawB):
        r.notch_filter([NOTCH_FREQ], picks="eeg", verbose=False)
        r.filter(HPF, LPF, picks="eeg", method="fir", verbose=False)
        r.resample(FS_TARGET, npad="auto", verbose=False)

    dataA,fs=rawA.get_data(),rawA.info["sfreq"]
    dataB    =rawB.get_data()

    idx=sliding_idx(dataA.shape[1], fs, WIN_LEN, WIN_STEP)
    t=np.array([rawA.times[s:e].mean() for (s,e) in idx])

    # Ingressi (E,S,C) da A
    EA, SA, CA = [], [], []
    for (s,e) in idx:
        WA, WB = dataA[:,s:e], dataB[:,s:e]
        ap=[bandpower(WA[ch],fs,*ALPHA_BAND) for ch in range(WA.shape[0])]
        tp=[total_power(WA[ch],fs) for ch in range(WA.shape[0])]
        EA.append(np.log((np.array(ap)/(np.array(tp)+1e-12)).mean()+1e-12))
        SA.append(np.mean([spectral_entropy(WA[ch],fs) for ch in range(WA.shape[0])]))
        CA.append(mean_coh_alpha(WA,fs,ALPHA_BAND))

    E = zscore(EA); S = zscore(SA); C = zscore(CA)
    U = np.vstack([E,S,C]).T  # N×3

    # Osservabile y (B)
    y_raw=[]
    for (s,e) in idx:
        WB=dataB[:,s:e]
        ap=[bandpower(WB[ch],fs,*ALPHA_BAND) for ch in range(WB.shape[0])]
        tp=[total_power(WB[ch],fs) for ch in range(WB.shape[0])]
        Eb = np.log((np.array(ap)/(np.array(tp)+1e-12)).mean()+1e-12)
        Sb = np.mean([spectral_entropy(WB[ch],fs) for ch in range(WB.shape[0])])
        Cb = mean_coh_alpha(WB,fs,ALPHA_BAND)
        y_raw.append(0.5*Eb + 0.5*Cb - 0.5*Sb)
    y = zscore(y_raw)

    return t, U, y

def make_lagged(U, y, lag_sec, dt):
    """Concatena colonne per ciascun lag; tronca in testa per allineare."""
    lags = [int(round(l/dt)) for l in lag_sec]
    Lmax = max(lags)
    U_lagged = []
    for L in lags:
        if L==0:
            U_lagged.append(U[L:])
        else:
            U_lagged.append(U[L:- (0) if L==0 else None])
    # allinea le dimensioni
    N = U.shape[0] - Lmax
    X = np.hstack([U[Lmax - L: Lmax - L + N] for L in lags])  # (N, 3*#lags)
    y2 = y[Lmax: Lmax + N]
    return X, y2, Lmax

# -------------------- Modello/ottimizzazione --------------------
def softplus(z): return np.log1p(np.exp(z))

def loglike_kf_lag(y, X, kappa, q, r, dt, n_lags):
    """
    Stato 1D: x_{k+1} = exp(-kappa*dt)*x_k + dt*( sum_l alpha_l E_{k-l} - sum_l beta_l S_{k-l} + sum_l gamma_l C_{k-l}) + w
    y_k = x_k + v
    X: (N, 3*n_lags) colonne in ordine [E_lags..., S_lags..., C_lags...]
    """
    a = np.exp(-min(kappa, KAPPA_MAX)*dt)
    q = max(q, Q_MIN); r = max(r, R_MIN)

    N = len(y)
    x=0.0; P=1.0; ll=0.0

    # parametri “lineari” (coefficienti) passati via closure nel nll
    global _COEFFS_TMP
    alpha_vec, beta_vec, gamma_vec = _COEFFS_TMP

    for k in range(N):
        Syy=P+r
        innov=y[k]-x
        ll += -0.5*(np.log(2*np.pi*Syy)+(innov**2)/Syy)
        K=P/(Syy+1e-12)
        x = x + K*innov
        P = (1-K)*P

        if k < N-1:
            E_part = np.dot(alpha_vec,  X[k, 0:n_lags])
            S_part = np.dot(beta_vec,   X[k, n_lags:2*n_lags])
            C_part = np.dot(gamma_vec,  X[k, 2*n_lags:3*n_lags])
            drive = dt*(E_part - S_part + C_part)
            x = a*x + drive
            P = a*a*P + q
    return ll

def nll_theta(theta, y, X, dt, n_lags, l2=1e-3):
    """
    theta = [alphas(n_lags), betas(n_lags), gammas(n_lags), kappa, q, r] in R (non vincolati).
    Tutti i parametri sono mappati con softplus per garantirne la non negatività.
    """
    al = softplus(theta[0:n_lags])
    be = softplus(theta[n_lags:2*n_lags])
    ga = softplus(theta[2*n_lags:3*n_lags])
    kappa = softplus(theta[3*n_lags + 0])
    q     = softplus(theta[3*n_lags + 1])
    r     = softplus(theta[3*n_lags + 2])

    # passaggio coeff ai loglike (closure “povera” ma chiara)
    global _COEFFS_TMP
    _COEFFS_TMP = (al, be, ga)

    ll = loglike_kf_lag(y, X, kappa, q, r, dt, n_lags)
    return -(ll - l2*np.sum(theta**2))

def estimate_params_lag(y, X, dt, n_lags):
    theta0 = np.concatenate([
        np.full(n_lags, 0.2),  # alphas
        np.full(n_lags, 0.2),  # betas
        np.full(n_lags, 0.2),  # gammas
        np.array([0.3, -2.0, -1.0])  # kappa, q, r (in spazio non vincolato)
    ]).astype(float)

    res = minimize(nll_theta, theta0, args=(y, X, dt, n_lags, 1e-3), method="L-BFGS-B")
    th = res.x
    al = softplus(th[0:n_lags])
    be = softplus(th[n_lags:2*n_lags])
    ga = softplus(th[2*n_lags:3*n_lags])
    kappa = min(softplus(th[3*n_lags + 0]), KAPPA_MAX)
    q     = max(softplus(th[3*n_lags + 1]), Q_MIN)
    r     = max(softplus(th[3*n_lags + 2]), R_MIN)
    return {"alpha_vec":al, "beta_vec":be, "gamma_vec":ga, "kappa":kappa, "q":q, "r":r}

def kf_filter_lag(y, X, pars, dt, n_lags):
    a=np.exp(-pars["kappa"]*dt)
    al, be, ga = pars["alpha_vec"], pars["beta_vec"], pars["gamma_vec"]
    q, r = pars["q"], pars["r"]
    N=len(y); xhat=np.zeros(N); x=0.0; P=1.0
    for k in range(N):
        Syy=P+r; K=P/(Syy+1e-12); innov=y[k]-x
        x = x + K*innov; P = (1-K)*P
        xhat[k]=x
        if k < N-1:
            drive = dt*( np.dot(al, X[k, 0:n_lags]) - np.dot(be, X[k, n_lags:2*n_lags]) + np.dot(ga, X[k, 2*n_lags:3*n_lags]) )
            x = a*x + drive; P = a*a*P + q
    return xhat

# -------------------- Bootstrap --------------------
def block_bootstrap_indices(N, block_len, rng):
    idx = []
    while len(idx) < N:
        s = rng.integers(0, max(1, N - block_len))
        idx.extend(list(range(s, min(N, s + block_len))))
    return np.array(idx[:N])

# -------------------- Main --------------------
def main():
    # 1) Feature A/B
    t, U, y = extract_AB_features()

    # 2) Lag causali
    X, y2, cut = make_lagged(U, y, LAG_SEC, DT)
    n_lags = len(LAG_SEC); N = len(y2)
    t2 = t[cut:cut+N]

    # 3) Train/Test split
    n_train = int(np.floor(TRAIN_FRAC * N))
    Xtr, ytr, ttr = X[:n_train], y2[:n_train], t2[:n_train]
    Xte, yte, tte = X[n_train:], y2[n_train:], t2[n_train:]

    # 4) Stima su TRAIN
    pars = estimate_params_lag(ytr, Xtr, DT, n_lags)

    # 5) Filtraggio su TRAIN e TEST (no ri-stima)
    xtr = kf_filter_lag(ytr, Xtr, pars, DT, n_lags)
    xte = kf_filter_lag(yte, Xte, pars, DT, n_lags)

    # 6) Metriche
    def metrics(y,x): 
        R=np.corrcoef(y,x)[0,1]; RMSE=np.sqrt(np.mean((y-x)**2)); return R,RMSE
    Rtr,RMSEtr = metrics(ytr, xtr)
    Rte,RMSEte = metrics(yte, xte)

    print("[PARAMETRI] kappa={kappa:.3f} q={q:.4f} r={r:.4f}".format(**pars))
    print("[ALPHA(lag s)]", dict(zip(LAG_SEC, pars["alpha_vec"])))
    print("[BETA(lag s)] ", dict(zip(LAG_SEC, pars["beta_vec"])))
    print("[GAMMA(lag s)]", dict(zip(LAG_SEC, pars["gamma_vec"])))
    print(f"[TRAIN] Corr={Rtr:.3f} RMSE={RMSEtr:.3f}   [TEST] Corr={Rte:.3f} RMSE={RMSEte:.3f}")

    # 7) Bootstrap su TRAIN → IC 95% + metriche TEST
    rng = np.random.default_rng(7)
    B = BOOTSTRAP_B
    block_len = int(round(BOOT_BLOCK_SEC/DT))
    rows=[]
    for b in range(B):
        boot_idx = block_bootstrap_indices(n_train, block_len, rng)
        Xb, yb = Xtr[boot_idx], ytr[boot_idx]
        p = estimate_params_lag(yb, Xb, DT, n_lags)
        # metrica test con param. ri-stimati
        xte_b = kf_filter_lag(yte, Xte, p, DT, n_lags)
        Rte_b, RMSEte_b = metrics(yte, xte_b)
        rows.append({
            **{f"alpha_{LAG_SEC[i]:.2f}": p["alpha_vec"][i] for i in range(n_lags)},
            **{f"beta_{LAG_SEC[i]:.2f}":  p["beta_vec"][i]  for i in range(n_lags)},
            **{f"gamma_{LAG_SEC[i]:.2f}": p["gamma_vec"][i] for i in range(n_lags)},
            "kappa": p["kappa"], "q": p["q"], "r": p["r"],
            "corr_test": Rte_b, "rmse_test": RMSEte_b
        })
    dfb = pd.DataFrame(rows)
    dfb.to_csv(OUT_PARAMS_BOOT, index=False, float_format="%.6f")

    def summary(s):
        return pd.Series({
            "mean": s.mean(), "std": s.std(),
            "p2.5": s.quantile(0.025), "p50": s.quantile(0.5), "p97.5": s.quantile(0.975)
        })
    dfs = dfb.apply(summary).T
    dfs.to_csv(OUT_PARAMS_SUMM, float_format="%.6f")

    # 8) Salvataggi serie complete (per tracciabilità)
    pd.DataFrame({
        "time_s": np.concatenate([ttr,tte]),
        **{f"E_lag{LAG_SEC[i]:.2f}": X[:, i] for i in range(n_lags)},
        **{f"S_lag{LAG_SEC[i]:.2f}": X[:, n_lags+i] for i in range(n_lags)},
        **{f"C_lag{LAG_SEC[i]:.2f}": X[:, 2*n_lags+i] for i in range(n_lags)},
        "y": y2,
        "xhat_QCI": np.concatenate([xtr,xte]),
        "split": (["train"]*len(ttr)) + (["test"]*len(tte))
    }).to_csv(OUT_SERIES, index=False, float_format="%.6f")

    # 9) Figure train/test
    plt.figure(figsize=(12,6)); plt.title("TRAIN: y vs x̂ (QCI) – hold-out + lag")
    plt.plot(ttr, ytr, label="y (B)"); plt.plot(ttr, xtr, label="x̂ (QCI)", linewidth=2)
    plt.xlabel("Time (s)"); plt.ylabel("z-score"); plt.legend(); plt.tight_layout()
    plt.savefig(OUT_FIT_TRAIN, dpi=150); plt.close()

    plt.figure(figsize=(12,6)); plt.title("TEST: y vs x̂ (QCI) – hold-out + lag")
    plt.plot(tte, yte, label="y (B)"); plt.plot(tte, xte, label="x̂ (QCI)", linewidth=2)
    plt.xlabel("Time (s)"); plt.ylabel("z-score"); plt.legend(); plt.tight_layout()
    plt.savefig(OUT_FIT_TEST, dpi=150); plt.close()

    # 10) Parametri punto-stima (comodi da leggere)
    point = {
        **{f"alpha_{LAG_SEC[i]:.2f}": pars["alpha_vec"][i] for i in range(n_lags)},
        **{f"beta_{LAG_SEC[i]:.2f}":  pars["beta_vec"][i]  for i in range(n_lags)},
        **{f"gamma_{LAG_SEC[i]:.2f}": pars["gamma_vec"][i] for i in range(n_lags)},
        "kappa": pars["kappa"], "q": pars["q"], "r": pars["r"],
        "corr_train": Rtr, "rmse_train": RMSEtr,
        "corr_test": Rte, "rmse_test": RMSEte
    }
    pd.DataFrame([point]).to_csv(OUT_PARAMS, index=False, float_format="%.6f")

    print(f"[OK] Serie:  {OUT_SERIES}")
    print(f"[OK] Params: {OUT_PARAMS}")
    print(f"[OK] Boot:   {OUT_PARAMS_BOOT}")
    print(f"[OK] Summ:   {OUT_PARAMS_SUMM}")
    print(f"[OK] Figure: {OUT_FIT_TRAIN}, {OUT_FIT_TEST}")

if __name__ == "__main__":
    main()
