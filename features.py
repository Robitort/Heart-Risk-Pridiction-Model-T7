import numpy as np
import pywt
import antropy as ant
import biosppy
from scipy.signal import welch




def extract_morphological_features(ecg_signal):
 
    mean_val = np.mean(ecg_signal)
    std_val = np.std(ecg_signal)
    max_val = np.max(ecg_signal)
    min_val = np.min(ecg_signal)
    return [mean_val, std_val, max_val, min_val]




def extract_wavelet_entropy(ecg_signal, wavelet='db4', level=4):
    """
    wavelet entropy for ECG signal
    """
    coeffs = pywt.wavedec(ecg_signal, wavelet, level=level)
    energy = np.array([np.sum(c**2) for c in coeffs])
    prob_energy = energy / np.sum(energy)
    entropy = -np.sum(prob_energy * np.log2(prob_energy + 1e-8))
    return [entropy]




def extract_spectral_entropy(ecg_signal, fs=500):
  
    freqs, psd = welch(ecg_signal, fs=fs)
    psd_norm = psd / np.sum(psd)
    entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-8))
    return [entropy]




def extract_nonlinear_features(ecg_signal):
    """
    Sample entropy, DFA
    """
    sampen = ant.sample_entropy(ecg_signal)
    dfa = ant.detrended_fluctuation(ecg_signal)
    return [sampen, dfa]




def extract_all_features(ecg_signal, fs=500):
    """
    Extracts all features for 1D ECG signal (one lead)
    """
    ecg_signal = ecg_signal[:, 0] if ecg_signal.ndim == 2 else ecg_signal  # lead I by default


    feats = []
    feats += extract_morphological_features(ecg_signal)
    feats += extract_wavelet_entropy(ecg_signal)
    feats += extract_spectral_entropy(ecg_signal, fs)
    feats += extract_nonlinear_features(ecg_signal)
    return np.array(feats, dtype=np.float32)


