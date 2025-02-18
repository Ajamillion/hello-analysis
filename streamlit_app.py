import streamlit as st
st.set_page_config(layout="wide")
import io, time, json, gzip, base64
import streamlit as st
import librosa, librosa.display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from matplotlib.patches import Wedge
from skimage.transform import resize

# Optional: Loudness measurement via pyloudnorm
try:
    import pyloudnorm as pyln
    LOUDNESS_AVAILABLE = True
except ImportError:
    LOUDNESS_AVAILABLE = False

####################################
# Helper Functions
####################################
def load_audio(file):
    y, sr = librosa.load(file, sr=None, mono=False)
    if y.ndim == 1:
        y = np.array([y])
    return y, sr

def compute_baseline_metrics(x, y, sr):
    """
    Compute baseline (non-serial) metrics from channel 1 (x) using standard units.
    - RMS is computed both as raw and in dBFS.
    - Peak is computed similarly.
    - Crest factor is computed as 20*log10(peak/RMS) (in dB).
    - Dynamic range is computed as (peak - RMS) [raw difference].
    """
    metrics = {}
    # Time-domain
    rms = np.mean(librosa.feature.rms(y=x))
    metrics['rms'] = float(rms)
    metrics['rms_db'] = 20 * np.log10(rms + 1e-9)
    peak = float(np.max(np.abs(x)))
    metrics['peak'] = peak
    metrics['peak_db'] = 20 * np.log10(peak + 1e-9)
    metrics['crest_db'] = 20 * np.log10((peak/(rms+1e-9)) + 1e-9)
    metrics['crest'] = peak/(rms+1e-9)
    metrics['zcr'] = float(np.mean(librosa.feature.zero_crossing_rate(y=x)))
    
    # Spectral metrics
    S = np.abs(librosa.stft(x))
    freqs = librosa.fft_frequencies(sr=sr)
    metrics['centroid'] = float(np.mean(librosa.feature.spectral_centroid(S=S, sr=sr)))
    metrics['bandwidth'] = float(np.mean(librosa.feature.spectral_bandwidth(S=S, sr=sr)))
    metrics['flatness'] = float(np.mean(librosa.feature.spectral_flatness(S=S)))
    metrics['rolloff'] = float(np.mean(librosa.feature.spectral_rolloff(S=S, sr=sr)))
    metrics['contrast'] = float(np.mean(librosa.feature.spectral_contrast(S=S, sr=sr)))
    mean_spec = np.mean(S, axis=1)
    log_freqs = np.log10(freqs + 1e-9)
    slope, _ = np.polyfit(log_freqs, np.log10(mean_spec + 1e-9), 1)
    metrics['slope'] = float(slope)
    
    # Harmonic-to-noise ratio (in dB)
    harmonic = librosa.effects.harmonic(x)
    noise = x - harmonic
    metrics['hnr'] = float(10 * np.log10(np.sum(harmonic**2) / (np.sum(noise**2) + 1e-9)))
    metrics['kurtosis'] = float(stats.kurtosis(mean_spec))
    
    # Frequency band ratios
    low_mask = freqs < 250
    mid_mask = (freqs >= 250) & (freqs < 4000)
    high_mask = freqs >= 4000
    low_energy = np.mean(S[low_mask, :], axis=0)
    mid_energy = np.mean(S[mid_mask, :], axis=0)
    high_energy = np.mean(S[high_mask, :], axis=0)
    metrics['low_mid_ratio'] = float(np.mean(low_energy) / (np.mean(mid_energy) + 1e-9))
    metrics['mid_high_ratio'] = float(np.mean(mid_energy) / (np.mean(high_energy) + 1e-9))
    
    # Dynamic Range (raw difference: peak - RMS)
    metrics['dynamic_range_raw'] = peak - rms
    
    # Loudness
    if LOUDNESS_AVAILABLE:
        meter = pyln.Meter(sr)
        mono_mix = np.mean(y, axis=0)
        metrics['integrated_loudness'] = float(meter.integrated_loudness(mono_mix))
    else:
        metrics['integrated_loudness'] = None
    return metrics

def compute_compound_metrics(baseline):
    """
    Compute compound metrics (normalized 0–1) from baseline values.
    The "Quality" metric is based on our previous formula (hnr * flatness / (zcr)/20).
    """
    compound = {}
    compound['Clarity'] = min(1, baseline['contrast'] / 30)
    compound['Mix Balance'] = min(1, (baseline['low_mid_ratio'] + baseline['mid_high_ratio']) / (baseline['rms'] + 1e-9))
    compound['Quality'] = min(1, baseline['hnr'] * baseline['flatness'] / (baseline['zcr'] + 1e-9) / 20)
    compound['Consistency'] = 1  # Placeholder
    compound['Spatial'] = 1      # For mono only
    compound['Dynamics'] = 1     # Placeholder
    compound['Complexity'] = min(1, baseline['kurtosis'] / 5)
    return compound

def compute_time_based_metrics(y, sr, window_sec=0.2, clip_threshold=0.99):
    """
    Compute five time-based metrics in non-overlapping 200ms windows.
    For each window, dynamic range is computed as (peak - RMS) [raw difference].
    """
    hop_length = int(sr * window_sec)
    n_frames = int(y.shape[1] // hop_length)
    times, sub_bass, clipping, dyn_range_raw, mid_side, stereo_corr = [], [], [], [], [], []
    freqs_stft = librosa.fft_frequencies(sr=sr)
    mask_sub = freqs_stft < 80
    for i in range(n_frames):
        start = i * hop_length; end = start + hop_length
        if y.shape[0] == 1:
            L = y[0, start:end]; R = L
        else:
            L = y[0, start:end]; R = y[1, start:end]
        times.append(i * window_sec)
        stft_L = np.abs(librosa.stft(L))
        stft_R = np.abs(librosa.stft(R))
        energy_L = np.sum(stft_L[mask_sub, :])
        energy_R = np.sum(stft_R[mask_sub, :])
        sub_bass.append(min(energy_L, energy_R) / (max(energy_L, energy_R) + 1e-9))
        clip_count = int(np.sum(np.abs(L) > clip_threshold) + np.sum(np.abs(R) > clip_threshold))
        clipping.append(clip_count)
        peak_val = max(np.max(np.abs(L)), np.max(np.abs(R)))
        rms_val = np.sqrt(np.mean(np.concatenate((L**2, R**2))))
        dyn_range_raw.append(peak_val - rms_val)
        mid = 0.5 * (L + R); side = 0.5 * (L - R)
        mid_side.append(np.sum(side**2) / (np.sum(mid**2) + 1e-9))
        stereo_corr.append(float(np.corrcoef(L, R)[0, 1]) if len(L) > 1 else 1.0)
    return np.array(times), np.array(sub_bass), np.array(clipping), np.array(dyn_range_raw), np.array(mid_side), np.array(stereo_corr)

def compute_summary_stats(arr):
    return {"mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr))}

def plot_radial_gauge(score, title="aQi (Audio Quality Index)"):
    fig, ax = plt.subplots(figsize=(4,4))
    ax.axis('equal'); ax.axis('off')
    bg = Wedge((0,0), 1, 0, 180, facecolor='lightgrey', edgecolor='none')
    ax.add_patch(bg)
    theta2 = 180 * score / 100
    fg = Wedge((0,0), 1, 0, theta2, facecolor='green', edgecolor='none')
    ax.add_patch(fg)
    circle = plt.Circle((0,0), 1, color="black", fill=False, linewidth=2)
    ax.add_artist(circle)
    ax.text(0, 0, f"{score:.1f}%", ha='center', va='center', fontsize=20)
    ax.set_title(title)
    return fig

def plot_radar_chart(metrics, title="Compound Metrics"):
    labels = list(metrics.keys())
    values = list(metrics.values())
    num_vars = len(labels)
    angles = np.linspace(0, 2*np.pi, num_vars, endpoint=False).tolist()
    values += values[:1]; angles += angles[:1]
    fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
    ax.plot(angles, values, color="blue", linewidth=2)
    ax.fill(angles, values, color="blue", alpha=0.25)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels)
    ax.set_title(title, y=1.1)
    return fig

def spectrogram_summary(S_db, target_shape=(100,100)):
    return resize(S_db, target_shape, anti_aliasing=True)

def compress_field(data):
    data_str = json.dumps(data)
    compressed = gzip.compress(data_str.encode("utf-8"))
    encoded = base64.b64encode(compressed).decode("utf-8")
    return encoded

def default_converter(o):
    if isinstance(o, (np.float32, np.float64)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {type(o)} is not JSON serializable")

####################################
# Main App
####################################
st.title("HMWMM aQi Advanced Analyzer")

uploaded_file = st.file_uploader("Upload an audio file (WAV, MP3, OGG)", type=["wav", "mp3", "ogg"])
if uploaded_file is not None:
    y, sr = load_audio(uploaded_file)
    n_channels = y.shape[0]
    duration_sec = y.shape[1] / sr
    st.metric("File Info", f"{n_channels} channels • {sr} Hz • {duration_sec:.1f} sec")
    
    # Stage 1: Baseline Metrics
    st.header("Stage 1: Baseline Summary")
    x = y[0]
    baseline = compute_baseline_metrics(x, y, sr)
    compound = compute_compound_metrics(baseline)
    baseline_aQi = np.mean(list(compound.values())) * 100
    st.metric("Baseline aQi", f"{baseline_aQi:.1f}%")
    with st.expander("Show Detailed Baseline Metrics"):
        st.write({
            "Time Metrics": {"RMS": baseline['rms'], "RMS (dBFS)": baseline['rms_db'], "Peak": baseline['peak'], "Peak (dBFS)": baseline['peak_db'],
                             "Crest (raw)": baseline['crest'], "Crest (dB)": baseline['crest_db'], "ZCR": baseline['zcr']},
            "Spectral Metrics": {"Centroid": baseline['centroid'], "Bandwidth": baseline['bandwidth'],
                                 "Flatness": baseline['flatness'], "Roll-off": baseline['rolloff'],
                                 "Contrast": baseline['contrast'], "Slope": baseline['slope'], "HNR": baseline['hnr'],
                                 "Kurtosis": baseline['kurtosis']},
            "Frequency Bands": {"Low/Mid Ratio": baseline['low_mid_ratio'], "Mid/High Ratio": baseline['mid_high_ratio']},
            "Dynamic Range (raw)": baseline['dynamic_range_raw'],
            "Loudness": baseline['integrated_loudness']
        })
        st.write("Compound Metrics:", compound)
    
    # Spectrogram display
    st.subheader("Spectrogram")
    fig_spec, ax_spec = plt.subplots(figsize=(10,4))
    S_full = np.abs(librosa.stft(x))
    S_db = librosa.amplitude_to_db(S_full, ref=np.max)
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log', ax=ax_spec)
    ax_spec.set_title("Spectrogram (dB)")
    st.pyplot(fig_spec)
    
    # Stage 2: Time-Based Metrics
    st.header("Stage 2: Time-Based Metrics")
    WINDOW_SEC = 0.2
    times, sub_bass, clipping, dyn_range_raw, mid_side, stereo_corr = compute_time_based_metrics(y, sr, window_sec=WINDOW_SEC, clip_threshold=0.99)
    
    sub_bass_cum = float(np.mean(sub_bass))
    clipping_cum = max(0, 1 - (float(np.mean(clipping)) / 10))
    dyn_range_cum = float(np.mean(dyn_range_raw))
    # For time-based, we keep the raw dynamic range (peak - RMS)
    dyn_range_norm = min(1, dyn_range_cum / 0.5)  # Adjust normalization scale as appropriate
    mid_side_cum = float(np.mean(mid_side))
    mid_side_norm = max(0, 1 - abs(mid_side_cum - 1))
    stereo_corr_cum = float(np.mean(stereo_corr))
    stereo_corr_norm = stereo_corr_cum if stereo_corr_cum >= 0 else (stereo_corr_cum + 1) / 2
    time_aQi = np.mean([sub_bass_cum, clipping_cum, dyn_range_norm, mid_side_norm, stereo_corr_norm]) * 100
    summary_stats = {
        "Sub-Bass Mono": compute_summary_stats(np.array(sub_bass)),
        "Clipping Count": compute_summary_stats(np.array(clipping)),
        "Dynamic Range (raw)": compute_summary_stats(np.array(dyn_range_raw)),
        "Mid/Side Ratio": compute_summary_stats(np.array(mid_side)),
        "Stereo Correlation": compute_summary_stats(np.array(stereo_corr))
    }
    
    progress_bar = st.progress(0)
    progress_text = st.empty()
    start_time = time.time()
    n_frames = len(times)
    for i in range(n_frames):
        progress = (i+1)/n_frames
        progress_bar.progress(progress)
        elapsed = time.time() - start_time
        est_total = elapsed / (i+1) * n_frames
        est_remaining = est_total - elapsed
        progress_text.text(f"Processing frame {i+1}/{n_frames}. Est. time remaining: {est_remaining:.1f} s")
        time.sleep(0.001)
    progress_text.empty()
    
    df_time = pd.DataFrame({
        "Time (s)": times,
        "Sub-Bass Mono": sub_bass,
        "Clipping Count": clipping,
        "Dynamic Range (raw)": dyn_range_raw,
        "Mid/Side Ratio": mid_side,
        "Stereo Corr": stereo_corr
    })
    with st.expander("Show Raw Time-Based Metrics"):
        st.dataframe(df_time)
    st.subheader("Time-Based Metric Trends")
    st.line_chart(df_time.set_index("Time (s)"))
    
    final_aQi = (baseline_aQi + time_aQi) / 2
    st.metric("Final aQi (Audio Quality Index)", f"{final_aQi:.1f}%")
    with st.expander("aQi Explanation"):
        st.write("""
        **aQi (Audio Quality Index) Calculation:**

        - **Baseline aQi:** Derived from compound metrics computed on channel 1 over the full file.
        - **Time-Based aQi:** Derived from cumulative normalized values computed from 200ms windows:
          - *Sub-Bass Mono Ratio:* Average similarity (ideal = 1).
          - *Clipping:* Normalized as 1 – (avg. clipping count/10).
          - *Dynamic Range (raw):* Computed as (peak - RMS) and normalized by an ideal scale (adjusted here to 0.5).
          - *Mid/Side Ratio:* Normalized as 1 – |avg – 1| (ideal = 1).
          - *Stereo Correlation:* Average correlation (normalized 0–1).
          
        The final aQi is the mean of these two values.
        """)
    
    # Export Data
    st.header("Export Metrics for AI Analysis")
    target_shape = (100,100)
    S_db_summary = spectrogram_summary(S_db, target_shape=target_shape)
    spec_export = {
        "frequencies": resize(librosa.fft_frequencies(sr=sr), (target_shape[0],), anti_aliasing=True).tolist(),
        "times": resize(librosa.frames_to_time(np.arange(S_full.shape[1]), sr=sr), (target_shape[1],), anti_aliasing=True).tolist(),
        "S_db_summary": S_db_summary.tolist()
    }
    buf_spec_img = io.BytesIO()
    fig_spec.savefig(buf_spec_img, format="png", bbox_inches="tight")
    buf_spec_img.seek(0)
    
    def compress_data(data):
        data_str = json.dumps(data)
        compressed = gzip.compress(data_str.encode("utf-8"))
        encoded = base64.b64encode(compressed).decode("utf-8")
        return encoded
    spec_export_compressed = compress_data(spec_export)
    
    metadata = {
        "baseline_metrics": "Time-domain and spectral features from channel 1. RMS is provided in raw and dBFS; Peak in raw and dBFS; Crest factor in raw and dB; Dynamic range as (peak - RMS).",
        "compound_metrics": "Normalized metrics (0–1) representing clarity, mix balance, quality (using HNR, flatness, and ZCR), consistency, spatial, dynamics, complexity.",
        "time_based_metrics": "Per-window (200ms) metrics. 'Raw' arrays are provided; 'summary' includes mean, median, std, min, and max.",
        "spectrogram": f"Spectrogram summary downsampled to shape {target_shape}. Frequencies (Hz), times (s), and amplitudes (dB) are downsampled. Also provided as a gzip-compressed, base64-encoded string in 'spectrogram_compressed'.",
        "aQi_explanation": "aQi is the average of the Baseline aQi (full-file analysis) and the Time-Based aQi (200ms windows)."
    }
    
    export_data = {
        "file_info": {"channels": n_channels, "sample_rate": sr, "duration_sec": duration_sec},
        "baseline_metrics": baseline,
        "compound_metrics": compound,
        "baseline_aQi": baseline_aQi,
        "time_based_metrics": {
            "raw": df_time.to_dict(orient="list"),
            "summary": summary_stats
        },
        "spectrogram": spec_export,
        "spectrogram_compressed": spec_export_compressed,
        "metadata": metadata,
        "aQi_explanation": (
            "Final aQi (Audio Quality Index) is computed as the average of the Baseline aQi and the Time-Based aQi.\n"
            "Baseline aQi is derived from normalized compound metrics computed on channel 1 over the full file.\n"
            "Time-Based aQi is computed from cumulative normalized values from 200ms windows, based on:\n"
            "  - Sub-Bass Mono Ratio (ideal = 1),\n"
            "  - Clipping (normalized as 1 - (avg clipping count/10)),\n"
            "  - Dynamic Range (raw) (normalized by an ideal scale, here set to 0.5),\n"
            "  - Mid/Side Ratio (ideal = 1),\n"
            "  - Stereo Correlation (normalized 0–1).\n"
            "The final aQi is the mean of these two values."
        )
    }
    
    export_json = json.dumps(export_data, default=default_converter, indent=2)
    st.download_button(label="Export Metrics as JSON",
                       data=export_json,
                       file_name="audio_metrics.json",
                       mime="application/json")
    st.download_button(label="Download Spectrogram Image",
                       data=buf_spec_img,
                       file_name="spectrogram.png",
                       mime="image/png")
