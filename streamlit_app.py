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
        if len(L) > 1 and len(R) > 1:
            corr_val = np.corrcoef(L, R)[0, 1]
        else:
            corr_val = 1.0
        stereo_corr.append(float(np.nan_to_num(corr_val, nan=0.0, posinf=0.0, neginf=0.0)))
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

def analyze_mix_health(baseline, compound, sub_bass, clipping, dyn_range_raw, mid_side, stereo_corr):
    """Derive heuristic mix observations for LLM prompt building."""

    def _safe_array(arr):
        if arr is None:
            return np.array([])
        cleaned = np.asarray(arr, dtype=float)
        if cleaned.ndim == 0:
            cleaned = cleaned.reshape(1)
        return cleaned

    arr_sub = _safe_array(sub_bass)
    arr_clipping = _safe_array(clipping)
    arr_dyn = _safe_array(dyn_range_raw)
    arr_mid_side = _safe_array(mid_side)
    arr_stereo = np.nan_to_num(_safe_array(stereo_corr), nan=0.0, posinf=0.0, neginf=0.0)

    issues = []
    strengths = []

    def add_issue(severity, text):
        issues.append({"severity": severity, "text": text})

    def add_strength(text):
        strengths.append(text)

    def safe_stat(arr, func, fallback=None):
        if arr.size == 0:
            return fallback
        value = func(arr)
        if np.isnan(value):
            return fallback
        return float(value)

    # Windowed statistics
    clip_windows_ratio = safe_stat(arr_clipping > 0, np.mean, 0.0)
    total_clipped_samples = int(np.sum(arr_clipping)) if arr_clipping.size else 0
    sub_bass_mean = safe_stat(arr_sub, np.mean)
    mid_side_mean = safe_stat(arr_mid_side, np.mean)
    stereo_corr_mean = safe_stat(arr_stereo, np.mean)
    dyn_range_mean = safe_stat(arr_dyn, np.mean)

    # Headroom & clipping
    peak_db = baseline.get('peak_db', 0.0)
    if clip_windows_ratio and clip_windows_ratio > 0:
        severity = "High" if clip_windows_ratio > 0.1 or baseline.get('peak', 0) >= 0.999 else "Moderate"
        add_issue(severity, f"Detected {total_clipped_samples} clipped samples across analysis windows (peak level {peak_db:.2f} dBFS).")
    else:
        add_strength(f"No clipping detected; peak level sits at {peak_db:.2f} dBFS.")

    # Crest factor & dynamics
    crest_db = baseline.get('crest_db')
    if crest_db is not None:
        crest_db = float(crest_db)
        if crest_db < 6:
            add_issue("High", f"Crest factor is {crest_db:.1f} dB, indicating restricted punch (typical mixes land around 8–14 dB).")
        elif crest_db < 8:
            add_issue("Moderate", f"Crest factor is {crest_db:.1f} dB; consider restoring a bit more transient headroom.")
        elif 8 <= crest_db <= 14:
            add_strength(f"Crest factor of {crest_db:.1f} dB sits in a healthy mix range.")

    if dyn_range_mean is not None:
        if dyn_range_mean is not None and dyn_range_mean < 0.05:
            add_issue("High", f"Average peak-to-RMS window range is {dyn_range_mean:.3f}, suggesting heavy limiting.")
        elif dyn_range_mean is not None and dyn_range_mean < 0.1:
            add_issue("Moderate", f"Average peak-to-RMS window range is {dyn_range_mean:.3f}; dynamics are a bit constrained.")
        elif dyn_range_mean and dyn_range_mean > 0.25:
            add_strength(f"Windowed peak-to-RMS range averages {dyn_range_mean:.3f}, retaining good punch.")

    # Spectral balance
    low_mid_ratio = baseline.get('low_mid_ratio')
    if low_mid_ratio is not None:
        low_mid_ratio = float(low_mid_ratio)
        if low_mid_ratio > 1.6:
            add_issue("Moderate", f"Low frequencies dominate the mids (low/mid ratio {low_mid_ratio:.2f}).")
        elif low_mid_ratio < 0.6:
            add_issue("Moderate", f"Low frequencies trail the mids (low/mid ratio {low_mid_ratio:.2f}).")
        elif 0.8 <= low_mid_ratio <= 1.2:
            add_strength(f"Low and mid energy stay balanced (ratio {low_mid_ratio:.2f}).")

    mid_high_ratio = baseline.get('mid_high_ratio')
    if mid_high_ratio is not None:
        mid_high_ratio = float(mid_high_ratio)
        if mid_high_ratio > 1.6:
            add_issue("Moderate", f"Midrange leans forward of highs (mid/high ratio {mid_high_ratio:.2f}).")
        elif mid_high_ratio < 0.6:
            add_issue("Moderate", f"Top end outweighs the mids (mid/high ratio {mid_high_ratio:.2f}).")
        elif 0.8 <= mid_high_ratio <= 1.2:
            add_strength(f"Mid and high balance looks even (ratio {mid_high_ratio:.2f}).")

    # Stereo behavior
    if sub_bass_mean is not None:
        if sub_bass_mean < 0.5:
            add_issue("High", f"Sub-bass mono compatibility averages {sub_bass_mean:.2f}; expect cancellations when summed to mono.")
        elif sub_bass_mean < 0.7:
            add_issue("Moderate", f"Sub-bass mono compatibility averages {sub_bass_mean:.2f}; consider tightening lows.")
        elif sub_bass_mean > 0.9:
            add_strength(f"Sub-bass mono compatibility is excellent (mean {sub_bass_mean:.2f}).")

    if mid_side_mean is not None:
        if mid_side_mean > 1.5:
            add_issue("Moderate", f"Mid/side energy ratio averages {mid_side_mean:.2f}; the sides dominate the stereo field.")
        elif mid_side_mean < 0.5:
            add_issue("Moderate", f"Mid/side energy ratio averages {mid_side_mean:.2f}; the mix may feel narrow.")
        else:
            add_strength(f"Mid/side energy ratio {mid_side_mean:.2f} keeps the stereo image balanced.")

    if stereo_corr_mean is not None:
        if stereo_corr_mean < 0.2:
            add_issue("High", f"Average stereo correlation is {stereo_corr_mean:.2f}; check for phase issues.")
        elif stereo_corr_mean < 0.6:
            add_issue("Moderate", f"Average stereo correlation is {stereo_corr_mean:.2f}; the image may drift or collapse in mono.")
        elif stereo_corr_mean > 0.8:
            add_strength(f"Stereo correlation averages {stereo_corr_mean:.2f}, showing cohesive left/right behavior.")

    # Loudness indicators
    rms_db = baseline.get('rms_db')
    if rms_db is not None:
        rms_db = float(rms_db)
        if rms_db > -8:
            add_issue("Moderate", f"RMS level sits at {rms_db:.1f} dBFS; watch that headroom isn't exhausted before mastering.")
        elif rms_db < -26:
            add_issue("Moderate", f"RMS level sits at {rms_db:.1f} dBFS; consider raising the mix bus if noise allows.")

    integrated_loudness = baseline.get('integrated_loudness')
    if integrated_loudness is not None:
        loudness_val = float(integrated_loudness)
        if loudness_val > -10:
            add_issue("High", f"Integrated loudness is {loudness_val:.1f} LUFS, above typical streaming limits (aim ≈ -14 LUFS).")
        elif loudness_val < -20:
            add_issue("Moderate", f"Integrated loudness is {loudness_val:.1f} LUFS, quieter than most references (aim ≈ -14 LUFS).")
        else:
            add_strength(f"Integrated loudness at {loudness_val:.1f} LUFS is in the streaming-ready window.")

    # Compound scores
    clarity = compound.get('Clarity')
    if clarity is not None:
        clarity = float(clarity)
        if clarity < 0.5:
            add_issue("Moderate", f"Clarity score is {clarity:.2f}; transient definition may need attention.")
        elif clarity > 0.7:
            add_strength(f"Clarity score of {clarity:.2f} suggests well-defined transients.")

    quality = compound.get('Quality')
    if quality is not None:
        quality = float(quality)
        if quality < 0.5:
            add_issue("Moderate", f"Quality score is {quality:.2f}; harmonic noise balance could be improved.")
        elif quality > 0.7:
            add_strength(f"Quality score of {quality:.2f} indicates a clean harmonic profile.")

    derived = {
        "clipped_window_pct": clip_windows_ratio * 100 if clip_windows_ratio is not None else None,
        "total_clipped_samples": total_clipped_samples,
        "sub_bass_mean": sub_bass_mean,
        "mid_side_mean": mid_side_mean,
        "stereo_corr_mean": stereo_corr_mean,
        "window_dynamic_range_mean": dyn_range_mean,
        "crest_db": crest_db,
        "rms_db": rms_db,
    }

    if integrated_loudness is not None:
        derived['integrated_loudness'] = float(integrated_loudness)

    return {
        "issues": issues,
        "strengths": strengths,
        "derived": derived
    }

def build_llm_prompt(export_data, health_summary):
    """Create a structured prompt string for downstream LLM analysis."""

    file_info = export_data.get("file_info", {})
    baseline = export_data.get("baseline_metrics", {})
    compound = export_data.get("compound_metrics", {})
    time_summary = export_data.get("time_based_metrics", {}).get("summary", {})

    baseline_aQi = export_data.get("baseline_aQi")
    time_aQi = export_data.get("time_aQi")
    final_aQi = export_data.get("final_aQi")

    def fmt_number(value):
        if value is None:
            return "N/A"
        try:
            value = float(value)
        except (TypeError, ValueError):
            return str(value)
        if abs(value) >= 1000 or (abs(value) > 0 and abs(value) < 0.001):
            return f"{value:.3e}"
        if abs(value) >= 100:
            return f"{value:.1f}"
        return f"{value:.3f}"

    prompt_lines = [
        "You are a seasoned mixing and mastering engineer. Review the supplied diagnostics and provide targeted mix guidance.",
        ""
    ]

    channels = file_info.get("channels")
    sample_rate = file_info.get("sample_rate")
    duration = file_info.get("duration_sec")
    mix_snapshot = " - ".join(
        [
            part for part in [
                f"Channels: {channels}" if channels is not None else None,
                f"Sample rate: {sample_rate} Hz" if sample_rate is not None else None,
                f"Duration: {duration:.1f} s" if isinstance(duration, (int, float)) else None,
            ] if part is not None
        ]
    )
    if mix_snapshot:
        prompt_lines.extend(["### Mix Snapshot", f"- {mix_snapshot}", ""])

    if baseline_aQi is not None and time_aQi is not None and final_aQi is not None:
        prompt_lines.append(f"- Baseline aQi: {baseline_aQi:.1f}% | Time-Based aQi: {time_aQi:.1f}% | Final aQi: {final_aQi:.1f}%")
    elif baseline_aQi is not None:
        prompt_lines.append(f"- Baseline aQi: {baseline_aQi:.1f}%")

    issues = health_summary.get("issues", [])
    strengths = health_summary.get("strengths", [])

    prompt_lines.append("")
    prompt_lines.append("### Notable Observations")
    if issues:
        for item in issues:
            severity = item.get("severity", "Info")
            text = item.get("text", "")
            prompt_lines.append(f"- [{severity}] {text}")
    else:
        prompt_lines.append("- No major technical issues were flagged by heuristics; verify with critical listening.")

    if strengths:
        prompt_lines.append("")
        prompt_lines.append("### Strengths to Preserve")
        for text in strengths:
            prompt_lines.append(f"- {text}")

    key_baseline = {
        "RMS (dBFS)": baseline.get("rms_db"),
        "Peak (dBFS)": baseline.get("peak_db"),
        "Crest Factor (dB)": baseline.get("crest_db"),
        "Spectral Centroid": baseline.get("centroid"),
        "Spectral Flatness": baseline.get("flatness"),
        "Low/Mid Ratio": baseline.get("low_mid_ratio"),
        "Mid/High Ratio": baseline.get("mid_high_ratio"),
        "Dynamic Range (raw)": baseline.get("dynamic_range_raw"),
        "Integrated Loudness (LUFS)": baseline.get("integrated_loudness"),
    }

    prompt_lines.append("")
    prompt_lines.append("### Key Baseline Metrics")
    for label, value in key_baseline.items():
        prompt_lines.append(f"- {label}: {fmt_number(value)}")

    prompt_lines.append("")
    prompt_lines.append("### Time-Based Metrics (200 ms windows)")
    for metric_name, stats in time_summary.items():
        mean_val = stats.get("mean")
        median_val = stats.get("median")
        max_val = stats.get("max")
        min_val = stats.get("min")
        prompt_lines.append(
            f"- {metric_name}: mean {fmt_number(mean_val)} | median {fmt_number(median_val)} | min {fmt_number(min_val)} | max {fmt_number(max_val)}"
        )

    if compound:
        prompt_lines.append("")
        prompt_lines.append("### Compound Scores (0–1)")
        for name, value in compound.items():
            prompt_lines.append(f"- {name}: {fmt_number(value)}")

    derived = health_summary.get("derived", {})
    if derived:
        label_map = {
            "clipped_window_pct": "Windows with clipping (%)",
            "total_clipped_samples": "Total clipped samples",
            "sub_bass_mean": "Sub-bass mono mean",
            "mid_side_mean": "Mid/side ratio mean",
            "stereo_corr_mean": "Stereo correlation mean",
            "window_dynamic_range_mean": "Window peak–RMS mean",
            "crest_db": "Crest factor (dB)",
            "rms_db": "RMS (dBFS)",
            "integrated_loudness": "Integrated loudness (LUFS)",
        }
        prompt_lines.append("")
        prompt_lines.append("### Derived Indicators")
        for key, label in label_map.items():
            if key in derived:
                prompt_lines.append(f"- {label}: {fmt_number(derived.get(key))}")

    prompt_lines.append("")
    prompt_lines.append("### Requested LLM Output")
    prompt_lines.append("1. Summarize the mix health, referencing the most relevant metrics above.")
    prompt_lines.append("2. Recommend high-impact mix adjustments (EQ, dynamics, stereo, balance) that address the flagged issues.")
    prompt_lines.append("3. Suggest final loudness and dynamic targets suited for mainstream streaming platforms unless specified otherwise.")
    prompt_lines.append("4. List any additional checks the producer should run before handing the mix to a mastering engineer.")

    return "\n".join(prompt_lines)

####################################
# Main App
####################################
st.title("aQi Deep")

uploaded_file = st.file_uploader("Processing is INTENSE (currently 1 minute per minute of audio) Upload an audio file (WAV, MP3, OGG)", type=["wav", "mp3", "ogg"])
if uploaded_file is not None:
    y, sr = load_audio(uploaded_file)
    n_channels = y.shape[0]
    duration_sec = y.shape[1] / sr
    st.metric("File Info", f"{n_channels} channels • {sr} Hz • {duration_sec:.1f} sec")
    
    # Stage 1: Baseline Metrics
    st.header("Stage 1: Baseline Summary (50%)")
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

    health_summary = analyze_mix_health(baseline, compound, sub_bass, clipping, dyn_range_raw, mid_side, stereo_corr)

    # Export Data & Prompting
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

    export_data["time_aQi"] = time_aQi
    export_data["final_aQi"] = final_aQi
    export_data["health_summary"] = health_summary

    prompt_text = build_llm_prompt(export_data, health_summary)
    export_data["llm_prompt_template"] = prompt_text

    export_json = json.dumps(export_data, default=default_converter, indent=2)

    st.subheader("LLM Prompt Builder")
    col_issues, col_strengths = st.columns(2)
    with col_issues:
        st.markdown("**Flagged Issues**")
        if health_summary["issues"]:
            for item in health_summary["issues"]:
                st.write(f"- [{item['severity']}] {item['text']}")
        else:
            st.write("- No major issues detected by heuristics.")
    with col_strengths:
        st.markdown("**Strengths**")
        if health_summary["strengths"]:
            for text in health_summary["strengths"]:
                st.write(f"- {text}")
        else:
            st.write("- No specific strengths flagged.")

    st.markdown("Copy or download the prompt below to brief your LLM co-pilot:")
    st.code(prompt_text, language="markdown")
    st.download_button(label="Download LLM Prompt", data=prompt_text, file_name="mix_prompt.md", mime="text/markdown")

    st.subheader("Raw Data Exports")
    st.download_button(label="Export Metrics as JSON",
                       data=export_json,
                       file_name="audio_metrics.json",
                       mime="application/json")
    st.download_button(label="Download Spectrogram Image",
                       data=buf_spec_img,
                       file_name="spectrogram.png",
                       mime="image/png")
