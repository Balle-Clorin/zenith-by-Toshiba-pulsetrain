"""
TOSH-9001R Pulse Train L/R Channel Timing (Zenith / HTA) Analyzer -- Streamlit web app
--------------------------------------------------------------------------------------
Web-app version of tosh9001r_pulse_azimuth_analyzer.py. Same DSP core:
pulse detection, FFT upsampling for sub-sample L/R peak timing, MAD outlier
rejection, burst splitting, along-groove offset (um) vs radius, burst trend fits.

Note: this is a zenith/HTA test, not an azimuth test.
"""

import gc
import io
from pathlib import Path

import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import find_peaks, resample

# ----------------------------------------------------------------
# Script identity (shown in header of the result page)
# ----------------------------------------------------------------
SCRIPT_NAME = "tosh9001r_pulse_zenith_analyzer"  # display name (file on GitHub is app.py)
SCRIPT_REV = "REV2"

# ----------------------------------------------------------------
# Defaults (adjustable in sidebar)
# ----------------------------------------------------------------
DEF_PEAK_THRESHOLD_FRAC = 0.3
DEF_MIN_PEAK_DISTANCE = 50
DEF_HALF_WINDOW = 60
DEF_UPSAMPLE = 50
DEF_BURST_GAP = 5000
DEF_MAD_MULT = 6.0
DEF_RADIUS_START = 125.0
DEF_RADIUS_END = 119.5
DEF_RPM = 33.33
DEF_DESCRIPTION = "JVC QL-Y5F SHURE V15-IV SASB VTA NO SHIM PH-8, TOSH-9001R pulse zenith/HTA test"

BASE_FONT = 13
PLOT_HALF_WINDOW_US = 500   # single-pulse plot: +/- this many us around window centre (display only)


# ----------------------------------------------------------------
# DSP core
# ----------------------------------------------------------------
def load_audio_bytes(raw):
    sr, data = wavfile.read(io.BytesIO(raw))
    if data.ndim == 1:
        raise ValueError("File is mono -- this test requires a stereo L/R recording.")
    data = data.astype(np.float64)
    return sr, data[:, 0].copy(), data[:, 1].copy()


def detect_pulses(L, R, thr_frac, min_dist):
    env = np.abs(L) + np.abs(R)
    thresh = thr_frac * np.max(env)
    peaks, _ = find_peaks(env, height=thresh, distance=min_dist)
    return peaks


def measure_offsets(L, R, peaks, sr, half_win, up):
    dt_list, valid = [], []
    for p in peaks:
        lo, hi = p - half_win, p + half_win
        if lo < 0 or hi >= len(L):
            continue
        segL, segR = L[lo:hi], R[lo:hi]
        upL = resample(segL, len(segL) * up)
        upR = resample(segR, len(segR) * up)
        iL = np.argmax(np.abs(upL))
        iR = np.argmax(np.abs(upR))
        dt_list.append((iR - iL) / up / sr * 1e6)
        valid.append(p)
    return np.array(valid), np.array(dt_list)


def reject_outliers(peaks, dt_us, mad_mult):
    median = np.median(dt_us)
    mad = np.median(np.abs(dt_us - median)) + 1e-12
    keep = np.abs(dt_us - median) < mad_mult * mad
    return peaks[keep], dt_us[keep]


def split_into_bursts(peaks, dt_us, gap):
    if len(peaks) == 0:
        return []
    gap_idx = np.where(np.diff(peaks) > gap)[0]
    bounds = [0] + list(gap_idx + 1) + [len(peaks)]
    return [(peaks[bounds[i]:bounds[i + 1]], dt_us[bounds[i]:bounds[i + 1]])
            for i in range(len(bounds) - 1)]


def radius_at_time(t, t0, t1, r0, r1):
    frac = np.clip((t - t0) / (t1 - t0), 0.0, 1.0)
    return r0 + frac * (r1 - r0)


def groove_velocity_mps(radius_mm, rpm):
    return radius_mm / 1000.0 * (rpm / 60.0 * 2 * np.pi)


def compute_track_offsets_um(peaks, dt_us, sr, r0, r1, rpm):
    t = peaks / sr
    radius_mm = radius_at_time(t, t[0], t[-1], r0, r1)
    vel = groove_velocity_mps(radius_mm, rpm)
    offset_um = vel * dt_us * 1e-6 * 1e6
    return radius_mm, vel, offset_um


def compute_burst_trend(bursts, radius_mm, offset_um, sr):
    bt, br, bdt, bof = [], [], [], []
    idx = 0
    for pk, dt in bursts:
        n = len(dt)
        if n == 0:
            continue
        bt.append(np.mean(pk / sr))
        br.append(np.mean(radius_mm[idx:idx + n]))
        bdt.append(np.mean(dt))
        bof.append(np.mean(offset_um[idx:idx + n]))
        idx += n
    bt, br, bdt, bof = map(np.array, (bt, br, bdt, bof))
    dt_fit = np.polyfit(bt, bdt, 1) if len(bt) >= 2 else np.array([np.nan, np.nan])
    of_fit = np.polyfit(br, bof, 1) if len(br) >= 2 else np.array([np.nan, np.nan])
    return dict(t_mean=bt, radius_mean=br, dt_mean=bdt, offset_mean=bof,
                dt_fit=dt_fit, offset_fit=of_fit)


# ----------------------------------------------------------------
# Plot
# ----------------------------------------------------------------
def make_figure(L, R, sr, peaks, dt_us, bursts, radius_mm, offset_um, trend,
                half_win, up, r0, r1, rpm, title_lines):
    plt.rcParams.update({"font.size": BASE_FONT})
    plt.style.use("dark_background")
    fig, axes = plt.subplots(5, 1, figsize=(13, 19), facecolor="#1e1e1e")
    for a in axes:
        a.set_facecolor("#1e1e1e")
    fig.suptitle(title_lines, fontsize=BASE_FONT, color="white")

    t = np.arange(len(L)) / sr
    ax = axes[0]
    step = max(1, len(L) // 400000)  # decimate for plotting only
    ax.plot(t[::step], L[::step], color="#00e5ff", linewidth=0.4, label="L")
    ax.plot(t[::step], R[::step], color="#ff5252", linewidth=0.4, label="R", alpha=0.7)
    ax.plot(peaks / sr, np.zeros_like(peaks), "|", color="yellow", markersize=8, label="detected pulses")
    ax.set_title("Full waveform overview")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("amplitude")
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[1]
    ex = np.argmin(np.abs(dt_us - np.median(dt_us)))
    p = peaks[ex]
    segL, segR = L[p - half_win:p + half_win], R[p - half_win:p + half_win]
    upL, upR = resample(segL, len(segL) * up), resample(segR, len(segR) * up)
    tt = (np.arange(len(upL)) - len(upL) / 2) / (sr * up) * 1e6
    iL, iR = np.argmax(np.abs(upL)), np.argmax(np.abs(upR))
    ax.plot(tt, upL, color="#00e5ff", linewidth=1.0, label="L (upsampled)")
    ax.plot(tt, upR, color="#ff5252", linewidth=1.0, label="R (upsampled)")
    ax.axvline(tt[iL], color="#00e5ff", linestyle="--", linewidth=0.8)
    ax.axvline(tt[iR], color="#ff5252", linestyle="--", linewidth=0.8)
    ax.set_title(f"Example single pulse, upsampled {up}x  "
                 f"(this pulse: dt = {(iR - iL) / up / sr * 1e6:+.3f} us)")
    ax.set_xlabel("time relative to window center (us)")
    ax.set_ylabel("amplitude")
    ax.set_xlim(-PLOT_HALF_WINDOW_US, PLOT_HALF_WINDOW_US)
    in_view = np.abs(tt) <= PLOT_HALF_WINDOW_US
    if np.any(in_view):
        ymax = max(np.max(np.abs(upL[in_view])), np.max(np.abs(upR[in_view]))) * 1.08
        ax.set_ylim(-ymax, ymax)
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[2]
    colors = plt.cm.plasma(np.linspace(0.15, 0.9, max(len(bursts), 1)))
    for i, (pk, dt) in enumerate(bursts):
        if len(dt):
            ax.plot(pk / sr, dt, ".", color=colors[i], markersize=3, label=f"burst {i + 1}")
    ax.axhline(np.median(dt_us), color="white", linestyle="--", linewidth=0.8, label="overall median")
    ax.plot(trend["t_mean"], trend["dt_mean"], "D", color="white", markersize=7,
            markeredgecolor="black", label="burst average", zorder=5)
    for tx, dy in zip(trend["t_mean"], trend["dt_mean"]):
        ax.annotate(f"{dy:+.2f}", (tx, dy), xytext=(0, 12), textcoords="offset points",
                    ha="center", fontsize=14, fontweight="bold", color="white", zorder=6)
    if len(trend["t_mean"]) >= 2:
        tl = np.array([trend["t_mean"].min(), trend["t_mean"].max()])
        ax.plot(tl, np.polyval(trend["dt_fit"], tl), "-", color="cyan", linewidth=1.5,
                label=f"trend: {trend['dt_fit'][0]:+.3f} us/s")
    ax.set_title("Per-pulse L/R timing offset across the recording")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("dt = t_R - t_L (us)")
    ax.legend(loc="upper right", fontsize=7, ncol=2)

    ax = axes[3]
    ax.hist(dt_us, bins=40, color="#69f0ae", edgecolor="black", linewidth=0.3)
    ax.axvline(np.median(dt_us), color="yellow", linestyle="--", linewidth=1.2,
               label=f"median = {np.median(dt_us):+.3f} us")
    ax.set_title("Distribution of per-pulse timing offsets")
    ax.set_xlabel("dt = t_R - t_L (us)")
    ax.set_ylabel("count")
    leg = ax.legend(loc="upper right", fontsize=20)
    for tx in leg.get_texts():
        tx.set_fontweight("bold")
        tx.set_color("white")

    ax = axes[4]
    ax.plot(radius_mm, offset_um, ".", color="#ffab40", markersize=3)
    ax.axhline(np.median(offset_um), color="white", linestyle="--", linewidth=0.8,
               label=f"median = {np.median(offset_um):+.3f} um")
    ax.plot(trend["radius_mean"], trend["offset_mean"], "D", color="white", markersize=7,
            markeredgecolor="black", label="burst average", zorder=5)
    for rx, oy in zip(trend["radius_mean"], trend["offset_mean"]):
        ax.annotate(f"{oy:+.2f}", (rx, oy), xytext=(0, 12), textcoords="offset points",
                    ha="center", fontsize=14, fontweight="bold", color="white", zorder=6)
    if len(trend["radius_mean"]) >= 2:
        rl = np.array([trend["radius_mean"].min(), trend["radius_mean"].max()])
        ax.plot(rl, np.polyval(trend["offset_fit"], rl), "-", color="cyan", linewidth=1.5,
                label=f"trend: {trend['offset_fit'][0]:+.4f} um/mm")
    ax.invert_xaxis()
    ax.set_title(f"Along-groove L/R contact-point offset vs. groove radius "
                 f"({r0:.1f}-{r1:.1f} mm, {rpm:.2f} RPM)")
    ax.set_xlabel("groove radius (mm)")
    ax.set_ylabel("offset (um)")
    ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


# ----------------------------------------------------------------
# UI
# ----------------------------------------------------------------
st.set_page_config(page_title="TOSH-9001R Zenith/HTA Analyzer", layout="wide")

st.title("TOSH-9001R Pulse Train L/R Timing (Zenith / HTA) Analyzer")
st.caption(f"Script: **{SCRIPT_NAME}**  |  Revision: **{SCRIPT_REV}**")

with st.sidebar:
    st.header("Input")
    uploaded = st.file_uploader("Stereo WAV recording", type=["wav"])
    description = st.text_area("User description", value=DEF_DESCRIPTION, height=100)

    st.header("Groove model")
    r_start = st.number_input("Radius at first pulse (mm)", value=DEF_RADIUS_START, step=0.1)
    r_end = st.number_input("Radius at last pulse (mm)", value=DEF_RADIUS_END, step=0.1)
    rpm = st.number_input("Turntable RPM", value=DEF_RPM, step=0.01, format="%.2f")

    with st.expander("Detection / DSP settings"):
        thr = st.slider("Peak threshold (fraction of max envelope)", 0.05, 0.9, DEF_PEAK_THRESHOLD_FRAC, 0.05)
        min_dist = st.number_input("Min pulse spacing (samples)", value=DEF_MIN_PEAK_DISTANCE, min_value=1, step=1)
        half_win = st.number_input("Half window (samples)", value=DEF_HALF_WINDOW, min_value=8, step=1)
        up = st.number_input("Upsample factor", value=DEF_UPSAMPLE, min_value=1, step=1)
        gap = st.number_input("Burst gap (samples)", value=DEF_BURST_GAP, min_value=1, step=100)
        mad_mult = st.number_input("Outlier MAD multiplier", value=DEF_MAD_MULT, min_value=1.0, step=0.5)

    run = st.button("Analyze", type="primary", disabled=uploaded is None)

if uploaded is None:
    st.info("Upload a stereo WAV of the TOSH-9001R pulse track in the sidebar, then press Analyze.")
    st.stop()

if run:
    try:
        raw = uploaded.getvalue()
        sr, L, R = load_audio_bytes(raw)
        del raw

        with st.spinner("Detecting pulses and measuring L/R offsets..."):
            peaks = detect_pulses(L, R, thr, int(min_dist))
            if len(peaks) < 3:
                st.error("Too few pulses detected -- lower the peak threshold.")
                st.stop()
            peaks, dt_us = measure_offsets(L, R, peaks, sr, int(half_win), int(up))
            if len(peaks) < 3:
                st.error("Too few valid pulses after windowing.")
                st.stop()
            peaks, dt_us = reject_outliers(peaks, dt_us, mad_mult)
            bursts = split_into_bursts(peaks, dt_us, int(gap))
            radius_mm, vel, offset_um = compute_track_offsets_um(peaks, dt_us, sr, r_start, r_end, rpm)
            trend = compute_burst_trend(bursts, radius_mm, offset_um, sr)

        # ---- Result page header ----
        st.divider()
        st.subheader("Results")
        h1, h2 = st.columns(2)
        h1.markdown(f"**Script:** {SCRIPT_NAME}  \n**Revision:** {SCRIPT_REV}")
        h2.markdown(f"**File:** {uploaded.name}  \n**Description:** {description}")

        m = st.columns(5)
        m[0].metric("Sample rate", f"{sr} Hz")
        m[1].metric("Pulses analyzed", f"{len(dt_us)}")
        m[2].metric("Median dt (R-L)", f"{np.median(dt_us):+.3f} us")
        m[3].metric("Median offset", f"{np.median(offset_um):+.3f} um")
        m[4].metric("Std dev dt", f"{np.std(dt_us):.3f} us")
        st.caption("R lags L (R arrives later)" if np.median(dt_us) > 0
                   else "R leads L (R arrives earlier)")

        # ---- Burst table ----
        rows, idx = [], 0
        for i, (pk, dt) in enumerate(bursts, start=1):
            n = len(dt)
            if n == 0:
                continue
            seg_r, seg_um = radius_mm[idx:idx + n], offset_um[idx:idx + n]
            idx += n
            rows.append({
                "Burst": i, "n": n, "Radius (mm)": round(float(np.mean(seg_r)), 2),
                "dt mean (us)": round(float(np.mean(dt)), 3),
                "dt median (us)": round(float(np.median(dt)), 3),
                "dt std (us)": round(float(np.std(dt)), 3),
                "offset mean (um)": round(float(np.mean(seg_um)), 3),
                "offset std (um)": round(float(np.std(seg_um)), 3),
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)

        if len(trend["t_mean"]) >= 2:
            st.write(f"Trend across burst averages: dt vs time **{trend['dt_fit'][0]:+.4f} us/s**, "
                     f"offset vs radius **{trend['offset_fit'][0]:+.4f} um/mm**")

        # ---- Figure ----
        title = (f"TOSH-9001R Pulse Zenith/HTA Timing Analysis  |  {SCRIPT_NAME}  {SCRIPT_REV}\n"
                 f"{description}\n{uploaded.name}")
        fig = make_figure(L, R, sr, peaks, dt_us, bursts, radius_mm, offset_um, trend,
                          int(half_win), int(up), r_start, r_end, rpm, title)
        st.pyplot(fig)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, facecolor=fig.get_facecolor())
        st.download_button("Download figure (PNG)", buf.getvalue(),
                           file_name=f"{Path(uploaded.name).stem}_zenith_hta_timing.png",
                           mime="image/png")

        csv = "pulse_sample,radius_mm,dt_us,offset_um\n" + "\n".join(
            f"{p},{r:.4f},{d:.4f},{o:.4f}" for p, r, d, o in zip(peaks, radius_mm, dt_us, offset_um))
        st.download_button("Download per-pulse data (CSV)", csv, file_name="pulse_offsets.csv",
                           mime="text/csv")

        plt.close("all")
        del L, R, fig, buf
        gc.collect()

    except Exception as e:
        st.error(f"Analysis failed: {e}")
else:
    st.info("Press **Analyze** in the sidebar.")
