# TOSH-9001R Pulse Train L/R Timing (Zenith / HTA) Analyzer

Streamlit web app for measuring L/R channel timing offsets from a recording of the
Toshiba TOSH-9001R pulse (click) train test record.

**Script:** `tosh9001r_pulse_zenith_analyzer` (deployed as `app.py`) | **Revision:** REV1

## What it does

The pulse train is cut mono into both groove walls. With perfect stylus zenith angle
(horizontal tracking angle, HTA), the L and R pulses arrive at the same instant. Any
zenith/HTA error shifts one channel's contact point along the groove, which shows up
as a small, repeatable L/R timing offset.

This is a **zenith/HTA** test, not an azimuth test. Azimuth mainly affects crosstalk
and channel balance, not L/R timing.

### Method

1. Load a stereo WAV.
2. Detect individual pulses on the combined envelope (|L| + |R|).
3. Cut a window around each pulse and FFT-upsample it (default 50x) to find the
   sub-sample peak time in L and R.
4. dt = t_R - t_L in microseconds. Positive: R lags L. Negative: R leads L.
5. Reject outliers with a robust MAD filter.
6. Split pulses into bursts at silent gaps and report per-burst statistics.
7. Convert dt to along-groove distance (um) using groove velocity at the pulse's
   radius (radius modelled as linear in time between first and last pulse).
8. Fit linear trends across burst averages (dt vs time, offset vs radius).

### Output

- Script name and revision, WAV file name and your description in the result header
- Summary metrics and per-burst table
- Five-panel figure (waveform overview, example upsampled pulse, per-pulse dt over time,
  histogram, along-groove offset vs radius)
- Downloads: figure (PNG) and per-pulse data (CSV)

## Notes

- The result is a timing offset, not an angle in degrees. Use the sign and
  repeatability of dt as feedback while adjusting zenith/HTA: drive dt toward zero and
  minimise its spread.
- No RIAA equalisation is needed; pulse arrival timing is unaffected by it.

## Run locally

```
pip install -r requirements.txt
streamlit run app.py
```

Upload a stereo WAV in the sidebar, adjust the description, groove model (radii, RPM)
and detection settings if needed, then press **Analyze**.

## License

PolyForm Noncommercial License 1.0.0 - see [LICENSE](LICENSE).
