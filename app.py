"""
DGA Analyzer — Flask Backend API
Wraps temp_pred.py train_model() and recursive_forecast() for REST consumption.

Deploy this on Railway / Render / any Python host.
The Netlify frontend calls POST /api/analyze.
"""

import os
import sys
import json
import tempfile
import traceback
import numpy as np

import matplotlib
matplotlib.use('Agg')
               
from flask import Flask, request, jsonify
from flask_cors import CORS

# ── Import the model code ──────────────────────────────────────────
# temp_pred.py must sit in the same directory as app.py
sys.path.insert(0, os.path.dirname(__file__))
from temp_pred import (
    load_and_preprocess_data,
    predict_gas_concentration,
    recursive_forecast,
    create_sequences,
    train_bilstm_component,
    classify_sample,
)
import torch

app = Flask(__name__)

# ── CORS: allow your Netlify domain (update after deploy) ──────────
# During development allow all origins; tighten before production.
CORS(app, origins=["*"])

GAS_THRESHOLDS = {
    'H2': 100, 'CH4': 50, 'C2H6': 65,
    'C2H4': 50, 'C2H2': 35, 'CO': 350, 'CO2': 2500,
}


def _to_python(obj):
    """Recursively convert numpy types to native Python for JSON."""
    if isinstance(obj, dict):
        return {k: _to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_python(i) for i in obj]
    if isinstance(obj, np.ndarray):
        return [_to_python(x) for x in obj.tolist()]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


# ── Health check ───────────────────────────────────────────────────
@app.route("/", methods=["GET"])
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "DGA Backend running"}), 200


# ── Main analysis endpoint ─────────────────────────────────────────
@app.route("/api/analyze", methods=["POST"])
def analyze():
    """
    Accepts multipart/form-data:
      file            — CSV file
      use_optimization — '1' or '0'  (BKA)
      gpu             — '1' or '0'
      num_gases       — '' or integer string
      forecast_steps  — integer string (default 60)

    Returns JSON:
      {
        gas_columns: [...],
        metrics:     { GAS: {R2, RMSE, MAE} },
        predictions: { GAS: [float, ...] },       ← BiLSTM test predictions
        ground_truth:{ GAS: [float, ...] },       ← actual test values
        recursive:   { GAS: { predictions:[float,...],
                               time_to_threshold_hours: float|null,
                               ggr_ppm_day: float,
                               immediate_alert: bool } },
        duval: { zone, ch4_ppm, c2h4_ppm, c2h2_ppm,
                 ch4_percent, c2h4_percent, c2h2_percent }
      }
    """
    # ── Validate file ──────────────────────────────────────────────
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    f = request.files["file"]
    if not f.filename.lower().endswith(".csv"):
        return jsonify({"error": "Only CSV files accepted"}), 400

    # ── Parse options ──────────────────────────────────────────────
    use_bka       = request.form.get("use_optimization", "0") == "1"
    use_gpu       = request.form.get("gpu", "1") == "1"
    num_gases_raw = request.form.get("num_gases", "").strip()
    num_gases     = int(num_gases_raw) if num_gases_raw.isdigit() else None
    forecast_steps = int(request.form.get("forecast_steps", "60"))

    device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")

    # ── Save uploaded CSV to a temp file ──────────────────────────
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="wb") as tmp:
        f.save(tmp)
        tmp_path = tmp.name

    try:
        # ── Load & preprocess ──────────────────────────────────────
        df, gas_columns = load_and_preprocess_data(tmp_path)
        if num_gases:
            gas_columns = gas_columns[:num_gases]

        metrics_out      = {}
        predictions_out  = {}
        ground_truth_out = {}
        recursive_out    = {}

        # ── Train BiLSTM per gas ───────────────────────────────────
        for gas in gas_columns:
            gas_data = df[gas].values
            try:
                preds, met, _ = predict_gas_concentration(
                    gas_data, window=6, device=device, use_bka=use_bka
                )
                # Ground truth for test split
                X, y       = create_sequences(gas_data, window=6)
                split_idx  = int(len(X) * 0.8)
                y_test     = y[split_idx: split_idx + len(preds)]

                metrics_out[gas]      = {
                    "R2":   float(met.get("R2",   0)),
                    "RMSE": float(met.get("RMSE", 0)),
                    "MAE":  float(met.get("MAE",  0)),
                }
                predictions_out[gas]  = [float(x) for x in preds]
                ground_truth_out[gas] = [float(x) for x in y_test]

            except Exception as e:
                print(f"[WARN] Gas {gas} training failed: {e}")
                metrics_out[gas]      = {"R2": 0, "RMSE": 0, "MAE": 0, "error": str(e)}
                predictions_out[gas]  = []
                ground_truth_out[gas] = []

        # ── Recursive multi-step forecast per gas ─────────────────
        for gas in gas_columns:
            gas_col = next((c for c in df.columns if c.lower() == gas.lower()), None)
            if not gas_col:
                continue
            series = df[gas_col].values.astype(float)
            window = 6
            try:
                X, y      = create_sequences(series, window)
                split_idx = int(len(X) * 0.8)
                model, scaler_X, scaler_y, _, _ = train_bilstm_component(
                    X[:split_idx], y[:split_idx],
                    X[split_idx:], y[split_idx:],
                    learning_rate=0.001, epochs=100,
                    hidden_size1=64, hidden_size2=64,
                    device=device, verbose=False, use_huber_loss=True,
                )
                init_window     = series[-window:]
                thr_val         = GAS_THRESHOLDS.get(gas.upper())
                threshold_dict  = {gas.lower(): thr_val} if thr_val else {}

                rec = recursive_forecast(
                    model=model, scaler_X=scaler_X, scaler_y=scaler_y,
                    init_window=init_window, steps=forecast_steps,
                    time_step_hours=1.0, threshold_ppm=threshold_dict,
                    ggr_threshold=2.0, history=series, device=device,
                )

                # time_to_threshold_hours is a dict keyed by gas.lower()
                ttt = rec["time_to_threshold_hours"]
                ttt_val = ttt.get(gas.lower()) if isinstance(ttt, dict) else ttt

                recursive_out[gas] = {
                    "predictions":             [float(x) for x in rec["predictions"]],
                    "time_to_threshold_hours": float(ttt_val) if ttt_val is not None else None,
                    "ggr_ppm_day":             float(rec.get("ggr_ppm_day", 0)),
                    "immediate_alert":         bool(rec.get("immediate_alert", False)),
                }
            except Exception as e:
                print(f"[WARN] Recursive forecast failed for {gas}: {e}")
                recursive_out[gas] = {
                    "predictions": [], "time_to_threshold_hours": None,
                    "ggr_ppm_day": 0, "immediate_alert": False,
                }

        # ── Duval Triangle classification ──────────────────────────
        duval_out = {}
        try:
            def _median_preds(g):
                p = predictions_out.get(g, predictions_out.get(g.lower(), []))
                return float(np.median(p[-10:])) if len(p) >= 1 else 0.0

            ch4_v  = _median_preds("CH4")
            c2h4_v = _median_preds("C2H4")
            c2h2_v = _median_preds("C2H2")

            if ch4_v > 0 or c2h4_v > 0 or c2h2_v > 0:
                zone, ch4_pct, c2h4_pct, c2h2_pct = classify_sample(ch4_v, c2h4_v, c2h2_v)
                duval_out = {
                    "zone":         zone,
                    "ch4_ppm":      ch4_v,
                    "c2h4_ppm":     c2h4_v,
                    "c2h2_ppm":     c2h2_v,
                    "ch4_percent":  ch4_pct,
                    "c2h4_percent": c2h4_pct,
                    "c2h2_percent": c2h2_pct,
                }
        except Exception as e:
            print(f"[WARN] Duval analysis failed: {e}")

        # ── Build response ─────────────────────────────────────────
        response = {
            "gas_columns":  gas_columns,
            "metrics":      metrics_out,
            "predictions":  predictions_out,
            "ground_truth": ground_truth_out,
            "recursive":    recursive_out,
            "duval":        duval_out,
        }
        return jsonify(_to_python(response)), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

    finally:
        # Clean up temp file
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
