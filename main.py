# ── main.py ──────────────────────────────────────────────────────────────────
"""
Entry point. Run:  python main.py

What it does
────────────
1. Downloads stock data for each ticker via yfinance
2. Engineers features (SMA, RSI), scales, splits 80/20
3. Trains RFR, ANN, LSTM
4. Saves all 3 models to disk  ← NEW (Streamlit uses these)
5. Evaluates on test set (MAE, RMSE, MAPE)
6. Saves all plots to outputs/
7. Saves loss curves for ANN and LSTM  ← NEW
8. Saves feature importance plot for RFR  ← NEW
9. Saves metrics.csv summary
"""
import random
import numpy as np
import tensorflow as tf

# Lock all random seeds
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

import logging
import os
import sys
import time

import config as cfg
from data_loader import load_ticker, prepare_data, make_sequences
from models      import train_rfr, train_ann, train_lstm, save_models
from evaluate    import (
    metrics, print_table,
    plot_single, plot_all, plot_residuals,
    plot_loss_curves, plot_feature_importance,
    plot_rmse_bar, plot_all_companies_comparison,
    save_csv, save_rfr_gridsearch_params,
)

# ─── Logging setup ───────────────────────────────────────────────────────────
# Logs go to both terminal and a log file simultaneously.
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(cfg.OUTPUT_DIR, "run.log"), mode="w"),
    ],
)
log = logging.getLogger(__name__)


# ─── Pipeline for one ticker ─────────────────────────────────────────────────

def run_ticker(ticker: str) -> dict:
    log.info("-" * 55)
    log.info("  Processing: %s", ticker)
    log.info("-" * 55)

    # Step 1 — Download + feature engineering
    df = load_ticker(ticker, cfg.START_DATE, cfg.END_DATE)

    # Step 2 — Scale + split (no data leakage)
    X_tr, X_te, y_tr, y_te, x_scaler, y_scaler = prepare_data(df)

    # Step 3 — Build LSTM sequences (60-day windows)
    X_tr_seq, y_tr_seq = make_sequences(X_tr, y_tr, cfg.SEQUENCE_LEN)
    X_te_seq, y_te_seq = make_sequences(X_te, y_te, cfg.SEQUENCE_LEN)

    # Step 4 — Train all three models
    rfr             = train_rfr(X_tr, y_tr)
    ann,  ann_hist  = train_ann(X_tr, y_tr)
    lstm, lstm_hist = train_lstm(X_tr_seq, y_tr_seq)

    # Step 5 — Save models to disk for Streamlit
    save_models(rfr, ann, lstm, ticker)

    # Step 6 — Predict
    # LSTM uses seq windows so its test set is shorter by SEQUENCE_LEN rows.
    # We align RFR and ANN to the same shorter window for a fair comparison.
    offset = cfg.SEQUENCE_LEN

    rfr_scaled  = rfr.predict(X_te[offset:])
    ann_scaled  = ann.predict(X_te[offset:], verbose=0).flatten()
    lstm_scaled = lstm.predict(X_te_seq, verbose=0).flatten()

    # Step 7 — Inverse transform: convert 0-1 scaled values back to real prices
    def inv(arr):
        return y_scaler.inverse_transform(arr.reshape(-1, 1)).flatten()

    actual    = inv(y_te[offset:])
    rfr_pred  = inv(rfr_scaled)
    ann_pred  = inv(ann_scaled)
    lstm_pred = inv(lstm_scaled)

    preds = {"ANN": ann_pred, "RFR": rfr_pred, "LSTM": lstm_pred}

    # Save predictions to disk for Streamlit to load instantly
    import pandas as pd
    pred_df = pd.DataFrame({
        "Actual": actual,
        "ANN": ann_pred,
        "RFR": rfr_pred,
        "LSTM": lstm_pred
    })
    pred_csv_path = os.path.join(cfg.OUTPUT_DIR, f"{ticker}_predictions.csv")
    pred_df.to_csv(pred_csv_path, index=False)
    log.info("Saved predictions -> %s", pred_csv_path)

    # Step 8 — Metrics
    results = {name: metrics(actual, pred) for name, pred in preds.items()}
    print_table(ticker, results)

    # Step 9 — All plots
    for name, pred in preds.items():
        plot_single(actual, pred, name, ticker)

    plot_all(actual, preds, ticker)
    plot_residuals(actual, preds, ticker)

    # Loss curves (shows training vs validation loss over epochs)
    plot_loss_curves(ann_hist,  "ANN",  ticker)
    plot_loss_curves(lstm_hist, "LSTM", ticker)

    # Feature importance (which input mattered most for RFR)
    plot_feature_importance(rfr, ticker, rfr.best_params_)

    return results, rfr.best_params_


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    all_results = {}
    all_best_params = {}

    for ticker in cfg.TICKERS:
        try:
            results, best_params = run_ticker(ticker)
            all_results[ticker] = results
            all_best_params[ticker] = best_params
            time.sleep(3)  # Wait 3 seconds to avoid rate limiting
        except FileNotFoundError as exc:
            log.error("Skipping %s — %s", ticker, exc)
        except Exception as exc:
            log.error("Failed on %s — %s", ticker, exc, exc_info=True)

    if all_results:
        save_csv(all_results, all_best_params)
        save_rfr_gridsearch_params(all_best_params)
        plot_rmse_bar(all_results)
        plot_all_companies_comparison(all_results)

    log.info("All done. Check the outputs/ folder for plots and metrics.csv")


if __name__ == "__main__":
    main()
