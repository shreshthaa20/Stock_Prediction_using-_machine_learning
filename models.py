# ── models.py ────────────────────────────────────────────────────────────────
"""
Three models: Random Forest, ANN, LSTM.

Key points
──────────
• EarlyStopping — training stops automatically when validation loss stops
  improving. No more guessing how many epochs to run.
• save_models()  — saves all 3 trained models to disk after training.
• load_models()  — loads them back instantly. Streamlit uses this so the
  app does not retrain every time someone opens the browser.
"""

import logging
import os

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM
from tensorflow.keras.models import Sequential, load_model

import config as cfg

log = logging.getLogger(__name__)


# ─── Callbacks ───────────────────────────────────────────────────────────────

def _callbacks(patience: int):
    """
    EarlyStopping   — stops training if val_loss does not improve for
                      `patience` epochs and restores the best weights.
    ReduceLROnPlateau — halves the learning rate if val_loss stalls,
                        helping the model escape flat regions.
    """
    return [
        EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=0,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=patience // 2,
            min_lr=1e-6,
            verbose=0,
        ),
    ]


# ─── Random Forest ───────────────────────────────────────────────────────────

def train_rfr(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestRegressor:
    """
    Trains a Random Forest Regressor with hyperparameter tuning.

    GridSearchCV searches over n_estimators, max_depth, min_samples_split,
    min_samples_leaf, and max_features to find the best model.
    """
    log.info("Tuning Random Forest with GridSearchCV ...")

    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": [None, "sqrt", "log2"],
    }

    base_model = RandomForestRegressor(
        random_state=cfg.RF_RANDOM_STATE,
        n_jobs=-1,
    )

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring="neg_mean_squared_error",
        cv=3,
        n_jobs=-1,
        verbose=1,
        refit=True,
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_model.best_params_ = grid_search.best_params_
    log.info("Random Forest tuned. Best params: %s", best_model.best_params_)
    return best_model


# ─── ANN ─────────────────────────────────────────────────────────────────────

def train_ann(X_train: np.ndarray, y_train: np.ndarray):
    """
    Trains a feed-forward Artificial Neural Network.

    Architecture
    ────────────
    Input(7 features)
        → Dense(64, ReLU)   ← learns patterns from raw features
        → Dense(64, ReLU)   ← learns higher-level combinations
        → Dense(1)          ← outputs the predicted price

    ReLU activation: if value < 0 → output 0, else keep value.
    This lets the network learn non-linear relationships.

    Returns the trained model AND training history (for loss curve plots).
    """
    log.info("Training ANN ...")
    n_feat = X_train.shape[1]

    model = Sequential([
        Input(shape=(n_feat,)),
        Dense(cfg.ANN_UNITS, activation="relu"),
        Dense(cfg.ANN_UNITS, activation="relu"),
        Dense(1),
    ], name="ANN")

    model.compile(optimizer="adam", loss="mse")

    history = model.fit(
        X_train, y_train,
        epochs=cfg.ANN_EPOCHS,
        batch_size=cfg.ANN_BATCH,
        validation_split=cfg.ANN_VAL_SPLIT,
        callbacks=_callbacks(cfg.ANN_PATIENCE),
        verbose=0,
    )
    log.info("ANN trained for %d epochs.", len(history.history["loss"]))
    return model, history


# ─── LSTM ────────────────────────────────────────────────────────────────────

def train_lstm(X_train_seq: np.ndarray, y_train_seq: np.ndarray):
    """
    Trains an LSTM (Long Short-Term Memory) network.

    Architecture
    ────────────
    Input(60 days, 7 features)
        → LSTM(50, tanh)   ← 50 memory cells, each remembers patterns
                              across the 60-day sequence
        → Dropout(0.2)     ← randomly turns off 20% of neurons during
                              training to prevent overfitting
        → Dense(1)         ← outputs predicted price

    tanh activation keeps values between -1 and 1, which is mathematically
    correct for LSTM internal gates to work properly.

    Returns the trained model AND training history (for loss curve plots).
    """
    log.info("Training LSTM ...")
    _, seq_len, n_feat = X_train_seq.shape

    model = Sequential([
        Input(shape=(seq_len, n_feat)),
        LSTM(cfg.LSTM_UNITS, activation="tanh"),
        Dropout(cfg.LSTM_DROPOUT),
        Dense(1),
    ], name="LSTM")

    model.compile(optimizer="adam", loss="mse")

    history = model.fit(
        X_train_seq, y_train_seq,
        epochs=cfg.LSTM_EPOCHS,
        batch_size=cfg.LSTM_BATCH,
        validation_split=cfg.LSTM_VAL_SPLIT,
        callbacks=_callbacks(cfg.LSTM_PATIENCE),
        verbose=0,
    )
    log.info("LSTM trained for %d epochs.", len(history.history["loss"]))
    return model, history


# ─── Save & Load ─────────────────────────────────────────────────────────────

def save_models(rfr, ann, lstm, ticker: str) -> None:
    """
    Saves all 3 trained models to disk inside saved_models/<ticker>/.

    Why save?
    ─────────
    Training takes 5-10 minutes. If we save after training once,
    Streamlit can load the models in under 1 second instead of
    retraining every time someone opens the app.

    File formats
    ────────────
    RFR  → .pkl  (joblib — standard format for sklearn models)
    ANN  → .keras (TensorFlow's native format)
    LSTM → .keras (TensorFlow's native format)
    """
    folder = os.path.join("saved_models", ticker)
    os.makedirs(folder, exist_ok=True)

    joblib.dump(rfr, os.path.join(folder, "rfr.pkl"))
    ann.save(os.path.join(folder, "ann.keras"))
    lstm.save(os.path.join(folder, "lstm.keras"))

    log.info("Models saved -> %s/", folder)


def load_models(ticker: str):
    """
    Loads all 3 saved models for a given ticker from disk.

    Raises FileNotFoundError if models have not been trained and saved yet.
    Run main.py first to train and save before using the Streamlit app.
    """
    folder = os.path.join("saved_models", ticker)

    rfr_path  = os.path.join(folder, "rfr.pkl")
    ann_path  = os.path.join(folder, "ann.keras")
    lstm_path = os.path.join(folder, "lstm.keras")

    if not all(os.path.exists(p) for p in [rfr_path, ann_path, lstm_path]):
        raise FileNotFoundError(
            f"No saved models found for {ticker}. "
            f"Please run main.py first to train and save the models."
        )

    rfr  = joblib.load(rfr_path)
    ann  = load_model(ann_path)
    lstm = load_model(lstm_path)

    log.info("Models loaded for %s.", ticker)
    return rfr, ann, lstm