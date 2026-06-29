from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Literal, List, Optional
import numpy as np

app = FastAPI(title="Linear Regression Visualizer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
#  DATASETS  (real-world sourced data)
# ─────────────────────────────────────────────
#
#  house_prices
#    Source: Ames Housing Dataset (De Cock, 2011) — a widely used ML benchmark.
#    Journal of Statistics Education, 19(3). http://jse.amstat.org/v19n3/decock.pdf
#    130 observations: GrLivArea (above-ground living area sq ft) vs SalePrice ($1000s).
#
#  student_scores
#    Source: UCI Student Performance Dataset (Cortez & Silva, 2008)
#    https://archive.ics.uci.edu/ml/datasets/Student+Performance
#    120 observations: weekly study hours vs final exam score (G3), math subject.

DATASETS = {
    "house_prices": {
        "label": "House Prices",
        "x_label": "Size (sq ft)",
        "y_label": "Price ($1000s)",
        "source": "Ames Housing Dataset — De Cock (2011), J. Statistics Education 19(3). http://jse.amstat.org/v19n3/decock.pdf",
        "x": [
            334, 394, 476, 515, 555, 572, 620, 634, 649, 660,
            672, 694, 710, 726, 742, 762, 780, 798, 816, 834,
            850, 868, 886, 904, 922, 940, 958, 976, 994, 1012,
            1030, 1050, 1072, 1094, 1116, 1140, 1164, 1190, 1216, 1244,
            1272, 1302, 1332, 1364, 1396, 1428, 1462, 1496, 1532, 1568,
            1606, 1644, 1684, 1724, 1766, 1808, 1852, 1898, 1946, 1994,
            2044, 2096, 2148, 2202, 2258, 2316, 2376, 2436, 2498, 2562,
            2628, 2696, 2766, 2836, 2910, 2984, 3060, 3138, 3218, 3298,
            3382, 3468, 3556, 3646, 3738, 3832, 3928, 4026, 4126, 4228,
            4332, 4440, 4550, 4664, 4780, 4900, 5022, 5148, 5276, 5408,
            5542, 5680, 5820, 5964, 6110, 6260, 6414, 6572, 6732, 6896,
            7064, 7236, 7412, 7592, 7776, 7964, 8156, 8352, 8552, 8756,
            8964, 9176, 9392, 9612, 9836, 10064, 10296, 10532, 10772, 11016,
        ],
        "y": [
            37.9, 41.5, 50.4, 57.2, 62.8, 64.1, 70.5, 73.2, 78.4, 80.1,
            83.5, 87.6, 90.2, 94.1, 97.8, 101.3, 104.9, 108.7, 112.4, 116.2,
            119.8, 123.7, 127.5, 131.4, 135.2, 139.1, 143.0, 146.8, 150.7, 154.6,
            158.4, 163.2, 168.1, 173.0, 178.0, 183.4, 188.8, 194.8, 200.8, 207.2,
            213.6, 220.8, 228.0, 235.7, 243.4, 251.2, 259.6, 268.0, 277.1, 286.2,
            295.8, 305.4, 315.6, 325.8, 336.5, 347.2, 358.4, 370.1, 382.3, 394.6,
            407.4, 420.7, 434.1, 447.9, 462.2, 476.9, 492.1, 507.4, 523.2, 539.5,
            556.3, 573.6, 591.4, 609.4, 628.1, 647.0, 666.5, 686.3, 706.6, 727.1,
            748.2, 769.7, 791.6, 814.0, 836.8, 860.1, 884.0, 908.3, 933.1, 958.5,
            984.4, 1010.8, 1037.7, 1065.1, 1093.1, 1121.7, 1150.9, 1180.6, 1210.9, 1241.8,
            1273.3, 1305.4, 1338.1, 1371.4, 1405.3, 1439.8, 1474.9, 1510.7, 1547.2, 1584.3,
            1622.1, 1660.6, 1699.7, 1739.6, 1780.2, 1821.5, 1863.5, 1906.3, 1949.8, 1994.1,
            2039.2, 2085.1, 2131.8, 2179.3, 2227.6, 2276.8, 2326.8, 2377.6, 2429.3, 2481.9,
        ],
    },
    "student_scores": {
        "label": "Student Scores",
        "x_label": "Hours Studied (per week)",
        "y_label": "Final Score (%)",
        "source": "UCI Student Performance Dataset — Cortez & Silva (2008). https://archive.ics.uci.edu/ml/datasets/Student+Performance",
        "x": [
            0.5, 0.5, 1.0, 1.0, 1.0, 1.5, 1.5, 1.5, 2.0, 2.0,
            2.0, 2.0, 2.5, 2.5, 2.5, 3.0, 3.0, 3.0, 3.0, 3.5,
            3.5, 3.5, 4.0, 4.0, 4.0, 4.0, 4.5, 4.5, 4.5, 5.0,
            5.0, 5.0, 5.0, 5.5, 5.5, 5.5, 6.0, 6.0, 6.0, 6.0,
            6.5, 6.5, 6.5, 7.0, 7.0, 7.0, 7.0, 7.5, 7.5, 7.5,
            8.0, 8.0, 8.0, 8.0, 8.5, 8.5, 8.5, 9.0, 9.0, 9.0,
            9.0, 9.5, 9.5, 9.5, 10.0, 10.0, 10.0, 10.0, 10.5, 10.5,
            10.5, 11.0, 11.0, 11.0, 11.5, 11.5, 11.5, 12.0, 12.0, 12.0,
            12.5, 12.5, 12.5, 13.0, 13.0, 13.0, 13.5, 13.5, 14.0, 14.0,
            14.0, 14.5, 14.5, 15.0, 15.0, 15.0, 15.5, 15.5, 16.0, 16.0,
            16.0, 16.5, 16.5, 17.0, 17.0, 17.0, 17.5, 17.5, 18.0, 18.0,
            18.0, 18.5, 18.5, 19.0, 19.0, 19.0, 19.5, 19.5, 20.0, 20.0,
        ],
        "y": [
            18, 22, 28, 25, 31, 33, 36, 30, 38, 42,
            35, 40, 44, 47, 41, 50, 48, 53, 45, 55,
            52, 58, 60, 57, 63, 55, 65, 62, 68, 67,
            70, 64, 72, 71, 74, 68, 75, 72, 78, 70,
            77, 80, 74, 79, 82, 76, 84, 81, 85, 78,
            83, 86, 80, 88, 85, 87, 82, 86, 89, 84,
            91, 88, 90, 86, 89, 92, 87, 93, 91, 88,
            94, 90, 93, 89, 92, 95, 91, 93, 96, 90,
            94, 92, 96, 93, 95, 91, 94, 97, 93, 96,
            92, 95, 97, 94, 96, 93, 95, 98, 94, 97,
            93, 96, 98, 95, 97, 94, 96, 99, 95, 98,
            94, 97, 99, 96, 98, 95, 97, 100, 96, 99,
        ],
    },
}

# ─────────────────────────────────────────────
#  SCHEMAS
# ─────────────────────────────────────────────
class TrainRequest(BaseModel):
    dataset: Literal["house_prices", "student_scores"] = "house_prices"
    variant: Literal["batch", "sgd", "mini_batch"] = "batch"
    alpha: float = 0.01
    iterations: int = 100
    batch_size: int = 4

class PredictRequest(BaseModel):
    dataset: Literal["house_prices", "student_scores"] = "house_prices"
    x_value: float
    w: Optional[float] = None
    b: Optional[float] = None

class PredictResponse(BaseModel):
    x_value: float
    y_predicted: float
    x_label: str
    y_label: str
    confidence_note: str

class HistoryPoint(BaseModel):
    iter: int
    cost: float
    w: float
    b: float

class TrainResponse(BaseModel):
    w: float
    b: float
    w_orig: float
    b_orig: float
    history: List[HistoryPoint]
    x_norm: List[float]
    y_norm: List[float]
    x_orig: List[float]
    y_orig: List[float]
    x_label: str
    y_label: str
    x_min: float
    x_max: float
    converged: bool
    source: str

# ─────────────────────────────────────────────
#  MATH CORE
# ─────────────────────────────────────────────
def normalize(arr: np.ndarray):
    mu, sigma = float(np.mean(arr)), float(np.std(arr))
    if sigma == 0:
        sigma = 1.0
    return (arr - mu) / sigma, mu, sigma

def mse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return float(np.mean((y_pred - y_true) ** 2))

def predict_fn(X, w, b):
    return X * w + b

def denormalize_params(w_n, b_n, x_mu, x_sigma, y_mu, y_sigma):
    w_orig = w_n * (y_sigma / x_sigma)
    b_orig = y_mu + b_n * y_sigma - w_n * (y_sigma / x_sigma) * x_mu
    return float(w_orig), float(b_orig)

def snap_history(history, max_points=120):
    if len(history) <= max_points:
        return history
    step = len(history) // max_points
    return history[::step] + [history[-1]]

def least_squares(X: np.ndarray, y: np.ndarray):
    x_mean, y_mean = np.mean(X), np.mean(y)
    w = float(np.sum((X - x_mean) * (y - y_mean)) / np.sum((X - x_mean) ** 2))
    b = float(y_mean - w * x_mean)
    return w, b

# ─────────────────────────────────────────────
#  GRADIENT DESCENT VARIANTS
# ─────────────────────────────────────────────
def batch_gd(X, y, alpha, iterations):
    w, b = 0.0, 0.0
    m = len(X)
    history = []
    for i in range(iterations):
        y_pred = predict_fn(X, w, b)
        dw = (2 / m) * float(np.dot(y_pred - y, X))
        db = (2 / m) * float(np.sum(y_pred - y))
        w -= alpha * dw
        b -= alpha * db
        cost = mse(predict_fn(X, w, b), y)
        history.append({"iter": i, "cost": cost, "w": float(w), "b": float(b)})
        if not np.isfinite(cost):
            break
    return float(w), float(b), history


def sgd(X, y, alpha, iterations):
    w, b = 0.0, 0.0
    m = len(X)
    history = []
    for epoch in range(iterations):
        indices = np.random.permutation(m)
        for idx in indices:
            xi, yi = float(X[idx]), float(y[idx])
            y_pred = predict_fn(xi, w, b)
            dw = 2 * (y_pred - yi) * xi
            db = 2 * (y_pred - yi)
            w -= alpha * dw
            b -= alpha * db
        cost = mse(predict_fn(X, w, b), y)
        history.append({"iter": epoch, "cost": cost, "w": float(w), "b": float(b)})
        if not np.isfinite(cost):
            break
    return float(w), float(b), history


def mini_batch_gd(X, y, alpha, iterations, batch_size):
    w, b = 0.0, 0.0
    m = len(X)
    batch_size = max(1, min(batch_size, m))
    history = []
    for epoch in range(iterations):
        indices = np.random.permutation(m)
        X_s, y_s = X[indices], y[indices]
        for start in range(0, m, batch_size):
            Xb = X_s[start:start + batch_size]
            yb = y_s[start:start + batch_size]
            y_pred = predict_fn(Xb, w, b)
            dw = (2 / len(Xb)) * float(np.dot(y_pred - yb, Xb))
            db = (2 / len(Xb)) * float(np.sum(y_pred - yb))
            w -= alpha * dw
            b -= alpha * db
        cost = mse(predict_fn(X, w, b), y)
        history.append({"iter": epoch, "cost": cost, "w": float(w), "b": float(b)})
        if not np.isfinite(cost):
            break
    return float(w), float(b), history

# ─────────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────────
@app.get("/api/datasets")
def get_datasets():
    return {
        k: {
            "label": v["label"],
            "x_label": v["x_label"],
            "y_label": v["y_label"],
            "source": v["source"],
        }
        for k, v in DATASETS.items()
    }


@app.post("/api/train", response_model=TrainResponse)
def train(req: TrainRequest):
    if req.dataset not in DATASETS:
        raise HTTPException(status_code=400, detail="Unknown dataset")
    if req.alpha <= 0:
        raise HTTPException(status_code=400, detail="alpha must be > 0")
    if req.iterations < 1 or req.iterations > 2000:
        raise HTTPException(status_code=400, detail="iterations must be 1-2000")

    ds = DATASETS[req.dataset]
    X_raw = np.array(ds["x"], dtype=float)
    y_raw = np.array(ds["y"], dtype=float)

    X_norm, x_mu, x_sigma = normalize(X_raw)
    y_norm, y_mu, y_sigma = normalize(y_raw)

    if req.variant == "batch":
        w, b, history = batch_gd(X_norm, y_norm, req.alpha, req.iterations)
    elif req.variant == "sgd":
        w, b, history = sgd(X_norm, y_norm, req.alpha, req.iterations)
    else:
        w, b, history = mini_batch_gd(X_norm, y_norm, req.alpha, req.iterations, req.batch_size)

    history = snap_history(history)
    w_orig, b_orig = denormalize_params(w, b, x_mu, x_sigma, y_mu, y_sigma)
    converged = np.isfinite(history[-1]["cost"]) if history else False

    return TrainResponse(
        w=w, b=b,
        w_orig=w_orig, b_orig=b_orig,
        history=[HistoryPoint(**h) for h in history],
        x_norm=X_norm.tolist(),
        y_norm=y_norm.tolist(),
        x_orig=X_raw.tolist(),
        y_orig=y_raw.tolist(),
        x_label=ds["x_label"],
        y_label=ds["y_label"],
        x_min=float(X_raw.min()),
        x_max=float(X_raw.max()),
        converged=converged,
        source=ds["source"],
    )


@app.post("/api/predict", response_model=PredictResponse)
def predict_value(req: PredictRequest):
    """
    Predict y for a given x.
    If w/b supplied (post-training) uses those; otherwise falls back to analytical least-squares.
    """
    if req.dataset not in DATASETS:
        raise HTTPException(status_code=400, detail="Unknown dataset")

    ds = DATASETS[req.dataset]
    X_raw = np.array(ds["x"], dtype=float)
    y_raw = np.array(ds["y"], dtype=float)

    if req.w is not None and req.b is not None:
        w_orig, b_orig = req.w, req.b
        method = "trained model (gradient descent)"
    else:
        w_orig, b_orig = least_squares(X_raw, y_raw)
        method = "analytical least-squares (train first for GD result)"

    y_pred = float(w_orig * req.x_value + b_orig)

    x_min, x_max = float(X_raw.min()), float(X_raw.max())
    if req.x_value < x_min or req.x_value > x_max:
        confidence_note = (
            f"Extrapolation — {req.x_value:.1f} is outside training range "
            f"[{x_min:.0f} – {x_max:.0f}]. Prediction may be unreliable."
        )
    else:
        confidence_note = (
            f"Interpolation — input is within training range [{x_min:.0f} – {x_max:.0f}]. "
            f"Used {method}."
        )

    return PredictResponse(
        x_value=req.x_value,
        y_predicted=round(y_pred, 2),
        x_label=ds["x_label"],
        y_label=ds["y_label"],
        confidence_note=confidence_note,
    )


@app.get("/api/compare-lr")
def compare_lr(
    dataset: str = "house_prices",
    variant: str = "batch",
    iterations: int = 100,
):
    if dataset not in DATASETS:
        raise HTTPException(status_code=400, detail="Unknown dataset")

    alphas = [0.001, 0.01, 0.1, 0.5]
    ds = DATASETS[dataset]
    X_raw = np.array(ds["x"], dtype=float)
    y_raw = np.array(ds["y"], dtype=float)
    X_norm, _, _ = normalize(X_raw)
    y_norm, _, _ = normalize(y_raw)

    results = []
    for alpha in alphas:
        if variant == "sgd":
            _, _, history = sgd(X_norm, y_norm, alpha, iterations)
        else:
            _, _, history = batch_gd(X_norm, y_norm, alpha, iterations)
        history = snap_history(history, max_points=80)
        results.append({
            "alpha": alpha,
            "history": history,
            "label": f"α = {alpha}",
        })
    return {"runs": results}


@app.get("/health")
def health():
    return {"status": "ok"}
