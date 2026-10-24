"""Entraînement du modèle de valeur V(après-coup) → score final de manche.

- Découpage par partie (GroupShuffleSplit) : toutes les lignes d'une même partie sont soit en entraînement,
  soit en test. Sans cela, des coups quasi identiques d'une même manche fuiraient d'un côté à l'autre.
- Deux références : l'estimation naïve (cartes visibles + cachées × espérance) que l'Expert utilise, et
  une régression linéaire. Le modèle n'a d'intérêt que s'il les bat nettement.
- Importance par permutation sur le jeu de test : quelles caractéristiques le modèle utilise vraiment.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from skyjo.ml.features import FEATURES

TARGET = "round_score"


@dataclass
class TrainResult:
    metrics: dict[str, dict[str, float]]
    importances: list[tuple[str, float, float]]
    n_train: int
    n_test: int
    params: dict[str, Any] = field(default_factory=dict)


def _metrics(y, pred) -> dict[str, float]:
    from sklearn.metrics import mean_absolute_error, r2_score

    return {"mae": float(mean_absolute_error(y, pred)), "r2": float(r2_score(y, pred))}


def train(df, out: str | Path, test_size: float = 0.2, seed: int = 0, max_iter: int = 120,
          learning_rate: float = 0.15, max_leaf_nodes: int = 31, importance_rows: int = 20000) -> TrainResult:
    """Modèle volontairement compact : à précision égale (vérifié), 3x plus rapide à jouer qu'un modèle à
    400 arbres, ce qui compte car chaque décision évalue des centaines d'après-coups."""
    import joblib
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.inspection import permutation_importance
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import GroupShuffleSplit

    X = df[list(FEATURES)].to_numpy(dtype=np.float32)
    y = df[TARGET].to_numpy(dtype=np.float32)
    groups = df["game"].to_numpy()
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed).split(X, y, groups))

    params = {"max_iter": max_iter, "learning_rate": learning_rate, "max_leaf_nodes": max_leaf_nodes,
              "l2_regularization": 1.0, "early_stopping": True, "validation_fraction": 0.1, "random_state": seed}
    t0 = time.perf_counter()
    model = HistGradientBoostingRegressor(**params).fit(X[tr], y[tr])
    fit_s = time.perf_counter() - t0

    est_col = FEATURES.index("my_est")
    linear = LinearRegression().fit(X[tr], y[tr])
    metrics = {
        "naive_estimate": _metrics(y[te], X[te, est_col]),
        "linear": _metrics(y[te], linear.predict(X[te])),
        "gradient_boosting": _metrics(y[te], model.predict(X[te])),
    }

    rng = np.random.default_rng(seed)
    sub = rng.choice(te, size=min(importance_rows, len(te)), replace=False)
    imp = permutation_importance(model, X[sub], y[sub], n_repeats=5, random_state=seed,
                                 scoring="neg_mean_absolute_error")
    importances = sorted(((f, float(m), float(s)) for f, m, s in
                          zip(FEATURES, imp.importances_mean, imp.importances_std, strict=True)),
                         key=lambda t: -t[1])

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": list(FEATURES), "target": TARGET}, out)
    report = {
        "features": list(FEATURES), "target": TARGET, "params": params, "fit_seconds": round(fit_s, 1),
        "n_train": int(len(tr)), "n_test": int(len(te)), "n_iter": int(model.n_iter_),
        "metrics": metrics, "permutation_importance_mae": [
            {"feature": f, "mean": m, "std": s} for f, m, s in importances],
    }
    out.with_suffix(".json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return TrainResult(metrics, importances, int(len(tr)), int(len(te)), params)
