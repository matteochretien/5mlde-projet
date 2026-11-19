"""Métriques d'évaluation, orientées exploitation.

La performance prédictive n'est pas notée, mais les métriques doivent servir à
*décider* : c'est sur elles que repose la règle de mise en service. On suit donc
trois niveaux de lecture :

* `mae` — erreur moyenne globale, comparable à la baseline ;
* `mae_lt30` — erreur sur les lignes dont la vraie RUL est inférieure à 30 jours.
  C'est la zone qui compte : se tromper de 10 jours sur un disque qui a 3 mois à
  vivre est sans conséquence, se tromper de 10 jours sur un disque qui lâche
  dans 5 jours en a ;
* `alert_precision` / `alert_recall` — qualité de l'alerte métier « RUL prédite
  < 30 jours », c'est-à-dire ce que l'exploitant voit réellement.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass
class EvaluationMetrics:
    n_samples: int
    mae: float
    rmse: float
    mae_lt30: float
    n_lt30: int
    alert_precision: float
    alert_recall: float
    alert_f1: float
    mean_prediction: float
    mean_target: float

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def evaluate_predictions(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    *,
    alert_threshold_days: int = 30,
) -> EvaluationMetrics:
    """Calcule les métriques globales, la MAE de la zone critique et l'alerte."""
    truth = np.asarray(y_true, dtype=float)
    prediction = np.asarray(y_pred, dtype=float)
    if truth.shape != prediction.shape:
        raise ValueError(f"tailles incompatibles: {truth.shape} vs {prediction.shape}")
    if truth.size == 0:
        raise ValueError("aucune observation à évaluer")

    errors = np.abs(truth - prediction)
    critical = truth < alert_threshold_days

    alert_true = truth < alert_threshold_days
    alert_pred = prediction < alert_threshold_days
    true_positive = float(np.sum(alert_true & alert_pred))
    false_positive = float(np.sum(~alert_true & alert_pred))
    false_negative = float(np.sum(alert_true & ~alert_pred))
    precision = _safe_ratio(true_positive, true_positive + false_positive)
    recall = _safe_ratio(true_positive, true_positive + false_negative)

    return EvaluationMetrics(
        n_samples=int(truth.size),
        mae=float(errors.mean()),
        rmse=float(np.sqrt(np.mean((truth - prediction) ** 2))),
        mae_lt30=float(errors[critical].mean()) if critical.any() else float("nan"),
        n_lt30=int(critical.sum()),
        alert_precision=precision,
        alert_recall=recall,
        alert_f1=_safe_ratio(2 * precision * recall, precision + recall),
        mean_prediction=float(prediction.mean()),
        mean_target=float(truth.mean()),
    )
