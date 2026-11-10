"""Módulo de evaluación de modelos de clasificación y matriz de confusión.

Este módulo provee funciones para calcular métricas exigidas por el Proyecto Final
(Accuracy, Precision, Recall, F1-Score) y generar la representación tabular y gráfica
de la matriz de confusión.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.logger_setup import get_logger

logger = get_logger(__name__)


def evaluate_classification(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    y_prob: Optional[np.ndarray | pd.Series] = None,
) -> Dict[str, Any]:
    """Calcula exhaustivamente las métricas de clasificación para evaluación de modelos.

    Computa Accuracy, Precision, Recall y F1-Score (tanto ponderado como por clase),
    además del área bajo la curva ROC (ROC-AUC) si se proveen probabilidades continuas.

    Args:
        y_true (np.ndarray | pd.Series): Vector con las etiquetas reales (0 o 1).
        y_pred (np.ndarray | pd.Series): Vector con las etiquetas predichas por el modelo.
        y_prob (Optional[np.ndarray | pd.Series]): Vector o matriz de probabilidades estimadas
            para la clase positiva. Si se proporciona, se calcula ROC-AUC.

    Returns:
        Dict[str, Any]: Diccionario con las siguientes claves:
            - 'accuracy': float, exactitud global del modelo.
            - 'precision': float, precisión ponderada (weighted).
            - 'recall': float, exhaustividad ponderada (weighted).
            - 'f1_score': float, media armónica F1 ponderada (weighted).
            - 'f1_macro': float, media armónica F1 macro promedio.
            - 'roc_auc': Optional[float], área bajo la curva ROC (None si y_prob no se pasa).
            - 'confusion_matrix': np.ndarray, matriz de confusión numérica (2x2).
            - 'classification_report': dict, reporte completo por clase y globales.

    Raises:
        ValueError: Si los tamaños de y_true y y_pred no coinciden o están vacíos.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Dimensiones incompatibles: y_true tiene {len(y_true)} elementos y y_pred {len(y_pred)}"
        )
    if len(y_true) == 0:
        raise ValueError("Los vectores de etiquetas están vacíos.")

    logger.info("Calculando métricas de clasificación sobre %d muestras...", len(y_true))

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    rec = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    f1_w = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    f1_m = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    roc_auc: Optional[float] = None
    if y_prob is not None:
        try:
            # Si y_prob es 2D con dos clases, tomar la probabilidad de la clase positiva
            prob_positive = y_prob[:, 1] if getattr(y_prob, "ndim", 1) == 2 else y_prob
            roc_auc = float(roc_auc_score(y_true, prob_positive))
        except Exception as err:
            logger.warning("No se pudo calcular ROC-AUC: %s", err)

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    logger.info(
        "Métricas calculadas: Accuracy=%.4f | Precision=%.4f | Recall=%.4f | F1=%.4f | ROC-AUC=%s",
        acc,
        prec,
        rec,
        f1_w,
        f"{roc_auc:.4f}" if roc_auc is not None else "N/A",
    )

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1_w,
        "f1_macro": f1_m,
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
        "classification_report": report,
    }


def plot_and_save_confusion_matrix(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    labels: Optional[List[str]] = None,
    output_path: Optional[Path] = None,
    title: str = "Matriz de Confusión",
) -> plt.Figure:
    """Genera una visualización en mapa de calor de la matriz de confusión y la guarda opcionalmente.

    Args:
        y_true (np.ndarray | pd.Series): Etiquetas verdaderas.
        y_pred (np.ndarray | pd.Series): Etiquetas predichas.
        labels (Optional[List[str]]): Nombres descriptivos para las clases.
            Por defecto ['Estándar (0)', 'Gama Alta (1)'].
        output_path (Optional[Path]): Ruta de archivo para guardar la imagen generada.
        title (str): Título que se mostrará en la figura.

    Returns:
        plt.Figure: Objeto de figura matplotlib generado.
    """
    if labels is None:
        labels = ["Estándar (0)", "Gama Alta (1)"]

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        cbar=False,
    )
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Clase Predicha", fontsize=10)
    ax.set_ylabel("Clase Real", fontsize=10)
    plt.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300)
        logger.info("Matriz de confusión guardada en: %s", output_path)

    return fig
