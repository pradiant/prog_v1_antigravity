"""Módulo de generación y exportación de gráficos para análisis y modelos.

Implementa visualizaciones con matplotlib y seaborn:
- Matrices de correlación entre variables numéricas.
- Distribución de clases de la variable objetivo.
- Importancia de características en modelos de ensamble (Random Forest).
- Distribución de valores por barrio.
"""

from pathlib import Path
from typing import List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.logger_setup import get_logger

logger = get_logger(__name__)


def plot_correlation_matrix(
    df: pd.DataFrame,
    output_path: Optional[Path] = None,
    figsize: tuple = (10, 8),
) -> plt.Figure:
    """Calcula y grafica el mapa de calor (heatmap) de correlación entre variables numéricas.

    Args:
        df (pd.DataFrame): DataFrame con variables numéricas.
        output_path (Optional[Path], optional): Ruta para guardar la imagen PNG. Por defecto None.
        figsize (tuple, optional): Tamaño de la figura (ancho, alto). Por defecto (10, 8).

    Returns:
        plt.Figure: Objeto de figura matplotlib generado.
    """
    logger.info("Generando mapa de correlación...")
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        square=True,
        ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Matriz de Correlación de Variables Numéricas", fontsize=12, fontweight="bold")
    plt.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300)
        logger.info("Mapa de correlación guardado en: %s", output_path)

    return fig


def plot_target_distribution(
    df: pd.DataFrame,
    target_col: str = "ES_GAMA_ALTA",
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """Grafica la distribución y proporción de las clases en la variable objetivo.

    Args:
        df (pd.DataFrame): DataFrame que contiene la columna objetivo.
        target_col (str, optional): Nombre de la variable objetivo. Por defecto 'ES_GAMA_ALTA'.
        output_path (Optional[Path], optional): Ruta para exportar la figura. Por defecto None.

    Returns:
        plt.Figure: Objeto de figura matplotlib.

    Raises:
        KeyError: Si target_col no existe en el DataFrame.
    """
    if target_col not in df.columns:
        raise KeyError(f"La columna objetivo '{target_col}' no existe en el DataFrame.")

    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df[target_col].value_counts().sort_index()
    labels = ["Estándar (0)", "Gama Alta (1)"]

    sns.barplot(x=labels, y=counts.values, palette=["#3498db", "#e74c3c"], ax=ax)
    for i, v in enumerate(counts.values):
        pct = (v / len(df)) * 100
        ax.text(i, v + (max(counts.values) * 0.02), f"{v:,} ({pct:.1f}%)", ha="center", fontweight="bold")

    ax.set_title(f"Distribución de la Variable Objetivo: {target_col}", fontsize=11, fontweight="bold")
    ax.set_ylabel("Cantidad de Propiedades", fontsize=10)
    ax.set_ylim(0, max(counts.values) * 1.15)
    plt.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300)
        logger.info("Gráfico de distribución guardado en: %s", output_path)

    return fig


def plot_feature_importances(
    feature_names: List[str],
    importances: np.ndarray,
    top_n: int = 15,
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """Grafica la importancia relativa de las principales variables de un modelo de árboles.

    Args:
        feature_names (List[str]): Lista de nombres de las características transformadas.
        importances (np.ndarray): Vector de importancias (e.g. model.feature_importances_).
        top_n (int, optional): Cantidad de variables a exhibir en el ranking. Por defecto 15.
        output_path (Optional[Path], optional): Ruta para guardar la figura. Por defecto None.

    Returns:
        plt.Figure: Figura matplotlib con el gráfico de barras horizontales.
    """
    df_imp = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(top_n)
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(data=df_imp, x="importance", y="feature", palette="viridis", ax=ax)
    ax.set_title(f"Top {top_n} Variables Más Importantes (Random Forest)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Importancia Relativa", fontsize=10)
    ax.set_ylabel("Característica", fontsize=10)
    plt.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300)
        logger.info("Gráfico de importancia de variables guardado en: %s", output_path)

    return fig
