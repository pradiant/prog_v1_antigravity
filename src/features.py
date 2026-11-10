"""Módulo de ingeniería de características y pipelines de preprocesamiento scikit-learn.

Este módulo implementa el ColumnTransformer para preprocesar de forma modular variables
numéricas y categóricas según lo requerido por la consigna oficial:
- Imputación de datos faltantes (mediana en numéricas, moda en categóricas).
- Escalado estándar (StandardScaler) en variables numéricas.
- Codificación One-Hot (OneHotEncoder) en variables categóricas.
"""

from typing import List, Optional, Tuple
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.logger_setup import get_logger

logger = get_logger(__name__)

# Conjuntos de características predeterminadas
DEFAULT_NUMERIC_FEATURES = [
    "M2",
    "AMBIENTES",
    "BAÑOS",
    "ANTIGUEDAD",
    "COCHERA",
    "BAULERA",
    "LAVADERO",
    "TERRAZA",
]

DEFAULT_CATEGORICAL_FEATURES = [
    "BARRIO",
    "COMUNA",
]


def build_preprocessor(
    numeric_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
) -> ColumnTransformer:
    """Construye un ColumnTransformer de scikit-learn para preprocesamiento integral.

    Aplica una secuencia de transformaciones sin filtración de datos (data leakage):
    - Numéricas: Imputación por mediana seguida de estandarización (media 0, varianza 1).
    - Categóricas: Imputación por valor más frecuente seguida de codificación One-Hot
      con soporte para categorías desconocidas en inferencia (handle_unknown='ignore').

    Args:
        numeric_features (Optional[List[str]]): Lista de nombres de columnas numéricas.
            Si es None, usa DEFAULT_NUMERIC_FEATURES.
        categorical_features (Optional[List[str]]): Lista de nombres de columnas categóricas.
            Si es None, usa DEFAULT_CATEGORICAL_FEATURES.

    Returns:
        ColumnTransformer: Transformador compuesto de scikit-learn listo para
            ser acoplado a un Pipeline con un estimador.
    """
    if numeric_features is None:
        numeric_features = DEFAULT_NUMERIC_FEATURES
    if categorical_features is None:
        categorical_features = DEFAULT_CATEGORICAL_FEATURES

    logger.info(
        "Construyendo preprocesador: %d variables numéricas, %d categóricas.",
        len(numeric_features),
        len(categorical_features),
    )

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )

    return preprocessor


def split_features_and_target(
    df: pd.DataFrame,
    target_column: str = "ES_GAMA_ALTA",
    features: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Separa la matriz de características (X) y el vector objetivo (y).

    Elimina deliberadamente variables filtradas o derivadas directamente del target
    como 'U_S_M2' y 'DOLARES' para evitar data leakage en el entrenamiento del modelo.

    Args:
        df (pd.DataFrame): DataFrame completo con características y variable objetivo.
        target_column (str, optional): Nombre de la columna objetivo. Por defecto 'ES_GAMA_ALTA'.
        features (Optional[List[str]], optional): Lista explícita de columnas predictoras.
            Si es None, toma DEFAULT_NUMERIC_FEATURES + DEFAULT_CATEGORICAL_FEATURES.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Tupla (X, y) donde X es la matriz de predictoras
            e y es la serie de etiquetas binarias.

    Raises:
        KeyError: Si target_column o alguna característica no existe en df.
    """
    if target_column not in df.columns:
        raise KeyError(f"La columna objetivo '{target_column}' no se encuentra en el DataFrame.")

    if features is None:
        features = DEFAULT_NUMERIC_FEATURES + DEFAULT_CATEGORICAL_FEATURES

    missing_cols = [c for c in features if c not in df.columns]
    if missing_cols:
        raise KeyError(f"Columnas predictoras ausentes en el DataFrame: {missing_cols}")

    X = df[features].copy()
    y = df[target_column].copy()

    logger.info("Separación X / y completada: X=%s, y=%s", X.shape, y.shape)
    return X, y
