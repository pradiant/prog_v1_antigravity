"""Módulo de entrenamiento, optimización de hiperparámetros y selección de modelos.

Implementa la construcción de pipelines con clasificadores (Regresión Logística y
Random Forest), la optimización con GridSearchCV mediante validación cruzada estratificada,
y la persistencia en disco mediante joblib.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from src.evaluation import evaluate_classification
from src.logger_setup import get_logger

logger = get_logger(__name__)


def create_dummy_classifier_pipeline(
    preprocessor: ColumnTransformer,
    strategy: str = "most_frequent",
    random_state: int = 42,
    **kwargs: Any,
) -> Pipeline:
    """Crea un Pipeline con un estimador DummyClassifier como baseline ingenuo de referencia.

    Sirve como control experimental para comprobar si los modelos reales superan
    una regla heurística trivial (por ejemplo, predecir siempre la clase mayoritaria).

    Args:
        preprocessor (ColumnTransformer): Transformador de características.
        strategy (str, optional): Estrategia de predicción trivial ('most_frequent', 'stratified', etc.).
            Por defecto 'most_frequent'.
        random_state (int, optional): Semilla para reproducibilidad. Por defecto 42.
        **kwargs: Parámetros adicionales para DummyClassifier.

    Returns:
        Pipeline: Flujo acoplado de preprocesamiento y clasificador dummy.
    """
    classifier = DummyClassifier(strategy=strategy, random_state=random_state, **kwargs)
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )
    logger.info("Pipeline de DummyClassifier (strategy='%s') creado.", strategy)
    return pipeline


def create_logistic_regression_pipeline(
    preprocessor: ColumnTransformer,
    random_state: int = 42,
    max_iter: int = 1000,
    class_weight: str = "balanced",
    **kwargs: Any,
) -> Pipeline:
    """Crea un Pipeline de scikit-learn con el preprocesador y un clasificador LogisticRegression.

    Args:
        preprocessor (ColumnTransformer): Transformador de características (numéricas y categóricas).
        random_state (int, optional): Semilla para reproducibilidad. Por defecto 42.
        max_iter (int, optional): Cantidad máxima de iteraciones del solver. Por defecto 1000.
        class_weight (str, optional): Estrategia de ponderación de clases ('balanced' o None).
            Por defecto 'balanced'.
        **kwargs: Parámetros adicionales para LogisticRegression.

    Returns:
        Pipeline: Flujo acoplado de preprocesamiento y clasificación lineal.
    """
    classifier = LogisticRegression(
        random_state=random_state,
        max_iter=max_iter,
        class_weight=class_weight,
        **kwargs,
    )
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )
    logger.info("Pipeline de Regresión Logística creado.")
    return pipeline


def create_random_forest_pipeline(
    preprocessor: ColumnTransformer,
    random_state: int = 42,
    n_estimators: int = 100,
    class_weight: str = "balanced",
    n_jobs: int = -1,
    **kwargs: Any,
) -> Pipeline:
    """Crea un Pipeline de scikit-learn con el preprocesador y un clasificador RandomForestClassifier.

    Args:
        preprocessor (ColumnTransformer): Transformador de características.
        random_state (int, optional): Semilla para reproducibilidad. Por defecto 42.
        n_estimators (int, optional): Número de árboles en el bosque. Por defecto 100.
        class_weight (str, optional): Ponderación de clases. Por defecto 'balanced'.
        n_jobs (int, optional): Hilos paralelos (-1 para todos los cores). Por defecto -1.
        **kwargs: Parámetros adicionales para RandomForestClassifier.

    Returns:
        Pipeline: Flujo acoplado de preprocesamiento y ensamble de árboles.
    """
    classifier = RandomForestClassifier(
        random_state=random_state,
        n_estimators=n_estimators,
        class_weight=class_weight,
        n_jobs=n_jobs,
        **kwargs,
    )
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )
    logger.info("Pipeline de Random Forest creado.")
    return pipeline


def optimize_with_gridsearch(
    pipeline: Pipeline,
    param_grid: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_splits: int = 5,
    scoring: str = "f1_weighted",
    n_jobs: int = -1,
) -> GridSearchCV:
    """Optimiza los hiperparámetros de un pipeline mediante búsqueda en grilla y validación cruzada.

    Aplica StratifiedKFold para garantizar que cada partición mantenga la proporción
    original de las clases 'Gama Alta' y 'Estándar'.

    Args:
        pipeline (Pipeline): Pipeline compuesto por preprocesador y modelo.
        param_grid (Dict[str, Any]): Espacio de hiperparámetros a explorar
            (e.g., {'classifier__n_estimators': [50, 100], 'classifier__max_depth': [10, 20]}).
        X_train (pd.DataFrame): Datos de entrenamiento.
        y_train (pd.Series): Etiquetas reales de entrenamiento.
        cv_splits (int, optional): Cantidad de folds en validación cruzada. Por defecto 5.
        scoring (str, optional): Métrica objetivo a maximizar. Por defecto 'f1_weighted'.
        n_jobs (int, optional): Hilos para paralelizar la búsqueda. Por defecto -1.

    Returns:
        GridSearchCV: Objeto ajustado con el mejor estimador y resultados de la búsqueda.
    """
    logger.info(
        "Iniciando GridSearchCV (%d combinaciones, %d-fold cross-validation, métrica='%s')...",
        len(param_grid),
        cv_splits,
        scoring,
    )
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        refit=True,
        verbose=1,
    )
    grid_search.fit(X_train, y_train)

    logger.info("Mejores hiperparámetros: %s", grid_search.best_params_)
    logger.info("Mejor score (%s) en CV: %.4f", scoring, grid_search.best_score_)
    return grid_search


def compare_models(
    models_dict: Dict[str, Pipeline],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[str, Pipeline, Dict[str, Dict[str, Any]]]:
    """Evalúa múltiples modelos sobre el conjunto de test y selecciona al ganador por F1-Score.

    Args:
        models_dict (Dict[str, Pipeline]): Diccionario nombre -> pipeline entrenado.
        X_test (pd.DataFrame): Matriz de características de prueba.
        y_test (pd.Series): Vector objetivo verdadero de prueba.

    Returns:
        Tuple[str, Pipeline, Dict[str, Dict[str, Any]]]:
            - Nombre del mejor modelo.
            - Instancia del mejor pipeline.
            - Diccionario con las métricas calculadas para cada modelo evaluado.
    """
    all_metrics: Dict[str, Dict[str, Any]] = {}
    best_name: str = ""
    best_score: float = -1.0
    best_pipeline: Optional[Pipeline] = None

    for name, model in models_dict.items():
        logger.info("Evaluando modelo candidato: %s", name)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
        metrics = evaluate_classification(y_test, y_pred, y_prob=y_prob)
        all_metrics[name] = metrics

        score = metrics["f1_score"]
        if score > best_score:
            best_score = score
            best_name = name
            best_pipeline = model

    logger.info("Modelo ganador seleccionado: '%s' con F1-Score ponderado de %.4f", best_name, best_score)
    assert best_pipeline is not None
    return best_name, best_pipeline, all_metrics


def save_model(model: Any, filepath: Path) -> None:
    """Serializa y persiste un modelo entrenado en disco usando joblib.

    Args:
        model (Any): Pipeline o estimador de scikit-learn a exportar.
        filepath (Path): Ruta de destino (e.g., 'models/best_model.joblib').
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, filepath)
    logger.info("Modelo serializado exitosamente en: %s", filepath)


def load_model(filepath: Path) -> Any:
    """Carga un modelo serializado previamente desde el disco.

    Args:
        filepath (Path): Ruta del archivo joblib.

    Returns:
        Any: Pipeline o modelo recuperado en memoria.

    Raises:
        FileNotFoundError: Si la ruta especificada no existe.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"No se encontró el modelo en: {filepath}")
    model = joblib.load(filepath)
    logger.info("Modelo cargado exitosamente desde: %s", filepath)
    return model
