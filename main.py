"""Pipeline principal del proyecto inmobiliario CABA.

Orquesta de forma automatizada y secuencial:
1. Carga y estandarización de los 16 datasets anuales de BA Data (2001-2016).
2. Limpieza de datos, filtrado de outliers y creación del target de clasificación (ES_GAMA_ALTA).
3. Persistencia en SQLite exclusivamente del dataset limpio (departamentos_limpios).
4. Partición estratificada en Train/Test y construcción del ColumnTransformer scikit-learn.
5. Entrenamiento del modelo baseline (LogisticRegression).
6. Optimización de hiperparámetros de RandomForestClassifier con GridSearchCV.
7. Comparación formal de modelos y selección del ganador por F1-Score.
8. Persistencia en MongoDB de resultados (métricas/predicciones) y parámetros óptimos.
9. Serialización del modelo ganador para la aplicación Streamlit.
"""

from pathlib import Path
import sys
from typing import Tuple
import numpy as np
from sklearn.model_selection import train_test_split

from config import (
    BASE_DIR,
    YEARS,
    SQLITE_DB_PATH,
    SQLITE_TABLE_CLEAN,
    MONGO_URI,
    MONGO_DB_NAME,
)
from src.data_loader import build_unified_dataset
from src.cleaning import clean_and_prepare_dataset
from src.features import build_preprocessor, split_features_and_target
from src.models import (
    create_dummy_classifier_pipeline,
    create_logistic_regression_pipeline,
    create_random_forest_pipeline,
    optimize_with_gridsearch,
    compare_models,
    save_model,
)
from src.evaluation import plot_and_save_confusion_matrix
from src.db_sql import save_clean_dataset
from src.db_mongo import (
    save_model_results_mongo,
    save_model_parameters_mongo,
)
from src.logger_setup import get_logger

logger = get_logger("pipeline_main")


def run_pipeline() -> None:
    """Ejecuta el flujo completo punta a punta del proyecto.

    Lee los archivos crudos, genera las transformaciones y almacena todos los
    artefactos y registros exigidos por la consigna del Proyecto Final:
    - SQLite: tabla 'departamentos_limpios'.
    - MongoDB: colecciones 'resultados', 'parametros' y 'logs'.

    Raises:
        Exception: Si ocurre algún error fatal durante la ejecución del pipeline.
    """
    logger.info("==================================================================")
    logger.info(" INICIANDO PIPELINE: CLASIFICACIÓN DE PROPIEDADES EN CABA")
    logger.info("==================================================================")

    model_output_path = BASE_DIR / "app" / "modelo_ganador.joblib"
    cm_output_path = BASE_DIR / "logs" / "matriz_confusion_ganador.png"

    # 1. Carga y unificación de datos
    logger.info("[Paso 1/8] Cargando y unificando 16 datasets anuales...")
    df_raw = build_unified_dataset(YEARS)

    # 2. Limpieza, outliers y target de clasificación
    logger.info("[Paso 2/8] Ejecutando limpieza, tratamiento de outliers y target ES_GAMA_ALTA...")
    df_clean = clean_and_prepare_dataset(df_raw, quantile_threshold=0.75)

    # 3. Persistencia en SQLite (exclusivamente el dataset limpio)
    logger.info("[Paso 3/8] Persistiendo dataset limpio en SQLite (tabla '%s')...", SQLITE_TABLE_CLEAN)
    save_clean_dataset(df_clean, SQLITE_DB_PATH, SQLITE_TABLE_CLEAN)

    # 4. Separación de variables y partición Train / Test
    logger.info("[Paso 4/8] Dividiendo datos en Train (80%) y Test (20%) estratificado...")
    X, y = split_features_and_target(df_clean, target_column="ES_GAMA_ALTA")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # 5. Construcción del ColumnTransformer
    logger.info("[Paso 5/8] Construyendo ColumnTransformer con imputadores, StandardScaler y OneHotEncoder...")
    preprocessor = build_preprocessor()

    # 6. Entrenamiento de Modelos
    logger.info("[Paso 6/8] Entrenando Modelo Base Ingenuo: DummyClassifier (Most Frequent)...")
    pipe_dummy = create_dummy_classifier_pipeline(preprocessor, strategy="most_frequent")
    pipe_dummy.fit(X_train, y_train)

    logger.info("[Paso 6/8] Entrenando Modelo 1: Regresión Logística (Baseline Lineal)...")
    pipe_lr = create_logistic_regression_pipeline(preprocessor)
    pipe_lr.fit(X_train, y_train)

    logger.info("[Paso 6/8] Entrenando Modelo 2: Random Forest con GridSearchCV...")
    pipe_rf = create_random_forest_pipeline(preprocessor)
    param_grid = {
        "classifier__n_estimators": [50, 100],
        "classifier__max_depth": [10, 20],
    }

    # Muestra estratificada representativa para agilizar la búsqueda de hiperparámetros
    sample_size = min(30000, len(X_train))
    sample_idx = X_train.sample(n=sample_size, random_state=42).index
    grid_search = optimize_with_gridsearch(
        pipe_rf,
        param_grid,
        X_train.loc[sample_idx],
        y_train.loc[sample_idx],
        cv_splits=3,
    )
    best_rf = grid_search.best_estimator_
    best_rf.fit(X_train, y_train)

    # 7. Comparación de modelos y selección del ganador
    logger.info("[Paso 7/8] Comparando modelos sobre conjunto de Test...")
    candidatos = {
        "DummyClassifier": pipe_dummy,
        "LogisticRegression": pipe_lr,
        "RandomForest_Tuned": best_rf,
    }
    winner_name, winner_pipeline, all_metrics = compare_models(candidatos, X_test, y_test)
    winner_metrics = all_metrics[winner_name]

    # Exportar matriz de confusión visual
    y_pred_winner = winner_pipeline.predict(X_test)
    y_prob_winner = (
        winner_pipeline.predict_proba(X_test)
        if hasattr(winner_pipeline, "predict_proba")
        else None
    )
    plot_and_save_confusion_matrix(
        y_test,
        y_pred_winner,
        output_path=cm_output_path,
        title=f"Matriz de Confusión - {winner_name}",
    )

    # 8. Persistencia en MongoDB (resultados y parámetros)
    logger.info("[Paso 8/8] Persistiendo resultados y parámetros en MongoDB...")
    save_model_results_mongo(
        y_true=y_test,
        y_pred=y_pred_winner,
        y_prob=y_prob_winner,
        metrics=winner_metrics,
        model_name=winner_name,
        uri=MONGO_URI,
        db_name=MONGO_DB_NAME,
    )
    save_model_parameters_mongo(
        model_name=winner_name,
        best_params=grid_search.best_params_,
        uri=MONGO_URI,
        db_name=MONGO_DB_NAME,
    )

    # Serializar modelo ganador para la app de Streamlit
    save_model(winner_pipeline, model_output_path)

    logger.info("==================================================================")
    logger.info(" ¡PIPELINE FINALIZADO CON ÉXITO!")
    logger.info(" Modelo Ganador: %s", winner_name)
    logger.info(" Accuracy: %.4f | Precision: %.4f | Recall: %.4f | F1-Score: %.4f",
                winner_metrics['accuracy'], winner_metrics['precision'],
                winner_metrics['recall'], winner_metrics['f1_score'])
    logger.info(" SQLite: dataset limpio guardado en %s (tabla '%s')", SQLITE_DB_PATH, SQLITE_TABLE_CLEAN)
    logger.info(" MongoDB: resultados y parámetros guardados en %s (db: %s)", MONGO_URI, MONGO_DB_NAME)
    logger.info(" Para lanzar la aplicación web: streamlit run app/app.py")
    logger.info("==================================================================")


if __name__ == "__main__":
    try:
        run_pipeline()
    except Exception as exc:
        logger.exception("Error fatal durante la ejecución del pipeline: %s", exc)
        sys.exit(1)
