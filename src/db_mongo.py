"""Módulo de persistencia no relacional en MongoDB (resultados, parámetros, logs).

Conforme a la especificación de folders.txt:
- db_mongo.py -> MongoDB: resultados, parámetros, logs.
Almacena en colecciones dedicadas de MongoDB:
1. Colección 'resultados': Métricas de desempeño globales (Accuracy, Precision, Recall, F1, ROC-AUC)
   y detalle de predicciones sobre el conjunto de test.
2. Colección 'parametros': Hiperparámetros óptimos y configuración del modelo.
3. Colección 'logs': Gestionada directamente a través de MongoHandler en logger_setup.py.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

from config import (
    MONGO_URI,
    MONGO_DB_NAME,
    MONGO_COLLECTION_RESULTS,
    MONGO_COLLECTION_PARAMS,
)
from src.logger_setup import get_logger

logger = get_logger(__name__)


def get_mongo_client(
    uri: str = MONGO_URI,
    timeout_ms: int = 2000,
) -> Optional[MongoClient]:
    """Crea y valida una conexión con el servidor de MongoDB.

    Verifica disponibilidad con ping rápido (timeout_ms) para evitar demoras
    si el servicio no se encuentra activo localmente.

    Args:
        uri (str, optional): Cadena de conexión URI de MongoDB.
        timeout_ms (int, optional): Tiempo límite de espera en ms. Por defecto 2000.

    Returns:
        Optional[MongoClient]: Cliente MongoDB conectado o None si no responde.
    """
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=timeout_ms)
        client.admin.command("ping")
        return client
    except (ConnectionFailure, ServerSelectionTimeoutError) as err:
        logger.warning("MongoDB no disponible en %s: %s", uri, err)
        return None


def save_model_results_mongo(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray],
    metrics: Dict[str, Any],
    model_name: str,
    uri: str = MONGO_URI,
    db_name: str = MONGO_DB_NAME,
    collection_name: str = MONGO_COLLECTION_RESULTS,
    sample_predictions_limit: int = 500,
) -> Optional[str]:
    """Persiste los resultados y métricas del modelo en la colección 'resultados' de MongoDB.

    Args:
        y_true (pd.Series | np.ndarray): Etiquetas verdaderas.
        y_pred (np.ndarray): Etiquetas predichas por el modelo.
        y_prob (Optional[np.ndarray]): Probabilidades estimadas para la clase positiva.
        metrics (Dict[str, Any]): Diccionario de métricas (Accuracy, Precision, Recall, F1).
        model_name (str): Nombre del algoritmo evaluado.
        uri (str, optional): URI de conexión de MongoDB.
        db_name (str, optional): Nombre de la base NoSQL.
        collection_name (str, optional): Colección destino ('resultados').
        sample_predictions_limit (int, optional): Máximo de predicciones individuales
            a incluir en el documento para optimizar tamaño. Por defecto 500.

    Returns:
        Optional[str]: ID del documento insertado o None si MongoDB no respondió.
    """
    client = get_mongo_client(uri)
    if client is None:
        logger.warning("No se pudo conectar a MongoDB. Omitiendo guardado en '%s'.", collection_name)
        return None

    try:
        db = client[db_name]
        collection = db[collection_name]

        # Extraer probabilidades de la clase positiva
        prob_pos = None
        if y_prob is not None:
            prob_pos = y_prob[:, 1] if getattr(y_prob, "ndim", 1) == 2 else y_prob

        # Muestra de predicciones fila por fila
        num_samples = min(len(y_true), sample_predictions_limit)
        predictions_sample = [
            {
                "id": int(i),
                "real": int(y_true.iloc[i] if hasattr(y_true, "iloc") else y_true[i]),
                "prediccion": int(y_pred[i]),
                "probabilidad": float(np.round(prob_pos[i], 4)) if prob_pos is not None else None,
            }
            for i in range(num_samples)
        ]

        document = {
            "timestamp": datetime.now().isoformat(),
            "modelo": model_name,
            "total_muestras_test": len(y_true),
            "metricas": {
                "accuracy": float(metrics.get("accuracy", 0.0)),
                "precision": float(metrics.get("precision", 0.0)),
                "recall": float(metrics.get("recall", 0.0)),
                "f1_score": float(metrics.get("f1_score", 0.0)),
                "f1_macro": float(metrics.get("f1_macro", 0.0)),
                "roc_auc": float(metrics["roc_auc"]) if metrics.get("roc_auc") is not None else None,
            },
            "matriz_confusion": metrics.get("confusion_matrix").tolist() if hasattr(metrics.get("confusion_matrix"), "tolist") else None,
            "muestra_predicciones": predictions_sample,
        }

        res = collection.insert_one(document)
        logger.info("Resultados del modelo guardados en MongoDB [Colección '%s', _id: %s]", collection_name, res.inserted_id)
        return str(res.inserted_id)
    except Exception as err:
        logger.warning("Error al guardar resultados en MongoDB: %s", err)
        return None


def save_model_parameters_mongo(
    model_name: str,
    best_params: Dict[str, Any],
    uri: str = MONGO_URI,
    db_name: str = MONGO_DB_NAME,
    collection_name: str = MONGO_COLLECTION_PARAMS,
) -> Optional[str]:
    """Persiste los hiperparámetros óptimos y configuración del modelo en la colección 'parametros' de MongoDB.

    Args:
        model_name (str): Nombre del algoritmo (e.g. 'RandomForestClassifier').
        best_params (Dict[str, Any]): Diccionario de hiperparámetros devuelto por GridSearchCV.
        uri (str, optional): URI de conexión.
        db_name (str, optional): Nombre de base de datos NoSQL.
        collection_name (str, optional): Colección destino ('parametros').

    Returns:
        Optional[str]: ID del documento insertado o None si falló.
    """
    client = get_mongo_client(uri)
    if client is None:
        logger.warning("No se pudo conectar a MongoDB. Omitiendo guardado en '%s'.", collection_name)
        return None

    try:
        db = client[db_name]
        collection = db[collection_name]

        # Sanitizar parámetros para asegurar compatibilidad BSON
        clean_params = {k: (int(v) if isinstance(v, (np.integer,)) else float(v) if isinstance(v, (np.floating,)) else str(v) if not isinstance(v, (int, float, bool, str)) else v) for k, v in best_params.items()}

        document = {
            "timestamp": datetime.now().isoformat(),
            "modelo": model_name,
            "hiperparametros": clean_params,
        }

        res = collection.insert_one(document)
        logger.info("Parámetros del modelo guardados en MongoDB [Colección '%s', _id: %s]", collection_name, res.inserted_id)
        return str(res.inserted_id)
    except Exception as err:
        logger.warning("Error al guardar parámetros en MongoDB: %s", err)
        return None
