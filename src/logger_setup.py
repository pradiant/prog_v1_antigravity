"""Módulo de configuración centralizada de logging.

Provee un logger con salida simultánea a:
1. Consola (StreamHandler).
2. Archivo rotativo en disco (RotatingFileHandler en logs/pipeline.log).
3. Base NoSQL MongoDB (MongoHandler en colección 'logs', con fallback elegante).
"""

from datetime import datetime
import logging
from logging.handlers import RotatingFileHandler
from typing import Optional
from pymongo import MongoClient

from config import (
    LOGS_DIR,
    MONGO_URI,
    MONGO_DB_NAME,
    MONGO_COLLECTION_LOGS,
)

FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


class MongoHandler(logging.Handler):
    """Handler personalizado para registrar eventos de logging directamente en MongoDB."""

    def __init__(
        self,
        uri: str = MONGO_URI,
        db_name: str = MONGO_DB_NAME,
        collection_name: str = MONGO_COLLECTION_LOGS,
        level: int = logging.INFO,
    ) -> None:
        """Inicializa el handler verificando conectividad de forma no bloqueante.

        Args:
            uri (str): URI de conexión de MongoDB.
            db_name (str): Nombre de la base de datos.
            collection_name (str): Colección para guardar los logs.
            level (int): Nivel mínimo de logging a capturar.
        """
        super().__init__(level)
        self.uri = uri
        self.db_name = db_name
        self.collection_name = collection_name
        self._client: Optional[MongoClient] = None
        self._collection = None
        self._disabled = False

    def _get_collection(self):
        """Obtiene la colección de MongoDB con lazy initialization y timeout rápido."""
        if self._disabled:
            return None
        if self._collection is None:
            try:
                self._client = MongoClient(self.uri, serverSelectionTimeoutMS=1000)
                self._client.admin.command("ping")
                db = self._client[self.db_name]
                self._collection = db[self.collection_name]
            except Exception:
                # Si Mongo no está disponible, se deshabilita para evitar demoras
                self._disabled = True
                return None
        return self._collection

    def emit(self, record: logging.LogRecord) -> None:
        """Inserta un documento de log en MongoDB.

        Args:
            record (logging.LogRecord): Registro de log generado por la aplicación.
        """
        try:
            coll = self._get_collection()
            if coll is not None:
                log_entry = {
                    "timestamp": datetime.fromtimestamp(record.created).isoformat(),
                    "level": record.levelname,
                    "logger": record.name,
                    "message": record.getMessage(),
                    "module": record.module,
                    "line": record.lineno,
                }
                coll.insert_one(log_entry)
        except Exception:
            # Fallback seguro: el fallo de Mongo nunca debe detener la aplicación
            pass


def get_logger(name: str) -> logging.Logger:
    """Configura y retorna una instancia de Logger con consola, archivo y MongoDB.

    Args:
        name (str): Nombre del módulo para identificar la procedencia de los mensajes.

    Returns:
        logging.Logger: Instancia de logger lista para registrar eventos.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter(FORMAT)

    # 1. Handler de Consola
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)
    logger.addHandler(console)

    # 2. Handler de Archivo Rotativo
    file_handler = RotatingFileHandler(
        LOGS_DIR / "pipeline.log",
        maxBytes=2_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    # 3. Handler de MongoDB (NoSQL)
    mongo_handler = MongoHandler(level=logging.INFO)
    logger.addHandler(mongo_handler)

    return logger
