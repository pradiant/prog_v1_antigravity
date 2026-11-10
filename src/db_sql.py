"""Módulo de persistencia en base de datos relacional (SQLite).

Conforme a la especificación de folders.txt:
- db_sql.py -> SQLite: tabla de datos limpios.

Se encarga exclusivamente de almacenar y recuperar el dataset unificado y depurado
en la tabla 'departamentos_limpios' dentro de data/inmuebles.db.
"""

from pathlib import Path
import sqlite3
import pandas as pd

from config import SQLITE_DB_PATH, SQLITE_TABLE_CLEAN
from src.logger_setup import get_logger

logger = get_logger(__name__)


def get_sqlite_connection(db_path: Path = SQLITE_DB_PATH) -> sqlite3.Connection:
    """Abre o crea una conexión con la base de datos SQLite local.

    Crea el directorio contenedor si no existe previamente.

    Args:
        db_path (Path, optional): Ruta al archivo SQLite. Por defecto SQLITE_DB_PATH.

    Returns:
        sqlite3.Connection: Objeto de conexión SQLite abierto.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(db_path)


def save_clean_dataset(
    df: pd.DataFrame,
    db_path: Path = SQLITE_DB_PATH,
    table_name: str = SQLITE_TABLE_CLEAN,
) -> None:
    """Persiste exclusivamente el dataset unificado y limpio en la base de datos SQLite.

    Guarda los registros filtrados, sin outliers y con la variable objetivo en la
    tabla especificada (por defecto 'departamentos_limpios').

    Args:
        df (pd.DataFrame): DataFrame con registros de propiedades limpios.
        db_path (Path, optional): Ruta al archivo SQLite. Por defecto SQLITE_DB_PATH.
        table_name (str, optional): Nombre de la tabla. Por defecto 'departamentos_limpios'.
    """
    logger.info("Guardando dataset limpio en SQLite (%s -> tabla '%s'): %d filas.", db_path.name, table_name, len(df))
    with get_sqlite_connection(db_path) as conn:
        df.to_sql(table_name, conn, if_exists="replace", index=False)
    logger.info("Dataset limpio guardado exitosamente en SQLite.")


def load_clean_dataset(
    db_path: Path = SQLITE_DB_PATH,
    table_name: str = SQLITE_TABLE_CLEAN,
) -> pd.DataFrame:
    """Carga los datos limpios almacenados previamente en la tabla SQLite.

    Args:
        db_path (Path, optional): Ruta al archivo de base de datos SQLite.
        table_name (str, optional): Nombre de la tabla a consultar.

    Returns:
        pd.DataFrame: DataFrame con los registros limpios recuperados de la base de datos.
    """
    with get_sqlite_connection(db_path) as conn:
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    logger.info("Se cargaron %d registros desde '%s' en %s", len(df), table_name, db_path.name)
    return df
