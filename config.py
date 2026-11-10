"""Configuración general del proyecto: rutas, constantes, parámetros y URIs de base de datos."""

from pathlib import Path

# Rutas del sistema
BASE_DIR = Path(__file__).resolve().parent
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
LOGS_DIR = BASE_DIR / "logs"

# Base de Datos Relacional (SQLite: exclusivamente para dataset limpio)
SQLITE_DB_PATH = BASE_DIR / "data" / "inmuebles.db"
SQLITE_TABLE_CLEAN = "departamentos_limpios"

# Base de Datos NoSQL (MongoDB: resultados, parámetros y logs)
MONGO_URI = "mongodb://localhost:27017"
MONGO_DB_NAME = "taller_inmobiliario"
MONGO_COLLECTION_RESULTS = "resultados"
MONGO_COLLECTION_PARAMS = "parametros"
MONGO_COLLECTION_LOGS = "logs"

# URLs y parámetros de descarga de BA Data
ROOT_URL = "https://data.buenosaires.gob.ar/dataset/departamentos-venta/resource/juqdkmgo-70"
RESOURCE_IDS = list(range(1, 32, 2))          # 1,3,5,...,31
YEARS = list(range(2001, 2017))               # 2001..2016
YEAR_TO_RESOURCE = dict(zip(YEARS, RESOURCE_IDS))

RAW_FILE_PATTERN = "departamentos-en-venta-{year}.csv"
CSV_PARAMS = dict(encoding="utf-8-sig", sep=";")

# Mapeo de normalización de columnas
COLUMN_RENAME_MAP = {
    "LATITUD": "LAT",
    "LONGITUD": "LON",
    "M2CUB": "M2_CUBIERTOS",
    "BAÃOS": "BAÑOS",
    "BAÑO": "BAÑOS",
    "BAUL": "BAULERA",
    "BARRIOS": "BARRIO",
    "A_ESTRENAR": "ES_ESTRENAR",
}

# Columnas esperadas del dataset unificado
TARGET_COLUMNS = [
    "AÑO", "CALLE", "NUMERO", "M2", "M2_CUBIERTOS", "DOLARES", "U_S_M2",
    "AMBIENTES", "DORMITORIO", "PISO", "ANTIGUEDAD", "ORIENT", "BAULERA",
    "COCHERA", "BAÑOS", "LAVADERO", "TERRAZA", "BARRIO", "COMUNA",
    "DIRECCION_NORMALIZADA", "LAT", "LON",
]
