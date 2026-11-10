"""Módulo de limpieza de datos, filtrado de outliers y creación del target de clasificación.

Provee funciones para depurar coordenadas geográficas dentro del polígono de CABA,
filtrar valores extremos anómalos (outliers) en superficie y precio, normalizar variables
categóricas y binarias, y calcular la variable objetivo binaria (ES_GAMA_ALTA).
"""

from typing import Tuple
import numpy as np
import pandas as pd

from src.logger_setup import get_logger

logger = get_logger(__name__)

# Límites rectangulares aproximados de la Ciudad Autónoma de Buenos Aires (CABA)
LAT_MIN, LAT_MAX = -34.72, -34.52
LON_MIN, LON_MAX = -58.54, -58.33


def filter_caba_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """Filtra las filas cuyas coordenadas geográficas caen dentro del perímetro de CABA.

    Descarta registros con latitudes o longitudes fuera del rango representativo
    de la Ciudad Autónoma de Buenos Aires o valores no numéricos.

    Args:
        df (pd.DataFrame): DataFrame original que contiene las columnas 'LAT' y 'LON'.

    Returns:
        pd.DataFrame: DataFrame filtrado únicamente con registros válidos de CABA.

    Raises:
        KeyError: Si 'LAT' o 'LON' no están presentes en el DataFrame.
    """
    if "LAT" not in df.columns or "LON" not in df.columns:
        raise KeyError("El DataFrame debe contener las columnas 'LAT' y 'LON'.")

    initial_len = len(df)
    lat = pd.to_numeric(df["LAT"], errors="coerce")
    lon = pd.to_numeric(df["LON"], errors="coerce")

    # Si faltan coordenadas pero tiene barrio o comuna válidos, se puede conservar
    # pero las que tienen coordenadas deben ser coherentes con CABA
    valid_coords = lat.between(LAT_MIN, LAT_MAX) & lon.between(LON_MIN, LON_MAX)
    has_no_coords = lat.isna() | lon.isna()
    
    # Conservamos coordenadas correctas o registros sin coordenadas pero con barrio
    mask = valid_coords | (has_no_coords & df["BARRIO"].notna())
    df_clean = df[mask].copy()

    logger.info(
        "Filtro CABA: %d registros iniciales -> %d conservados (%d descartados).",
        initial_len,
        len(df_clean),
        initial_len - len(df_clean),
    )
    return df_clean


def handle_missing_and_outliers(
    df: pd.DataFrame,
    min_m2: float = 15.0,
    max_m2: float = 800.0,
    min_price: float = 10000.0,
    max_price: float = 4000000.0,
    min_usd_m2: float = 200.0,
    max_usd_m2: float = 15000.0,
) -> pd.DataFrame:
    """Elimina registros con valores atípicos y nulos críticos en precio y superficie.

    Calcula la consistencia entre 'DOLARES' y 'M2' para derivar 'U_S_M2' si estuviese ausente.
    Normaliza columnas binarias (COCHERA, BAULERA, LAVADERO, TERRAZA) a 0 y 1.

    Args:
        df (pd.DataFrame): DataFrame a depurar.
        min_m2 (float, optional): Superficie mínima válida en m². Por defecto 15.0.
        max_m2 (float, optional): Superficie máxima válida en m². Por defecto 800.0.
        min_price (float, optional): Precio mínimo en dólares. Por defecto 10,000.0.
        max_price (float, optional): Precio máximo en dólares. Por defecto 4,000,000.0.
        min_usd_m2 (float, optional): Valor mínimo admisible de U$S/m². Por defecto 200.0.
        max_usd_m2 (float, optional): Valor máximo admisible de U$S/m². Por defecto 15,000.0.

    Returns:
        pd.DataFrame: DataFrame depurado con tipos homogéneos y sin valores extremos.

    Raises:
        KeyError: Si faltan columnas requeridas ('M2', 'DOLARES').
    """
    initial_len = len(df)
    clean_df = df.copy()

    # Conversión a tipos numéricos
    clean_df["M2"] = pd.to_numeric(clean_df["M2"], errors="coerce")
    clean_df["DOLARES"] = pd.to_numeric(clean_df["DOLARES"], errors="coerce")

    # Si U_S_M2 falta o es cero, se deriva
    clean_df["U_S_M2"] = pd.to_numeric(clean_df["U_S_M2"], errors="coerce")
    calc_us_m2 = clean_df["DOLARES"] / clean_df["M2"]
    clean_df["U_S_M2"] = clean_df["U_S_M2"].fillna(calc_us_m2)

    # Filtrar valores no positivos y rangos razonables
    valid_mask = (
        clean_df["M2"].between(min_m2, max_m2)
        & clean_df["DOLARES"].between(min_price, max_price)
        & clean_df["U_S_M2"].between(min_usd_m2, max_usd_m2)
    )
    clean_df = clean_df[valid_mask].copy()

    # Normalizar texto en BARRIO y COMUNA
    clean_df["BARRIO"] = (
        clean_df["BARRIO"].astype(str).str.strip().str.upper().replace({"NAN": "DESCONOCIDO", "": "DESCONOCIDO"})
    )
    clean_df["COMUNA"] = pd.to_numeric(clean_df["COMUNA"], errors="coerce").fillna(0).astype(int)

    # Normalizar columnas binarias (SI/NO -> 1/0)
    binary_cols = ["COCHERA", "BAULERA", "LAVADERO", "TERRAZA"]
    for col in binary_cols:
        if col in clean_df.columns:
            str_col = clean_df[col].astype(str).str.strip().str.upper()
            clean_df[col] = np.where(str_col.isin(["SI", "S", "1", "1.0", "TRUE"]), 1, 0)

    # Imputar variables numéricas estructuradas
    for col in ["AMBIENTES", "BAÑOS", "ANTIGUEDAD"]:
        if col in clean_df.columns:
            clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")

    logger.info(
        "Filtro de outliers: %d iniciales -> %d válidos (%d eliminados).",
        initial_len,
        len(clean_df),
        initial_len - len(clean_df),
    )
    return clean_df


def create_target(
    df: pd.DataFrame,
    quantile_threshold: float = 0.75,
    target_column_name: str = "ES_GAMA_ALTA",
) -> pd.DataFrame:
    """Genera la variable objetivo binaria de clasificación basada en el valor por metro cuadrado.

    Para evitar distorsiones macroeconómicas (inflación y devaluación histórica del dólar
    entre 2001 y 2016), el umbral de corte se calcula dinámicamente según el percentil
    del precio por m² (U_S_M2) de cada año. Las propiedades en el percentil superior
    (>= quantile_threshold) se catalogan como 'Gama Alta' (1) y el resto como 'Estándar' (0).

    Args:
        df (pd.DataFrame): DataFrame depurado con 'AÑO' y 'U_S_M2'.
        quantile_threshold (float, optional): Percentil de corte entre 0 y 1. Por defecto 0.75.
        target_column_name (str, optional): Nombre de la columna resultante. Por defecto 'ES_GAMA_ALTA'.

    Returns:
        pd.DataFrame: DataFrame que incluye la columna objetivo binaria (int: 0 o 1).

    Raises:
        KeyError: Si 'U_S_M2' o 'AÑO' no existen en el DataFrame.
        ValueError: Si quantile_threshold no está entre 0 y 1.
    """
    if not (0 < quantile_threshold < 1):
        raise ValueError("El quantile_threshold debe estar estrictamente entre 0 y 1.")
    if "U_S_M2" not in df.columns or "AÑO" not in df.columns:
        raise KeyError("Se requieren las columnas 'AÑO' y 'U_S_M2' para crear el target.")

    result_df = df.copy()

    # Cálculo del percentil anual de corte
    threshold_per_year = result_df.groupby("AÑO")["U_S_M2"].transform(
        lambda s: s.quantile(quantile_threshold)
    )

    result_df[target_column_name] = (result_df["U_S_M2"] >= threshold_per_year).astype(int)

    counts = result_df[target_column_name].value_counts().to_dict()
    logger.info(
        "Target '%s' generado exitosamente (percentil=%.2f): Clases: %s",
        target_column_name,
        quantile_threshold,
        counts,
    )
    return result_df


def clean_and_prepare_dataset(
    df: pd.DataFrame,
    quantile_threshold: float = 0.75,
) -> pd.DataFrame:
    """Función de alto nivel que encadena el flujo completo de limpieza y creación del target.

    Args:
        df (pd.DataFrame): DataFrame unificado de departamentos.
        quantile_threshold (float, optional): Percentil para el corte de Gama Alta. Por defecto 0.75.

    Returns:
        pd.DataFrame: DataFrame listo para ingeniería de variables y modelado.
    """
    logger.info("Iniciando pipeline de limpieza completo...")
    df_coords = filter_caba_coordinates(df)
    df_no_outliers = handle_missing_and_outliers(df_coords)
    df_with_target = create_target(df_no_outliers, quantile_threshold=quantile_threshold)
    logger.info("Pipeline de limpieza finalizado con éxito: %d filas resultantes.", len(df_with_target))
    return df_with_target
