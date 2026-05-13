"""
Utilidades para Clase 2 - Mineria de Datos - UPCh 2026A
Funciones reutilizables para los 8 ejercicios sobre ENIGH Chiapas.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import pairwise_distances

# Paleta y estilo (consistente con las slides del curso)
NAVY = "#1C3257"
TERRA = "#AA4B37"
SAND = "#F4EFE6"
INK = "#1A1A1A"

PLOTLY_TEMPLATE = "plotly_white"
PLOTLY_FONT = dict(family="Helvetica, Arial, sans-serif", color=INK, size=13)


def clasificar_atributos(
    df: pd.DataFrame,
    hints: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Tabla con dtype, tipo conceptual, n_unique, n_missing y ejemplos por columna.

    Hints permite override manual: {'folioviv': 'nominal', 'tam_loc': 'ordinal'}.
    Heuristica cuando no hay hint:
      - object/string  -> nominal
      - numerico con n_unique == 2 -> binario
      - entero con n_unique <= 10  -> revisar (probable nominal/ordinal codificado)
      - resto numerico -> numerico
    """
    hints = hints or {}
    filas = []
    for col in df.columns:
        s = df[col]
        n_unique = int(s.nunique(dropna=True))
        n_missing = int(s.isna().sum())
        dtype = str(s.dtype)

        if col in hints:
            tipo = hints[col]
        elif pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
            tipo = "nominal"
        elif n_unique == 2:
            tipo = "binario"
        elif pd.api.types.is_integer_dtype(s) and n_unique <= 10:
            tipo = "revisar"
        elif pd.api.types.is_numeric_dtype(s):
            tipo = "numerico"
        else:
            tipo = "revisar"

        ejemplos = s.dropna().unique()[:4]
        filas.append(
            {
                "columna": col,
                "dtype_pandas": dtype,
                "tipo_conceptual": tipo,
                "n_unique": n_unique,
                "n_missing": n_missing,
                "ejemplo_valores": list(ejemplos),
            }
        )
    return pd.DataFrame(filas)


def detectar_outliers_iqr(s: pd.Series, k: float = 1.5) -> pd.Series:
    """Mascara booleana: True si el valor es outlier segun la regla k*IQR.

    NaN se marca como False (no es outlier, simplemente falta).
    """
    if not pd.api.types.is_numeric_dtype(s):
        raise ValueError(f"detectar_outliers_iqr requiere serie numerica, recibio {s.dtype}")
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    low = q1 - k * iqr
    high = q3 + k * iqr
    return ((s < low) | (s > high)).fillna(False)


def chi_square(df: pd.DataFrame, col_a: str, col_b: str) -> dict:
    """Chi-cuadrado de independencia entre dos columnas categoricas.

    Devuelve dict con keys: chi2, p_value, dof, observed, expected.
    """
    for col in (col_a, col_b):
        if col not in df.columns:
            raise ValueError(f"Columna '{col}' no esta en el DataFrame")
    sub = df[[col_a, col_b]].dropna()
    if sub.empty:
        raise ValueError(f"No hay filas con valores en ambas columnas {col_a}, {col_b}")
    observed = pd.crosstab(sub[col_a], sub[col_b])
    chi2, p, dof, expected = stats.chi2_contingency(observed.values)
    expected_df = pd.DataFrame(expected, index=observed.index, columns=observed.columns)
    return {
        "chi2": float(chi2),
        "p_value": float(p),
        "dof": int(dof),
        "observed": observed,
        "expected": expected_df,
    }


def aplicar_normalizaciones(s: pd.Series) -> pd.DataFrame:
    """DataFrame con columnas: original, minmax, zscore, decimal.

    minmax  -> (x - min) / (max - min) en [0, 1]
    zscore  -> (x - media) / std
    decimal -> x / 10^j, con j = ceil(log10(max(|x|)))
    """
    if not pd.api.types.is_numeric_dtype(s):
        raise ValueError("aplicar_normalizaciones requiere una Serie numerica")
    x = s.astype(float).reset_index(drop=True)
    rango = x.max() - x.min()
    minmax = (x - x.min()) / rango if rango != 0 else pd.Series(np.zeros(len(x)))
    sigma = x.std()
    zscore = (x - x.mean()) / sigma if sigma != 0 else pd.Series(np.zeros(len(x)))
    max_abs = x.abs().max()
    j = math.ceil(math.log10(max_abs)) if max_abs > 0 else 0
    decimal = x / (10 ** j) if j > 0 else x.copy()
    return pd.DataFrame(
        {
            "original": x,
            "minmax": minmax,
            "zscore": zscore,
            "decimal": decimal,
        }
    )


def distancia_matriz(df: pd.DataFrame, metrica: str = "euclidean") -> np.ndarray:
    """Matriz de distancia n x n entre las filas de df (todas las columnas deben ser numericas).

    metrica en {'euclidean', 'manhattan', 'chebyshev', 'cosine'}.
    """
    permitidas = {"euclidean", "manhattan", "chebyshev", "cosine"}
    if metrica not in permitidas:
        raise ValueError(f"metrica debe ser una de {permitidas}, recibio '{metrica}'")
    no_numericas = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    if no_numericas:
        raise ValueError(f"distancia_matriz requiere columnas numericas; no numericas: {no_numericas}")
    return pairwise_distances(df.values, metric=metrica)


def imputar_por_grupo(
    df: pd.DataFrame,
    target: str,
    group: str,
    estrategia: str = "median",
) -> pd.Series:
    """Imputa NaN de df[target] con la mediana/media del grupo definido por df[group].

    estrategia en {'median', 'mean'}.
    """
    if estrategia not in {"median", "mean"}:
        raise ValueError(f"estrategia debe ser 'median' o 'mean', recibio '{estrategia}'")
    for col in (target, group):
        if col not in df.columns:
            raise ValueError(f"Columna '{col}' no esta en el DataFrame")
    func = "median" if estrategia == "median" else "mean"
    valores_grupo = df.groupby(group)[target].transform(func)
    imputada = df[target].fillna(valores_grupo)
    # Fallback global por si algun grupo entero es NaN
    if imputada.isna().any():
        fallback = df[target].median() if estrategia == "median" else df[target].mean()
        imputada = imputada.fillna(fallback)
    return imputada
