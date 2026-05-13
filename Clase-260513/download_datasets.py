"""
download_datasets.py
====================

Descarga los dos datasets para la Clase 2 de Minería de Datos (UPCh, Mayo 2026).

  1. ENIGH 2024 Chiapas — para demostración en vivo (Ramsés)
     Fuente: INEGI · tabla concentradohogar · nueva serie 2024
     Filtra a entidad 07 (Chiapas) y selecciona ~16 columnas pedagógicamente útiles.

  2. Telco Customer Churn — para que los estudiantes repliquen por su cuenta
     Fuente: IBM (mirror público, no requiere auth de Kaggle)
     7043 filas · 21 columnas · mix completo de tipos · missing values naturales.

Uso:
    python download_datasets.py
    python download_datasets.py --output-dir ./data

Requisitos:
    pip install pandas requests
"""

from __future__ import annotations

import argparse
import io
import sys
import zipfile
from pathlib import Path

import pandas as pd
import requests


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ENIGH_URL = (
    "https://www.inegi.org.mx/contenidos/programas/enigh/nc/2024/"
    "microdatos/enigh2024_ns_concentradohogar_csv.zip"
)

# Mirror público estable mantenido por IBM
TELCO_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)

# Columnas pedagógicas en concentradohogar — cubren los 4 tipos de atributo
ENIGH_COLS_TARGET = [
    # Identificadores / nominales
    "folioviv",       # ID vivienda (nominal · 10 dígitos, primeros 2 = entidad)
    "ubica_geo",      # geo (nominal · 9 dígitos)
    "clase_hog",      # tipo de hogar (nominal · 1-5)
    # Ordinales
    "tam_loc",        # tamaño localidad (ordinal · 1-4)
    "est_socio",      # estrato socioeconómico (ordinal · 1-4)
    "educa_jefe",     # nivel educativo del jefe (ordinal · 0-11)
    # Binario
    "sexo_jefe",      # 1=hombre, 2=mujer
    # Numéricos discretos
    "edad_jefe",
    "tot_integ",
    "hombres",
    "mujeres",
    "mayores",
    "menores",
    # Numéricos continuos (ratio)
    "ing_cor",        # ingreso corriente total trimestral
    "ingtrab",        # ingreso por trabajo
    "gasto_mon",      # gasto monetario total
    "alimentos",
    "educa_espa",     # gasto educación y esparcimiento
    "salud",
    "transporte",
]

CHIAPAS_ENTIDAD_CODE = "07"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def log(msg: str, level: str = "info") -> None:
    prefix = {"info": "→", "ok": "✓", "warn": "!", "err": "✗"}[level]
    print(f"  {prefix} {msg}", flush=True)


def download_bytes(url: str, label: str) -> bytes:
    log(f"Descargando {label} …")
    log(f"  URL: {url}")
    r = requests.get(url, stream=True, timeout=120, headers={
        "User-Agent": "Mozilla/5.0 (educational/UPCh-mineria-2026a)"
    })
    r.raise_for_status()
    content = r.content
    size_mb = len(content) / 1024 / 1024
    log(f"  {size_mb:.2f} MB recibidos", "ok")
    return content


# ---------------------------------------------------------------------------
# ENIGH
# ---------------------------------------------------------------------------

def extract_concentrado_from_zip(zip_bytes: bytes) -> pd.DataFrame:
    """Extrae la tabla concentradohogar del ZIP descargado de INEGI."""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        # Buscar el CSV dentro del ZIP (suele estar en una subcarpeta)
        csv_members = [
            n for n in zf.namelist()
            if n.lower().endswith(".csv") and "concentrado" in n.lower()
        ]
        if not csv_members:
            # fallback: cualquier CSV
            csv_members = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not csv_members:
            raise RuntimeError(
                f"No se encontró CSV dentro del ZIP. Contenido: {zf.namelist()}"
            )

        member = csv_members[0]
        log(f"  Leyendo {member} del ZIP …")
        with zf.open(member) as f:
            # INEGI usa latin-1 a veces; probar utf-8 primero
            raw = f.read()
            for enc in ("utf-8", "latin-1", "cp1252"):
                try:
                    df = pd.read_csv(io.BytesIO(raw), encoding=enc, low_memory=False)
                    log(f"  Encoding: {enc}", "ok")
                    return df
                except UnicodeDecodeError:
                    continue
            raise RuntimeError("No se pudo decodificar el CSV con encodings conocidos")


def prepare_enigh_chiapas(df: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    log(f"  Filas originales (nacional): {len(df):,}")
    log(f"  Columnas originales: {df.shape[1]}")

    # Normalizar nombre de columnas a minúsculas (defensivo)
    df.columns = [c.lower() for c in df.columns]

    # Filtrar a Chiapas vía folioviv (primeros 2 dígitos = entidad)
    df["folioviv"] = df["folioviv"].astype(str).str.zfill(10)
    df_chis = df[df["folioviv"].str[:2] == CHIAPAS_ENTIDAD_CODE].copy()
    log(f"  Filas Chiapas: {len(df_chis):,}", "ok")

    # Seleccionar solo las columnas pedagógicas que sí existen
    available = [c for c in ENIGH_COLS_TARGET if c in df_chis.columns]
    missing = [c for c in ENIGH_COLS_TARGET if c not in df_chis.columns]
    if missing:
        log(f"  Columnas no encontradas (se omiten): {missing}", "warn")

    df_clean = df_chis[available].copy()

    # Guardar
    df_clean.to_csv(out_path, index=False, encoding="utf-8")
    log(f"  Guardado: {out_path}  ({len(df_clean):,} filas × {df_clean.shape[1]} cols)", "ok")
    return df_clean


# ---------------------------------------------------------------------------
# Telco
# ---------------------------------------------------------------------------

def prepare_telco(csv_bytes: bytes, out_path: Path) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(csv_bytes))
    df.to_csv(out_path, index=False, encoding="utf-8")
    log(f"  Guardado: {out_path}  ({len(df):,} filas × {df.shape[1]} cols)", "ok")
    return df


# ---------------------------------------------------------------------------
# Resumen pedagógico
# ---------------------------------------------------------------------------

def print_summary(df: pd.DataFrame, name: str) -> None:
    print()
    print(f"  === {name} ===")
    print(f"  shape: {df.shape}")
    print()
    print("  dtypes:")
    for col, dtype in df.dtypes.items():
        n_unique = df[col].nunique(dropna=True)
        n_missing = df[col].isna().sum()
        print(f"    {col:20s}  {str(dtype):10s}  unique={n_unique:<6}  missing={n_missing}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("./data"),
        help="Carpeta donde se guardarán los CSVs limpios (default: ./data)",
    )
    parser.add_argument(
        "--skip-enigh", action="store_true",
        help="No descargar ENIGH (útil si ya lo tienes)",
    )
    parser.add_argument(
        "--skip-telco", action="store_true",
        help="No descargar Telco Churn",
    )
    parser.add_argument(
        "--summary", action="store_true",
        help="Imprimir resumen de tipos y missing values al final",
    )
    args = parser.parse_args()

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nCarpeta de salida: {out_dir.resolve()}\n")

    enigh_df = None
    telco_df = None

    # -------- ENIGH --------
    if not args.skip_enigh:
        print("[1/2] ENIGH 2024 · concentradohogar · Chiapas")
        print("─" * 60)
        try:
            zip_bytes = download_bytes(ENIGH_URL, "ENIGH 2024 ZIP")
            df_full = extract_concentrado_from_zip(zip_bytes)
            enigh_df = prepare_enigh_chiapas(
                df_full, out_dir / "enigh_chiapas_2024.csv"
            )
        except Exception as e:
            log(f"ENIGH falló: {e}", "err")
            print()
            print("  El servidor de INEGI a veces requiere navegador (DescargaMasiva).")
            print("  Si la descarga directa falla, opciones:")
            print("    1. Visita https://www.inegi.org.mx/programas/enigh/nc/2024/")
            print("       → sección 'Microdatos' → 'Concentrado de hogar' → CSV")
            print("    2. Descarga manual del ZIP, luego corre este script con --skip-enigh")
            print("       y procesa el ZIP local con el módulo extract_concentrado_from_zip.")

    # -------- Telco --------
    if not args.skip_telco:
        print()
        print("[2/2] Telco Customer Churn (Kaggle · mirror IBM público)")
        print("─" * 60)
        try:
            csv_bytes = download_bytes(TELCO_URL, "Telco Churn CSV")
            telco_df = prepare_telco(csv_bytes, out_dir / "telco_churn.csv")
        except Exception as e:
            log(f"Telco falló: {e}", "err")

    # -------- Resumen --------
    if args.summary:
        if enigh_df is not None:
            print_summary(enigh_df, "ENIGH Chiapas 2024")
        if telco_df is not None:
            print_summary(telco_df, "Telco Customer Churn")

    print()
    print("Listo. Para inspección rápida:")
    print(f"   python -c \"import pandas as pd; print(pd.read_csv('{out_dir}/enigh_chiapas_2024.csv').describe())\"")
    print(f"   python -c \"import pandas as pd; print(pd.read_csv('{out_dir}/telco_churn.csv').head())\"")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
