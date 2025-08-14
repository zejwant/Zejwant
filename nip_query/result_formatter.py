"""
result_formatter.py
-------------------

Formats raw DataFrame query results into multiple enterprise-ready output formats.

Features:
    - Plain-English summaries using descriptive stats & optional trend detection.
    - Chart generation (Matplotlib for static, Plotly for interactive).
    - Downloadable CSV / Excel export.
    - Optional analysis methods (trend detection, correlations).
    - Localization-ready output (multi-language summaries).
    - Full logging for observability.

Author: Varun-engineer mode (20+ years experience)
"""

from __future__ import annotations
import logging
import io
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from typing import Dict, Any, Optional, Literal
from datetime import datetime
import locale

logger = logging.getLogger(__name__)

# ---------- CONFIG ----------
DEFAULT_LANG = "en_US"  # Can be changed to "fr_FR", "es_ES", etc.

try:
    locale.setlocale(locale.LC_ALL, DEFAULT_LANG)
except locale.Error:
    logger.warning(f"Locale {DEFAULT_LANG} not found. Falling back to system default.")


# ---------- CORE FUNCTIONS ----------

def summarize_dataframe(df: pd.DataFrame, lang: str = DEFAULT_LANG) -> str:
    """
    Create a plain-English summary of the DataFrame.
    """
    if df.empty:
        return _translate("No results found.", lang)

    num_rows, num_cols = df.shape
    col_info = ", ".join([f"{col} ({df[col].dtype})" for col in df.columns[:5]])
    summary = (
        f"{_translate('The query returned', lang)} {num_rows} {_translate('rows and', lang)} {num_cols} {_translate('columns.', lang)} "
        f"{_translate('Sample columns are', lang)}: {col_info}."
    )

    try:
        numeric_cols = df.select_dtypes(include=["number"]).columns
        if len(numeric_cols) > 0:
            stats = df[numeric_cols].describe().to_dict()
            for col, values in stats.items():
                summary += f" {_translate('For', lang)} '{col}', {_translate('the mean is', lang)} {values.get('mean'):.2f}."
    except Exception as e:
        logger.error(f"Error generating numeric summary: {e}")

    return summary


def generate_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    chart_type: Literal["line", "bar", "scatter"] = "line",
    interactive: bool = False
):
    """
    Generate a chart from DataFrame.
    """
    if df.empty:
        raise ValueError("Cannot generate chart from empty DataFrame.")

    if interactive:
        fig = getattr(px, chart_type)(df, x=x, y=y, title=f"{chart_type.title()} Chart")
        return fig
    else:
        plt.figure(figsize=(8, 5))
        getattr(plt, chart_type)(df[x], df[y])
        plt.xlabel(x)
        plt.ylabel(y)
        plt.title(f"{chart_type.title()} Chart")
        plt.grid(True)
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plt.close()
        return buf


def export_dataframe(
    df: pd.DataFrame,
    format: Literal["csv", "excel"] = "csv"
) -> bytes:
    """
    Export DataFrame to CSV or Excel.
    """
    if format == "csv":
        return df.to_csv(index=False).encode("utf-8")
    elif format == "excel":
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
            df.to_excel(writer, index=False)
        return buf.getvalue()
    else:
        raise ValueError("Unsupported export format. Use 'csv' or 'excel'.")


# ---------- OPTIONAL ANALYSIS ----------
def detect_trend(df: pd.DataFrame, date_col: str, value_col: str) -> str:
    """
    Basic trend detection: identifies increasing/decreasing trend in a time-series.
    """
    if df.empty or date_col not in df or value_col not in df:
        return "No trend detected."

    try:
        df_sorted = df.sort_values(by=date_col)
        slope = (df_sorted[value_col].iloc[-1] - df_sorted[value_col].iloc[0]) / len(df_sorted)
        if slope > 0:
            return "Upward trend detected."
        elif slope < 0:
            return "Downward trend detected."
        else:
            return "No significant trend detected."
    except Exception as e:
        logger.error(f"Trend detection failed: {e}")
        return "Error detecting trend."


# ---------- LOCALIZATION ----------
def _translate(text: str, lang: str) -> str:
    """
    Stub for translation. Replace with i18n service for production.
    """
    translations = {
        "fr_FR": {
            "No results found.": "Aucun résultat trouvé.",
            "The query returned": "La requête a renvoyé",
            "rows and": "lignes et",
            "columns.": "colonnes.",
            "Sample columns are": "Les colonnes d'exemple sont",
            "For": "Pour",
            "the mean is": "la moyenne est"
        }
    }
    return translations.get(lang, {}).get(text, text)


# ---------- UNIT TEST STUB ----------
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    df_test = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=5, freq="D"),
        "sales": [100, 120, 130, 150, 170],
        "region": ["North", "North", "South", "South", "East"]
    })

    print(summarize_dataframe(df_test))
    print(detect_trend(df_test, "date", "sales"))

    csv_data = export_dataframe(df_test, "csv")
    print(f"CSV Export Size: {len(csv_data)} bytes")

    chart_buf = generate_chart(df_test, "date", "sales", "line", interactive=False)
    print(f"Static chart generated: {len(chart_buf.getvalue())} bytes")
  
