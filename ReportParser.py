"""
report_parser.py
────────────────
Extracts financial figures from uploaded business reports
using only local tools: pdfminer, openpyxl, pandas, spaCy, and regex.

Zero internet calls. Zero external AI. Runs entirely on your machine.
"""

import re
import os
import pandas as pd
import spacy
from pdfminer.high_level import extract_text
from typing import Optional

# Load the small English model (downloaded in Dockerfile)
# This model runs locally — no network needed
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    # Fallback if not yet downloaded (dev without Docker)
    nlp = None


# ─── Regex patterns for financial figures ──────────────────────────────────────
# Matches values like: $1,200,000 | 1.2M | 1,200,000.00 | (500,000) [loss]

NUMBER_PATTERN = re.compile(
    r"""
    [$£€]?                          # optional currency symbol
    (\(?)                           # optional opening bracket (means negative)
    (\d{1,3}(?:,\d{3})*(?:\.\d+)?) # main number with optional commas + decimals
    \)?                             # optional closing bracket
    \s*([KkMmBb](?:illion)?)?       # optional scale: K M B
    """,
    re.VERBOSE,
)

# Keywords that signal which financial line item a number belongs to
FIELD_KEYWORDS = {
    "revenue":           ["revenue", "total revenue", "net revenue", "sales", "total sales", "turnover"],
    "net_profit":        ["net income", "net profit", "profit after tax", "net earnings", "pat"],
    "gross_profit":      ["gross profit", "gross income"],
    "operating_income":  ["operating income", "operating profit", "ebit"],
    "ebitda":            ["ebitda", "earnings before interest"],
    "total_assets":      ["total assets"],
    "total_liabilities": ["total liabilities"],
    "total_equity":      ["total equity", "shareholders equity", "stockholders equity"],
    "cash_flow":         ["cash flow", "operating cash flow", "net cash"],
    "revenue_growth":    ["revenue growth", "sales growth", "growth rate"],
    "profit_margin":     ["profit margin", "net margin"],
    "debt_to_equity":    ["debt to equity", "debt/equity", "d/e ratio"],
    "current_ratio":     ["current ratio"],
    "return_on_equity":  ["return on equity", "roe"],
    "return_on_assets":  ["return on assets", "roa"],
}


def _parse_number(raw: str) -> Optional[float]:
    """Convert a matched raw string like '(1,200,000)' or '1.2M' to a float."""
    if not raw:
        return None
    raw = raw.strip()
    negative = raw.startswith("(") and raw.endswith(")")
    raw = raw.replace("(", "").replace(")", "").replace("$", "").replace("£", "").replace("€", "").replace(",", "").strip()

    scale = 1.0
    if raw.upper().endswith("B"):
        scale = 1_000_000_000
        raw = raw[:-1]
    elif raw.upper().endswith("M"):
        scale = 1_000_000
        raw = raw[:-1]
    elif raw.upper().endswith("K"):
        scale = 1_000
        raw = raw[:-1]

    try:
        value = float(raw) * scale
        return -value if negative else value
    except ValueError:
        return None


def _extract_from_text(text: str) -> dict:
    """
    Scan plain text line by line.
    If a line contains a known keyword, grab the first number on that line.
    """
    results = {}
    lines = text.lower().split("\n")

    for line in lines:
        line_clean = line.strip()
        for field, keywords in FIELD_KEYWORDS.items():
            if field in results:
                continue  # already found this field
            for kw in keywords:
                if kw in line_clean:
                    matches = NUMBER_PATTERN.findall(line_clean)
                    for m in matches:
                        # m is a tuple (bracket, number, scale)
                        raw = f"{m[0]}{m[1]}{m[2]}"
                        value = _parse_number(raw)
                        if value is not None:
                            results[field] = value
                            break
                    break

    return results


def parse_pdf(file_path: str) -> dict:
    """Extract financial data from a PDF report."""
    try:
        text = extract_text(file_path)
        return _extract_from_text(text)
    except Exception as e:
        return {"error": str(e)}


def parse_excel(file_path: str) -> dict:
    """
    Extract financial data from an Excel file.
    Reads all sheets and scans for keyword-value pairs.
    """
    results = {}
    try:
        xl = pd.ExcelFile(file_path)
        for sheet in xl.sheet_names:
            df = xl.parse(sheet, header=None)
            # Convert to string and concatenate as text, then reuse text extractor
            text = df.to_string()
            sheet_results = _extract_from_text(text)
            for k, v in sheet_results.items():
                if k not in results:
                    results[k] = v
    except Exception as e:
        results["error"] = str(e)
    return results


def parse_csv(file_path: str) -> dict:
    """Extract financial data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        text = df.to_string()
        return _extract_from_text(text)
    except Exception as e:
        return {"error": str(e)}


def parse_report(file_path: str, file_type: str) -> dict:
    """
    Main entry point.
    Dispatches to the correct parser based on file type.
    Returns a dict of extracted financial fields.
    """
    file_type = file_type.lower().lstrip(".")

    if file_type == "pdf":
        raw = parse_pdf(file_path)
    elif file_type in ("xlsx", "xls"):
        raw = parse_excel(file_path)
    elif file_type == "csv":
        raw = parse_csv(file_path)
    else:
        return {"error": f"Unsupported file type: {file_type}"}

    # Compute derived ratios if base values are available
    if raw.get("net_profit") and raw.get("revenue") and raw["revenue"] != 0:
        raw.setdefault("profit_margin", raw["net_profit"] / raw["revenue"])

    return raw