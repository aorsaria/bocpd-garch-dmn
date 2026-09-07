"""Prepared-data contract for the dissertation experiments.

This module contains only the fixed study boundaries and futures universe used
by the BOCPD and deep-momentum experiments.  It intentionally contains no raw
data reconstruction, cleaning, repair, or archive-writing logic.  The command
line programs expect a previously prepared ZIP archive whose members follow
the contract documented in ``README.md``.
"""

from __future__ import annotations

HISTORY_START = "1987-01-01"
"""Earliest permitted date, used only to initialise lagged model features."""

STUDY_START = "1990-01-01"
"""First date that may enter estimation or evaluation samples."""

END_DATE = "2024-12-31"
"""Final date of the dissertation sample."""

FIRST_TEST_START = "1995-01-01"
"""Start of the first out-of-sample deep-momentum test window."""

CSV_COLUMNS = ("date", "open", "high", "low", "close", "vol", "oi")
"""Column names, in order, for each headerless CSV archive member."""

# Each value is (asset class, first eligible out-of-sample test year).  The
# order is fixed because it also defines deterministic portfolio construction.
UNIVERSE: dict[str, tuple[str, int]] = {
    # Commodities
    "CC": ("CM", 1995),
    "DA": ("CM", 2000),
    "GI": ("CM", 1995),
    "JO": ("CM", 1995),
    "KC": ("CM", 1995),
    "KW": ("CM", 1995),
    "LB": ("CM", 1995),
    "NR": ("CM", 1995),
    "SB": ("CM", 1995),
    "ZA": ("CM", 1995),
    "ZC": ("CM", 1995),
    "ZF": ("CM", 1995),
    "ZG": ("CM", 1995),
    "ZH": ("CM", 1995),
    "ZI": ("CM", 1995),
    "ZK": ("CM", 1995),
    "ZL": ("CM", 1995),
    "ZN": ("CM", 1995),
    "ZO": ("CM", 1995),
    "ZP": ("CM", 1995),
    "ZR": ("CM", 1995),
    "ZT": ("CM", 1995),
    "ZU": ("CM", 1995),
    "ZW": ("CM", 1995),
    "ZZ": ("CM", 1995),
    # Equity indices
    "CA": ("EQ", 2000),
    "EN": ("EQ", 2005),
    "ER": ("EQ", 2005),
    "ES": ("EQ", 2000),
    "LX": ("EQ", 1995),
    "MD": ("EQ", 1995),
    "SC": ("EQ", 2000),
    "SP": ("EQ", 1995),
    "XU": ("EQ", 2005),
    "XX": ("EQ", 2005),
    "YM": ("EQ", 2005),
    "NK": ("EQ", 1995),
    # Fixed income
    "DT": ("FI", 1995),
    "FB": ("FI", 1995),
    "TY": ("FI", 1995),
    "UB": ("FI", 2005),
    "US": ("FI", 1995),
    # Foreign exchange
    "AN": ("FX", 1995),
    "BN": ("FX", 1995),
    "CN": ("FX", 1995),
    "DX": ("FX", 1995),
    "FN": ("FX", 1995),
    "JN": ("FX", 1995),
    "MP": ("FX", 2000),
    "SN": ("FX", 1995),
}

TICKERS = tuple(UNIVERSE)
"""Ordered tuple of the 50 contract identifiers."""

CLASS_ORDER = ("CM", "EQ", "FI", "FX")
"""Canonical asset-class order used in tabular outputs."""
