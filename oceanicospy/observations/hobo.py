from __future__ import annotations

import glob
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd


@dataclass
class CleaningRules:
    """
    Container for simple quality-control thresholds.

    Values are inclusive ranges; set to ``None`` to disable the bound.

    Attributes
    ----------
    t_min, t_max : float or None
        Acceptable temperature range in °C. Default −2 to 45.
    c_min, c_max : float or None
        Acceptable conductivity range in µS/cm. Default 0 to 1 000 000.
    """

    t_min: Optional[float] = -2.0
    t_max: Optional[float] = 45.0
    c_min: Optional[float] = 0.0
    c_max: Optional[float] = 1_000_000.0


class HOBOBase(ABC):
    """
    Abstract base class for HOBO data logger readers.

    Provides a common interface for loading, standardizing, and cleaning
    HOBO CSV exports from a directory. Subclasses implement format-specific
    column detection for Temperature Logger (TL) and Conductivity Logger
    (CL) files by overriding :attr:`_file_prefix` and
    :meth:`_standardize_columns`.

    Parameters
    ----------
    directory_path : str
        Path to the directory containing HOBO CSV files.
    start_dt : datetime-like, optional
        Lower bound for trimming records (inclusive).
    end_dt : datetime-like, optional
        Upper bound for trimming records (inclusive).
    rules : CleaningRules, optional
        QC thresholds applied in :meth:`get_clean_records`.
        Defaults to the standard thresholds defined in :class:`CleaningRules`.
    encoding_main : str, optional
        Primary encoding for reading CSV files. Default ``'utf-8-sig'``
        handles BOM headers produced by HOBOware.
    encoding_fallback : str, optional
        Fallback encoding used if the primary one fails. Default ``'latin-1'``.

    Notes
    -----
    **Development history**

    - 01-Jun-2025 : Origination — Daniela Rosero
    - 24-Mar-2026 : Refactored into abstract + concrete classes — Franklin Ayala
    """

    def __init__(
        self,
        filepath: str,
        start_dt=None,
        end_dt=None,
        rules: Optional[CleaningRules] = None,
        encoding_main: str = "utf-8-sig",
        encoding_fallback: str = "latin-1",
    ) -> None:
        self.filepath = filepath
        self.rules = rules or CleaningRules()
        self.encoding_main = encoding_main
        self.encoding_fallback = encoding_fallback
        self.start_dt = pd.to_datetime(start_dt) if start_dt is not None else None
        self.end_dt = pd.to_datetime(end_dt) if end_dt is not None else None

    @property
    @abstractmethod
    def _file_prefix(self) -> str:
        """
        Filename prefix used to filter matching files (e.g. ``'TL'`` or ``'CL'``).

        Only CSV files whose basenames start with this prefix (case-insensitive)
        are loaded by :meth:`get_raw_records`.
        """
        pass

    @abstractmethod
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect and rename instrument-specific columns to the standard schema.

        Subclasses must implement this to map raw HOBOware column names to:
        ``date``, ``temperature[C]``, ``conductivity[uS/cm]`` (CL only),
        and ``seq`` (if present).

        Parameters
        ----------
        df : pandas.DataFrame
            Raw DataFrame produced by :meth:`_read_single_file` before any
            renaming (title row already skipped).

        Returns
        -------
        pandas.DataFrame
            DataFrame containing only the columns listed above, with those
            names applied.
        """
        pass

    def get_raw_records(self) -> pd.DataFrame:
        """
        Load, standardize, and concatenate all matching HOBO CSV files.

        Files are filtered by :attr:`_file_prefix`, read individually via
        :meth:`_read_single_file`, merged, sorted chronologically,
        deduplicated by timestamp, and optionally trimmed to
        [``start_dt``, ``end_dt``].

        Returns
        -------
        pandas.DataFrame
            Combined records with standardized columns.

        Raises
        ------
        FileNotFoundError
            If no CSV files matching the prefix are found in ``directory_path``.
        """

        df = self._load_raw_dataframe()
        return df

    def get_clean_records(self) -> pd.DataFrame:
        """
        Return QC-filtered records.

        Calls :meth:`get_raw_records` and applies the thresholds in
        :attr:`rules` — out-of-range values are replaced with ``NaN`` and
        rows where all measured variables are ``NaN`` are dropped.

        Returns
        -------
        pandas.DataFrame
            Cleaned DataFrame ready for downstream analysis.
        """
        df = self._load_raw_dataframe()
        df = self._standardize_columns(df)
        df = self._parse_dates_and_trim(df)
        # df = self._apply_qc(df)
        return df

    def _load_raw_dataframe(self) -> pd.DataFrame:
        """
        Read a single HOBO CSV (skip the HOBOware title row), then delegate
        column renaming to :meth:`_standardize_columns` and parse types.
        """
        try:
            df = pd.read_csv(self.filepath, skiprows=1, encoding=self.encoding_main)
        except Exception:
            df = pd.read_csv(self.filepath, skiprows=1, encoding=self.encoding_fallback)

        return df

    def _parse_dates_and_trim(self, df: pd.DataFrame) -> pd.DataFrame:
        """Trim records to [``start_dt``, ``end_dt``] when either bound is set."""
        if self.start_dt is None and self.end_dt is None:
            return df
        return df[self.start_dt:self.end_dt]

    def _apply_qc(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply QC thresholds and drop rows where all measurement columns are NaN."""
        if df.empty:
            return df

        value_cols = [c for c in ("temperature[C]", "conductivity[uS/cm]") if c in df.columns]
        if value_cols:
            df = df.dropna(subset=value_cols, how="all").copy()

        if "temperature[C]" in df.columns:
            if self.rules.t_min is not None:
                df.loc[df["temperature[C]"] < self.rules.t_min, "temperature[C]"] = pd.NA
            if self.rules.t_max is not None:
                df.loc[df["temperature[C]"] > self.rules.t_max, "temperature[C]"] = pd.NA

        if "conductivity[uS/cm]" in df.columns:
            if self.rules.c_min is not None:
                df.loc[df["conductivity[uS/cm]"] < self.rules.c_min, "conductivity[uS/cm]"] = pd.NA
            if self.rules.c_max is not None:
                df.loc[df["conductivity[uS/cm]"] > self.rules.c_max, "conductivity[uS/cm]"] = pd.NA

        for col in value_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        return self._order_columns(df).reset_index(drop=True)

    @staticmethod
    def _order_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Return a copy with deterministic column ordering."""
        if df.empty:
            return df
        ordered = ["date"]
        for col in ("temperature[C]", "conductivity[uS/cm]", "seq"):
            if col in df.columns:
                ordered.append(col)
        return df[ordered].copy()

    @staticmethod
    def _normalize(text: str) -> str:
        """
        Normalize a header string for robust keyword matching.

        Strips encoding artefacts (``Â``, ``µ`` variants), reduces to ASCII,
        and collapses whitespace.
        """
        if text is None:
            return text
        t = (
            text.replace("Â", "")
                .replace("µ", "u")
                .replace("Î¼", "u")
                .encode("latin-1", "ignore")
                .decode("latin-1")
        )
        t = t.encode("ascii", "ignore").decode("ascii")
        return re.sub(r"\s+", " ", t).strip()


class HOBO_Temp(HOBOBase):
    """
    Reader for HOBO Temperature Logger (TL) CSV files.

    Loads all ``TL*.csv`` files from ``directory_path``, parses temperature
    and (optionally) sequence columns, and exposes them through the standard
    :meth:`get_raw_records` / :meth:`get_clean_records` interface.

    Parameters
    ----------
    directory_path : str
        Path to the directory containing TL CSV files.
    start_dt, end_dt, rules, encoding_main, encoding_fallback
        See :class:`HOBOBase`.

    """

    @property
    def _file_prefix(self) -> str:
        return "TL"

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect and rename TL-specific columns to ``date``, ``temperature[C]``,
        and ``seq``.

        Parameters
        ----------
        df : pandas.DataFrame
            Raw DataFrame (title row already skipped).

        Returns
        -------
        pandas.DataFrame
            DataFrame with only the relevant columns under standard names.
        
        Notes
        -----
        Date is recorded in local time so no further timezone conversion is needed. The date column is parsed as datetime and set as index, sorted in chronological order.
        """
        norm = [self._normalize(c) for c in df.columns]
        cols = df.columns.tolist()

        dt_idx = next(
            (i for i, c in enumerate(norm)
             if "Fecha Tiempo" in c or c.lower().startswith("date") or "Tiempo" in c),
            1,
        )
        seq_idx = next((i for i, c in enumerate(norm) if c.startswith("N")), None)
        temp_idx = next(
            (i for i, c in enumerate(norm) if "Temp" in c and ("C" in c or "degC" in c)),
            None,
        )

        keep = [cols[dt_idx]]
        rename = {cols[dt_idx]: "date"}
        if seq_idx is not None:
            keep.insert(0, cols[seq_idx])
            rename[cols[seq_idx]] = "seq"
        if temp_idx is not None:
            keep.append(cols[temp_idx])
            rename[cols[temp_idx]] = "temperature[C]"

        df = df[keep].rename(columns=rename)
        df['date'] = pd.to_datetime(df['date'],
                                    format = '%m/%d/%y %I:%M:%S %p',
                                    errors='coerce')
        df = df.set_index('date').sort_index()

        return df


class HOBO_TempCond(HOBOBase):
    """
    Reader for HOBO Conductivity Logger (CL) CSV files.

    Loads all ``CL*.csv`` files from ``directory_path``, parses temperature,
    conductivity, and (optionally) sequence columns, and exposes them through
    the standard :meth:`get_raw_records` / :meth:`get_clean_records` interface.

    Parameters
    ----------
    directory_path : str
        Path to the directory containing CL CSV files.
    start_dt, end_dt, rules, encoding_main, encoding_fallback
        See :class:`HOBOBase`.
    """

    @property
    def _file_prefix(self) -> str:
        return "CL"

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect and rename CL-specific columns to ``date``, ``temperature[C]``,
        ``conductivity[uS/cm]``, and ``seq``.

        Parameters
        ----------
        df : pandas.DataFrame
            Raw DataFrame (title row already skipped).

        Returns
        -------
        pandas.DataFrame
            DataFrame with only the relevant columns under standard names.
        """
        norm = [self._normalize(c) for c in df.columns]
        cols = df.columns.tolist()

        dt_idx = next(
            (i for i, c in enumerate(norm)
             if "Fecha Tiempo" in c or c.lower().startswith("date") or "Tiempo" in c),
            1,
        )
        seq_idx = next((i for i, c in enumerate(norm) if c.startswith("N")), None)
        temp_idx = next(
            (i for i, c in enumerate(norm) if "Temp" in c and ("C" in c or "degC" in c)),
            None,
        )
        cond_idx = next(
            (i for i, c in enumerate(norm)
             if "S/cm" in c and ("Rango" in c or "Conductivity" in c)),
            None,
        )

        keep = [cols[dt_idx]]
        rename = {cols[dt_idx]: "date"}
        if seq_idx is not None:
            keep.insert(0, cols[seq_idx])
            rename[cols[seq_idx]] = "seq"
        if temp_idx is not None:
            keep.append(cols[temp_idx])
            rename[cols[temp_idx]] = "temperature[C]"
        if cond_idx is not None:
            keep.append(cols[cond_idx])
            rename[cols[cond_idx]] = "conductivity[uS/cm]"

        df = df[keep].rename(columns=rename)
        df['date'] = pd.to_datetime(df['date'], 
                                    format = '%m/%d/%y %I:%M:%S %p',
                                    errors='coerce')
        df = df.set_index('date').sort_index()

        return df