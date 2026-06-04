from __future__ import annotations

import geopandas as gpd
import pandas as pd

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

__all__ = [
    "XYZFormatSpec",
    "PointFileIO",
]

# Constants

#: Delimiters tried during format inference, in evaluation order.
_CANDIDATE_DELIMITERS: List[str] = [",", ";", "\t", "|"]

#: Prefix characters that identify comment lines (skipped during inference).
_COMMENT_PREFIXES: tuple[str, ...] = ("#", "//")

#: Vector file extensions handled directly by GeoPandas.
_VECTOR_EXTENSIONS: tuple[str, ...] = (".shp", ".geojson", ".gpkg")


def _is_float_token(token: str) -> bool:
    """
    Return ``True`` if *token* can be interpreted as a floating-point number.

    Parameters
    ----------
    token : str
        A single whitespace-stripped token from a line of text.

    Returns
    -------
    bool
    """
    try:
        float(token)
        return True
    except ValueError:
        return False

def _normalize_epsg(crs: Union[str, int]) -> str:
    """
    Normalize a CRS value to a canonical ``"EPSG:XXXX"`` string.

    Accepts integer EPSG codes and string representations in any of the
    forms ``4326``, ``"4326"``, ``"EPSG:4326"`` or ``"epsg:4326"``.
    The returned string is always upper-case and accepted by both
    GeoPandas and pyproj.

    This helper is also imported by ``crs_tools`` so that CRS
    normalization is handled in a single place across the subpackage.

    Parameters
    ----------
    crs : str or int
        EPSG code to normalize.

    Returns
    -------
    str
        Canonical ``"EPSG:XXXX"`` string.

    Raises
    ------
    ValueError
        If *crs* is a string that cannot be interpreted as an EPSG code.
    """
    if isinstance(crs, int):
        return f"EPSG:{crs}"

    cleaned = crs.strip().upper()
    if cleaned.startswith("EPSG:"):
        return cleaned

    try:
        return f"EPSG:{int(cleaned)}"
    except ValueError:
        raise ValueError(
            f"Cannot interpret '{crs}' as an EPSG code. "
            "Expected an integer or a string like '4326' or 'EPSG:4326'."
        )

def _infer_format(
    file_path: Union[str, Path],
    sample_size: int = 50,
) -> "XYZFormatSpec":
    """
    Automatically detect the format of an XYZ file from a sample of lines.

    The function reads up to *sample_size* non-empty, non-comment lines
    and attempts to detect:

    * The column delimiter (comma, semicolon, tab, pipe, or whitespace).
    * Whether the first data line is a header or numeric data.
    * Column names when a header row is detected.

    When the detection is inconclusive the function returns the default
    :class:`XYZFormatSpec` (space-delimited, no header, columns ``x/y/z``).

    Parameters
    ----------
    file_path : str or pathlib.Path
        Path to the XYZ-like text file.
    sample_size : int, optional
        Maximum number of lines to read during detection.

    Returns
    -------
    XYZFormatSpec
        Detected format specification.

    Notes
    -----
    This function is intentionally conservative.  When in doubt it
    returns safe defaults rather than an incorrect guess.  Pass an
    explicit :class:`XYZFormatSpec` to :class:`PointFileIO` whenever
    the file format is known in advance.
    """
    sample_lines: List[str] = []

    with open(file_path, "r", encoding="utf-8", errors="ignore") as fh:
        for raw_line in fh:
            stripped = raw_line.strip()
            if not stripped or stripped.startswith(_COMMENT_PREFIXES):
                continue
            sample_lines.append(stripped)
            if len(sample_lines) >= sample_size:
                break

    if not sample_lines:
        return XYZFormatSpec()

    # --- delimiter detection ------------------------------------------------
    delim_scores: Dict[str, int] = {d: 0 for d in _CANDIDATE_DELIMITERS}
    for line in sample_lines:
        for delim in _CANDIDATE_DELIMITERS:
            delim_scores[delim] += line.count(delim)

    best_delim = max(_CANDIDATE_DELIMITERS, key=lambda d: delim_scores[d])
    delimiter = best_delim if delim_scores[best_delim] > 0 else " "

    # --- header detection ---------------------------------------------------
    first_line = sample_lines[0]
    tokens = first_line.split() if delimiter == " " else first_line.split(delimiter)
    has_header = any(not _is_float_token(tok) for tok in tokens)

    if has_header:
        x_col = tokens[0] if len(tokens) > 0 else "x"
        y_col = tokens[1] if len(tokens) > 1 else "y"
        z_col = tokens[2] if len(tokens) > 2 else "z"
        return XYZFormatSpec(
            delimiter=delimiter,
            has_header=True,
            x_column=x_col,
            y_column=y_col,
            z_column=z_col,
        )

    return XYZFormatSpec(delimiter=delimiter, has_header=False)

@dataclass
class XYZFormatSpec:
    """
    Descriptor for the on-disk layout of an XYZ point file.

    All reading and writing operations in :class:`PointFileIO` accept an
    ``XYZFormatSpec`` so that format details are declared once and reused
    consistently across I/O calls.
    """

    #: Column separator. Use ``" "`` for whitespace, ``","`` for CSV,
    #: ``";"`` for semicolon-separated, or ``"\\t"`` for tab-separated.
    delimiter: str = " "

    #: ``True`` if the first non-comment line contains column names.
    #: ``False`` if the file starts directly with data.
    has_header: bool = False

    #: Name of the column that stores the X coordinate.
    x_column: str = "x"

    #: Name of the column that stores the Y coordinate.
    y_column: str = "y"

    #: Name of the column that stores the Z value (elevation or depth).
    z_column: str = "z"

    #: File encoding passed to :func:`open` and :func:`pandas.read_csv`.
    encoding: str = "utf-8"

    def column_order(self) -> List[str]:
        """
        Return ``[x_column, y_column, z_column]`` in write order.

        Returns
        -------
        list of str
            The three coordinate column names in the canonical x→y→z order.
        """
        return [self.x_column, self.y_column, self.z_column]

class PointFileIO:
    """
    Read and write point cloud data from XYZ or vector files.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the input or output file.
    format_spec : XYZFormatSpec, optional
        Layout descriptor for XYZ files.  When ``None`` the format is
        detected automatically from the file content.  Ignored for
        vector file formats.
    crs : str or int, optional
        CRS to assign when loading XYZ files, e.g. ``9377`` or
        ``"EPSG:9377"``.  For vector files the CRS is read from the
        file metadata.  When ``None`` no CRS is assigned.

    Notes
    -----
    The supported file formats are:

    - XYZ text files (``.xyz``, ``.txt``): plain-text tables with X, Y, Z
      columns separated by whitespace, comma, semicolon, tab or pipe.
    - Vector files (``.shp``, ``.geojson``, ``.gpkg``): point layers
      supported by GeoPandas.
    """

    def __init__(
        self,
        path: Union[str, Path],
        format_spec: Optional[XYZFormatSpec] = None,
        crs: Optional[Union[str, int]] = None,
    ) -> None:
        self.path = Path(path)
        self.crs = _normalize_epsg(crs) if crs is not None else None

        # Infer format only for XYZ files; vector files do not need it
        if format_spec is not None:
            self.format_spec = format_spec
        elif self.path.suffix.lower() not in _VECTOR_EXTENSIONS and self.path.exists():
            self.format_spec = _infer_format(self.path)
        else:
            self.format_spec = XYZFormatSpec()

    def read(self) -> pd.DataFrame:
        """
        Read the file into a :class:`pandas.DataFrame`.

        For XYZ files, columns are parsed according to :attr:`format_spec`.
        For vector files, all attribute columns are returned and the
        geometry column is dropped.

        Returns
        -------
        pandas.DataFrame
            Point data with at least ``x``, ``y`` and ``z`` columns.

        Raises
        ------
        ImportError
            If a vector file is requested but GeoPandas is not installed.
        """
        if self.path.suffix.lower() in _VECTOR_EXTENSIONS:
            return self._read_vector()
        return self._read_xyz()

    def read_as_geodataframe(self) -> "gpd.GeoDataFrame":
        """
        Read the file and return a :class:`geopandas.GeoDataFrame`.

        For XYZ files, Point geometries are built from the X and Y columns
        and the CRS is assigned from :attr:`crs`.  For vector files, the
        GeoDataFrame is returned as loaded by GeoPandas with its embedded
        CRS.

        Returns
        -------
        geopandas.GeoDataFrame
            Point GeoDataFrame with geometry built from X and Y.

        Raises
        ------
        ImportError
            If GeoPandas is not installed.
        ValueError
            If loading an XYZ file and :attr:`crs` is ``None``.
        """
        if self.path.suffix.lower() in _VECTOR_EXTENSIONS:
            return gpd.read_file(self.path)

        if self.crs is None:
            raise ValueError(
                "crs must be provided to load an XYZ file as a GeoDataFrame. "
                "Pass crs when constructing PointFileIO."
            )

        df = self._read_xyz()
        return gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(
                df[self.format_spec.x_column],
                df[self.format_spec.y_column],
            ),
            crs=self.crs,
        )

    def write(
        self,
        df: pd.DataFrame,
        float_format: str = "%.3f",
        include_header: Optional[bool] = None,
    ) -> None:
        """
        Write a :class:`pandas.DataFrame` to an XYZ plain-text file.

        The output path is taken from :attr:`path` set at construction.

        Parameters
        ----------
        df : pandas.DataFrame
            Source data.  Must contain the columns declared in
            :attr:`format_spec`.
        float_format : str, optional
            ``printf``-style format string for floating-point values.
        include_header : bool, optional
            Override whether a header row is written.  When ``None`` the
            value is taken from :attr:`format_spec`.

        Raises
        ------
        ValueError
            If any required column is absent from *df*.
        """
        if include_header is None:
            include_header = self.format_spec.has_header

        missing = [
            c for c in self.format_spec.column_order() if c not in df.columns
        ]
        if missing:
            raise ValueError(
                f"DataFrame is missing the following columns required by "
                f"XYZFormatSpec: {missing}"
            )

        df[self.format_spec.column_order()].to_csv(
            self.path,
            sep=self.format_spec.delimiter,
            header=include_header,
            index=False,
            float_format=float_format,
            encoding=self.format_spec.encoding,
        )

    def write_from_geodataframe(
        self,
        gdf: "gpd.GeoDataFrame",
        z_column: str = "z",
        float_format: str = "%.3f",
        include_header: Optional[bool] = None,
    ) -> None:
        """
        Write a :class:`geopandas.GeoDataFrame` to an XYZ plain-text file.

        Coordinates are extracted from the geometry column (X and Y) and
        from *z_column* (Z).  The output path is taken from :attr:`path`
        set at construction.

        Parameters
        ----------
        gdf : geopandas.GeoDataFrame
            Source GeoDataFrame with Point geometries.
        z_column : str, optional
            Name of the attribute column holding elevation or depth values.
        float_format : str, optional
            ``printf``-style format string for floating-point values.
        include_header : bool, optional
            Override whether a header row is written.  When ``None`` the
            value is taken from :attr:`format_spec`.

        Raises
        ------
        ImportError
            If GeoPandas is not installed.
        ValueError
            If *z_column* is not present in *gdf*.
        """

        if z_column not in gdf.columns:
            raise ValueError(
                f"Column '{z_column}' not found in GeoDataFrame. "
                f"Available columns: {list(gdf.columns)}"
            )

        df = pd.DataFrame({
            self.format_spec.x_column: gdf.geometry.x,
            self.format_spec.y_column: gdf.geometry.y,
            self.format_spec.z_column: gdf[z_column].to_numpy(),
        })

        self.write(df, float_format=float_format, include_header=include_header)

    def _read_xyz(self) -> pd.DataFrame:
        """Read an XYZ text file using :attr:`format_spec`."""
        header = 0 if self.format_spec.has_header else None
        names = (
            None if self.format_spec.has_header
            else self.format_spec.column_order()
        )
        return pd.read_csv(
            self.path,
            sep=self.format_spec.delimiter,
            header=header,
            names=names,
            encoding=self.format_spec.encoding,
            engine="python",
        )

    def _read_vector(self) -> pd.DataFrame:
        """
        Read a vector file via GeoPandas and return a plain DataFrame.

        The geometry column is dropped; X and Y are extracted from it
        and added as explicit columns.
        """
        gdf = gpd.read_file(self.path)
        df = pd.DataFrame(gdf.drop(columns="geometry"))
        df["x"] = gdf.geometry.x
        df["y"] = gdf.geometry.y
        return df