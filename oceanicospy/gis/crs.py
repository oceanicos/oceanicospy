from __future__ import annotations
 
import geopandas as gpd

from pathlib import Path
from typing import Optional, Union
 
from .point_io import (
    XYZFormatSpec,
    PointFileIO,
    _normalize_epsg,
)
 
__all__ = [
    "PointFileReprojector",
    "reproject_points",
]
  
class PointFileReprojector:
    """
    Load point data from a vector or XYZ file, reproject it, and export
    the result as an XYZ plain-text file.
 
    All file I/O is delegated to :class:`~point_io.PointFileIO`.
  
    Parameters
    ----------
    input_path : str or pathlib.Path
        Path to the input file.
    z_column : str, optional
        Name of the attribute column that holds elevation or depth values.
        For XYZ files this must match the column name in *format_spec*.
    source_epsg : str or int, optional
        CRS of the input data (e.g. ``4326`` or ``"EPSG:4326"``).  Required
        for XYZ files.  For vector files it overrides the file metadata
        only when no CRS is embedded.
    format_spec : XYZFormatSpec, optional
        Layout descriptor used for both reading XYZ input files and writing
        XYZ output files.  When ``None`` a default
        :class:`~point_io.XYZFormatSpec` is used.  Ignored for vector
        input formats.
  
    Raises
    ------
    ValueError
        If *source_epsg* is not provided for an XYZ input file.
    ValueError
        If *z_column* is not found in the loaded data.
    
    Notes
    -----
    The supported input formats
    
    - Vector files (``.shp``, ``.geojson``, ``.gpkg``): CRS is read from
      file metadata when available.  Pass *source_epsg* only when the
      file lacks a CRS definition.
    - XYZ text files (``.xyz``, ``.txt``): *source_epsg* is required
      because plain-text files carry no CRS metadata.
    """
 
    def __init__(
        self,
        input_path: Union[str, Path],
        z_column: str = "z",
        source_epsg: Optional[Union[str, int]] = None,
        format_spec: Optional[XYZFormatSpec] = None,
    ) -> None:
        self.input_path = Path(input_path)
        self.z_column = z_column
        self.source_epsg = source_epsg
        self.format_spec = format_spec or XYZFormatSpec()
        self.gdf: gpd.GeoDataFrame = self._load()
  
    @property
    def crs(self):
        """
        CRS of the internal GeoDataFrame.
 
        Returns
        -------
        pyproj.CRS or None
            The current CRS as managed by GeoPandas.
        """
        return self.gdf.crs
  
    def _load(self) -> gpd.GeoDataFrame:
        """
        Load the input file into a GeoDataFrame via
        :class:`~point_io.PointFileIO`.
 
        Returns
        -------
        geopandas.GeoDataFrame
 
        Raises
        ------
        ValueError
            If *z_column* is absent from the loaded data.
        """
        gdf = PointFileIO(
            path=self.input_path,
            format_spec=self.format_spec,
            crs=self.source_epsg,
        ).read_as_geodataframe()
 
        if self.z_column not in gdf.columns:
            raise ValueError(
                f"Column '{self.z_column}' not found in "
                f"'{self.input_path.name}'. "
                f"Available columns: {list(gdf.columns)}"
            )
 
        return gdf
  
    def reproject_to_epsg(self, target_epsg: Union[str, int]) -> None:
        """
        Reproject the internal GeoDataFrame to a target CRS in place.
 
        Parameters
        ----------
        target_epsg : str or int
            Target EPSG code (e.g. ``9377`` or ``"EPSG:9377"``).
 
        Raises
        ------
        ValueError
            If the internal GeoDataFrame has no CRS defined.
        """
        if self.gdf.crs is None:
            raise ValueError(
                "Source CRS is not defined. "
                "Assign source_epsg before reprojecting."
            )
        self.gdf = self.gdf.to_crs(_normalize_epsg(target_epsg))
 
    def to_xyz(
        self,
        output_path: Union[str, Path],
        format_spec: Optional[XYZFormatSpec] = None,
        float_format: str = "%.3f",
        include_header: Optional[bool] = None,
    ) -> None:
        """
        Export the current GeoDataFrame to an XYZ plain-text file.
 
        Delegates to :meth:`~point_io.PointFileIO.write_from_geodataframe`.
        The effective format spec defaults to the one set at construction.
 
        Parameters
        ----------
        output_path : str or pathlib.Path
            Destination file path.
        format_spec : XYZFormatSpec, optional
            Layout descriptor for the output file.  When ``None`` the
            instance's *format_spec* (set at construction) is reused.
        float_format : str, optional
            ``printf``-style format string for floating-point values.
        include_header : bool, optional
            Override whether a header row is written.  When ``None`` the
            value is taken from the effective *format_spec*.
 
        Raises
        ------
        ValueError
            If *z_column* is not present in the internal GeoDataFrame.
        """
        PointFileIO(
            path=output_path,
            format_spec=format_spec or self.format_spec,
        ).write_from_geodataframe(
            gdf=self.gdf,
            z_column=self.z_column,
            float_format=float_format,
            include_header=include_header,
        )
 
 
# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------
 
def reproject_points(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    target_epsg: Union[str, int],
    source_epsg: Optional[Union[str, int]] = None,
    z_column: str = "z",
    format_spec: Optional[XYZFormatSpec] = None,
    float_format: str = "%.3f",
    include_header: Optional[bool] = None,
) -> None:
    """
    Reproject a point file from one CRS to another in a single call.
 
    Accepts both XYZ text files and vector files (``.shp``, ``.geojson``,
    ``.gpkg``) as input.  The output is always an XYZ plain-text file.
 
    This is a thin convenience wrapper around :class:`PointFileReprojector`.
    Use :class:`PointFileReprojector` directly when you need to inspect or
    modify the data between loading and exporting.
 
    Parameters
    ----------
    input_path : str or pathlib.Path
        Path to the input file (XYZ or vector).
    output_path : str or pathlib.Path
        Path to the output XYZ file.
    target_epsg : str or int
        Target CRS (e.g. ``9377`` or ``"EPSG:9377"``).
    source_epsg : str or int, optional
        CRS of the input file.  Required for XYZ files.  For vector files
        it is only needed when no CRS is embedded in the file metadata.
    z_column : str, optional
        Name of the column containing elevation or depth values.
    format_spec : XYZFormatSpec, optional
        Layout descriptor used for both reading and writing.  When
        ``None`` a default :class:`~point_io.XYZFormatSpec` is used.
    float_format : str, optional
        ``printf``-style format string for floating-point values.
    include_header : bool, optional
        Override whether a header row is written in the output file.
    """
    reprojector = PointFileReprojector(
        input_path=input_path,
        z_column=z_column,
        source_epsg=source_epsg,
        format_spec=format_spec,
    )
    reprojector.reproject_to_epsg(target_epsg)
    reprojector.to_xyz(
        output_path=output_path,
        float_format=float_format,
        include_header=include_header,
    )