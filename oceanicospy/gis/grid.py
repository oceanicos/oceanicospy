from __future__ import annotations

import numpy as np
import pandas as pd
import shapefile

from pathlib import Path
from typing import Optional, Union

__all__ = ["Grid"]

class Grid:
    """
    2D spatial grid for numerical model domains.

    Use :meth:`from_coordinates` or :meth:`from_shapefile` to construct
    an instance.  Direct instantiation via ``__init__`` is intentionally
    not supported.

    Attributes
    ----------
    dx : float
        Grid spacing in the x direction.
    dy : float
        Grid spacing in the y direction.
    xvar : bool
        Whether variable x-spacing is used.
    crs : str or None
        CRS string for metadata purposes (e.g. ``"EPSG:4326"``).
        No coordinate transformation is performed.
    """

    def __init__(self) -> None:
        # Not for direct use.  Call from_coordinates or from_shapefile.
        self.dx: float = 0.0
        self.dy: float = 0.0
        self.xvar: bool = False
        self.crs: Optional[str] = None

        self.x_start: Optional[float] = None
        self.x_end: Optional[float] = None
        self.y_start: Optional[float] = None
        self.y_end: Optional[float] = None

        self.shapefile_path: Optional[Path] = None

        self.x_1d: Optional[np.ndarray] = None
        self.y_1d: Optional[np.ndarray] = None
        self.x_2d: Optional[np.ndarray] = None
        self.y_2d: Optional[np.ndarray] = None

    @classmethod
    def from_coordinates(
        cls,
        x_start: float,
        x_end: float,
        y_start: float,
        y_end: float,
        dx: float,
        dy: Optional[float] = None,
        xvar: bool = False,
        crs: Optional[str] = None,
    ) -> "Grid":
        """
        Build a uniform grid from explicit bounding-box coordinates.

        Parameters
        ----------
        x_start : float
            Western (minimum x) boundary of the domain.
        x_end : float
            Eastern (maximum x) boundary of the domain.
        y_start : float
            Southern (minimum y) boundary of the domain.
        y_end : float
            Northern (maximum y) boundary of the domain.
        dx : float
            Grid spacing in the x direction.  Must be strictly positive.
        dy : float
            Grid spacing in the y direction.  Must be strictly positive.
        xvar : bool, optional
            If ``True``, variable x-spacing will be used (not yet
            implemented).  Defaults to ``False``.
        crs : str, optional
            CRS string for metadata purposes (e.g. ``"EPSG:4326"``).

        Returns
        -------
        Grid

        Raises
        ------
        ValueError
            If *dx* or *dy* are not strictly positive, or if *x_end* <=
            *x_start* or *y_end* <= *y_start*.
        """
        if dx <= 0.0:
            raise ValueError(f"dx must be strictly positive, got {dx}.")
        
        if dy is None:
            dy = dx
        else:
            if dy <= 0.0:
                raise ValueError(f"dy must be strictly positive, got {dy}.")
            
        if x_end <= x_start:
            raise ValueError(
                f"x_end ({x_end}) must be greater than x_start ({x_start})."
            )
        if y_end <= y_start:
            raise ValueError(
                f"y_end ({y_end}) must be greater than y_start ({y_start})."
            )

        obj = cls()
        obj.x_start = x_start
        obj.x_end = x_end
        obj.y_start = y_start
        obj.y_end = y_end
        obj.dx = dx
        obj.dy = dy
        obj.xvar = xvar
        obj.crs = crs
        obj._build_uniform(x_start, x_end, y_start, y_end, dx, dy)
        return obj

    @classmethod
    def from_shapefile(
        cls,
        shapefile_path: Union[str, Path],
        dx: float,
        dy: Optional[float] = None,
        xvar: bool = False,
        crs: Optional[str] = None,
    ) -> "Grid":
        """
        Build a grid from the bounding box of a shapefile.

        The first shape in the shapefile defines the domain extent.
        Grid coordinates are expressed relative to the lower-left corner
        of the bounding box so that the origin is ``(0, 0)``.

        Parameters
        ----------
        shapefile_path : str or Path
            Path to the ``.shp`` file.
        dx : float
            Grid spacing in the x direction.  Must be strictly positive.
        dy : float
            Grid spacing in the y direction.  Must be strictly positive.
        xvar : bool, optional
            If ``True``, variable x-spacing will be used (not yet
            implemented).  Defaults to ``False``.
        crs : str, optional
            CRS string for metadata purposes (e.g. ``"EPSG:4326"``).

        Returns
        -------
        Grid

        Raises
        ------
        FileNotFoundError
            If *shapefile_path* does not exist.
        ImportError
            If the ``pyshp`` package is not installed.
        ValueError
            If *dx* or *dy* are not strictly positive.

        Notes
        -----
        The shapefile is assumed to be in Cartesian coordinates (in meters). No coordinate transformation is performed.  
        """
        if dx <= 0.0:
            raise ValueError(f"dx must be strictly positive, got {dx}.")

        if dy is None:
            dy = dx
        else:
            if dy <= 0.0:
                raise ValueError(f"dy must be strictly positive, got {dy}.")

        obj = cls()
        obj.dx = dx
        obj.dy = dy
        obj.xvar = xvar
        obj.crs = crs
        obj.shapefile_path = Path(shapefile_path)
        obj._build_from_shapefile(Path(shapefile_path), dx, dy, xvar)
        return obj

    @property
    def nx(self) -> int:
        """
        Number of grid points in the x direction.

        Returns
        -------
        int
            Length of the 1-D x coordinate array.

        Raises
        ------
        ValueError
            If the grid has not been built yet.
        """
        if self.x_1d is None:
            raise ValueError("x-component grid array is not available.")
        return len(self.x_1d)-1

    @property
    def ny(self) -> int:
        """
        Number of grid points in the y direction.

        Returns
        -------
        int
            Length of the 1-D y coordinate array.

        Raises
        ------
        ValueError
            If the grid has not been built yet.
        """
        if self.y_1d is None:
            raise ValueError("y-component grid array is not available.")
        return len(self.y_1d)-1

    @property
    def relative_x_coordinates(self) -> pd.DataFrame:
        """
        X coordinates of every grid point relative to the domain origin.

        Each value is the distance in the x direction from *x_start* to
        the corresponding grid point, so the first column always starts
        at ``0.0``.

        Returns
        -------
        pandas.DataFrame
            Shape ``(ny, nx)``.  Column labels are integer indices.

        Raises
        ------
        ValueError
            If the grid has not been built yet.
        """
        if self.x_2d is None:
            raise ValueError("x-component grid array is not available.")
        return pd.DataFrame(self.x_2d - self.x_start)

    @property
    def relative_y_coordinates(self) -> pd.DataFrame:
        """
        Y coordinates of every grid point relative to the domain origin.

        Each value is the distance in the y direction from *y_start* to
        the corresponding grid point, so the first row always starts
        at ``0.0``.

        Returns
        -------
        pandas.DataFrame
            Shape ``(ny, nx)``.  Column labels are integer indices.

        Raises
        ------
        ValueError
            If the grid has not been built yet.
        """
        if self.y_2d is None:
            raise ValueError("y-component grid array is not available.")
        return pd.DataFrame(self.y_2d - self.y_start)

    @property
    def absolute_x_coordinates(self) -> pd.DataFrame:
        """
        Absolute x coordinates of every grid point in the original CRS.

        For grids built from a shapefile the values are in the same
        projected units as the shapefile (typically metres).  For grids
        built from explicit coordinates they match the values passed to
        :meth:`from_coordinates`.

        Returns
        -------
        pandas.DataFrame
            Shape ``(ny, nx)``.  Column labels are integer indices.

        Raises
        ------
        ValueError
            If the grid has not been built yet.
        """
        if self.x_2d is None:
            raise ValueError("x-component grid array is not available.")
        if self.x_start is None:
            raise ValueError("x_start must be set to compute absolute coordinates.")
        return pd.DataFrame(self.x_2d)

    @property
    def absolute_y_coordinates(self) -> pd.DataFrame:
        """
        Absolute y coordinates of every grid point in the original CRS.

        For grids built from a shapefile the values are in the same
        projected units as the shapefile (typically metres).  For grids
        built from explicit coordinates they match the values passed to
        :meth:`from_coordinates`.

        Returns
        -------
        pandas.DataFrame
            Shape ``(ny, nx)``.  Column labels are integer indices.

        Raises
        ------
        ValueError
            If the grid has not been built yet.
        """
        if self.y_2d is None:
            raise ValueError("y-component grid array is not available.")
        if self.y_start is None:
            raise ValueError("y_start must be set to compute absolute coordinates.")
        return pd.DataFrame(self.y_2d)

    def _build_uniform(
        self,
        x_start: float,
        x_end: float,
        y_start: float,
        y_end: float,
        dx: float,
        dy: float,
    ) -> None:
        self.x_1d = np.arange(x_start, x_end + dx, dx)
        self.y_1d = np.arange(y_start, y_end + dy, dy)
        self.x_2d, self.y_2d = np.meshgrid(self.x_1d, self.y_1d)

    def _build_from_shapefile(
        self,
        shapefile_path: Path,
        dx: float,
        dy: float,
        xvar: bool,
    ) -> None:
        if not shapefile_path.exists():
            raise FileNotFoundError(
                f"Shapefile not found at '{shapefile_path}'. "
                "Check input folder and name."
            )

        sf = shapefile.Reader(str(shapefile_path))
        shape = sf.shapes()[0]

        # assuming shapefile is in cartesian coordinates.
        min_easting, min_northing, max_easting, max_northing = shape.bbox

        self.x_start = min_easting
        self.x_end = max_easting
        self.y_start = min_northing
        self.y_end = max_northing

        if not xvar:
            self._build_uniform(min_easting, max_easting, min_northing, max_northing, dx, dy)
        else:
            raise NotImplementedError("Variable x-spacing is not yet implemented.")