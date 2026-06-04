from __future__ import annotations

import warnings
import numpy as np
import pandas as pd

from scipy.spatial import cKDTree

from .profile_axis import ProfileAxis

__all__ = ["ProfileInterpolator"]

#: Default number of nearest neighbours used for Z interpolation.
_DEFAULT_K_NEIGHBORS: int = 1

#: Default half-width of the spatial corridor filter in CRS units (metres).
_DEFAULT_CORRIDOR_WIDTH: float = 500.0

class ProfileInterpolator:
    """
    Sample Z values from a scattered XYZ point cloud along a profile axis.

    The interpolation pipeline consists of four steps:

    1. Validate that *axis* was built with
       :meth:`~profile_axis.ProfileAxis.from_coordinates`.
    2. Optionally verify CRS consistency between *axis* and *xyz*.
    3. Verify that the profile endpoints fall within the spatial domain
       of *xyz*.
    4. Pre-filter *xyz* to a corridor around the profile line, build a
       :class:`~scipy.spatial.cKDTree`, and query Z values.

    The results are accessible as two DataFrames via :attr:`profile_sz`
    and :attr:`profile_xyz`.

    Parameters
    ----------
    axis : ProfileAxis
        Sampling axis built with
        :meth:`~profile_axis.ProfileAxis.from_coordinates`.
    xyz : pandas.DataFrame or geopandas.GeoDataFrame
        Point cloud with at least columns ``"x"``, ``"y"`` and ``"z"``.
        Load with :meth:`~point_io.PointFileIO.read` or
        :meth:`~point_io.PointFileIO.read_as_geodataframe`.  Passing a
        :class:`~geopandas.GeoDataFrame` enables CRS consistency checking.
    k_neighbors : int, optional
        Number of nearest neighbours used to interpolate Z.  When
        ``k_neighbors > 1`` the interpolated value is the mean of the
        *k* nearest Z values.  Defaults to ``1`` (nearest neighbour).
    corridor_width : float, optional
        Half-width of the spatial corridor (in CRS units, typically
        metres) used to pre-filter *xyz* before building the KDTree.
        Larger values increase robustness in sparse datasets; smaller
        values reduce computational cost.  Defaults to ``500.0``.

    Raises
    ------
    ValueError
        If *axis* was built with
        :meth:`~profile_axis.ProfileAxis.from_length`.
    ValueError
        If any profile endpoint falls outside the spatial domain of *xyz*.
    ValueError
        If fewer than *k_neighbors* points remain after corridor filtering.
    """

    def __init__(
        self,
        axis: ProfileAxis,
        xyz: pd.DataFrame,
        k_neighbors: int = _DEFAULT_K_NEIGHBORS,
        corridor_width: float = _DEFAULT_CORRIDOR_WIDTH,
    ) -> None:
        self.axis = axis
        self.k_neighbors = k_neighbors
        self.corridor_width = corridor_width

        self._validate_axis()
        self._validate_crs(xyz)
        self._validate_domain(xyz)

        xyz_filtered = self._filter_corridor(xyz)
        self._z_interpolated: np.ndarray = self._interpolate(xyz_filtered)

    @property
    def profile_sz(self) -> pd.DataFrame:
        """
        Profile as cumulative distance and interpolated Z.

        Returns
        -------
        pandas.DataFrame
            DataFrame with columns:

            - ``"s"`` : cumulative distance from the profile origin [m].
            - ``"z"`` : interpolated elevation or depth at each position.
        """
        return pd.DataFrame({
            "s": self.axis._s_axis,
            "z": self._z_interpolated,
        })

    @property
    def profile_xyz(self) -> pd.DataFrame:
        """
        Profile as planar coordinates and interpolated Z.

        The column layout (``"x"``, ``"y"``, ``"z"``) matches the default
        :class:`~point_io.XYZFormatSpec`, so this DataFrame can be passed
        directly to :meth:`~point_io.PointFileIO.write` without any
        transformation.

        Returns
        -------
        pandas.DataFrame
            DataFrame with columns:

            - ``"x"`` : planar X coordinate of each profile point.
            - ``"y"`` : planar Y coordinate of each profile point.
            - ``"z"`` : interpolated elevation or depth at each position.
        """
        coords = self.axis.coordinates
        return pd.DataFrame({
            "x": coords["x"].to_numpy(),
            "y": coords["y"].to_numpy(),
            "z": self._z_interpolated,
        })

    def _validate_axis(self) -> None:
        """
        Raise :exc:`ValueError` if *axis* has no planar coordinates.

        Raises
        ------
        ValueError
            If *axis* was built with
            :meth:`~profile_axis.ProfileAxis.from_length`.
        """
        if not self.axis._has_coordinates:
            raise ValueError(
                "ProfileInterpolator requires a ProfileAxis built with "
                "from_coordinates. Axes built with from_length carry no "
                "planar coordinates and cannot be used for interpolation."
            )

    def _validate_crs(self, xyz: pd.DataFrame) -> None:
        """
        Check CRS consistency between *axis* and *xyz*.

        If *axis* has no CRS, a :class:`UserWarning` is emitted and
        execution continues.  If *xyz* is a
        :class:`~geopandas.GeoDataFrame` with a CRS attribute, it is
        compared against ``axis.crs``; a mismatch raises
        :exc:`ValueError`.

        Parameters
        ----------
        xyz : pandas.DataFrame or geopandas.GeoDataFrame
            Point cloud passed to :meth:`__init__`.

        Raises
        ------
        ValueError
            If *axis* and *xyz* have different CRS definitions.
        """
        if self.axis.crs is None:
            warnings.warn(
                "No CRS was provided to ProfileAxis. Cannot verify that "
                "the profile coordinates and the XYZ dataset share the "
                "same coordinate reference system. Ensure both are in the "
                "same projected CRS before interpolating.",
                UserWarning,
                stacklevel=3,
            )
            return

        xyz_crs = getattr(xyz, "crs", None)
        if xyz_crs is None:
            warnings.warn(
                f"ProfileAxis has CRS '{self.axis.crs}' but the XYZ "
                "DataFrame carries no CRS metadata. Load the point cloud "
                "with PointFileIO.read_as_geodataframe to enable CRS "
                "consistency checking.",
                UserWarning,
                stacklevel=3,
            )
            return

        axis_crs_str = self.axis.crs.strip().upper()
        xyz_crs_str = str(xyz_crs).strip().upper()

        if axis_crs_str != xyz_crs_str:
            raise ValueError(
                f"CRS mismatch: ProfileAxis has '{self.axis.crs}' but "
                f"the XYZ dataset has '{xyz_crs}'. Reproject one of them "
                "before interpolating."
            )

    def _validate_domain(self, xyz: pd.DataFrame) -> None:
        """
        Verify that the profile endpoints fall within the XYZ domain.

        The domain is defined as the axis-aligned bounding box of *xyz*.
        Both the profile start and the adjusted end must fall inside.

        Parameters
        ----------
        xyz : pandas.DataFrame
            Point cloud with columns ``"x"`` and ``"y"``.

        Raises
        ------
        ValueError
            If either endpoint falls outside the bounding box of *xyz*.
        """
        x_min, x_max = float(xyz["x"].min()), float(xyz["x"].max())
        y_min, y_max = float(xyz["y"].min()), float(xyz["y"].max())

        start = self.axis._start
        end_point = (
            self.axis._start + self.axis._s_axis[-1] * self.axis._direction
        )

        for label, point in [("start", start), ("end", end_point)]:
            px, py = float(point[0]), float(point[1])
            if not (x_min <= px <= x_max and y_min <= py <= y_max):
                raise ValueError(
                    f"Profile {label} point ({px:.2f}, {py:.2f}) falls "
                    f"outside the XYZ domain "
                    f"[{x_min:.2f} – {x_max:.2f}, "
                    f"{y_min:.2f} – {y_max:.2f}]. "
                    "Check that the profile coordinates and the XYZ "
                    "dataset are in the same CRS and overlap spatially."
                )

    def _filter_corridor(self, xyz: pd.DataFrame) -> pd.DataFrame:
        """
        Retain only XYZ points within the profile corridor.

        The corridor is a band of half-width :attr:`corridor_width`
        around the profile line, extended by the same amount beyond
        each endpoint.

        Parameters
        ----------
        xyz : pandas.DataFrame
            Full point cloud with columns ``"x"``, ``"y"``, ``"z"``.

        Returns
        -------
        pandas.DataFrame
            Filtered point cloud.

        Raises
        ------
        ValueError
            If fewer than :attr:`k_neighbors` points remain after
            filtering.
        """
        points = xyz[["x", "y"]].to_numpy()
        rel = points - self.axis._start
        u = self.axis._direction
        L = float(self.axis._s_axis[-1])

        s_proj = rel @ u
        perp = rel[:, 0] * u[1] - rel[:, 1] * u[0]

        mask = (
            (s_proj >= -self.corridor_width)
            & (s_proj <= L + self.corridor_width)
            & (np.abs(perp) <= self.corridor_width)
        )

        filtered = xyz.loc[mask].reset_index(drop=True)

        if len(filtered) < self.k_neighbors:
            raise ValueError(
                f"Only {len(filtered)} point(s) found within the profile "
                f"corridor (corridor_width={self.corridor_width}), but "
                f"k_neighbors={self.k_neighbors} is required. "
                "Increase corridor_width or verify XYZ coverage."
            )

        return filtered

    def _interpolate(self, xyz_filtered: pd.DataFrame) -> np.ndarray:
        """
        Build a KDTree and query Z values at each profile position.

        Parameters
        ----------
        xyz_filtered : pandas.DataFrame
            Pre-filtered point cloud with columns ``"x"``, ``"y"``,
            ``"z"``.

        Returns
        -------
        numpy.ndarray
            Interpolated Z values at each profile sampling point.
        """
        tree = cKDTree(xyz_filtered[["x", "y"]].to_numpy())

        query_coords = self.axis.coordinates[["x", "y"]].to_numpy()
        _, idx = tree.query(query_coords, k=self.k_neighbors)

        z_values = xyz_filtered["z"].to_numpy()

        if self.k_neighbors == 1:
            return z_values[idx]

        # Average over k neighbours when k > 1
        return z_values[idx].mean(axis=1)