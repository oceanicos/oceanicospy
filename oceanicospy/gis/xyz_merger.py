from __future__ import annotations

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from shapely.ops import unary_union

from pathlib import Path
from typing import List, Optional, Union

from .point_io import XYZFormatSpec, PointFileIO, _normalize_epsg

__all__ = ["XYZMerger"]

# Constants for internal use only. These are not part of the public API.

#: Decimal places used when snapping XY coordinates to the coverage grid.
_XY_ROUND_DECIMALS: int = 6

#: Number of points sampled to estimate the coverage grid cell size.
_SAMPLE_SIZE_COVERAGE: int = 2000

#: Grid cell size as a multiple of the estimated point spacing.
_GRID_FACTOR: float = 1.0

#: Default buffer radius as a multiple of the estimated point spacing.
_BUFFER_FACTOR: float = 0.75

#: Default morphological closing factor applied after the buffer union.
_CLOSING_FACTOR: float = 1.0

class _XYZTile:
    """
    Internal container for a single XYZ tile.

    Holds the point data, CRS, priority rank and coverage polygon for one
    input file.  Instances are created and managed exclusively by
    :class:`XYZMerger` and are never exposed to the caller.

    Parameters
    ----------
    path : str or pathlib.Path
        Original file path, kept for diagnostic messages only.
    points : pandas.DataFrame
        Point data with columns ``x``, ``y``, ``z``.
    crs : str
        Canonical CRS string (e.g. ``"EPSG:9377"``).
    priority_rank : int
        Rank assigned by :class:`XYZMerger`.  Rank ``0`` = highest priority.
    buffer_factor : float, optional
        Buffer radius as a multiple of the estimated point spacing.
    closing_factor : float, optional
        Morphological closing factor applied after the buffer union.
    """

    def __init__(
        self,
        path: Union[str, Path],
        points: pd.DataFrame,
        crs: str,
        priority_rank: int,
        buffer_factor: float = _BUFFER_FACTOR,
        closing_factor: float = _CLOSING_FACTOR,
    ) -> None:
        self.path = Path(path)
        self.points = points
        self.crs = crs
        self.priority_rank = priority_rank
        self._buffer_factor = buffer_factor
        self._closing_factor = closing_factor
        self.coverage_polygon: gpd.GeoDataFrame = self._compute_coverage_polygon()

    def _estimate_spacing(self) -> float:
        """
        Estimate a representative XY point spacing from a random sample.

        Uses the median nearest-neighbour distance over at most
        ``_SAMPLE_SIZE_COVERAGE`` points.

        Returns
        -------
        float
            Estimated spacing in CRS units.  Returns ``1e-4`` when the
            tile contains fewer than two points.
        """
        n = len(self.points)
        if n < 2:
            return 1e-4

        sample = self.points.sample(
            n=min(_SAMPLE_SIZE_COVERAGE, n),
            random_state=42,
            replace=False,
        )
        coords = sample[["x", "y"]].to_numpy()

        from scipy.spatial import cKDTree  # local import — optional dependency
        tree = cKDTree(coords)
        dists, _ = tree.query(coords, k=2)
        return float(np.median(dists[:, 1]))

    def _compute_coverage_polygon(self) -> gpd.GeoDataFrame:
        """
        Build a concave-like coverage footprint via grid-snapping and
        buffered union of unique grid cells.

        Returns
        -------
        geopandas.GeoDataFrame
            Single-row GeoDataFrame containing the coverage polygon in
            ``self.crs``.
        """
        df = self.points[["x", "y"]].copy()
        df["x"] = df["x"].round(_XY_ROUND_DECIMALS)
        df["y"] = df["y"].round(_XY_ROUND_DECIMALS)

        spacing = self._estimate_spacing()
        cell = max(1e-12, _GRID_FACTOR * spacing)
        buf_r = max(1e-12, self._buffer_factor * spacing)
        close_r = buf_r * self._closing_factor

        # --- edge cases ---------------------------------------------------
        if len(df) == 0:
            return gpd.GeoDataFrame({"geometry": []}, crs=self.crs)

        if len(df) == 1:
            point_buf = Point(
                float(df["x"].iloc[0]),
                float(df["y"].iloc[0]),
            ).buffer(max(1e-12, 0.5 * buf_r))
            return gpd.GeoDataFrame({"geometry": [point_buf]}, crs=self.crs)

        # --- grid snapping ------------------------------------------------
        grid_ix = np.floor(df["x"].to_numpy() / cell).astype(np.int64)
        grid_iy = np.floor(df["y"].to_numpy() / cell).astype(np.int64)

        unique_cells, _ = np.unique(
            np.stack([grid_ix, grid_iy], axis=1),
            axis=0,
            return_index=True,
        )

        cell_cx = (unique_cells[:, 0].astype(float) + 0.5) * cell
        cell_cy = (unique_cells[:, 1].astype(float) + 0.5) * cell

        # --- buffered union + morphological closing -----------------------
        geoms = [Point(xy).buffer(buf_r) for xy in zip(cell_cx, cell_cy)]
        polygon = unary_union(geoms).buffer(close_r).buffer(-close_r)

        return gpd.GeoDataFrame({"geometry": [polygon]}, crs=self.crs)
    
class XYZMerger:
    """
    Merge a collection of XYZ tiles into a single point dataset.

    Overlapping regions are resolved by priority: points from
    lower-priority tiles that fall within the coverage footprint of any
    higher-priority tile are removed.  The caller declares the priority
    order explicitly via the *priority* parameter.

    All input tiles must share the same CRS and on-disk format.  Use
    :mod:`point_io` and :mod:`crs` to standardize files beforehand.

    Parameters
    ----------
    input_dir : str or pathlib.Path
        Directory containing the ``.xyz`` files to merge.
    priority : list of str
        Filenames ordered from highest to lowest priority.  Only the
        files listed here will be loaded; other ``.xyz`` files present
        in *input_dir* are ignored.
    crs : str or int
        Common CRS of all input tiles (e.g. ``9377`` or ``"EPSG:9377"``).
    format_spec : XYZFormatSpec, optional
        On-disk layout shared by all input tiles.  Defaults to
        :class:`~point_io.XYZFormatSpec` (space-delimited, no header,
        columns ``x/y/z``).
    buffer_factor : float, optional
        Buffer radius used when building coverage polygons, expressed as a
        multiple of the estimated point spacing.  Increase this value if
        lower-priority points leak into high-priority areas; decrease it
        if too many valid points are discarded.  Defaults to ``0.75``.
    closing_factor : float, optional
        Morphological closing factor applied after the buffer union.
        Controls how aggressively small gaps inside a coverage polygon are
        filled.  Defaults to ``1.0``.

    Raises
    ------
    FileNotFoundError
        If any file declared in *priority* is not found in *input_dir*.
    ValueError
        If any tile is missing the required ``x``, ``y``, ``z`` columns.
    """

    def __init__(
        self,
        input_dir: Union[str, Path],
        priority: List[str],
        crs: Union[str, int],
        format_spec: Optional[XYZFormatSpec] = None,
        buffer_factor: float = _BUFFER_FACTOR,
        closing_factor: float = _CLOSING_FACTOR,
    ) -> None:
        self.input_dir = Path(input_dir)
        self.priority = priority
        self.crs = _normalize_epsg(crs)
        self.format_spec = format_spec or XYZFormatSpec()
        self.buffer_factor = buffer_factor
        self.closing_factor = closing_factor

        self._tiles: List[_XYZTile] = []
        self._file_paths: List[Path] = []

    def run_merge(self, output_path: Union[str, Path]) -> pd.DataFrame:
        """
        Run the full merge pipeline and export the result.

        The pipeline executes five stages in order:

        1. Discover files declared in *priority* within *input_dir*.
        2. Load each file into an internal ``_XYZTile``.
        3. Assign priority ranks from the *priority* list.
        4. Merge tiles, discarding lower-priority points in overlap zones.
        5. Export the merged dataset to *output_path*.

        Parameters
        ----------
        output_path : str or pathlib.Path
            Destination file path for the merged XYZ file.

        Returns
        -------
        pandas.DataFrame
            Merged point data with columns ``x``, ``y``, ``z``.

        Raises
        ------
        FileNotFoundError
            If any file declared in *priority* is not found in *input_dir*.
        ValueError
            If any file is missing the required ``x``, ``y``, ``z`` columns.
        """
        self._discover_tiles()
        self._load_tiles()
        self._assign_priorities()
        merged_df = self._merge()
        self._export(merged_df, output_path)
        return merged_df

    def _discover_tiles(self) -> None:
        """
        Verify that all files declared in *priority* exist in *input_dir*
        and populate the internal file path list in priority order.

        Raises
        ------
        FileNotFoundError
            If any file declared in *priority* is not found in *input_dir*.
        """
        self._file_paths = []
        missing_files = []
        for ref in self.priority:
            path = self.input_dir / Path(ref).name
            if not path.exists():
                missing_files.append(ref)
            else:
                self._file_paths.append(path)

        if missing_files:
            raise FileNotFoundError(
                f"The following files declared in priority were not found "
                f"in '{self.input_dir}': {missing_files}"
            )

    def _load_tiles(self) -> None:
        """
        Load each ``.xyz`` file via :class:`~point_io.PointFileIO` and
        build an ``_XYZTile`` instance.

        Raises
        ------
        ValueError
            If a loaded file is missing the ``x``, ``y`` or ``z`` columns.
        """
        self._tiles = []
        for path in self._file_paths:
            df = PointFileIO(path, format_spec=self.format_spec).read()

            missing = [c for c in ("x", "y", "z") if c not in df.columns]
            if missing:
                raise ValueError(
                    f"'{path.name}' is missing required columns: {missing}. "
                    "Verify that format_spec matches the file layout."
                )

            self._tiles.append(
                _XYZTile(
                    path=path,
                    points=df,
                    crs=self.crs,
                    priority_rank=-1,
                    buffer_factor=self.buffer_factor,
                    closing_factor=self.closing_factor,
                )
            )

    def _assign_priorities(self) -> None:
        """
        Assign ``priority_rank`` to each tile from the *priority* list
        and sort tiles by ascending rank.

        Rank ``0`` is highest priority.
        """
        rank_map = {Path(ref).name: rank for rank, ref in enumerate(self.priority)}
        for tile in self._tiles:
            tile.priority_rank = rank_map[tile.path.name]
        self._tiles.sort(key=lambda t: t.priority_rank)

    def _merge(self) -> pd.DataFrame:
        """
        Merge tiles by discarding lower-priority points inside
        higher-priority coverage polygons.

        Returns
        -------
        pandas.DataFrame
            Merged point data sorted by ``y`` then ``x``.
        """
        tile_gdfs = [
            gpd.GeoDataFrame(
                tile.points[["x", "y", "z"]].copy(),
                geometry=gpd.points_from_xy(tile.points["x"], tile.points["y"]),
                crs=self.crs,
            )
            for tile in self._tiles
        ]
        coverage_gdfs = [tile.coverage_polygon for tile in self._tiles]

        kept: List[pd.DataFrame] = []

        for i, gdf in enumerate(tile_gdfs):
            if i == 0:
                kept.append(gdf[["x", "y", "z"]].copy())
                continue

            higher_coverage = pd.concat(coverage_gdfs[:i], ignore_index=True)

            if higher_coverage.empty:
                kept.append(gdf[["x", "y", "z"]].copy())
                continue

            joined = gpd.sjoin(gdf, higher_coverage, how="left", predicate="within")
            outside_mask = joined["index_right"].isna()
            kept.append(joined.loc[outside_mask, ["x", "y", "z"]].copy())

        return (
            pd.concat(kept, ignore_index=True)
            .sort_values(by=["y", "x"])
            .reset_index(drop=True)
        )

    def _export(self, df: pd.DataFrame, output_path: Union[str, Path]) -> None:
        """
        Write the merged DataFrame to an XYZ file via
        :class:`~point_io.PointFileIO`.

        Parameters
        ----------
        df : pandas.DataFrame
            Merged point data.
        output_path : str or pathlib.Path
            Destination file path.
        """
        PointFileIO(
            path=output_path,
            format_spec=self.format_spec,
        ).write(df, float_format="%.6f")