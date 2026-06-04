from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "AxisAlignedRectangle",
    "XYZRectangleMask",
]

@dataclass(frozen=True)
class AxisAlignedRectangle:
    """
    Immutable axis-aligned rectangle defined by two opposite corners.

    The order of *p1* and *p2* is irrelevant; minimum and maximum bounds
    are computed internally.  This geometry operates purely in the XY
    plane and carries no CRS information.
    """

    #: First corner as (x, y).
    p1: Tuple[float, float]

    #: Opposite corner as (x, y).
    p2: Tuple[float, float]

    @property
    def x_min(self) -> float:
        """Minimum X bound."""
        return min(self.p1[0], self.p2[0])

    @property
    def x_max(self) -> float:
        """Maximum X bound."""
        return max(self.p1[0], self.p2[0])

    @property
    def y_min(self) -> float:
        """Minimum Y bound."""
        return min(self.p1[1], self.p2[1])

    @property
    def y_max(self) -> float:
        """Maximum Y bound."""
        return max(self.p1[1], self.p2[1])

    def contains(self, x: float, y: float) -> bool:
        """
        Return ``True`` if ``(x, y)`` lies inside or on the boundary.

        Parameters
        ----------
        x : float
            X coordinate to test.
        y : float
            Y coordinate to test.

        Returns
        -------
        bool
        """
        return (self.x_min <= x <= self.x_max) and (self.y_min <= y <= self.y_max)

class XYZRectangleMask:
    """
    Apply axis-aligned rectangular masks to an XYZ point DataFrame.

    Two modes are supported:

    - ``"keep"`` — retain only points that fall **inside** at least one
      rectangle (default).
    - ``"exclude"`` — remove points that fall **inside** any rectangle.

    Parameters
    ----------
    rectangles : list of AxisAlignedRectangle
        One or more rectangles defining the inclusion or exclusion zones.
    mode : {"keep", "exclude"}, optional
        Filtering mode.  Defaults to ``"keep"``.

    Raises
    ------
    ValueError
        If *mode* is not ``"keep"`` or ``"exclude"``.
    """

    def __init__(
        self,
        rectangles: List[AxisAlignedRectangle],
        mode: str = "keep",
    ) -> None:
        if mode not in ("keep", "exclude"):
            raise ValueError(
                f"Invalid mode '{mode}'. Expected 'keep' or 'exclude'."
            )
        self.rectangles = rectangles
        self.mode = mode

    def filter_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter *df* according to the rectangles and the selected mode.

        Parameters
        ----------
        df : pandas.DataFrame
            Input point data.  Must contain at least the columns
            ``"x"`` and ``"y"``.

        Returns
        -------
        pandas.DataFrame
            Filtered copy of *df* with the index reset.

        Raises
        ------
        ValueError
            If ``"x"`` or ``"y"`` columns are absent from *df*.
        """
        missing = [c for c in ("x", "y") if c not in df.columns]
        if missing:
            raise ValueError(
                f"DataFrame is missing required columns: {missing}"
            )

        x = df["x"].to_numpy()
        y = df["y"].to_numpy()

        inside_any = np.zeros(len(df), dtype=bool)
        for rect in self.rectangles:
            inside_any |= (
                (x >= rect.x_min) & (x <= rect.x_max) &
                (y >= rect.y_min) & (y <= rect.y_max)
            )

        mask = inside_any if self.mode == "keep" else ~inside_any
        return df.loc[mask].reset_index(drop=True)