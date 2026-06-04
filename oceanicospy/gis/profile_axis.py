from __future__ import annotations

import numpy as np
import pandas as pd

from typing import Dict, Optional, Tuple, Union

__all__ = ["ProfileAxis"]

#: Default behaviour when the spacing does not divide the length exactly.
_DEFAULT_AUTO_EXTEND: bool = True

class ProfileAxis:
    """
    1D sampling axis for a straight-line profile.

    Use the class methods :meth:`from_coordinates` or :meth:`from_length`
    to construct an instance.  Direct instantiation via ``__init__`` is
    intentionally not supported.

    Attributes
    ----------
    crs : str or None
        CRS string assigned at construction (e.g. ``"EPSG:9377"``).
        ``None`` when no CRS was provided or when the axis was built
        from a length only.
    auto_extend : bool
        Whether the axis was extended so that the last interval equals
        the last spacing value.
    """
    def __init__(self) -> None:
        # Not for direct use.  Call from_coordinates or from_length.
        self._s_axis: np.ndarray = np.array([])
        self._start: Optional[np.ndarray] = None
        self._direction: Optional[np.ndarray] = None
        self._has_coordinates: bool = False
        self.crs: Optional[str] = None
        self.auto_extend: bool = _DEFAULT_AUTO_EXTEND

    @classmethod
    def from_coordinates(
        cls,
        start: Tuple[float, float],
        end: Tuple[float, float],
        dx: Union[float, Dict[float, float]],
        auto_extend: bool = _DEFAULT_AUTO_EXTEND,
        crs: Optional[str] = None,
    ) -> "ProfileAxis":
        """
        Build a profile axis from two planar endpoint coordinates.

        The Euclidean distance between *start* and *end* defines the
        nominal profile length.  When *auto_extend* is ``True`` and the
        spacing does not divide the length exactly, the axis is extended
        beyond *end* so that every interval has exactly the last spacing
        value used.  The adjusted endpoint can be retrieved via
        :attr:`adjusted_end`.

        Parameters
        ----------
        start : tuple of (float, float)
            Planar coordinates ``(x, y)`` of the profile origin.
            Must be in a projected CRS (metres).
        end : tuple of (float, float)
            Nominal planar coordinates ``(x, y)`` of the profile end.
            Must be in the same CRS as *start*.
        dx : float or dict of {float: float}
            Spacing definition along the profile:

            - ``float`` — constant spacing.
            - ``dict``  — piecewise spacing ``{position: dx}`` where
              each key is the absolute distance from *start* (in metres)
              up to which the associated spacing is used.
        auto_extend : bool, optional
            If ``True`` (default), the axis is extended so that the last
            interval equals the last spacing value.  If ``False``, the
            axis stops at the last exact multiple of the spacing and the
            nominal end may not be reached.
        crs : str, optional
            CRS string for metadata purposes only (e.g. ``"EPSG:9377"``).
            No coordinate transformation is performed.

        Returns
        -------
        ProfileAxis

        Raises
        ------
        ValueError
            If *start* and *end* are identical (zero-length profile).
        """
        obj = cls()
        obj.auto_extend = auto_extend
        obj.crs = crs
        obj._has_coordinates = True

        start_arr = np.array(start, dtype=float)
        end_arr = np.array(end, dtype=float)

        vec = end_arr - start_arr
        length = float(np.linalg.norm(vec))

        if length == 0.0:
            raise ValueError(
                "start and end coordinates are identical. "
                "Profile length is zero."
            )

        obj._start = start_arr
        obj._direction = vec / length
        obj._s_axis = obj._build_axis(
            start=0.0,
            end=length,
            dx=dx,
            auto_extend=auto_extend,
        )

        return obj

    @classmethod
    def from_length(
        cls,
        length: float,
        dx: Union[float, Dict[float, float]],
        auto_extend: bool = _DEFAULT_AUTO_EXTEND,
    ) -> "ProfileAxis":
        """
        Build a profile axis from a nominal total length.

        Use this constructor when planar coordinates are not available
        or not needed.  The resulting axis starts at ``s = 0`` and
        extends along the profile distance.  The properties
        :attr:`coordinates` and :attr:`adjusted_end` are not available
        in this mode.

        Parameters
        ----------
        length : float
            Nominal total profile length in the same units as *dx*.
        dx : float or dict of {float: float}
            Spacing definition.  See :meth:`from_coordinates` for details.
        auto_extend : bool, optional
            If ``True`` (default), the axis is extended so that the last
            interval equals the last spacing value.

        Returns
        -------
        ProfileAxis

        Notes
        -----
        Profile is assumed to start at s=0 and extend along the profile distance.

        Raises
        ------
        ValueError
            If *length* is not strictly positive.
        """
        if length <= 0.0:
            raise ValueError(
                f"length must be strictly positive, got {length}."
            )

        obj = cls()
        obj.auto_extend = auto_extend
        obj._has_coordinates = False
        obj._s_axis = obj._build_axis(
            start=0.0,
            end=length,
            dx=dx,
            auto_extend=auto_extend,
        )

        return obj

    @property
    def distance_axis(self) -> pd.DataFrame:
        """
        Profile axis as cumulative distances from the origin.

        Available in both :meth:`from_coordinates` and
        :meth:`from_length` modes.

        Returns
        -------
        pandas.DataFrame
            Single-column DataFrame with column ``"s"`` containing the
            cumulative distance at each sampling point, starting at
            ``0.0``.
        """
        return pd.DataFrame({"s": self._s_axis})

    @property
    def coordinates(self) -> pd.DataFrame:
        """
        Planar coordinates of each sampling point along the profile.

        Only available when the axis was built with
        :meth:`from_coordinates`.

        Returns
        -------
        pandas.DataFrame
            DataFrame with columns ``"x"`` and ``"y"`` containing the
            planar coordinates of each sampling point.

        Raises
        ------
        ValueError
            If the axis was built with :meth:`from_length`.
        """
        if not self._has_coordinates:
            raise ValueError(
                "coordinates is not available when the axis was built "
                "with from_length. Use from_coordinates instead."
            )

        xy = self._start + np.outer(self._s_axis, self._direction)
        return pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1]})

    @property
    def adjusted_end(self) -> Dict[str, float]:
        """
        Recalculated endpoint after spacing adjustment.

        When *auto_extend* is ``True`` and the spacing does not divide
        the nominal length exactly, the actual endpoint differs from the
        one passed to :meth:`from_coordinates`.  This property returns
        the adjusted planar coordinate so the caller can verify the
        extent of the constructed profile.

        Only available when the axis was built with
        :meth:`from_coordinates`.

        Returns
        -------
        dict of {str: float}
            ``{"x_adjusted": float, "y_adjusted": float}`` — the
            planar coordinates of the last point on the constructed axis.

        Raises
        ------
        ValueError
            If the axis was built with :meth:`from_length`.
        """
        if not self._has_coordinates:
            raise ValueError(
                "adjusted_end is not available when the axis was built "
                "with from_length. Use from_coordinates instead."
            )

        end_point = self._start + self._s_axis[-1] * self._direction
        return {
            "x_adjusted": float(end_point[0]),
            "y_adjusted": float(end_point[1]),
        }

    @property
    def length(self) -> float:
        """
        Total length of the profile axis.

        Returns
        -------
        float
            Total length in the same units as the spacing definition.
        """
        if self._has_coordinates:
            length = np.sqrt((self._end[0] - self._start[0]) ** 2 + (self._end[1] - self._start[1]) ** 2)
        else:
            length = float(self._s_axis[-1])

        return length

    def _build_axis(
        self,
        start: float,
        end: float,
        dx: Union[float, Dict[float, float]],
        auto_extend: bool,
    ) -> np.ndarray:
        """
        Dispatch to the appropriate axis builder based on *dx* type.

        Parameters
        ----------
        start : float
            Starting position (usually ``0.0``).
        end : float
            Nominal end position.
        dx : float or dict of {float: float}
            Spacing definition.
        auto_extend : bool
            Whether to extend the axis beyond *end*.

        Returns
        -------
        numpy.ndarray

        Raises
        ------
        TypeError
            If *dx* is not a supported type.
        ValueError
            If any spacing value is not strictly positive.
        """
        if isinstance(dx, (int, float)):
            return self._build_from_float(start, end, float(dx), auto_extend)
        if isinstance(dx, dict):
            return self._build_from_dict(start, end, dx, auto_extend)
        raise TypeError(
            f"dx must be float or dict, got {type(dx).__name__}."
        )

    def _build_from_float(
        self,
        start: float,
        end: float,
        dx: float,
        auto_extend: bool,
    ) -> np.ndarray:
        """
        Build a uniform axis with constant spacing *dx*.

        Parameters
        ----------
        start : float
            Starting position.
        end : float
            Nominal end position.
        dx : float
            Constant spacing.  Must be strictly positive.
        auto_extend : bool
            If ``True``, the axis is extended so the last interval
            equals *dx* exactly.  If ``False``, the axis stops at the
            last exact multiple of *dx* before *end*.

        Returns
        -------
        numpy.ndarray

        Raises
        ------
        ValueError
            If *dx* is not strictly positive.
        """
        if dx <= 0.0:
            raise ValueError(f"dx must be strictly positive, got {dx}.")

        length = end - start
        n_cells = int(np.ceil(length / dx)) if auto_extend else int(np.floor(length / dx))
        n_cells = max(n_cells, 1)
        return start + np.arange(0, n_cells + 1) * dx

    def _build_from_dict(
        self,
        start: float,
        end: float,
        dx_dict: Dict[float, float],
        auto_extend: bool,
    ) -> np.ndarray:
        """
        Build a piecewise-constant axis from a ``{position: dx}`` dict.

        Each key is an absolute distance from *start* (in the same
        units as the axis) defining the boundary up to which the
        associated spacing is applied.  Keys are sorted automatically.

        Example: ``{200: 10, 500: 5}`` means use ``dx=10`` from 0 to
        200 m, then ``dx=5`` from 200 m to 500 m.

        Parameters
        ----------
        start : float
            Starting position.
        end : float
            Nominal end position.
        dx_dict : dict of {float: float}
            Piecewise spacing definition.  All spacing values must be
            strictly positive.
        auto_extend : bool
            Whether to extend the axis beyond *end*.

        Returns
        -------
        numpy.ndarray

        Raises
        ------
        ValueError
            If any spacing value is not strictly positive.
        """
        positions = [start]
        last_dx = None

        for boundary, dx in sorted(dx_dict.items()):
            if dx <= 0.0:
                raise ValueError(
                    f"All dx values must be strictly positive, got {dx}."
                )
            last_dx = dx
            segment_end = min(start + boundary, end)

            while positions[-1] + dx <= segment_end:
                positions.append(positions[-1] + dx)

            if positions[-1] >= end:
                break

        # Extend to end with last dx if needed
        if last_dx is None:
            raise ValueError("dx_dict must contain at least one entry.")

        while positions[-1] < end:
            next_pos = positions[-1] + last_dx
            if next_pos >= end:
                if auto_extend:
                    positions.append(next_pos)
                else:
                    positions.append(end)
                break
            positions.append(next_pos)

        return np.array(positions)