from __future__ import annotations

import numpy as np
import shutil
import warnings

from pathlib import Path
from typing import Dict, Optional, Tuple, Union
from .... import utils
from ....gis import Grid, ProfileAxis

class _ProfileBuilder:
    """
    Proxy that builds a 1-D XBeach profile axis and exports the grid files.

    Obtained via ``GridMaker.build_profile``; do not instantiate directly.
    Delegates coordinate geometry to :class:`~oceanicospy.gis.ProfileAxis`.
    """

    def __init__(self, gridmaker: "GridMaker") -> None:
        self._gm = gridmaker

    def from_coordinates(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        dx: Union[float, Dict[float, float]],
        auto_extend: bool = True,
        crs: Optional[str] = None,
    ) -> ProfileAxis:
        """
        Build a profile axis from two planar endpoint coordinates.

        Parameters
        ----------
        start : tuple of (float, float)
            Planar coordinates ``(x, y)`` of the profile origin.
        end : tuple of (float, float)
            Nominal planar coordinates ``(x, y)`` of the profile end.
        dx : float or dict of {float: float}
            Spacing definition. See :class:`~oceanicospy.gis.ProfileAxis`.
        auto_extend : bool, optional
            Extend the axis so the last interval equals the last spacing.
        crs : str, optional
            CRS string for metadata only (e.g. ``"EPSG:9377"``).

        Returns
        -------
        ProfileAxis
        """
        profile = ProfileAxis.from_coordinates(start, end, dx, auto_extend=auto_extend, crs=crs)
        self._gm._profile_axis = profile
        self._gm._grid_dict = self._export(profile)
        return profile

    def from_length(
        self,
        length: float,
        dx: Union[float, Dict[float, float]],
        auto_extend: bool = True,
    ) -> ProfileAxis:
        """
        Build a profile axis from a nominal total length.

        Parameters
        ----------
        length : float
            Nominal total profile length in the same units as *dx*.
        dx : float or dict of {float: float}
            Spacing definition. See :class:`~oceanicospy.gis.ProfileAxis`.
        auto_extend : bool, optional
            Extend the axis so the last interval equals the last spacing.

        Returns
        -------
        ProfileAxis
        """
        profile = ProfileAxis.from_length(length, dx, auto_extend=auto_extend)
        self._gm._profile_axis = profile
        self._gm._grid_dict = self._export(profile)
        return profile

    def _export(self, profile: ProfileAxis) -> dict:
        """
        Write the 1-D profile axis to ``x.grd`` / ``y.grd`` and return the grid metadata.

        When ``GridMaker.coordinate_type`` is ``"absolute"``, the actual planar
        (x, y) coordinates along the transect are written.  For ``"relative"``
        (default), cumulative cross-shore distances are written to ``x.grd`` and
        a zero array to ``y.grd``, which is the standard XBeach 1-D convention.

        Parameters
        ----------
        profile : ProfileAxis
            Axis object returned by :meth:`from_coordinates` or
            :meth:`from_length`.

        Returns
        -------
        dict
            Keys ``xfilepath``, ``yfilepath``, ``meshes_x``, ``meshes_y``
            (always ``0`` for 1-D profiles).
        """
        run_folder = Path(self._gm.init.dict_folders["run"])

        if self._gm.coordinate_type == "absolute":
            x_coords = profile.coordinates["x"].values
            y_coords = profile.coordinates["y"].values
        else:
            x_coords = profile.distance_axis["s"].values
            y_coords = np.zeros_like(x_coords)

        np.savetxt(run_folder / "x.grd", x_coords, fmt="%.4f")
        np.savetxt(run_folder / "y.grd", y_coords, fmt="%.4f")
        return {
            "xfilepath": "x.grd",
            "yfilepath": "y.grd",
            "meshes_x": len(x_coords) - 1,
            "meshes_y": 0,
        }

class _RectangularGridBuilder:
    """
    Proxy that builds a 2-D XBeach rectangular grid and exports the grid files.

    Obtained via ``GridMaker.build_rectangular_grid``; do not instantiate directly.
    Delegates coordinate geometry to :class:`~oceanicospy.gis.Grid`.
    """

    def __init__(self, gridmaker: "GridMaker") -> None:
        self._gm = gridmaker

    def from_shapefile(
        self,
        source_file: str,
        dx: float,
        dy: float = None,
        xvar: bool = False,
        crs=None,
    ) -> Grid:
        """
        Build a rectangular grid from the bounding box of a shapefile.

        Parameters
        ----------
        source_file : str
            Filename of the ``.shp`` file inside the project input folder.
        dx : float
            Grid spacing in the x direction.
        dy : float, optional
            Grid spacing in the y direction. Defaults to *dx*.
        xvar : bool, optional
            Use variable x-spacing (not yet implemented in Grid).
        crs : str, optional
            CRS string for metadata only (e.g. ``"EPSG:9377"``).

        Returns
        -------
        Grid
        """
        input_folder = self._gm.init.dict_folders["input"]

        shp_path = Path(input_folder) / source_file
        grid = Grid.from_shapefile(shp_path, dx=dx, dy=dy, xvar=xvar, crs=crs)

        self._gm._grid = grid
        self._gm._grid_dict = self._export(grid)
        return grid

    def _export(self, grid: Grid) -> dict:
        """
        Write the 2-D grid arrays to ``x.grd`` / ``y.grd`` and return the grid metadata.

        The arrays written depend on ``GridMaker.coordinate_type``:

        - ``"relative"`` (default) — coordinates measured from the lower-left
          corner of the bounding box, so the origin is always ``(0, 0)``.
        - ``"absolute"`` — coordinates in the original projected CRS units
          (typically metres).

        Parameters
        ----------
        grid : Grid
            Grid object returned by :meth:`from_shapefile`.

        Returns
        -------
        dict
            Keys ``xfilepath``, ``yfilepath``, ``meshes_x``, ``meshes_y``.
        """
        run_folder = Path(self._gm.init.dict_folders["run"])
        if self._gm.coordinate_type == "absolute":
            np.savetxt(run_folder / "x.grd", grid.absolute_x_coordinates, fmt="%.4f")
            np.savetxt(run_folder / "y.grd", grid.absolute_y_coordinates, fmt="%.4f")
        else:
            np.savetxt(run_folder / "x.grd", grid.relative_x_coordinates, fmt="%.4f")
            np.savetxt(run_folder / "y.grd", grid.relative_y_coordinates, fmt="%.4f")
        
        return {
            "xfilepath": "x.grd",
            "yfilepath": "y.grd",
            "meshes_x": grid.nx,
            "meshes_y": grid.ny,
        }

class GridMaker:
    """
    Creates a XBeach computational grid and fills grid information in params.txt.

    Use :attr:`build_profile` or :attr:`build_rectangular_grid` to generate
    the grid, then call :meth:`fill_grid_section` to write it to params.txt.

    Parameters
    ----------
    init : object
        Project initialization object with folder configuration.
    grid_params : dict
        Dictionary of additional grid parameters to be included in params.txt. The following keys must be included:

        - ``thetamax``: maximum angle of grid rotation (degrees)
        - ``thetamin``: minimum angle of grid rotation (degrees)
        - ``dtheta``: angle increment for grid rotation (degrees)
        - ``alfa``: grid stretching factor (dimensionless)

    coordinates_type : str, optional
        Type of coordinates to be written: "absolute" or "relative". "relative" is set as default
    """

    def __init__(self, init, grid_params: dict, coordinates_type: str = "relative"):
        self.init = init
        self.grid_params = grid_params
        self.coordinate_type = coordinates_type.lower()
        self._grid_dict = None
        self._profile_axis = None
        self._grid = None

        if self.coordinate_type not in ["absolute", "relative"]:
            raise ValueError("Invalid coordinate_type. Choose 'absolute' or 'relative'.")

    @property
    def metadata(self) -> dict:
        """
        Grid metadata dictionary used to fill the grid section in params.txt.

        Returns
        -------
        dict or None
            Keys ``xfilepath``, ``yfilepath``, ``meshes_x``, ``meshes_y``,
            or ``None`` if no grid has been generated yet.
        """
        return self._grid_dict

    @property
    def build_profile(self) -> _ProfileBuilder:
        """Return a builder for 1-D profile grids."""
        return _ProfileBuilder(self)

    @property
    def build_rectangular_grid(self) -> _RectangularGridBuilder:
        """Return a builder for 2-D rectangular grids."""
        return _RectangularGridBuilder(self)
    
    def load_existing_xbgrid(self,xgrid_filename,ygrid_filename) -> dict:
        """
        Load an existing grid from the configured input folder, validate it,
        copy the files into the run folder, and return a descriptor dictionary.

        Returns
        -------
        dict or None
            ``None`` if fewer than two ``.grd`` files are found.
            Otherwise a dict with ``xfilepath``, ``yfilepath``,
            ``meshes_x``, and ``meshes_y``.

        Raises
        ------
        ValueError
            If the loaded x and y arrays do not have the same shape.
        """
        input_folder = Path(self.init.dict_folders["input"])
        run_folder = Path(self.init.dict_folders["run"])

        fixed_x = input_folder / xgrid_filename
        fixed_y = input_folder / ygrid_filename

        if fixed_x.exists() and fixed_y.exists():
            x_file, y_file = fixed_x, fixed_y
        else:
            raise FileNotFoundError(
                f"Grid files not found in input folder: {fixed_x}, {fixed_y}"
            )

        x = np.loadtxt(x_file)
        y = np.loadtxt(y_file)

        if x.shape != y.shape:
            raise ValueError("Loaded .grd files do not have matching shapes.")

        is_2d = (x.ndim == 2)
        meshes_x = x.shape[1] - 1 if is_2d else x.shape[0] - 1
        meshes_y = x.shape[0] - 1 if is_2d else 0

        shutil.copy(str(x_file), str(run_folder / "x.grd"))
        shutil.copy(str(y_file), str(run_folder / "y.grd"))

        return {
            "xfilepath": "x.grd",
            "yfilepath": "y.grd",
            "meshes_x": meshes_x,
            "meshes_y": meshes_y,
        }

    def fill_grid_section(self) -> None:
        """Write the generated grid metadata to the params.txt file."""

        self._grid_dict.update(self.grid_params)

        for key in self._grid_dict:
            if isinstance(self._grid_dict[key], (int, float)):
                self._grid_dict[key] = str(self._grid_dict[key])

        if self._grid_dict is None:
            raise ValueError(
                "Grid has not been generated yet. "
                "Call build_profile or build_rectangular_grid first."
            )
        print("\n*** Adding/Editing grid information in params file ***\n")
        utils.fill_files(f'{self.init.dict_folders["run"]}params.txt', self._grid_dict)
