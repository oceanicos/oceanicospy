from typing import Any, Dict, Optional

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from hydromt_sfincs import SfincsModel


class GridMaker:
    """
    Constructor de la malla computacional (regular) para modelos SFINCS.

    Delega la construcción de la malla a HydroMT-SFINCS
    (``SfincsModel.setup_grid``) y conserva los parámetros usados para
    poder visualizarla.

    Parameters
    ----------
    model : SfincsModel
        Instancia del modelo SFINCS inicializada con HydroMT.

    Attributes
    ----------
    grid_info : dict
        Parámetros de la malla: ``x0, y0, dx, dy, nmax, mmax, rotation,
        epsg``.

    Notes
    -----
    - Malla regular (no quadtree, no subgrid); es el modo usado en todas las
      corridas de referencia de este proyecto.
    - ``x0``/``y0`` son la esquina del primer punto de grilla, **no** el
      centro de la primera celda (convención de HydroMT-SFINCS/SFINCS).
    - **Convención de ejes** (confirmada contra la documentación oficial de
      HydroMT-SFINCS): ``mmax`` es el número de celdas en dirección **x**
      (va con ``dx``), y ``nmax`` es el número de celdas en dirección **y**
      (va con ``dy``). Es fácil confundirlos porque hay documentación/código
      de referencia de este mismo proyecto que los intercambia.
    """

    def __init__(self, model: SfincsModel) -> None:
        self.model = model
        self.grid_info: Dict[str, Any] = {}

    def setup_grid(
        self,
        plot: bool = True,
        plot_kwargs: Optional[Dict[str, Any]] = None,
        **grid_kwargs,
    ) -> Dict[str, Any]:
        """
        Configura la malla del modelo SFINCS.

        Parameters
        ----------
        plot : bool, optional
            Si es True, genera una visualización de la malla.
        plot_kwargs : dict, optional
            Argumentos adicionales para :meth:`plot_grid`.
        **grid_kwargs : dict
            Parámetros de la malla, pasados tal cual a
            ``SfincsModel.setup_grid``:

            - x0, y0 : float — esquina del primer punto de grilla (m,
              sistema proyectado).
            - dx, dy : float — resolución espacial en x/y (m).
            - mmax : int — número de celdas en dirección **x**.
            - nmax : int — número de celdas en dirección **y**.
            - rotation : float — rotación antihoraria desde el eje x/este
              (grados).
            - epsg : int — código EPSG del sistema de referencia.

        Returns
        -------
        dict
            Diccionario con los parámetros de la malla.

        Notes
        -----
        HydroMT-SFINCS recomienda limitar el número total de celdas activas
        a un orden de ~3 millones; ``mmax * nmax`` es el número de celdas
        totales (antes de descontar las inactivas por la máscara), así que
        aquí solo se emite una advertencia informativa si se excede ese
        orden de magnitud - no es un límite estricto.

        Examples
        --------
        >>> grid = GridMaker(model)
        >>> grid.setup_grid(
        ...     x0=423966, y0=1389496,
        ...     dx=15, dy=15,
        ...     mmax=75, nmax=65,
        ...     rotation=38, epsg=32617,
        ... )
        """
        mmax = grid_kwargs.get("mmax")
        nmax = grid_kwargs.get("nmax")
        if mmax is not None and nmax is not None and mmax * nmax > 3_000_000:
            print(
                f"[GridMaker] Aviso: mmax*nmax = {mmax * nmax:,} celdas. "
                "HydroMT-SFINCS recomienda mantener el total de celdas "
                "activas en un orden de ~3 millones."
            )

        # --------------------------------------------------
        # 1) Construcción de la malla con HydroMT
        # --------------------------------------------------
        self.model.setup_grid(**grid_kwargs)

        # --------------------------------------------------
        # 2) Guardar parámetros de la malla
        # --------------------------------------------------
        self.grid_info = {
            "x0": grid_kwargs.get("x0"),
            "y0": grid_kwargs.get("y0"),
            "dx": grid_kwargs.get("dx"),
            "dy": grid_kwargs.get("dy"),
            "nmax": nmax,
            "mmax": mmax,
            "rotation": grid_kwargs.get("rotation"),
            "epsg": grid_kwargs.get("epsg"),
        }

        # --------------------------------------------------
        # 3) Visualización opcional
        # --------------------------------------------------
        if plot:
            self.plot_grid(**(plot_kwargs or {}))

        return self.grid_info

    def plot_grid(
        self,
        filename: str = "grid.png",
        subdir: str = "figures",
        figsize: tuple = (6, 5),
        line_width: float = 0.3,
    ) -> str:
        """
        Genera una visualización (solo malla, sin rotación) de los bordes
        de celda del modelo SFINCS.

        Parameters
        ----------
        filename : str, optional
            Nombre del archivo de salida.
        subdir : str, optional
            Subdirectorio (relativo a la raíz del modelo) donde se guardará
            la figura.
        figsize : tuple, optional
            Tamaño de la figura en pulgadas.
        line_width : float, optional
            Grosor de las líneas de la malla.

        Returns
        -------
        str
            Ruta completa de la imagen generada.

        Raises
        ------
        RuntimeError
            Si la malla no ha sido configurada previamente.

        Notes
        -----
        Es una visualización simplificada: dibuja los bordes de celda en
        los ejes locales x/y de la malla (sin aplicar la ``rotation`` de
        vuelta a coordenadas UTM), solo para inspección rápida de
        resolución y extensión.
        """
        if not self.grid_info:
            raise RuntimeError(
                "grid_info está vacío. Llama primero a setup_grid() antes de plot_grid()."
            )

        x0 = self.grid_info["x0"]
        y0 = self.grid_info["y0"]
        dx = self.grid_info["dx"]
        dy = self.grid_info["dy"]
        nmax = self.grid_info["nmax"]
        mmax = self.grid_info["mmax"]

        # --------------------------------------------------
        # 1) Calcular bordes de celdas
        #    mmax celdas en x (van con dx), nmax celdas en y (van con dy)
        # --------------------------------------------------
        x_edges = x0 + np.arange(mmax + 1) * dx
        y_edges = y0 + np.arange(nmax + 1) * dy

        # --------------------------------------------------
        # 2) Crear figura
        # --------------------------------------------------
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_aspect("equal")

        for xe in x_edges:
            ax.plot([xe, xe], [y_edges[0], y_edges[-1]], "k-", lw=line_width)

        for ye in y_edges:
            ax.plot([x_edges[0], x_edges[-1]], [ye, ye], "k-", lw=line_width)

        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title("Malla SFINCS")

        ax.set_xlim(x_edges[0], x_edges[-1])
        ax.set_ylim(y_edges[0], y_edges[-1])

        # --------------------------------------------------
        # 3) Guardado
        # --------------------------------------------------
        outdir = Path(self.model.root) / subdir
        outdir.mkdir(parents=True, exist_ok=True)
        fpath = outdir / filename

        fig.savefig(fpath, dpi=300, bbox_inches="tight")
        plt.close(fig)

        return str(fpath)
