from typing import Any, Dict, List, Optional
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import rasterio

from hydromt_sfincs import SfincsModel


class BathyMaker:
    """
    Configuración de topobatimetría y máscaras para modelos SFINCS.

    Esta clase se encarga de:
        - Construir el raster de elevación (`dep`), opcionalmente
          combinando varias capas topobatimétricas con las reglas de
          fusión nativas de HydroMT-SFINCS.
        - Definir la máscara activa del modelo (`msk == 1`)
        - Definir las celdas de frontera hidráulica (`msk == 2`)
        - Generar visualizaciones de los rasters del modelo

    Utiliza las herramientas de HydroMT-SFINCS para preparar el dominio
    espacial a partir de datasets definidos en el `data_catalog`.

    Parameters
    ----------
    model : SfincsModel
        Instancia del modelo SFINCS inicializada con HydroMT.
    data_catalog : str
        Ruta al archivo `data_catalog.yml` donde se definen los datasets
        (por ejemplo, topografía y batimetría).

    Notes
    -----
    - El raster `dep` contiene elevaciones (positivas en tierra, negativas en mar).
    - La máscara (`msk`) define:
        * 0 : celda inactiva
        * 1 : celda activa
        * 2 : frontera de entrada (waterlevel)
        * 3 : frontera de salida
    """

    def __init__(self, model: SfincsModel, data_catalog: str) -> None:
        self.model = model
        self.data_catalog = data_catalog

    # --------------------------------------------------
    # SETUP BATIMETRÍA Y MÁSCARAS
    # --------------------------------------------------
    def setup_bathy(
        self,
        datasets_dep: Optional[List[Dict[str, Any]]] = None,
        zmin_active: float = -4,
        zmax_bounds: float = -2,
        btype: str = "waterlevel",
        reset_mask: bool = True,
        reset_bounds: bool = True,
        plot: bool = False,
        plot_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Configura la topobatimetría y las máscaras del modelo SFINCS.

        Este método realiza:
            1. Generación del raster de elevación (`dep`)
            2. Definición de celdas activas
            3. Definición de celdas de frontera hidráulica

        Parameters
        ----------
        datasets_dep : list of dict, optional
            Lista de datasets topobatimétricos a fusionar, en el formato
            que espera ``SfincsModel.setup_dep`` - cada entrada es un dict
            con la clave ``"elevtn"`` (nombre del dataset en el
            ``data_catalog``) y, opcionalmente, argumentos de fusión como
            ``zmin``, ``zmax``, ``offset``, ``merge_method`` o
            ``reproj_method`` (ver
            :py:func:`hydromt.workflows.merge_multi_dataarrays`). Por
            ejemplo, para dar prioridad a un levantamiento fino sobre un
            DEM de relleno más grueso::

                [
                    {"elevtn": "topobathy_fino"},
                    {"elevtn": "topobathy_grueso", "merge_method": "first"},
                ]

            Si no se especifica, se usa un único dataset llamado
            ``"topobathy"`` (comportamiento por defecto, para
            compatibilidad con un `data_catalog` que ya trae un único
            ráster pre-fusionado).
        zmin_active : float, optional
            Elevación mínima (m) para definir celdas activas.
            Las celdas con elevación mayor o igual a este valor se
            consideran activas.
        zmax_bounds : float, optional
            Elevación máxima (m) para definir las celdas de frontera de
            nivel del mar. El manual de SFINCS recomienda ubicar la
            frontera costera en la zona de swash, aproximadamente a 2 m de
            profundidad - de ahí el valor por defecto de -2.
        btype : str, optional
            Tipo de frontera hidráulica (``"waterlevel"`` u
            ``"outflow"``). Por defecto ``"waterlevel"``.
        reset_mask : bool, optional
            Si es True, reinicia la máscara activa existente.
        reset_bounds : bool, optional
            Si es True, reinicia las fronteras existentes del tipo
            `btype` antes de fijar las nuevas.
        plot : bool, optional
            Si es True, genera una visualización del raster `dep`.
        plot_kwargs : dict, optional
            Argumentos adicionales para :meth:`plot_sfincs_raster`.

        Returns
        -------
        None

        Notes
        -----
        - Cada dataset referenciado en `datasets_dep` debe estar definido
          en el `data_catalog` (ver ``Initializer.write_data_catalog``).
        - La máscara activa controla dónde se simula la inundación.
        - Las fronteras (`msk == 2`) definen dónde se aplican forzantes
          como nivel del mar.

        Examples
        --------
        >>> bathy = BathyMaker(model, "data_catalog.yml")
        >>> bathy.setup_bathy(zmin_active=-5, zmax_bounds=-2, plot=True)
        """
        if datasets_dep is None:
            datasets_dep = [{"elevtn": "topobathy"}]

        # --------------------------------------------------
        # 1) Construcción del raster de elevación (dep)
        # --------------------------------------------------
        self.model.setup_dep(datasets_dep=datasets_dep)

        # --------------------------------------------------
        # 2) Definición de máscara activa
        # --------------------------------------------------
        self.model.setup_mask_active(
            zmin=zmin_active,
            reset_mask=reset_mask,
        )

        # --------------------------------------------------
        # 3) Definición de fronteras hidráulicas
        # --------------------------------------------------
        self.model.setup_mask_bounds(
            btype=btype,
            zmax=zmax_bounds,
            reset_bounds=reset_bounds,
        )

        # --------------------------------------------------
        # 4) Visualización opcional
        # --------------------------------------------------
        # plot_sfincs_raster reads <root>/gis/<name>.tif, which HydroMT only
        # writes to disk on model.write() - without this, plot=True would
        # always raise FileNotFoundError. Safe to call here: at this point
        # in the pipeline only grid/dep/msk are set (no forcings yet), and
        # SfincsCaseBuilder.build_basic_case already calls model.write()
        # again right after setup_bathy(), so writing twice is harmless.
        if plot:
            self.model.write()
            self.plot_sfincs_raster("dep", **(plot_kwargs or {}))

    # --------------------------------------------------
    # PLOT DE RASTERS HYDROMT
    # --------------------------------------------------
    def plot_sfincs_raster(
        self,
        raster_name: str,
        filename: Optional[str] = None,
        subdir: Optional[str] = "figures",
        figsize: tuple = (7, 6),
        cmap: Optional[str] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        add_colorbar: bool = True,
    ) -> str:
        """
        Genera una visualización de un raster del modelo SFINCS.

        Parameters
        ----------
        raster_name : str
            Nombre del raster (por ejemplo: ``"dep"``, ``"msk"``, ``"man"``).
        filename : str, optional
            Nombre del archivo de salida. Si no se especifica,
            se usa ``<raster_name>.png``.
        subdir : str or None, optional
            Subdirectorio (relativo a la raíz del modelo, salvo que sea
            una ruta absoluta) donde se guardará la figura.
        figsize : tuple, optional
            Tamaño de la figura en pulgadas.
        cmap : str, optional
            Colormap de matplotlib.
        vmin, vmax : float, optional
            Valores mínimo y máximo para la escala de colores.
        add_colorbar : bool, optional
            Si es True, añade barra de color.

        Returns
        -------
        str
            Ruta completa de la imagen generada.

        Raises
        ------
        FileNotFoundError
            Si el raster no existe.

        Notes
        -----
        - El raster se busca en: ``<model.root>/gis/<raster_name>.tif``.
        - **La malla del proyecto está rotada** (``rotation`` != 0 en todos
          los casos conocidos), y HydroMT-SFINCS escribe los `.tif` de
          salida con esa rotación genuina en la transformación afín (no
          "north-up"). ``matplotlib.imshow`` no puede representar una
          transformación afín rotada - solo coloca la imagen en una caja
          alineada a los ejes -, así que aquí se calculan las coordenadas
          x/y de cada celda a partir de la afín del ráster y se dibuja con
          ``pcolormesh``, que sí soporta una malla rotada.
        - Reemplaza la implementación previa (``rasterio`` + ``cartopy`` +
          ``imshow(extent=...)``), que para una malla rotada como la de
          este proyecto distorsionaba/deformaba la figura y no
          representaba la geometría real del dominio. Ya no depende de
          ``cartopy``.

        Examples
        --------
        >>> bathy.plot_sfincs_raster("dep", cmap="terrain")
        """
        raster_path = Path(self.model.root) / "gis" / f"{raster_name}.tif"
        if not raster_path.exists():
            raise FileNotFoundError(f"No existe {raster_name}.tif en {raster_path}")

        with rasterio.open(raster_path) as src:
            data = src.read(1, masked=True)
            transform = src.transform

        rows, cols = np.indices(data.shape)
        x_coords, y_coords = transform @ (cols + 0.5, rows + 0.5)

        fig, ax = plt.subplots(figsize=figsize)
        ax.set_aspect("equal")

        mesh = ax.pcolormesh(
            x_coords,
            y_coords,
            data,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            shading="auto",
        )

        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title(raster_name)

        if add_colorbar:
            plt.colorbar(mesh, ax=ax, shrink=0.8)

        # --------------------------------------------------
        # Guardado de figura
        # --------------------------------------------------
        if subdir is None:
            outdir = Path(self.model.root) / "figures"
        else:
            subdir_path = Path(subdir)
            outdir = subdir_path if subdir_path.is_absolute() else Path(self.model.root) / subdir_path

        outdir.mkdir(parents=True, exist_ok=True)

        if filename is None:
            filename = f"{raster_name}.png"

        outpath = outdir / filename
        fig.savefig(outpath, dpi=300, bbox_inches="tight")
        plt.close(fig)

        return str(outpath)
