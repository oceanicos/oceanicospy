from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize as rio_rasterize
from rasterio.transform import from_origin

from hydromt_sfincs import SfincsModel


class BottomFriction:
    """
    Rugosidad de Manning para modelos SFINCS.

    Ofrece dos niveles, ambos delegados a
    ``SfincsModel.setup_manning_roughness``:

    - :meth:`setup_uniform` - un único valor para tierra y otro para mar,
      separados por una elevación de referencia (equivalente físicamente a
      fijar ``manning_land``/``manning_sea`` a mano, pero por la vía oficial
      del paquete, que además deja registrado ``manningfile`` en vez de
      solo las claves escalares).
    - :meth:`setup_from_landcover` - rugosidad espacialmente variable a
      partir de un ráster de cobertura de suelo categórico + una tabla de
      reclasificación (clase -> Manning n). Cualquier celda sin dato en el
      ráster de cobertura (p. ej. fuera de la cuenca delimitada) se rellena
      con ``manning_land``/``manning_sea`` según la elevación, igual que en
      :meth:`setup_uniform`.

    :meth:`rasterize_landcover` y :meth:`write_reclass_table` son
    utilidades genéricas para construir esos dos insumos a partir de
    polígonos (p. ej. las subcuencas y zonas impermeables ya delimitadas
    para SWMM) - no son específicas de ningún proyecto en particular.

    Parameters
    ----------
    model : SfincsModel
        Instancia del modelo SFINCS inicializada con HydroMT.
    """

    #: Valores de Manning por defecto para un esquema de 2 clases
    #: verde/permeable vs. impermeable (calles y techos), como los delimitados
    #: para la cuenca de Los Almendros a partir de las subcuencas de SWMM.
    #: - Impermeable: rango "pavimentado" (0.010-0.015) referenciado en la
    #:   calibración manual de Manning ya hecha para este proyecto.
    #: - Verde/permeable: 0.035, el mismo valor de "manning_land" usado por
    #:   Gaido-Lasserre (2024) para un dominio SFINCS costero comparable.
    DEFAULT_LANDCOVER_MANNING: Dict[int, float] = {
        1: 0.035,  # verde / permeable
        2: 0.015,  # impermeable (calles, techos)
    }

    def __init__(self, model: SfincsModel) -> None:
        self.model = model

    # --------------------------------------------------
    # NIVEL 2: rugosidad uniforme tierra/mar
    # --------------------------------------------------
    def setup_uniform(
        self,
        manning_land: float = 0.04,
        manning_sea: float = 0.02,
        rgh_lev_land: float = 0.0,
    ) -> None:
        """
        Configura rugosidad uniforme, diferenciada tierra/mar.

        Parameters
        ----------
        manning_land : float, optional
            Manning n para celdas con elevación >= `rgh_lev_land`.
        manning_sea : float, optional
            Manning n para celdas con elevación < `rgh_lev_land`.
        rgh_lev_land : float, optional
            Elevación de referencia (m, mismo datum que `dep`) que separa
            tierra de mar para este propósito. Por defecto 0.0.
        """
        self.model.setup_manning_roughness(
            manning_land=manning_land,
            manning_sea=manning_sea,
            rgh_lev_land=rgh_lev_land,
        )

    # --------------------------------------------------
    # NIVEL 3: rugosidad espacial por cobertura de suelo
    # --------------------------------------------------
    def setup_from_landcover(
        self,
        lulc_path: Union[str, Path],
        reclass_table: Union[str, Path],
        manning_land: float = 0.035,
        manning_sea: float = 0.02,
        rgh_lev_land: float = 0.0,
    ) -> None:
        """
        Configura rugosidad espacialmente variable a partir de un ráster
        de cobertura de suelo categórico.

        Parameters
        ----------
        lulc_path : str or Path
            Ráster categórico de cobertura de suelo (una clase entera por
            celda, p. ej. generado con :meth:`rasterize_landcover`). No
            necesita estar registrado en el `data_catalog` - HydroMT acepta
            una ruta de archivo directamente.
        reclass_table : str or Path
            CSV con el mapeo clase -> Manning n (p. ej. generado con
            :meth:`write_reclass_table`).
        manning_land, manning_sea : float, optional
            Valores de respaldo para celdas fuera de la cobertura de suelo
            dada (nodata en `lulc_path`), separados por `rgh_lev_land`.
            Por defecto 0.035/0.02 - ver :attr:`DEFAULT_LANDCOVER_MANNING`
            para la justificación del valor de tierra.
        rgh_lev_land : float, optional
            Elevación de referencia (m) para el respaldo tierra/mar.

        Notes
        -----
        HydroMT interpola linealmente (``interp_method="linear"``) al
        remuestrear el ráster de cobertura sobre la malla SFINCS, así que
        las celdas justo en el borde entre dos clases pueden salir con un
        valor de Manning intermedio, no un salto abrupto entre los dos
        valores de la tabla de reclasificación. Esto es un comportamiento
        de HydroMT, no configurable desde aquí.
        """
        self.model.setup_manning_roughness(
            datasets_rgh=[{"lulc": str(lulc_path), "reclass_table": str(reclass_table)}],
            manning_land=manning_land,
            manning_sea=manning_sea,
            rgh_lev_land=rgh_lev_land,
        )

    # --------------------------------------------------
    # UTILIDADES GENÉRICAS DE CONSTRUCCIÓN DE INSUMOS
    # --------------------------------------------------
    @staticmethod
    def rasterize_landcover(
        class_polygons: Dict[int, List[gpd.GeoDataFrame]],
        out_path: Union[str, Path],
        resolution: float,
        bounds: tuple,
        crs,
        nodata: int = 0,
        dtype: str = "uint8",
    ) -> str:
        """
        Quema capas de polígonos en un ráster de cobertura de suelo
        categórico, para usar con :meth:`setup_from_landcover`.

        Las clases se queman en el orden en que aparecen en
        `class_polygons`: una clase posterior sobreescribe a una anterior
        donde se superpongan. Esto permite, por ejemplo, quemar primero el
        polígono completo de una subcuenca como clase "verde" y luego
        encima su parte impermeable como clase "impermeable", sin tener
        que calcular explícitamente la diferencia geométrica entre ambas
        capas.

        Parameters
        ----------
        class_polygons : dict of {int: list of GeoDataFrame}
            Mapeo de id de clase (entero > 0; 0 queda reservado para
            `nodata`) a una lista de GeoDataFrames con los polígonos de esa
            clase. Cada GeoDataFrame se reproyecta a `crs` si su CRS
            declarado es distinto.
        out_path : str or Path
            Ruta del GeoTIFF de salida.
        resolution : float
            Tamaño de celda (m) del ráster de salida.
        bounds : tuple of (xmin, ymin, xmax, ymax)
            Extensión del ráster de salida, en el sistema de referencia
            `crs`.
        crs : CRS-like
            Sistema de referencia del ráster de salida (p. ej. ``32617`` o
            ``"EPSG:32617"``).
        nodata : int, optional
            Valor de celda sin cobertura asignada (por defecto 0).
        dtype : str, optional
            Tipo de dato del ráster de salida (por defecto ``"uint8"``).

        Returns
        -------
        str
            Ruta del GeoTIFF generado.
        """
        xmin, ymin, xmax, ymax = bounds
        width = int(np.ceil((xmax - xmin) / resolution))
        height = int(np.ceil((ymax - ymin) / resolution))
        transform = from_origin(xmin, ymax, resolution, resolution)

        out_array = np.full((height, width), nodata, dtype=dtype)

        for class_id, gdfs in class_polygons.items():
            shapes = []
            for gdf in gdfs:
                if gdf.crs is not None and str(gdf.crs) != str(crs):
                    gdf = gdf.to_crs(crs)
                shapes.extend((geom, class_id) for geom in gdf.geometry if geom is not None)

            if not shapes:
                continue

            rio_rasterize(
                shapes,
                out_shape=(height, width),
                transform=transform,
                fill=nodata,
                out=out_array,
                dtype=dtype,
            )

        out_path = Path(out_path)
        with rasterio.open(
            out_path, "w", driver="GTiff",
            height=height, width=width, count=1,
            dtype=dtype, crs=crs, transform=transform, nodata=nodata,
        ) as dst:
            dst.write(out_array, 1)

        return str(out_path)

    @staticmethod
    def write_reclass_table(
        mapping: Dict[int, float],
        out_path: Union[str, Path],
    ) -> str:
        """
        Escribe una tabla de reclasificación clase -> Manning n en el
        formato que espera HydroMT-SFINCS (CSV con el id de clase como
        índice y una columna ``N`` con el valor de Manning).

        Parameters
        ----------
        mapping : dict of {int: float}
            Mapeo de id de clase a valor de Manning n. Por defecto puede
            usarse :attr:`DEFAULT_LANDCOVER_MANNING`.
        out_path : str or Path
            Ruta del CSV de salida.

        Returns
        -------
        str
            Ruta del CSV generado.
        """
        out_path = Path(out_path)
        lines = [",N"] + [f"{class_id},{n}" for class_id, n in mapping.items()]
        out_path.write_text("\n".join(lines) + "\n")
        return str(out_path)
