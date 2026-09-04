import os
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd
from hydromt_sfincs import SfincsModel, utils

GRAVITY = 9.81


# --------------------------------------------------
# FÍSICA DEL INTERCAMBIO 1D-2D (funciones puras, sin SWMM)
# --------------------------------------------------
def orifice_discharge(head_diff: float, area: float, cd: float = 0.6) -> float:
    """
    Caudal de orificio sumergido, según la formulación clásica
    (la misma usada por SFINCS para su propio `drnfile` tipo "culvert",
    ``par1 = Cd * A * sqrt(2g)``, y por Iber-SWMM para la condición
    sumergida en la conexión pozo-superficie: Sañudo et al. 2025).

    Parameters
    ----------
    head_diff : float
        Diferencia de carga entre los dos lados de la conexión (m).
        Puede ser negativa - el signo del resultado sigue el de
        `head_diff` (caudal positivo en la dirección de mayor a menor
        carga).
    area : float
        Área efectiva de la abertura (m²) - p. ej. el área de la rejilla
        o tapa del pozo.
    cd : float, optional
        Coeficiente de descarga adimensional. Por defecto 0.6 - valor de
        uso general para orificios, coincide con el que el manual de
        SFINCS recomienda para su formulación de weir (``Cd = 0.6``).

    Returns
    -------
    float
        Caudal (m³/s), con el mismo signo que `head_diff`.
    """
    return np.sign(head_diff) * cd * area * np.sqrt(2 * GRAVITY * abs(head_diff))


def weir_discharge(head: float, length: float, cw: float = 1.7) -> float:
    """
    Caudal de vertedero de pared delgada, para la condición no sumergida
    (el nivel del lado receptor está por debajo de la cresta/cota de
    referencia).

    Parameters
    ----------
    head : float
        Carga sobre la cresta del vertedero (m). Valores <= 0 devuelven
        caudal 0 (no hay flujo si el nivel no supera la cresta).
    length : float
        Longitud efectiva de la cresta (m) - p. ej. el perímetro de la
        rejilla o tapa del pozo.
    cw : float, optional
        Coeficiente de vertedero. Por defecto 1.7 (rango típico para
        vertedero de pared delgada en unidades SI, 1.7-1.84).

    Returns
    -------
    float
        Caudal (m³/s), siempre >= 0.
    """
    if head <= 0:
        return 0.0
    return cw * length * head**1.5


def compute_exchange_discharge(
    node_hgl: float,
    rim_elevation: float,
    surface_level: float,
    area: float,
    length: float,
    cd: float = 0.6,
    cw: float = 1.7,
) -> float:
    """
    Caudal de intercambio pozo-superficie en un instante dado, replicando
    la lógica de Iber-SWMM (Sañudo et al. 2025): vertedero cuando la
    superficie está por debajo de la cota del pozo/rejilla (no sumergido),
    orificio cuando ambos lados están sumergidos.

    Convención de signo: **positivo = desborde del pozo hacia la
    superficie** (agua saliendo de la red hacia la calle); **negativo =
    reingreso** (agua de la calle entrando de vuelta a la red, una vez
    hay capacidad).

    Parameters
    ----------
    node_hgl : float
        Línea de energía (nivel de agua) en el nodo SWMM, en el mismo
        datum vertical que `rim_elevation`/`surface_level`.
    rim_elevation : float
        Cota de la tapa/rejilla del pozo (mismo datum).
    surface_level : float
        Nivel de agua en la superficie junto al pozo (p. ej. calculado
        por SFINCS en la celda correspondiente), mismo datum.
    area : float
        Área efectiva de la abertura (m²), para el régimen de orificio.
    length : float
        Longitud efectiva de la cresta (m), para el régimen de vertedero.
    cd, cw : float, optional
        Coeficientes de descarga de orificio y vertedero.

    Returns
    -------
    float
        Caudal de intercambio (m³/s). Positivo = desborde hacia la
        superficie; negativo = reingreso hacia la red.

    Notes
    -----
    Esta función es independiente de SWMM y de SFINCS - solo necesita
    los tres niveles como argumentos. Quien la use es responsable de
    obtener `node_hgl` (p. ej. de ``pyswmm``) y `surface_level` (p. ej.
    de la salida de SFINCS) en el mismo datum vertical que el DEM.
    """
    surface_submerged = surface_level > rim_elevation

    if not surface_submerged:
        # Calle seca (o a lo sumo al nivel de la tapa): solo puede haber
        # flujo si el nodo se satura por encima de la cota del pozo -
        # vertedero no sumergido. Si el nodo tampoco supera la cota, no
        # hay ninguna conexión activa.
        if node_hgl > rim_elevation:
            return weir_discharge(node_hgl - rim_elevation, length, cw=cw)
        return 0.0

    # Calle ya inundada (nivel superficial por encima de la cota del pozo):
    # orificio sumergido en ambos lados, con signo dado por quién tiene
    # mayor carga - esto cubre tanto que el desborde continúe (nodo más
    # alto) como el reingreso (nodo baja por debajo del nivel de la calle).
    return orifice_discharge(node_hgl - surface_level, area, cd=cd)


# --------------------------------------------------
# FORZANTE DE DESCARGA PUNTUAL PARA SFINCS (genérico, sin SWMM)
# --------------------------------------------------
class DischargeForcing:
    """
    Generador de forzante de descarga puntual (`srcfile`/`disfile`) para
    SFINCS, a partir de una o más series de tiempo de caudal.

    Es completamente genérico respecto al origen de la serie: puede venir
    de un intercambio con SWMM (ver :func:`compute_exchange_discharge`),
    de una prueba sintética, o de cualquier otra fuente - esta clase solo
    necesita la serie y las coordenadas del punto. Esto permite probar el
    forzante de SFINCS de forma aislada, sin necesidad de tener SWMM
    corriendo.

    Parameters
    ----------
    model : SfincsModel
        Instancia del modelo SFINCS inicializada con HydroMT.
    """

    def __init__(self, model: SfincsModel) -> None:
        self.model = model
        #: Mapeo {id_punto original -> índice entero SFINCS}, poblado por
        #: la última llamada a :meth:`from_dataframe`.
        self.point_ids: Dict[object, int] = {}

    def from_dataframe(
        self,
        df: pd.DataFrame,
        locations: Dict[object, Tuple[float, float]],
    ) -> str:
        """
        Registra uno o más puntos de descarga a partir de un DataFrame.

        Parameters
        ----------
        df : pandas.DataFrame
            Índice de tiempo (hora local) y una columna de caudal (m³/s)
            por cada punto, con el mismo nombre que las llaves de
            `locations`.
        locations : dict
            Mapeo ``{id_punto: (x, y)}`` en el CRS del modelo. `id_punto`
            debe coincidir con el nombre de la columna correspondiente en
            `df` (p. ej. los nombres de nodo de SWMM: ``"R1"``, ``"R2"``...).
            HydroMT-SFINCS exige que las columnas de la serie y el índice
            de `locations` sean enteros (misma convención que usa SFINCS
            internamente para numerar puntos de frontera/descarga) - esta
            función hace la conversión por dentro, así que `id_punto`
            puede ser cualquier identificador legible (p. ej. el nombre
            del nodo SWMM); el mapeo usado queda en
            :attr:`point_ids` para poder rastrear qué índice SFINCS
            corresponde a cada punto original.

        Returns
        -------
        str
            Nombre del archivo de descarga (``disfile``) que SFINCS
            escribirá al llamar a ``model.write()`` - HydroMT controla el
            nombre real, no se escribe nada en esta llamada.

        Raises
        ------
        ValueError
            Si algún id en `locations` no tiene columna correspondiente
            en `df`.
        RuntimeError
            Si la serie queda vacía tras el recorte temporal.
        """
        missing = [pid for pid in locations if pid not in df.columns]
        if missing:
            raise ValueError(
                f"Faltan columnas en df para los puntos: {missing}"
            )

        tstart = utils.parse_datetime(self.model.config["tstart"])
        tstop = utils.parse_datetime(self.model.config["tstop"])

        ids = list(locations.keys())
        self.point_ids = {pid: i + 1 for i, pid in enumerate(ids)}

        df_out = df[ids].loc[tstart:tstop].dropna()

        if df_out.empty:
            raise RuntimeError(
                "La serie de descarga quedó vacía tras el recorte temporal"
            )

        df_out = df_out.rename(columns=self.point_ids)

        gdf = gpd.GeoDataFrame(
            index=[self.point_ids[pid] for pid in ids],
            geometry=gpd.points_from_xy(
                [locations[pid][0] for pid in ids],
                [locations[pid][1] for pid in ids],
            ),
            crs=self.model.crs,
        )

        self.model.setup_discharge_forcing(timeseries=df_out, locations=gdf)

        return self.model.config.get("disfile", "sfincs.dis")
