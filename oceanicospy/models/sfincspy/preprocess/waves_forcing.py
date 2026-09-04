import os
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr
from hydromt_sfincs import SfincsModel, utils


class WavesForcing:
    """
    Generador de forzante de oleaje (`bzifile`) para SFINCS a partir de
    salidas de XBeach (1D).

    Transforma la elevación de superficie libre calculada por XBeach en
    un punto del perfil en la señal ``bzi`` que espera SFINCS: una
    variación de nivel de agua **centrada en cero**, superpuesta al nivel
    lento de ``bzs``.

    Parameters
    ----------
    model : SfincsModel
        Instancia del modelo SFINCS inicializada con HydroMT.

    Notes
    -----
    - El forzante generado corresponde a `bzifile` en SFINCS.
    - Se asume que el NetCDF proviene de una simulación XBeach 1D.
    - Se extrae una serie temporal en un punto específico del dominio.
    - **No se transfiere el wave setup de XBeach hacia `bzs`.** Esa pieza
      (sumar el nivel medio/setup calculado por XBeach al nivel de marea,
      con separación entrante/saliente y una corrección de setup
      calibrada por sitio - ver Gaido-Lasserre 2024, Nederhoff 2024)
      queda pendiente como trabajo futuro: requiere un punto de
      referencia de mayor fidelidad (p. ej. XBeach-2D) o mediciones
      independientes de nivel/runup para calibrarla, que este proyecto no
      tiene todavía.
    """

    def __init__(self, model: SfincsModel) -> None:
        self.model = model

    def from_xbeach(
        self,
        nc_path: str,
        point_index: int = 0,
        nx_index: Optional[int] = None,
        out_filename: str = "sfincs.bzi",
    ) -> str:
        """
        Genera el forzante de oleaje a partir de un archivo NetCDF de XBeach.

        Parameters
        ----------
        nc_path : str
            Ruta al archivo NetCDF de salida de XBeach.
        point_index : int, optional
            Índice sobre ``ds["pointx"]`` - las coordenadas de los puntos
            de salida tipo "point" que se hayan configurado en XBeach
            (típicamente solo el `nrugauge`, casi siempre 1 solo punto).
            Se usa para hallar, por cercanía, el índice correspondiente en
            el arreglo espacial global (``globalx``/``nx``) del que
            realmente se extrae ``zs``. **No es un índice directo sobre el
            perfil completo** - si el `.nc` solo tiene 1 punto (el caso en
            todos los casos existentes del proyecto), `point_index` solo
            puede ser 0, y el punto real usado es el del runup gauge, no
            necesariamente el que se quiere para forzar SFINCS. Ignorado
            si se da `nx_index`.
        nx_index : int, optional
            Índice directo sobre el perfil espacial completo (``globalx``/
            ``nx``, todos los puntos guardados vía `nglobalvar` - no solo
            los puntos tipo "point"). Úsalo para forzar con un punto
            específico del perfil (p. ej. donde la batimetría cruza una
            profundidad dada) sin depender de qué puntos "point" haya
            configurado el caso de XBeach. Tiene prioridad sobre
            `point_index` si se da.
        out_filename : str, optional
            Nombre del archivo de salida (por defecto ``"sfincs.bzi"``).

        Returns
        -------
        str
            Ruta completa del archivo generado.

        Raises
        ------
        RuntimeError
            Si la serie queda vacía tras el recorte temporal.
        KeyError
            Si la variable ``zs`` no existe en el NetCDF.

        Notes
        -----
        Procesamiento realizado:

        1. Lectura del NetCDF de XBeach.
        2. Extracción de la elevación de superficie libre (``zs``) - no
           la altura de ola (``H``): ``bzi`` es una variación de nivel de
           agua, no una magnitud de ola, y el manual de SFINCS exige que
           oscile alrededor de cero.
        3. Selección del punto espacial más cercano.
        4. Recorte al periodo del modelo ``[tstart, tstop]``.
        5. Resta de la media de ``zs`` calculada **sobre esa misma
           ventana recortada** (no sobre todo el NetCDF), para que la
           señal quede centrada en cero como exige SFINCS.
        6. Conversión de tiempo a segundos desde ``tref``.
        7. Escritura en formato ASCII requerido por SFINCS.

        El archivo generado tiene formato:

        ``time[s]  bzi[m]``

        donde `bzi` representa la variación (anomalía) de nivel de agua
        inducida por el oleaje, alrededor de cero.

        Examples
        --------
        >>> waves = WavesForcing(model)
        >>> waves.from_xbeach(
        ...     nc_path="xbeach_output.nc",
        ...     point_index=0
        ... )
        """

        # --------------------------------------------------
        # 1) Tiempos del modelo
        # --------------------------------------------------
        tref = utils.parse_datetime(self.model.config["tref"])
        tstart = utils.parse_datetime(self.model.config["tstart"])
        tstop = utils.parse_datetime(self.model.config["tstop"])

        # --------------------------------------------------
        # 2) Abrir NetCDF XBeach
        # --------------------------------------------------
        ds = xr.open_dataset(nc_path)

        if "zs" not in ds:
            raise KeyError("La variable 'zs' no existe en el NetCDF")

        zs = ds["zs"]                        # (globaltime, ny, nx)
        time = ds["globaltime"].values       # segundos desde inicio
        X = ds["globalx"].values[0, :]

        # --------------------------------------------------
        # 3) Selección del punto espacial
        # --------------------------------------------------
        if nx_index is not None:
            ix = nx_index
        else:
            px = ds["pointx"].values
            xp = px[point_index]
            ix = np.argmin(np.abs(X - xp))
        zs_p = zs.isel(nx=ix, ny=0).values

        # --------------------------------------------------
        # 4) Tiempo absoluto y recorte al periodo del modelo
        # --------------------------------------------------
        time_dt = pd.to_datetime(time, unit="s", origin=tref)

        df = pd.DataFrame({"zs": zs_p}, index=time_dt)
        df = df.loc[tstart:tstop]

        if df.empty:
            raise RuntimeError(
                "La serie de nivel de agua (zs) de XBeach quedó vacía tras el recorte temporal"
            )

        # --------------------------------------------------
        # 5) Anomalía centrada en cero, sobre la ventana ya recortada
        # --------------------------------------------------
        df["bzi"] = df["zs"] - df["zs"].mean()

        # --------------------------------------------------
        # 6) Tiempo en segundos desde tref
        # --------------------------------------------------
        time_sec = (df.index - tref).total_seconds().astype(int)

        df_out = pd.DataFrame(
            {
                "time": time_sec,
                "bzi": df["bzi"].values,
            }
        )

        # --------------------------------------------------
        # 7) Escritura del archivo sfincs.bzi
        # --------------------------------------------------
        out_path = os.path.join(self.model.root, out_filename)

        df_out.to_csv(
            out_path,
            sep=" ",
            index=False,
            header=False,
            float_format="%.6f",
        )

        # --------------------------------------------------
        # 8) Registro en configuración SFINCS
        # --------------------------------------------------
        self.model.set_config("bzifile", out_filename)

        return out_path
