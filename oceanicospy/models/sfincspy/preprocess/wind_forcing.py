import os
from pathlib import Path

import numpy as np
import xarray as xr
import rioxarray
from rasterio.enums import Resampling
from pyproj import CRS
from hydromt_sfincs import SfincsModel, utils as hydromt_utils

from .... import utils as oceanicospy_utils
from ....utils.wind import download_era5_winds


class WindForcing:
    """
    Generador de forzante de viento espacial (ERA5) para SFINCS.

    Sigue la misma convención que ``xbeachpy.preprocess.WindForcing``: la
    descarga de ERA5 se hace con el mismo downloader compartido
    (``oceanicospy.utils.wind`` / ``ERA5Downloader``) y **siempre se ajusta
    a hora local** antes de usarse - a diferencia del ``wind_forcing.py``
    original de SFINCS, que leía el NetCDF de ERA5 tal cual (en UTC) y lo
    recortaba directamente con ``tstart``/``tstop`` (que están en hora
    local), lo que desalinea el forzante ~5 h respecto al evento simulado.

    Parameters
    ----------
    model : SfincsModel
        Instancia del modelo SFINCS inicializada con HydroMT.

    Notes
    -----
    El resultado (:meth:`from_era5_nc`) es un NetCDF con las variables:
        - amu (componente zonal del viento) [m/s]
        - amv (componente meridional del viento) [m/s]

    en el formato FEWS que espera SFINCS (``netamuamvfile``).
    """

    def __init__(self, model: SfincsModel) -> None:
        self.model = model

    # --------------------------------------------------
    # DESCARGA (ERA5, siempre ajustada a hora local)
    # --------------------------------------------------
    def get_winds_from_ERA5(
        self,
        wind_info: dict,
        input_dir: str,
        utc_offset_hours: float = -5,
        filename: str = "winds_era5.nc",
        override: bool = False,
    ) -> str:
        """
        Descarga viento ERA5 para la región del dominio y lo entrega ya
        ajustado a hora local - mismo comportamiento que
        ``xbeachpy.preprocess.WindForcing.get_winds_from_ERA5``.

        Reutiliza el mismo downloader compartido
        (``oceanicospy.utils.wind`` / ``ERA5Downloader``), así que la
        descarga cruda (UTC) y el ajuste a hora local son idénticos a los
        que usa XBeach para el mismo dominio y periodo.

        Parameters
        ----------
        wind_info : dict
            Configuración espacial del dominio de descarga. Debe contener:
            ``lon_ll_corner_wind``, ``lat_ll_corner_wind``, ``nx_wind``,
            ``ny_wind``, ``dx_wind``, ``dy_wind`` (mismas claves que en
            ``xbeachpy``).
        input_dir : str
            Carpeta ``input/`` del caso, donde se guarda el NetCDF crudo
            (UTC) y el ajustado a hora local (``<filename>_localtime.nc``).
        utc_offset_hours : float, optional
            Diferencia horaria a UTC (por defecto -5, Colombia).
        filename : str, optional
            Nombre del archivo crudo (UTC) a descargar.
        override : bool, optional
            Si es True, vuelve a descargar aunque el archivo ya exista.

        Returns
        -------
        str
            Ruta del NetCDF **ya ajustado a hora local**
            (``<filename sin extensión>_localtime.nc``), lista para pasar
            a :meth:`from_era5_nc`.
        """
        tstart = hydromt_utils.parse_datetime(self.model.config["tstart"])
        tstop = hydromt_utils.parse_datetime(self.model.config["tstop"])

        filepath = Path(f"{input_dir}{filename}")
        localtime_path = filepath.with_name(filepath.stem + "_localtime" + filepath.suffix)

        if not oceanicospy_utils.verify_file(str(filepath)) or override:
            download_era5_winds(
                wind_info, tstart, tstop, utc_offset_hours,
                str(filepath), format_localtime=True,
            )
            print("\t ERA5 wind data downloaded and shifted to local time")
        else:
            print("\t ERA5 wind data already exists, skipping download")

        return str(localtime_path)

    # --------------------------------------------------
    # PROCESAMIENTO (NetCDF ya en hora local -> forzante SFINCS)
    # --------------------------------------------------
    def from_era5_nc(
        self,
        nc_path: str,
        u_var: str = "u10",
        v_var: str = "v10",
        out_filename: str = "wind_era5.nc",
    ) -> str:
        """
        Genera un forzante de viento espacial a partir de un NetCDF de
        ERA5 **ya ajustado a hora local** (ver :meth:`get_winds_from_ERA5`).

        Parameters
        ----------
        nc_path : str
            Ruta al NetCDF de ERA5, en hora local.
        u_var : str, optional
            Nombre de la variable de viento zonal (por defecto ``"u10"``).
        v_var : str, optional
            Nombre de la variable de viento meridional (por defecto ``"v10"``).
        out_filename : str, optional
            Nombre del archivo de salida.

        Returns
        -------
        str
            Ruta completa del archivo NetCDF generado.

        Raises
        ------
        KeyError
            Si las variables de viento no existen en el dataset.

        Notes
        -----
        Flujo de procesamiento:

        1. Lectura del dataset ERA5 (se asume en hora local).
        2. Selección temporal (``tstart``-``tstop``, hora local).
        3. Limpieza de dimensiones incompatibles.
        4. Corrección de longitudes (0-360 -> -180-180).
        5. Reproyección al CRS del modelo (UTM).
        6. Interpolación a la malla SFINCS.
        7. Conversión de tiempo a segundos desde ``tref``.
        8. Reorganización de dimensiones (time, y, x).
        9. Limpieza de valores NaN.
        10. Escritura del NetCDF final.

        Examples
        --------
        >>> wind = WindForcing(model)
        >>> local_nc = wind.get_winds_from_ERA5(wind_info, input_dir="input/")
        >>> wind.from_era5_nc(local_nc)
        """

        # --------------------------------------------------
        # 1) Tiempos del modelo
        # --------------------------------------------------
        tref = hydromt_utils.parse_datetime(self.model.config["tref"])
        tstart = hydromt_utils.parse_datetime(self.model.config["tstart"])
        tstop = hydromt_utils.parse_datetime(self.model.config["tstop"])

        # --------------------------------------------------
        # 2) Abrir dataset ERA5 (hora local)
        # --------------------------------------------------
        ds = xr.open_dataset(nc_path)

        if "valid_time" in ds.dims:
            ds = ds.rename({"valid_time": "time"})

        if u_var not in ds or v_var not in ds:
            raise KeyError(
                f"No se encontraron variables '{u_var}' y/o '{v_var}' en el dataset"
            )

        ds = ds[[u_var, v_var]]
        ds = ds.sel(time=slice(tstart, tstop))

        # --------------------------------------------------
        # 3) Eliminar dimensiones no soportadas
        # --------------------------------------------------
        for dim in ["expver", "number"]:
            if dim in ds.dims:
                ds = ds.isel({dim: 0}).drop(dim)
            if dim in ds.coords:
                ds = ds.drop_vars(dim)

        # --------------------------------------------------
        # 4) Ajuste de longitudes
        # --------------------------------------------------
        if "longitude" in ds.coords and float(ds.longitude.max()) > 180:
            ds = ds.assign_coords(
                longitude=((ds.longitude + 180) % 360) - 180
            ).sortby("longitude")

        # --------------------------------------------------
        # 5) Reproyección a CRS del modelo
        # --------------------------------------------------
        epsg_model = int(self.model.config["epsg"])

        ds = ds.rio.write_crs("EPSG:4326", inplace=False)

        ds_utm = ds.rio.reproject(
            CRS.from_epsg(epsg_model),
            resampling=Resampling.bilinear,
        )

        # --------------------------------------------------
        # 6) Interpolación al grid SFINCS
        # --------------------------------------------------
        # La malla del proyecto está rotada (rotation != 0 en todos los
        # casos conocidos). Para una malla rotada, HydroMT-SFINCS no puede
        # representar la posición real de cada celda con arreglos 1D de
        # x/y - por eso `model.grid.x`/`.y` son solo índices de celda
        # (0,1,2,...), no coordenadas reales. Interpolar directamente con
        # esos índices como si fueran coordenadas UTM (como hacía el
        # código original) produce un campo totalmente fuera de rango:
        # verificado que da amu/amv = 0.0 en todo el dominio, sin ningún
        # error o aviso. Las coordenadas reales por celda sí existen en
        # `model.reggrid.coordinates` (xc/yc, 2D), así que se usa
        # interpolación "avanzada" (vectorizada) con esos arreglos 2D en
        # vez de los índices de `grid.x`/`grid.y`.
        grid = self.model.grid
        coords = self.model.reggrid.coordinates
        xc = xr.DataArray(coords["xc"][1], dims=coords["xc"][0])
        yc = xr.DataArray(coords["yc"][1], dims=coords["yc"][0])

        ds_utm = ds_utm.interp(
            x=xc,
            y=yc,
            method="linear",
        )

        # --------------------------------------------------
        # 7) Renombrar variables
        # --------------------------------------------------
        ds_utm = ds_utm.rename(
            {
                u_var: "amu",
                v_var: "amv",
            }
        )

        # --------------------------------------------------
        # 8) Conversión temporal
        # --------------------------------------------------
        time_seconds = (
            (ds_utm.time.values - np.datetime64(tref))
            / np.timedelta64(1, "s")
        ).astype("float64")

        ds_utm = ds_utm.assign_coords(
            time=("time", time_seconds)
        )

        ds_utm.time.attrs["units"] = (
            f"seconds since {tref:%Y-%m-%d %H:%M:%S}"
        )

        # --------------------------------------------------
        # 9) Ajuste de coordenadas espaciales
        # --------------------------------------------------
        # Aquí sí es correcto volver a usar grid.x/grid.y (los índices de
        # celda): es la misma convención que usa internamente HydroMT-SFINCS
        # para etiquetar las dimensiones x/y de una malla rotada - a
        # diferencia del paso 6, esto no se usa para interpolar, solo para
        # que el NetCDF final quede dimensionalmente consistente con lo que
        # SFINCS espera.
        ds_utm = ds_utm.assign_coords(
            x=("x", grid.x.values),
            y=("y", grid.y.values),
        )

        ds_utm["x"].attrs["units"] = "m"
        ds_utm["y"].attrs["units"] = "m"

        # --------------------------------------------------
        # 10) Orden requerido por SFINCS
        # --------------------------------------------------
        ds_utm = ds_utm.transpose("time", "y", "x")

        # --------------------------------------------------
        # 11) Limpieza de datos
        # --------------------------------------------------
        ds_utm["amu"] = ds_utm["amu"].fillna(0.0).astype("float32")
        ds_utm["amv"] = ds_utm["amv"].fillna(0.0).astype("float32")

        ds_utm["amu"].attrs["units"] = "m s-1"
        ds_utm["amv"].attrs["units"] = "m s-1"

        # --------------------------------------------------
        # 12) Guardar NetCDF
        # --------------------------------------------------
        out_path = os.path.join(self.model.root, out_filename)
        ds_utm.to_netcdf(out_path)

        # --------------------------------------------------
        # 13) Registrar en SFINCS
        # --------------------------------------------------
        self.model.set_forcing(ds_utm, name="netamuamv")
        self.model.set_config("netamuamvfile", out_filename)

        return out_path
