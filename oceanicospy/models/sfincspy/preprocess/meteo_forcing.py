import os

import pandas as pd
from hydromt_sfincs import SfincsModel, utils


class UniformMetForcingClassic:
    """
    Generador de forzantes meteorológicos uniformes en formato clásico de
    SFINCS, a partir de una estación en Excel. Compatible con:
        - Excel tipo AG5 (columna ``Date/Time``, incluye presión)
        - Excel generado (columna ``Fecha``, sin presión)

    Parameters
    ----------
    model : SfincsModel
        Instancia del modelo SFINCS inicializada con HydroMT.

    Notes
    -----
    - Se asume que la hora de la estación ya está en hora local (no se
      aplica ningún desplazamiento de zona horaria, a diferencia de
      :class:`WaterLevelForcing`/:class:`WindForcing`, que sí lo hacen
      para datos de UHSLC/ERA5).
    - **El viento no se escribe por defecto** (``write_wind=False``). La
      decisión de este proyecto es forzar viento con ERA5
      (:class:`WindForcing`), no con esta estación: la estación está a
      ~1.4 km del dominio y a una altura de sensor distinta de los 10 m
      estándar que asume la fórmula de arrastre de viento de SFINCS
      (Vatvani et al. 2012), mientras que ERA5 ya está nativamente a 10 m
      y es la misma fuente que usa XBeach. Ver la comparación cuantitativa
      hecha para esta decisión (estación vs. ERA5, r=0.43, sesgo +1.1 m/s).
      El parámetro ``write_wind`` se deja disponible únicamente como
      alternativa documentada, no como uso recomendado.
    """

    def __init__(self, model: SfincsModel) -> None:
        self.model = model

    def from_excel(
        self,
        excel_path: str,
        time_col: str = None,
        precip_col: str = "Precipitacion (mm)",
        pressure_col: str = "Presion Barometrica (hPa)",
        write_wind: bool = False,
        wind_speed_col: str = "Velocidad Viento (m/s)",
        wind_dir_col: str = "Direccion Viento (°)",
    ) -> None:
        """
        Genera forzantes meteorológicos uniformes a partir de un Excel de
        estación.

        Siempre escribe precipitación (``sfincs.prcp``). Escribe presión
        atmosférica (``sfincs.patm``) si `pressure_col` está presente en
        el archivo. **No** escribe viento a menos que se pida
        explícitamente con ``write_wind=True`` (ver Notes de la clase).

        Parameters
        ----------
        excel_path : str
            Ruta al archivo Excel de la estación.
        time_col : str, optional
            Nombre de la columna de tiempo. Si es ``None``, se detecta
            automáticamente (``"Date/Time"`` o ``"Fecha"``).
        precip_col : str, optional
            Nombre de la columna de precipitación (mm).
        pressure_col : str, optional
            Nombre de la columna de presión barométrica (hPa). Se
            convierte a Pa para SFINCS.
        write_wind : bool, optional
            Si es True, además escribe ``sfincs.wnd`` con la velocidad y
            dirección de viento de la estación. Por defecto ``False`` -
            ver Notes de la clase sobre por qué el viento de este
            proyecto viene de ERA5, no de esta estación.
        wind_speed_col, wind_dir_col : str, optional
            Nombres de columnas de viento, solo usados si
            ``write_wind=True``.

        Raises
        ------
        ValueError
            Si no se encuentra columna de tiempo.
        RuntimeError
            Si alguna de las series queda vacía tras el recorte temporal
            o la limpieza de nulos.
        """

        # --------------------------------------------------
        # 1) Leer tiempos del modelo
        # --------------------------------------------------
        tref = utils.parse_datetime(self.model.config["tref"])
        tstart = utils.parse_datetime(self.model.config["tstart"])
        tstop = utils.parse_datetime(self.model.config["tstop"])

        # --------------------------------------------------
        # 2) Leer archivo Excel
        # --------------------------------------------------
        df = pd.read_excel(excel_path)

        if time_col is None:
            if "Date/Time" in df.columns:
                time_col = "Date/Time"
            elif "Fecha" in df.columns:
                time_col = "Fecha"
            else:
                raise ValueError(
                    "No se encontró columna de tiempo ('Date/Time' o 'Fecha')"
                )

        df[time_col] = pd.to_datetime(df[time_col])
        df = df.set_index(time_col).sort_index()

        # --------------------------------------------------
        # 3) Recorte temporal
        # --------------------------------------------------
        df = df.loc[tstart:tstop]

        if df.empty:
            raise RuntimeError(
                "La serie meteorológica quedó vacía tras recortar por tstart-tstop"
            )

        # --------------------------------------------------
        # 4) Tiempo en segundos desde tref
        # --------------------------------------------------
        time_sec = (df.index - tref).total_seconds().astype(int)

        # --------------------------------------------------
        # 5) VIENTO -> sfincs.wnd (opcional, ver Notes de la clase)
        # --------------------------------------------------
        if write_wind:
            wnd = pd.DataFrame(
                {
                    "time": time_sec,
                    "vmag": df[wind_speed_col].astype(float),
                    "vdir": df[wind_dir_col].astype(float),
                }
            ).dropna()

            if wnd.empty:
                raise RuntimeError("Archivo de viento quedó vacío")

            wnd_path = os.path.join(self.model.root, "sfincs.wnd")
            wnd.to_csv(wnd_path, sep=" ", index=False, header=False)

            self.model.set_config("wndfile", "sfincs.wnd")

        # --------------------------------------------------
        # 6) PRECIPITACIÓN -> sfincs.prcp
        # --------------------------------------------------
        prcp = pd.DataFrame(
            {
                "time": time_sec,
                "prcp": df[precip_col].astype(float),
            }
        ).dropna()

        if prcp.empty:
            raise RuntimeError("Archivo de precipitación quedó vacío")

        prcp_path = os.path.join(self.model.root, "sfincs.prcp")
        prcp.to_csv(prcp_path, sep=" ", index=False, header=False)

        self.model.set_config("precipfile", "sfincs.prcp")

        # --------------------------------------------------
        # 7) PRESIÓN ATMOSFÉRICA -> opcional, según disponibilidad
        # --------------------------------------------------
        if pressure_col in df.columns:
            patm = pd.DataFrame(
                {
                    "time": time_sec,
                    "patm": df[pressure_col].astype(float) * 100.0,  # hPa -> Pa
                }
            ).dropna()

            if patm.empty:
                raise RuntimeError("Archivo de presión atmosférica quedó vacío")

            patm_path = os.path.join(self.model.root, "sfincs.patm")
            patm.to_csv(patm_path, sep=" ", index=False, header=False)

            self.model.set_config("patmfile", "sfincs.patm")

            print("\t Presión incluida (sfincs.patm generado)")
        else:
            print("\t No se encontró presión, no se genera sfincs.patm")
