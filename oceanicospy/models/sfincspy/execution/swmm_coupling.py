from typing import Callable, Dict, Optional

import pandas as pd

from ..preprocess.discharge_forcing import compute_exchange_discharge

#: SWMM's flow-unit systems that use liters/second (LPS) - the only ones this
#: class has been verified against, since the project's own SWMM model
#: (Los Almendros) declares FLOW_UNITS = LPS.
LPS_PER_CMS = 1000.0


class SwmmSurfaceExchange:
    """
    Orquesta un acople débil (offline, iterativo) entre SWMM y SFINCS en
    la conexión pozo-superficie, siguiendo la misma física que usa
    Iber-SWMM (Sañudo et al. 2025) para su acople fuerte (vertedero no
    sumergido / orificio sumergido), pero calculada externamente en cada
    ventana de tiempo en vez de en cada paso interno del solver 2D - ver
    la discusión de diseño de esta implementación para la justificación
    de por qué un acople fuerte no es viable con SFINCS (ejecutable
    cerrado, sin BMI operativo).

    Esta clase **no ejecuta SFINCS**. El nivel de agua superficial en
    cada nodo se obtiene mediante `surface_level_fn`, que quien la use
    debe proveer - así se puede probar de punta a punta con una función
    sintética, sin tener SFINCS corriendo, y más adelante conectarla a
    una función real que lea la salida de SFINCS de la ventana anterior.

    Parameters
    ----------
    swmm_inp_path : str
        Ruta al archivo `.inp` de SWMM.
    node_config : dict
        Mapeo ``{node_id: {"area": .., "length": .., "cd": 0.6 (opcional),
        "cw": 1.7 (opcional), "rim_elevation": .. (opcional, por defecto
        ``invert_elevation + full_depth`` del nodo)}}``. `area` y `length`
        son el área efectiva de la abertura (orificio) y la longitud
        efectiva de la cresta (vertedero) - ver
        :func:`~....preprocess.discharge_forcing.compute_exchange_discharge`.
    surface_level_fn : callable
        ``surface_level_fn(node_id, window_start, window_end) -> float``,
        nivel de agua superficial (m, mismo datum vertical que las cotas
        de SWMM) a usar para esa ventana.
    window_seconds : float, optional
        Duración de cada ventana de intercambio (por defecto 900 s = 15 min).
    cd, cw : float, optional
        Coeficientes de descarga por defecto (orificio/vertedero), usados
        cuando `node_config` no los especifica por nodo.

    Notes
    -----
    Dos detalles verificados empíricamente contra el motor real de SWMM
    (vía ``pyswmm``), no asumidos:

    - **Convención de signo**: ``compute_exchange_discharge`` devuelve
      positivo = desborde hacia la superficie (agua saliendo de la red).
      ``pyswmm``'s ``Node.generated_inflow`` interpreta un valor positivo
      como **agua entrando al nodo** - lo opuesto. Por eso aquí se le
      pasa el negativo del caudal de intercambio. Verificado
      directamente: un ``generated_inflow`` negativo sí baja el nivel de
      un nodo hasta 0 (no lo deja negativo), y uno positivo lo sube -
      confirma que la extracción de agua (nuestra "desborde") funciona
      como se espera.
    - **Unidades**: ``generated_inflow`` espera el caudal en las unidades
      de ``FLOW_UNITS`` del `.inp` (en el modelo de Los Almendros, LPS),
      no en m³/s. Esta clase hace la conversión explícita
      (``compute_exchange_discharge`` siempre devuelve m³/s).
    - Se asume que el sistema de unidades del `.inp` es LPS (o cualquier
      otro sistema métrico de caudal-en-litros); si el `.inp` usara
      unidades imperiales (CFS, MGD, etc.) esta conversión sería
      incorrecta y habría que revisarla explícitamente.
    """

    def __init__(
        self,
        swmm_inp_path: str,
        node_config: Dict[object, dict],
        surface_level_fn: Callable[[object, "pd.Timestamp", "pd.Timestamp"], float],
        window_seconds: float = 900.0,
        cd: float = 0.6,
        cw: float = 1.7,
    ) -> None:
        self.swmm_inp_path = swmm_inp_path
        self.node_config = node_config
        self.surface_level_fn = surface_level_fn
        self.window_seconds = window_seconds
        self.default_cd = cd
        self.default_cw = cw

    def run(self) -> pd.DataFrame:
        """
        Corre la simulación SWMM completa, calculando y aplicando el
        intercambio con la superficie en cada ventana.

        Returns
        -------
        pandas.DataFrame
            Índice de tiempo (fin de cada ventana) y, por cada nodo de
            `node_config`, dos columnas: ``<node_id>`` (caudal de
            intercambio, m³/s - positivo = desborde hacia la superficie,
            negativo = reingreso hacia la red) y ``<node_id>_hgl`` (nivel
            de agua en el nodo al inicio de la ventana, antes de aplicar
            el intercambio - mismo datum que `rim_elevation`). La(s)
            columna(s) de caudal están listas para pasar directamente a
            ``DischargeForcing.from_dataframe`` (ver
            :class:`~....preprocess.discharge_forcing.DischargeForcing`) -
            basta con seleccionar las columnas por nombre de nodo, sin
            las columnas ``_hgl``.
        """
        from pyswmm import Nodes, Simulation

        records = []

        with Simulation(self.swmm_inp_path) as sim:
            sim.step_advance(self.window_seconds)
            nodes = Nodes(sim)

            window_start = sim.start_time

            for _ in sim:
                window_end = sim.current_time
                row = {"time": window_end}

                for node_id, cfg in self.node_config.items():
                    node = nodes[node_id]
                    rim_elevation = cfg.get(
                        "rim_elevation", node.invert_elevation + node.full_depth
                    )
                    node_hgl = node.invert_elevation + node.depth
                    surface_level = self.surface_level_fn(node_id, window_start, window_end)

                    q = compute_exchange_discharge(
                        node_hgl=node_hgl,
                        rim_elevation=rim_elevation,
                        surface_level=surface_level,
                        area=cfg["area"],
                        length=cfg["length"],
                        cd=cfg.get("cd", self.default_cd),
                        cw=cfg.get("cw", self.default_cw),
                    )
                    row[node_id] = q
                    # HGL del nodo *antes* de aplicar el intercambio de esta
                    # ventana - útil para revisar qué tan cerca estuvo cada
                    # nodo de su cota de rebose a lo largo del evento.
                    row[f"{node_id}_hgl"] = node_hgl

                    # Signo invertido y conversión m3/s -> LPS: ver Notes.
                    node.generated_inflow(-q * LPS_PER_CMS)

                records.append(row)
                window_start = window_end

        return pd.DataFrame(records).set_index("time")
