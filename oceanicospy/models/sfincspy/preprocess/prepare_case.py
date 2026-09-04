from hydromt_sfincs import SfincsModel

from .gridmaker import GridMaker
from .bathymaker import BathyMaker
from .bottom_friction import BottomFriction
from .waterlevel_forcing import WaterLevelForcing
from .waves_forcing import WavesForcing
from .wind_forcing import WindForcing
from .meteo_forcing import UniformMetForcingClassic
from .discharge_forcing import DischargeForcing


class SfincsCaseBuilder:
    """
    Punto de entrada para armar un caso SFINCS.

    Crea el ``SfincsModel`` compartido y expone cada módulo de
    preprocesamiento ya conectado a él (`grid`, `bathy`, `friction`,
    `waterlevel`, `waves`, `wind`, `meteo`, `discharge`) para llamarlos
    explícitamente, en orden, desde el script del caso - el mismo patrón
    que ya usa ``main_xbeach.py`` con ``xbeachpy.preprocess.*``.

    Deliberadamente **no** ofrece un método único que arme "todo el
    caso" a partir de un diccionario de parámetros: cada forzante se
    decide y se llama a mano en el script, para que sea trazable qué
    insumo, qué rugosidad y qué fuente se usó en cada corrida - esto
    importa especialmente al reconstruir casos de calibración ya
    existentes, donde se necesita poder auditar cada decisión.

    Parameters
    ----------
    root : str
        Directorio raíz donde se construirá el modelo SFINCS (típicamente
        ``<caso>/run/``, creado antes con
        :class:`~oceanicospy.models.sfincspy.Initializer`).
    data_catalog : str, optional
        Ruta al ``data_catalog.yml`` del caso (ver
        :meth:`~oceanicospy.models.sfincspy.Initializer.write_data_catalog`).
    mode : str, optional
        Modo de escritura del modelo (por defecto ``"w+"``).

    Attributes
    ----------
    model : SfincsModel
        Instancia compartida por todos los módulos siguientes.
    grid : GridMaker
    bathy : BathyMaker
    friction : BottomFriction
    waterlevel : WaterLevelForcing
    waves : WavesForcing
    wind : WindForcing
    meteo : UniformMetForcingClassic
    discharge : DischargeForcing

    Examples
    --------
    >>> builder = SfincsCaseBuilder(root="CasoX/run", data_catalog="CasoX/input/data_catalog.yml")
    >>> builder.setup_time(tref="20250618 000000", tstart="20250618 000000", tstop="20250626 230000")
    >>> builder.grid.setup_grid(plot=False, x0=..., y0=..., dx=1, dy=1, mmax=..., nmax=..., rotation=38, epsg=32617)
    >>> builder.bathy.setup_bathy(zmin_active=-4, zmax_bounds=-2, plot=False)
    >>> builder.friction.setup_uniform(manning_land=0.04, manning_sea=0.02)
    >>> builder.waterlevel.from_dataframe(df_tide, column_name="depth[m]", x_bnd=423820, y_bnd=1389625)
    >>> builder.waves.from_xbeach(nc_path="E1_profile1D_Nov2008.nc", point_index=0)
    >>> builder.wind.from_era5_nc(nc_path=local_era5_nc)
    >>> builder.meteo.from_excel("AG5-0359.xlsx")
    >>> builder.write()
    """

    def __init__(
        self,
        root: str,
        data_catalog: str = "data_catalog.yml",
        mode: str = "w+",
    ) -> None:
        self.root = root
        self.data_catalog = data_catalog

        self.model = SfincsModel(
            data_libs=[data_catalog],
            root=root,
            mode=mode,
        )

        self.grid = GridMaker(self.model)
        self.bathy = BathyMaker(self.model, data_catalog)
        self.friction = BottomFriction(self.model)
        self.waterlevel = WaterLevelForcing(self.model)
        self.waves = WavesForcing(self.model)
        self.wind = WindForcing(self.model)
        self.meteo = UniformMetForcingClassic(self.model)
        self.discharge = DischargeForcing(self.model)

    def setup_time(self, tref: str, tstart: str, tstop: str) -> None:
        """
        Configura el dominio temporal del modelo.

        Debe llamarse antes que cualquier forzante (todos leen
        ``tref``/``tstart``/``tstop`` de ``model.config``).

        Parameters
        ----------
        tref : str
            Tiempo de referencia del modelo (``"AAAAMMDD HHMMSS"``).
        tstart : str
            Tiempo de inicio de la simulación.
        tstop : str
            Tiempo de fin de la simulación.
        """
        self.model.setup_config(tref=tref, tstart=tstart, tstop=tstop)

    def write(self) -> None:
        """Escribe todos los archivos nativos de SFINCS a `root`."""
        self.model.write()
        self.model.write_config()
