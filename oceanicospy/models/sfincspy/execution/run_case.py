import shutil
from pathlib import Path

from .... import utils


class CaseRunner:
    """
    Genera y envía el script SLURM para correr un caso SFINCS ya armado
    (ver :class:`~oceanicospy.models.sfincspy.preprocess.prepare_case.SfincsCaseBuilder`)
    en el clúster `fisica`, vía Singularity.

    Sigue el mismo patrón que
    ``xbeachpy.execution.CaseRunner.fill_slurm_file``, pero con una
    plantilla propia (``launcher_sfincs_cecc_base.slurm``) - a diferencia
    de XBeach, SFINCS se ejecuta con OpenMP dentro del contenedor
    Singularity, **sin** `mpirun` (confirmado contra los
    `slurm_launch.sh` reales de este proyecto en
    `SFINCSPY/Casos/*/run/slurm_launch.sh`), y no mueve archivos de
    resultado al terminar: `sfincs_map.nc`/`sfincs_his.nc`/`sfincs.log`
    quedan en la misma carpeta `run/` donde corrió (esa es la
    convención real ya usada en el proyecto), y solo la salida propia de
    SLURM (`.out`/`.err`) va a `output/`.

    Parameters
    ----------
    dict_folders : dict
        Diccionario de carpetas del caso, con al menos las llaves
        ``"run"`` y ``"output"`` (ver
        :class:`~oceanicospy.models.sfincspy.Initializer`).

    Notes
    -----
    SFINCS lee ``sfincs.inp`` del directorio de trabajo actual (no recibe
    la ruta del caso como argumento) - por eso la plantilla hace ``cd``
    a `run_path_case` antes de invocar el ejecutable.
    """

    def __init__(self, dict_folders: dict) -> None:
        self.dict_folders = dict_folders
        print('\n*** Initializing SFINCS Case Runner ***\n')

    def fill_slurm_file(
        self,
        case_name: str,
        ntasks: int,
        image_name: str = "sfincs-cpu_sfincs-v2.3.0-mt-Faber-Release.sif",
    ) -> str:
        """
        Copia la plantilla `.slurm` de SFINCS a la carpeta `run/` del
        caso y rellena sus campos.

        Parameters
        ----------
        case_name : str
            Nombre del caso (usado como ``--job-name``).
        ntasks : int
            Número de tareas por nodo (``--ntasks-per-node``) - en la
            práctica, el número de hilos OpenMP disponibles para SFINCS.
        image_name : str, optional
            Nombre del contenedor Singularity en ``/localapps`` (por
            defecto la imagen v2.3.0 "Faber" ya usada en producción para
            este proyecto).

        Returns
        -------
        str
            Ruta completa del script `.slurm` generado, listo para
            enviarse con ``sbatch``.
        """
        script_dir = Path(__file__).resolve().parent.parent.parent.parent
        data_dir = script_dir.parent / 'data'

        template_path = data_dir / 'hpc_slurm_templates' / 'launcher_sfincs_cecc_base.slurm'
        out_path = Path(self.dict_folders['run']) / 'launcher_sfincs.slurm'

        shutil.copy(template_path, out_path)

        launch_dict = dict(
            case_name=case_name,
            run_path_case=f'{self.dict_folders["run"]}',
            output_path_case=f'{self.dict_folders["output"]}',
            number_tasks=ntasks,
            image_name=image_name,
        )
        utils.fill_files(str(out_path), launch_dict)

        return str(out_path)
