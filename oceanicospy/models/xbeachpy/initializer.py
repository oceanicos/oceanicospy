import shutil
from pathlib import Path
from string import Template
from . import utils


class Initializer:
    """
    Utility class for setting up the directory structure and baseline configuration
    files required to run XBeach model simulations.

    It automates the creation and cleanup of the project folder tree and the
    generation of a ``params.txt`` file from a template stamped with user-supplied
    configuration flags.

    The directory layout produced by this class is:

    .. code-block:: text

        root_path/
        ├── input/      ← place your static input files here before running
        ├── pros/
        ├── run/
        │   └── params.txt     ← generated from the bundled template
        └── output/

    Parameters
    ----------
    root_path : str
        Root directory of the case.  All sub-folders are created inside this
        path.  Must end with a trailing slash (e.g. ``'path/to/case/'``).
    dict_ini_data : dict
        Case-level flags that override the package defaults in
        ``xbeachpy/utils/defaults.py`` and are substituted into
        ``params.txt``.  Typical keys:

        - ``case_description`` — free-form label (not used by XBeach)
        - ``act_morf`` — morphological updating (``0`` = off)
        - ``act_sedtrans`` — sediment transport (``0`` = off)
        - ``act_wavemodel`` — spectral wave model (``1`` = surfbeat)
        - ``dims`` — number of dimensions (``2`` = 2D)

    ini_date : datetime, optional
        Simulation start date.  Used downstream by preprocessing and
        execution classes to slice forcing datasets and compute ``tstop``.
    end_date : datetime, optional
        Simulation end date.  Same usage as ``ini_date``.
    """

    def __init__(self, root_path, dict_ini_data, ini_date=None, end_date=None):

        self.root_path = root_path
        self.ini_date = ini_date
        self.end_date = end_date
        self.dict_ini_data = dict_ini_data
        self.folder_names = ['input', 'pros', 'run', 'output']
        self.dict_folders = {
            name: f'{self.root_path}{name}/' for name in self.folder_names
        }

        print('*** Initializing XBeach model ***\n')

    def _generate_baseline_XBeach(self, template_in, template_out, replacement_dict):
        """
        Render an XBeach configuration template and write the result to disk.

        Uses :class:`string.Template` with ``safe_substitute`` so that any
        placeholder not present in *replacement_dict* is left unchanged rather
        than raising an error.

        Parameters
        ----------
        template_in : str or Path
            Path to the source template file (e.g. ``params_base.txt``).
        template_out : str or Path
            Destination path for the rendered file (e.g. ``run/params.txt``).
        replacement_dict : dict
            Key-value pairs used to fill ``$key`` placeholders inside the template.
            Unmapped placeholders are preserved as-is.
        """
        template_text = Path(template_in).read_text()
        filled_text = Template(template_text).safe_substitute(replacement_dict)
        Path(template_out).write_text(filled_text)

    def create_folders(self):
        """
        Create the full project folder structure.

        **Step 1 – top-level directories**

        Creates the four standard sub-directories under ``root_path``:
        ``input/``, ``pros/``, ``run/``, and ``output/``.  On a re-run,
        ``run/`` and ``output/`` are wiped with ``shutil.rmtree`` before being
        recreated so that stale files from a previous execution do not pollute
        the new case.  ``input/`` and ``pros/`` are always left untouched.

        All directories are created with ``Path.mkdir(parents=True,
        exist_ok=True)``, so missing intermediate paths are handled
        automatically.
        """
        print('\n\t*** Creating project folder structure ***\n')

        for folder_name in self.folder_names:
            folder_path = Path(self.dict_folders[folder_name])
            if folder_name in ['output', 'run'] and folder_path.exists():
                shutil.rmtree(folder_path)
            folder_path.mkdir(parents=True, exist_ok=True)

    def replace_ini_data(self):
        """
        Copy the base ``params.txt`` template into ``run/`` and substitute case flags.

        Locates the bundled ``params_base.txt`` template (under
        ``data/model_config_templates/xbeach/``), merges the package defaults
        from ``xbeachpy/utils/defaults.py`` with the user-supplied
        ``dict_ini_data`` (user values take precedence), and writes the
        rendered file to ``<run>/params.txt``.

        All values in ``dict_ini_data`` are cast to ``str`` before substitution
        as required by the :class:`string.Template` engine.

        Must be called after :meth:`create_folders` so that ``run/`` exists.
        """
        print('\n\t*** Copying base XBeach configuration file into run folder ***\n')

        self.script_dir = Path(__file__).resolve().parent
        self.data_dir = self.script_dir.parent.parent.parent / 'data'

        str_ini_data = {k: str(v) for k, v in self.dict_ini_data.items()}
        merged = {**utils.defaults, **str_ini_data}

        self._generate_baseline_XBeach(
            f'{self.data_dir}/model_config_templates/xbeach/params_base.txt',
            f'{self.dict_folders["run"]}params.txt',
            merged
        )
