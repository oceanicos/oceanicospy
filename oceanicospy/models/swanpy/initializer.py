import shutil
from pathlib import Path
from string import Template

class Initializer:
    """
    Utility class for setting up the directory structure and baseline configuration
    files required to run SWAN (Simulating WAves Nearshore) model simulations.

    It automates the creation and cleanup of the project folder tree, domain-specific
    sub-folders, and the generation of a per-domain ``run.swn`` file from a template.

    The directory layout produced by this class is:

    .. code-block:: text

        root_path/
        ├── input/
        │   └── domain_XX/      ← this needs to be populated by the user with input files before running the case
        ├── pros/
        ├── run/
        │   └── domain_XX/
        │       └── run.swn     ← generated from template
        └── output/
            └── domain_XX/

    Parameters
    ----------
    root_path : str
        Root directory path where the project structure is created.
    dict_ini_data : dict
        Dictionary with case metadata and configuration parameters for SWAN.
        Expected keys:

        * ``name`` (*str*) – case name.
        * ``case_number`` (*int*) – numeric identifier for the case.
        * ``case_description`` (*str*) – free-text description.
        * ``stat_id`` (*int*) – ``0`` for non-stationary, ``1`` for stationary.
        * ``number_domains`` (*int*) – total number of computational domains.
        * ``parent_domains`` (*dict*) – mapping ``{child_id: parent_id}`` where
          ``parent_id`` is ``None`` for top-level domains.

    ini_date : datetime or None
        Start date of the model run. Default is ``None``. 
    end_date : datetime or None
        End date of the model run. Default is ``None``.

    Notes
    -----
    * The ``run.swn`` template is selected based on the value of ``stat_id`` in ``dict_ini_data``:
        * If ``stat_id == 0``, the non-stationary template ``run_base_nonstat.swn`` is used.
        * If ``stat_id == 1``, the stationary template ``run_base_stat.swn`` is used.   
    """

    def __init__(self, root_path, dict_ini_data, ini_date = None, end_date = None):
        self.root_path = root_path
        self.ini_date = ini_date
        self.end_date = end_date
        self.dict_ini_data = dict_ini_data
        self.folder_names = ['input', 'pros', 'run', 'output']
        self.dict_folders = {
            name: f'{self.root_path}{name}/' for name in self.folder_names
        }

        print('*** Initializing SWAN model ***\n')

    def _generate_baseline_SWAN(self, template_in, template_out, replacement_dict):
        """
        Render a SWAN configuration template and write the result to disk.

        Uses :class:`string.Template` with ``safe_substitute`` so that any
        placeholder not present in *replacement_dict* is left unchanged rather
        than raising an error.

        Parameters
        ----------
        template_in : str or Path
            Path to the source template file (e.g. ``run_base_stat.swn``).
        template_out : str or Path
            Destination path for the rendered file (e.g. ``run/domain_01/run.swn``).
        replacement_dict : dict
            Key-value pairs used to fill ``$key`` placeholders inside the template.
            Unmapped placeholders are preserved as-is.
        """
        template_text = Path(template_in).read_text()
        filled_text = Template(template_text).safe_substitute(replacement_dict)
        Path(template_out).write_text(filled_text)

    def create_folders(self):
        """
        Create the full project folder structure for all computational domains.

        **Step 1 – top-level directories**

        Creates the four standard sub-directories under ``root_path``:
        ``input/``, ``pros/``, ``run/``, and ``output/``.  On a re-run,
        ``run/`` and ``output/`` are wiped with ``shutil.rmtree`` before being
        recreated so that stale files from a previous execution do not pollute
        the new case.  ``input/`` and ``pros/`` are always left untouched.

        **Step 2 – per-domain sub-directories**

        Iterates over ``[1, number_domains]`` (read from
        ``dict_ini_data['number_domains']``) and creates a ``domain_0{N}/``
        directory under both ``run/`` and ``output/``.  Each directory is
        created fresh — if it already exists it is removed first.

        All directories are created with ``Path.mkdir(parents=True,
        exist_ok=True)``, so missing intermediate paths are handled
        automatically.
        """
        print('\n\t*** Creating project folder structure ***\n')

        # Step 1 – top-level folders
        for folder_name in self.folder_names:
            folder_path = Path(self.dict_folders[folder_name])
            if folder_name in ['output', 'run'] and folder_path.exists():
                shutil.rmtree(folder_path)
            folder_path.mkdir(parents=True, exist_ok=True)

        # Step 2 – per-domain sub-folders under run/ and output/
        for domain in range(1, self.dict_ini_data['number_domains'] + 1):
            for top_folder in ['output', 'run']:
                domain_path = Path(self.dict_folders[top_folder]) / f'domain_0{domain}'
                if domain_path.exists():
                    shutil.rmtree(domain_path)
                domain_path.mkdir(parents=True, exist_ok=True)

    def replace_ini_data(self):
        """
        Generate per-domain ``run.swn`` files from the appropriate SWAN template.

        Selects the correct base template based on ``dict_ini_data['stat_id']``:

        * ``stat_id == 0`` → ``run_base_nonstat.swn``
        * ``stat_id == 1`` → ``run_base_stat.swn``

        The selected template is filled using :meth:`_generate_baseline_SWAN`
        with user-supplied ``dict_ini_data``.  User values take precedence
        over defaults for any key that appears in both.

        The rendered ``run.swn`` is written to
        ``run/domain_0{N}/run.swn`` for every domain in
        ``[1, number_domains]``.

        """
        self.stat_label = 'NONSTAT' if self.dict_ini_data['stat_id'] == 0 else 'STAT'
        self.dict_ini_data['stat_label'] = self.stat_label

        print('\n\t*** Copying base SWAN configuration file into run folder ***\n')

        self.script_dir = Path(__file__).resolve().parent
        self.data_dir = self.script_dir.parent.parent.parent / 'data'

        template_path = (
            f'{self.data_dir}/model_config_templates/swan/'
            f'run_base_{self.stat_label.lower()}.swn'
        )

        for domain in range(1, self.dict_ini_data['number_domains'] + 1):
            output_path = f'{self.dict_folders["run"]}domain_0{domain}/run.swn'
            self._generate_baseline_SWAN(template_path, output_path, self.dict_ini_data)
