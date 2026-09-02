import pandas as pd
from pathlib import Path
from datetime import datetime
import re

from .... import utils

class CaseRunner():
    """
    CaseRunner is a utility class for finalizing the configuration of a SWAN case and preparing it for execution.

    Parameters
    ----------
    init : object
        Initialization object containing configuration data and folder paths.
    domain_number : int
        Identifier for the domain being processed.
    dict_comp_data : dict
        Computation parameters for the SWAN run (dates, time step, stationary flag, etc.).
    all_domains : dict
        Dictionary with configuration info for all domains in the simulation.
    """

    def __init__(self,init,domain_number,dict_comp_data,all_domains=None):
        self.init = init
        self.domain_number = domain_number
        self.dict_comp_data = dict_comp_data
        self.all_domains = all_domains
        print(f'\n*** Initializing Case Runner for domain {self.domain_number} ***\n')

    def _delete_placeholder_leftover(self):
        """
        Replace any unfilled ``$placeholder`` tokens in the run file with whitespace.

        Scans the domain ``.swn`` run file and overwrites any remaining
        ``$word`` tokens that were not substituted during preprocessing.
        """
        run_file = Path(f'{self.init.dict_folders["run"]}domain_0{self.domain_number}/run.swn')
        text = run_file.read_text()
        text = re.sub(r"\$\w+", " ", text)  # wipe leftovers
        run_file.write_text(text)

    def define_output_from_file(self,filename=None):
        """
        Write a SWAN output location file from a CSV of point coordinates.

        Reads a CSV file with ``X`` and ``Y`` coordinate columns, wraps any
        negative longitudes to [0, 360), and writes the result to
        ``points.loc`` in the domain run directory.

        Parameters
        ----------
        filename : str
            Name of the CSV file located in the domain input folder. Must
            contain at least ``X`` and ``Y`` columns.
        """

        ds = pd.read_csv(f'{self.init.dict_folders["input"]}domain_0{self.domain_number}/{filename}',delimiter=',')
        ds = ds[['X','Y']]
        if (ds['X'] < 0).any():
            ds.loc[ds['X'] < 0, 'X'] += 360
        ds.to_csv(f'{self.init.dict_folders["run"]}domain_0{self.domain_number}/points.loc',
                  index=False, header=False, na_rep=0, float_format='%7.7f',sep=' ')
    
    def write_nest_section(self):
        """
        Write the nesting section of the SWAN run file for the current domain.

        Determines which domains are nested inside the current one. If there are
        no child domains, removes the ``NESTOUT`` and ``NGRID`` lines from the
        run file. Otherwise, fills in the grid and nest identifiers for each
        child domain, duplicating template lines as needed.
        """
        dict_parent_doms = self.init.dict_ini_data["parent_domains"]
        nested_doms = [child for child,parent in dict_parent_doms.items() if parent==self.domain_number]
        if len(nested_doms) == 0:
            utils.delete_line(f'{self.init.dict_folders["run"]}domain_0{self.domain_number}/run.swn','NESTOUT')
            utils.delete_line(f'{self.init.dict_folders["run"]}domain_0{self.domain_number}/run.swn','NGRID')
        else:
            nested_doms_info = []
            for nest_dom in nested_doms:
                nested_doms_info.append(self.all_domains[nest_dom]["grid"])

            if len(nested_doms)>1:
                if utils.count_NGRID_occurrences(f'{self.init.dict_folders["run"]}domain_0{self.domain_number}/run.swn')==1:
                    NGRID_line_number = utils.look_for_NGRID_linenumber(f'{self.init.dict_folders["run"]}domain_0{self.domain_number}/run.swn')
                    utils.duplicate_lines(f'{self.init.dict_folders["run"]}domain_0{self.domain_number}/run.swn', NGRID_line_number)
            for nested_dom_id,nested_dom_info in zip(nested_doms,nested_doms_info):
                nested_dom_info_=dict()
                for key in nested_dom_info.copy().keys():
                    nested_dom_info_[f'child_{key}']= nested_dom_info[key]
                nested_dom_info_.update(nest_id=f'n0{self.domain_number}_0{nested_dom_id}',nest_grid_file=f'child0{self.domain_number}_0{nested_dom_id}.NEST')
                utils.fill_files_only_once(f'{self.init.dict_folders["run"]}domain_0{self.domain_number}/run.swn',nested_dom_info_)

    def fill_computation_section(self):
        """
        Build and write the computation section of the SWAN run file.

        Cleans up any leftover placeholders, then constructs the ``COMP``
        command string based on whether the run is stationary or non-stationary.
        For non-stationary runs, a single ``COMP NONSTAT`` line is built. For
        stationary runs, one ``COMP STAT`` line is built per computation date,
        optionally separated by ``INIT`` commands. The result is written into
        the domain ``.swn`` file via key substitution.
        """
        #self._delete_placeholder_leftover()

        if self.dict_comp_data['stat_comp'] in (1,"1"): # If the computation is stationary
            if self.init.dict_ini_data["stat_id"] == 1:
                self.string_comp = 'COMP'
            else:
                self.stat_comp_label = 'NONSTAT'
                # TODO: review the logic for multiple COMP lines
                # self.string_comp = ''
                # for idx,date in enumerate(self.dict_comp_data['comp_dates']):
                #     self.date=date.strftime('%Y%m%d.%H%M%S')
                #     if idx == len(self.dict_comp_data['comp_dates']) - 1:
                #         self.string_comp += f'COMP {self.stat_comp_label} {self.date}'
                #     else:
                #         if self.dict_comp_data['init_intermediate']:
                #             self.string_comp += f'COMP {self.stat_comp_label} {self.date}\nINIT\n'
                #         else:
                #             self.string_comp += f'COMP {self.stat_comp_label} {self.date}\n'

        else:
            self.stat_comp_label = 'NONSTAT'
            ini = self.dict_comp_data["ini_comp_date"].strftime("%Y%m%d.%H%M%S")
            end = self.dict_comp_data["end_comp_date"].strftime("%Y%m%d.%H%M%S")
            dt_min = self.dict_comp_data["dt_min"]
            self.string_comp = f'COMP {self.stat_comp_label} {ini} {dt_min} MIN {end}'

        self.dict_comp_data['string_comp'] = self.string_comp
        for param in self.dict_comp_data:
            if isinstance(self.dict_comp_data[param], datetime):
                self.dict_comp_data[param] = self.dict_comp_data[param].strftime('%Y%m%d.%H%M%S')
            else:
                self.dict_comp_data[param] = str(self.dict_comp_data[param])

        print (f'\n \t*** Adding/Editing compilation information for domain {self.domain_number} in configuration file ***\n')
        utils.fill_files(f'{self.init.dict_folders["run"]}domain_0{self.domain_number}/run.swn',self.dict_comp_data)

    # def fill_slurm_file(self):
    #     """
    #     Populate the SLURM launcher script with simulation-specific parameters.

    #     Copies the base SLURM template to the run directory and fills in the
    #     case path, simulation name, number of domains, and parent-domain
    #     mapping as a bash associative array.
    #     """
    #     self.script_dir = Path(__file__).resolve().parent.parent
    #     self.data_dir = self.script_dir.parent.parent.parent / 'data'

    #     shutil.copy(f'{self.data_dir}/model_config_templates/swan/launcher_base_nest_cecc.slurm',
    #                 f'{self.init.dict_folders["run"]}/../launcher_swan.slurm')
        
    #     bash_code = "declare -a bash_dict\n"
    #     for key, value in self.init.dict_ini_data["parent_domains"].items():
    #         bash_value = "" if value is None else value
    #         bash_code += f'bash_dict[{key}]={bash_value}\n'

    #     launch_dict = {
    #         'path_case': self.init.root_path,
    #         'simulation_name': self.init.dict_ini_data["name"].replace(" ", "_"),
    #         'number_domains': self.init.dict_ini_data["number_domains"],
    #         'parent_domains': bash_code
    #     }
    #     utils.fill_files(f'{self.init.dict_folders["run"]}../launcher_swan.slurm', launch_dict, strict=False)

