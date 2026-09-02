import glob as glob
from pathlib import Path

from .... import utils

class BottomFrictionProcessor:
    """
    BottomFrictionProcessor is a utility class for generating and managing the bottom friction information for SWAN.

    Parameters
    ----------
    init : object
        An initialization object containing configuration data and folder paths.
    domain_number : int
        Identifier for the domain being processed.
    bottom_fric_info : dict or None, optional
        Dictionary containing bottom friction information. If None, friction must be provided via `filename`.
    use_link: bool, optional
        If True, creates symbolic links for the friction file instead of copying it. Defaults to True.
    """

    def __init__(self,init,domain_number,bottom_fric_info=None,use_link=None):
        self.init = init
        self.domain_number = domain_number
        self.bottom_fric_info = bottom_fric_info
        self.use_link = use_link
        print(f'\n*** Initializing BottomFrictionProcessor for domain {self.domain_number} ***\n')

    def use_ascii_file_from_user(self):
        """
        Handles the selection and linking or copying of a friction file for the current domain.
        This method searches for a `.fric` bottom friction file in the input directory for the specified domain.

        Returns
        -------
        dict or None
            The updated ``bottom_fric_info`` dictionary if it was provided at
            initialisation, otherwise ``None``.

        Raises
        ------
        FileNotFoundError
            If no ``.fric`` file is found in the expected input directory.
        """
    
        friction_filepaths = glob.glob(f'{self.init.dict_folders["input"]}domain_0{self.domain_number}/*.fric')
        if not friction_filepaths:
            raise FileNotFoundError(f'Friction file not found in {self.init.dict_folders["input"]}domain_0{self.domain_number}/ or file extension is not .fric')
        friction_filepath = Path(friction_filepaths[0])
        friction_filename = friction_filepath.name

        run_domain_dir = f'{self.init.dict_folders["run"]}domain_0{self.domain_number}/'

        utils.deploy_input_file(friction_filename, f'{self.init.dict_folders["input"]}domain_0{self.domain_number}/', run_domain_dir, self.use_link)

        if self.bottom_fric_info != None:
            self.bottom_fric_info.update({"friction_file":f"../../input/domain_0{self.domain_number}/{friction_filename}"})
            return self.bottom_fric_info

    def fill_friction_section(self):
        """
        Replaces and updates the .swn file with the bottom friction configuration for a specific domain.

        Raises
        ------
        ValueError
            If no friction information was provided at initialization.
        """

        if self.bottom_fric_info == None:
            raise ValueError(f'Friction information is not provided for domain {self.domain_number}.')

        print (f'\n \t*** Adding/Editing friction information for domain {self.domain_number} in configuration file ***\n')
        utils.fill_files(f'{self.init.dict_folders["run"]}domain_0{self.domain_number}/run.swn',self.bottom_fric_info)
