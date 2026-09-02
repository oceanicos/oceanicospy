    # def fill_slurm_file(self, case_name):
    #     """
    #     Fills the SLURM script with the necessary parameters for running the XBeach model.
    #     This includes paths, simulation name, number of domains, and parent domains.
    #     """
    #     self.script_dir = Path(__file__).resolve().parent.parent
    #     self.data_dir = self.script_dir.parent.parent.parent / 'data'
    #
    #     shutil.copy(f'{self.data_dir}/model_config_templates/xbeach/launcher_xbeach_base.slurm',
    #                 f'{self.init.dict_folders["run"]}launcher_xbeach.slurm')
    #
    #     launch_dict = dict(output_path_case=f'{self.init.dict_folders["output"]}', case_name=case_name)
    #     utils.fill_files(f'{self.init.dict_folders["run"]}launcher_xbeach.slurm', launch_dict, strict=False)
