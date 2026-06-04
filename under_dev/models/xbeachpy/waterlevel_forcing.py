# def txt_from_user(self):
#     """
#     Link an existing ``.wl`` file from the input folder into the run folder.

#     Returns
#     -------
#     dict
#         Dictionary with key ``'sealevelfilepath'`` set to the
#         water-level filename.
#     """
#     sealevel_file_path = glob.glob(f'{self.init.dict_folders["input"]}*.wl')[0]
#     sealevel_filename = Path(sealevel_file_path).name

#     if not utils.verify_link(sealevel_filename, self.init.dict_folders["run"]):
#         utils.create_link(
#             sealevel_filename,
#             self.init.dict_folders["input"],
#             self.init.dict_folders["run"],
#         )

#     self.wl_info = {"sealevelfilepath": sealevel_filename}
