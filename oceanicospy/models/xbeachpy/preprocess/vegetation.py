import numpy as np
import glob as glob
from scipy.interpolate import griddata
import os

from .... import utils

class Vegetation():
    """
    Prepare XBeach vegetation input files from species and location dictionaries.

    This class generates the three vegetation-related files expected by XBeach:

    - ``veggielist.txt`` — lists the per-species parameter file names.
    - ``<species>.txt`` — one file per species containing the XBeach vegetation
      parameters (height, density, drag coefficient, etc.).
    - ``veggiemapfile.txt`` — integer map over the x-grid assigning each cell to
      a species index (0 = no vegetation).

    Parameters
    ----------
    dict_species : dict of {str: dict}
        Mapping of species name to a dictionary of XBeach vegetation parameters.
        Each inner dictionary has parameter names as keys and their values as
        values.  List values are written as space-separated sequences.

        Example::

            {
                'mangrove': {'Hveg': 1.0, 'Dveg': 0.05, 'Cd': 1.0, 'N': [10, 10]},
                'seagrass': {'Hveg': 0.3, 'Dveg': 0.01, 'Cd': 0.5, 'N': [50, 50]},
            }

    dict_locations : dict of {str: dict}
        Mapping of patch name to a location descriptor.  Each descriptor must
        contain:

        - ``'loc'`` (*float*) — signed offset from the maximum x coordinate of
          the grid (negative means seaward).
        - ``'length'`` (*float*) — alongshore extent of the patch in metres.
    """

    def __init__(self, dict_species, dict_locations, *args, **kwargs):
        """
        Parameters
        ----------
        dict_species : dict of {str: dict}
            Species parameter dictionaries.  See class docstring for details.
        dict_locations : dict of {str: dict}
            Patch location descriptors.  See class docstring for details.
        """
        self.dict_species = dict_species
        self.dict_locations = dict_locations
        self.dict_veggie = {}

    def definition_species(self):
        """
        Create ``veggielist.txt`` and populate the vegetation metadata dictionary.

        Writes one line per species to ``<run>/veggielist.txt``, where each line
        is the filename ``<species_name>.txt`` that XBeach will read for the
        corresponding species parameters.  Also stores ``number_species`` and
        ``vegetation_file`` in :attr:`dict_veggie` for later use by
        :meth:`fill_vegetation_section`.
        """
        os.system(f'touch {self.dict_folders["run"]}veggielist.txt')

        with open(f'{self.dict_folders["run"]}veggielist.txt', 'w') as f:
            for specie in self.dict_species.keys():
                f.write(f'{specie}.txt\n')

        self.dict_veggie = {
            'number_species': str(len(self.dict_species.keys())),
            'vegetation_file': 'veggielist.txt',
        }

        for key, value in self.dict_veggie.items():
            self.dict_veggie[key] = str(value)

    def params_per_specie(self):
        """
        Write one parameter file per species into the run folder.

        For each species in :attr:`dict_species`, creates
        ``<run>/<species_name>.txt`` and writes every parameter as
        ``key = value``.  List values are serialised as space-separated strings
        so that XBeach can parse multi-layer inputs (e.g. stem density per layer).
        """
        for specie in self.dict_species.keys():
            with open(f'{self.dict_folders["run"]}{specie}.txt', 'w') as f:
                for key, value in self.dict_species[specie].items():
                    if type(value) == list:
                        value_to_write = ' '.join([str(i) for i in value])
                    else:
                        value_to_write = value
                    f.write(f'{key} = {value_to_write}\n')

    def create_veggie_map(self):
        """
        Build the integer vegetation map and write it to ``veggiemapfile.txt``.

        Reads the x-grid from ``<run>/x_profile.grd``, then for each patch in
        :attr:`dict_locations` assigns the patch species index (1-based) to every
        grid cell that falls within the patch bounds.  Cells outside all patches
        receive index ``0`` (no vegetation).  The result is saved as
        ``<run>/veggiemapfile.txt`` and the key ``vegetation_map_file`` is added
        to :attr:`dict_veggie`.

        Notes
        -----
        Patch boundaries are computed relative to the maximum x coordinate of
        the grid: ``abscissa_start = max(x) + loc``.  Negative ``loc`` values
        therefore place the patch seaward of the shoreline.
        """
        x = np.genfromtxt(f'{self.dict_folders["run"]}x_profile.grd')
        max_x = np.nanmax(x)
        veggie_locs = np.zeros(x.shape)
        for idx, dic_space in enumerate(self.dict_locations.values()):
            abscisa_start = max_x + dic_space['loc']
            abscisa_end = abscisa_start + dic_space['length']
            index_end = np.argmin(np.abs(x - abscisa_end))
            index_start = np.argmin(np.abs(x - abscisa_start))
            veggie_locs[index_start:index_end] = int(idx + 1)

        np.savetxt(f'{self.dict_folders["run"]}veggiemapfile.txt', veggie_locs, fmt='%d')
        self.dict_veggie.update({'vegetation_map_file': 'veggiemapfile.txt'})

    def fill_vegetation_section(self):
        """
        Write vegetation configuration into the XBeach ``params.txt`` file.

        Converts every value in :attr:`dict_veggie` to ``str`` (required by the
        placeholder-substitution engine) and calls :func:`utils.fill_files` to
        replace the corresponding ``$placeholder`` tokens in
        ``<run>/params.txt``.

        Must be called after :meth:`definition_species` (and optionally
        :meth:`create_veggie_map`) have populated :attr:`dict_veggie`.
        """
        for param in self.dict_veggie:
            self.dict_veggie[param] = str(self.dict_veggie[param])
        utils.fill_files(f'{self.dict_folders["run"]}params.txt', self.dict_veggie)