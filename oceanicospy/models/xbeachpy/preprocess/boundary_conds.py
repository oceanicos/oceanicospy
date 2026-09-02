from wavespectra import read_swan
import numpy as np
import pandas as pd
import re
import os

from .... import utils

class BoundaryConditions:
    """
    Generate and register XBeach spectral boundary conditions from SWAN output.

    Starting from a SWAN spectral output file (``SpecSWAN.out``), this class
    converts the spectra to XBeach-compatible ``.sp2`` files, writes the
    corresponding ``filelist_<n>.txt`` and ``loclist.txt`` files under
    ``<run>/bounds_conds/``, and populates the boundary block of
    ``params.txt`` via :meth:`fill_boundaries_section`.

    Parameters
    ----------
    init : object
        Case initialization object.  Must expose ``ini_date``, ``end_date``,
        and ``dict_folders`` (a mapping with at least ``"input"`` and ``"run"``
        keys pointing to the respective directories).

    """
    def __init__ (self,init):
        self.init = init
        self._purger()
        print('*** Initializing Boundary Conditions ***')

    def _purger(self):
        """
        Remove any previously generated boundary-condition files.

        Deletes the ``<run>/bounds_conds/`` directory and its entire contents
        so that each run starts from a clean slate.  Called automatically by
        :meth:`__init__`.
        """
        print('*** Cleaning Boundary Conditions ***')
        os.system(f'rm -rf {self.init.dict_folders["run"]}bounds_conds')

    # -------------------------------------------------------------------------
    # spectra_from_swan helpers
    # -------------------------------------------------------------------------

    def _load_dataset(self, input_filename,point_indexes=None):
        """
        Load a SWAN spectra output file and restrict it to the simulation time window.

        Parameters
        ----------
        input_filename : str
            Name of the SWAN output file (without extension) located in
            ``self.init.dict_folders["input"]``.
        point_indexes : array-like of int, optional
            Indices of the sites within the SWAN output to use as boundary
            locations.  When ``None``, all sites present in the file are used.

        Returns
        -------
        xarray.Dataset
            Dataset containing the spectral energy ``efth`` and coordinate
            variables ``time``, ``site``, ``lon``, and ``lat``, sliced to
            ``[self.init.ini_date, self.init.end_date]``.

        Raises
        ------
        RuntimeError
            If the loaded dataset does not contain a ``time`` coordinate.
        """
        self.dataset = read_swan(f'{self.init.dict_folders["input"]}{input_filename}')
        if "time" not in self.dataset.coords:
            raise RuntimeError("Loaded dataset has no 'time' coordinate to filter on.")
        start = np.datetime64(self.init.ini_date)
        end = np.datetime64(self.init.end_date)

        if point_indexes is not None:
            self.dataset = self.dataset.isel(site=point_indexes)
        return self.dataset.sel(time=slice(start, end))

    def _write_sp2_header(self, forigin, fdest, lon, lat):
        """
        Copy and transform the header block from a SWAN spectra file into an sp2 file.

        Reads ``forigin`` line-by-line until the ``'date and time'`` sentinel is
        encountered, writing each line to ``fdest`` after applying two in-place
        transformations:

        - **number of locations**: rewrites the location count to ``1`` and retains
          only the coordinate line that matches ``(lon, lat)``.
        - **number of directions**: reads the following 36 direction values and
          offsets each by 270 degrees to convert from SWAN's convention to XBeach's.

        Parameters
        ----------
        forigin : file-like
            Opened ``SpecSWAN.out`` file positioned at the beginning of a
            header block.
        fdest : file-like
            Opened destination ``.sp2`` file to which the transformed header
            is written.
        lon : float
            Longitude of the target site, used to identify its coordinate line
            in the ``'number of locations'`` block.
        lat : float
            Latitude of the target site (formatted to 5 decimal places for
            string matching).

        Notes
        -----
        The function stops reading ``forigin`` as soon as the ``'date and time'``
        line is found, leaving the file cursor positioned immediately after that
        line so that the caller can continue reading the spectral data.
        """
        while True:
            line = forigin.readline()
            if 'date and time' in line:
                break
            # If the line contains the number of locations, we need to rewrite it to 1 
            # and find the matching location line
            elif 'number of locations' in line: 
                line = re.sub(r'\d+', "1", line)
                next_line = forigin.readline()
                while next_line.strip().isalpha() == False:
                    next_line_list = [float(x) for x in next_line.split('   ') if x]
                    if (lon in next_line_list) and (lat in next_line_list):
                        line += next_line
                        break
                    next_line = forigin.readline()
                fdest.write(line)

            # if the line contains the number of directions, we need to read the following 36 lines 
            # and add 270 to each direction value
            elif 'number of directions' in line:
                for _ in range(36):
                    next_line = forigin.readline()
                    line = line + str(float(next_line) + 270) + '\n'
                fdest.write(line)

            # if the line has two numeric values separated by three spaces, we assume it's a coordinate line 
            # and skip it (except for the matching location line handled above)
            elif all(x.strip()!= '' and x.isdigit() for x in line.split('   ')):
                continue
            else:
                fdest.write(line)

    def _write_sp2_spectrum(self, fdest, spec_matrix, time_str):
        """
        Write the spectral data block to an sp2 file.

        Appends the timestamp, the ``FACTOR`` header with its scaling value,
        and the normalised spectral energy matrix to ``fdest``.

        Parameters
        ----------
        fdest : file-like
            Opened destination ``.sp2`` file, already containing the header
            block written by :meth:`_write_sp2_header`.
        spec_matrix : numpy.ndarray, shape (n_freq, n_dir)
            Spectral energy array normalised by the FACTOR value (``0.1E-05``),
            so that each element is written as a rounded integer.
        time_str : str
            Timestamp string in SWAN format ``'YYYYMMDD.HHMMSS'``.
        """
        fdest.write(time_str + '\n')
        fdest.write('FACTOR\n')
        fdest.write('0.1E-05\n')
        for row in spec_matrix:
            np.savetxt(fdest, row, fmt='%7.0f')

    def _write_sp2_file(self, site_idx, time_idx, lon, lat):
        """
        Write a single ``.sp2`` file for one site/time-step combination.

        Opens the source ``SpecSWAN.out`` and the destination ``.sp2`` file,
        then delegates to :meth:`_write_sp2_header` and
        :meth:`_write_sp2_spectrum` to produce a self-contained SWAN spectra
        file for a single location and timestamp.

        Parameters
        ----------
        site_idx : int
            Index of the boundary site within ``self.dataset.site``.
        time_idx : int
            Index of the time step within ``self.dataset.time``.
        lon : float
            Longitude of the site, forwarded to :meth:`_write_sp2_header`
            for location-line matching.
        lat : float
            Latitude of the site, forwarded to :meth:`_write_sp2_header`
            for location-line matching.

        Notes
        -----
        The output file is written to
        ``<run>/bounds_conds/point_<site_idx>/spec_time<time_idx>_point<site_idx>.sp2``.
        The spectral energy is normalised by ``0.1E-05`` before being passed
        to :meth:`_write_sp2_spectrum`.
        """
        spec_matrix = np.matrix(self.data_spectra[time_idx, site_idx, :, :]) / 0.1e-5

        time_str = pd.to_datetime(self.dataset.time.values[time_idx]).strftime('%Y%m%d.%H%M%S')
        sp2_path = (
            f"{self.init.dict_folders['run']}"
            f"bounds_conds/point_{site_idx}/spec_time{time_idx}_point{site_idx}.sp2"
        )
        with open(f"{self.init.dict_folders['input']}SpecSWAN.out") as forigin:
            with open(sp2_path, "w") as fdest:
                self._write_sp2_header(forigin, fdest, lon, lat)
                self._write_sp2_spectrum(fdest, spec_matrix, time_str)

    def _write_site_filelist(self, site_idx):
        """
        Create the output directory and ``filelist_<site_idx>.txt`` for one boundary site.

        For each time step in ``self.dataset.time``, calls :meth:`_write_sp2_file`
        to produce the individual ``.sp2`` spectra files and then records each
        entry (duration, directional spreading, relative path) in the filelist.

        Parameters
        ----------
        site_idx : int
            Index of the boundary site within ``self.dataset.site``.  Must be
            greater than or equal to 3 (sites 0–2 are reserved and skipped by
            :meth:`spectra_from_swan`).

        Notes
        -----
        The output directory ``<run>/bounds_conds/point_<site_idx>/`` is created
        with ``exist_ok=True``, so re-running is safe.  The filelist is written to
        ``<run>/bounds_conds/filelist_<site_idx>.txt`` with one entry per time step
        in the format expected by XBeach (``3600 0.2 '<relative_sp2_path>'``)
        
        Notice that the time step per site is defined as 0.2.
        """
        bounds_conds_path = os.path.join(
            self.init.dict_folders["run"], "bounds_conds", f"point_{site_idx}"
        )
        os.makedirs(bounds_conds_path, exist_ok=True)

        self.lon = self.dataset.lon[site_idx]
        self.lat = self.dataset.lat[site_idx]
        filelist_path = f"{self.init.dict_folders['run']}bounds_conds/filelist_{site_idx}.txt"

        with open(filelist_path, "w") as filelist:
            filelist.write('FILELIST\n')
            for idx_time in range(len(self.dataset.time)):
                self._write_sp2_file(site_idx, idx_time, self.lon.values, self.lat.values)
                filelist.write(
                    f"3600 0.2 'bounds_conds/point_{site_idx}/"
                    f"spec_time{idx_time}_point{site_idx}.sp2'\n"
                )

    def _write_loclist(self, location_points):
        """
        Write ``loclist.txt`` mapping each boundary site to its filelist.

        Iterates over all ``self.number_spectrum_locs`` sites and writes one
        entry per site to ``<run>/bounds_conds/loclist.txt``.  Each entry
        places the site at ``x = 0`` and ``y = -(site_idx) * 100`` and points
        to the corresponding ``filelist_<site_idx>.txt``.
        """
        loclist_path = f"{self.init.dict_folders['run']}bounds_conds/loclist.txt"
        with open(loclist_path, "w") as floc:
            floc.write('LOCLIST\n')
            for idx_site in range(self.number_spectrum_locs):
                x, y = location_points[idx_site]
                floc.write(f"{x} {y} 'bounds_conds/filelist_{idx_site}.txt'\n")


    def spectra_from_swan(self, input_filename,location_points, point_indexes=None):
        """
        Convert a SWAN spectral output file into XBeach boundary condition files.

        Orchestrates the full boundary-condition workflow:

        1. Loads the SWAN spectral dataset via :meth:`_load_dataset` and slices
           it to the simulation time window.
        2. For each boundary site, creates a subdirectory under
           ``<run>/bounds_conds/``, writes one ``.sp2`` file per time step, and
           records all entries in ``filelist_<site_idx>.txt``.
        3. Writes ``loclist.txt`` that maps every site to its filelist.
        4. Populates :attr:`dict_boundaries` with the ``params.txt`` keys
           required to activate spectral wave boundaries in XBeach
           (``w_bc_version``, ``n_spectrum_loc``, ``bcfilepath``).

        Parameters
        ----------
        input_filename : str
            Name of the SWAN output file (without directory) located in
            ``init.dict_folders["input"]``.  Must be readable by
            :func:`wavespectra.read_swan`.
        point_indexes : array-like of int, optional
            Indices of the sites within the SWAN output to use as boundary
            locations.  When ``None``, all sites present in the file are used.

        Notes
        -----
        When only one boundary site is found, a warning is printed reminding
        the user to remove the ``loclist`` section and ``nspectrumloc`` command
        from ``params.txt``; :attr:`dict_boundaries` is not populated in that
        case.
        """
        self.dataset = self._load_dataset(input_filename, point_indexes)
        self.data_spectra = self.dataset.efth
        self.number_spectrum_locs = len(self.dataset.site)

        self.dict_boundaries = {
            'w_bc_version': 3,
            'n_spectrum_loc': self.number_spectrum_locs,
            'bcfilepath': 'bounds_conds/loclist.txt',
        }

        for idx_site in range(self.number_spectrum_locs):
            self._write_site_filelist(idx_site)

        if self.number_spectrum_locs != len(location_points):
            raise ValueError(f"Number of location points ({len(location_points)}) does not match number of spectrum locations ({self.number_spectrum_locs}).")

        self._write_loclist(location_points)

    def fill_boundaries_section(self):
        """
        Write the boundary-condition parameters to ``params.txt``.

        Converts every value in :attr:`dict_boundaries` to ``str`` (required
        by the placeholder-substitution engine) and calls
        :func:`utils.fill_files` to replace the corresponding ``$placeholder``
        tokens in ``<run>/params.txt``.

        Must be called after :meth:`spectra_from_swan` has populated
        :attr:`dict_boundaries`.
        """
        for param in self.dict_boundaries:
            self.dict_boundaries[param]=str(self.dict_boundaries[param])

        print (f'\n*** Adding/Editing boundary information for domain in configuration file ***\n')
        utils.fill_files(f'{self.init.dict_folders["run"]}params.txt',self.dict_boundaries)
