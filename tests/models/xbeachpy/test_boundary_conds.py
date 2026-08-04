import datetime as dt
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from oceanicospy.models.xbeachpy.preprocess.boundary_conds import BoundaryConditions

# ---------------------------------------------------------------------------
# Synthetic SpecSWAN.out fixture
# ---------------------------------------------------------------------------
#
# Mirrors the real SWAN output format (SET NAUT -> "spectral nautical
# directions"), but with a single site/timestep and the direction bins
# deliberately written in SWAN's raw, non-ascending order -- exactly like
# the real file, whose NDIR block runs 265, 255, ..., 5, 355, ..., 275
# instead of ascending. All the energy is planted at a single, known
# direction (85 deg) so a header/data mismatch is unambiguous to detect.

LON, LAT = 100.000000, 10.000000
# A second, dry (NODATA) point with a different lon AND lat. wavespectra
# treats a SWAN file as a rectangular "grid" (dims lat/lon) instead of a
# flat "site" list whenever len(unique(lons))*len(unique(lats)) == n_points
# -- true for a single point (1*1==1), which breaks the site-based code
# path BoundaryConditions relies on. A second, non-collinear point avoids
# that and forces the site-list path, matching the real multi-point files.
LON2, LAT2 = 100.010000, 10.030000
FREQS = [0.0400, 0.1000, 0.2000]
N_DIRS = 36
RAW_DIRS = [(265 - 10 * k) % 360 for k in range(N_DIRS)]  # same pattern as real SpecSWAN.out
ENERGY_DIR = 85.0
ENERGY_RAW_COL = RAW_DIRS.index(ENERGY_DIR)
ENERGY_VALUE = 1000


def _spec_row(energy_col, value):
    row = [0] * N_DIRS
    row[energy_col] = value
    return " ".join(f"{v:.0f}" for v in row) + "\n"


def _build_specswan_text() -> str:
    lines = [
        "SWAN   1                                Swan standard spectral file, version\n",
        "$   synthetic fixture for boundary_conds tests\n",
        "TIME                                    time-dependent data\n",
        "     1                                  time coding option\n",
        "LONLAT                                  locations in spherical coordinates\n",
        "     2                                  number of locations\n",
        f"  {LON:.6f}   {LAT:.6f}\n",
        f"  {LON2:.6f}   {LAT2:.6f}\n",
        "AFREQ                                   absolute frequencies in Hz\n",
        f"    {len(FREQS)}                                  number of frequencies\n",
    ]
    lines += [f"    {f:.4f}\n" for f in FREQS]
    lines += [
        "NDIR                                    spectral nautical directions in degr\n",
        f"    {N_DIRS}                                  number of directions\n",
    ]
    lines += [f"  {d:.4f}\n" for d in RAW_DIRS]
    lines += [
        "QUANT\n",
        "     1                                  number of quantities in table\n",
        "VaDens                                  variance densities in m2/Hz/degr\n",
        "m2/Hz/degr                              unit\n",
        "   -0.9900E+02                          exception value\n",
        "20250101.000000                         date and time\n",
        "FACTOR\n",
        "    1.00000000E+00\n",
    ]
    lines += [_spec_row(ENERGY_RAW_COL, ENERGY_VALUE) for _ in FREQS]
    lines.append("NODATA\n")  # second (dry) point, no spectrum
    return "".join(lines)


@pytest.fixture
def fake_init(tmp_path):
    input_dir = tmp_path / "input"
    run_dir = tmp_path / "run"
    input_dir.mkdir()
    run_dir.mkdir()
    (input_dir / "SpecSWAN.out").write_text(_build_specswan_text())
    return SimpleNamespace(
        dict_folders={"input": str(input_dir) + "/", "run": str(run_dir) + "/"},
        ini_date=dt.datetime(2025, 1, 1, 0),
        end_date=dt.datetime(2025, 1, 1, 0),
    )


def _read_generated_sp2(sp2_path: Path):
    """Parse a generated .sp2 back into (declared directions, direction of peak energy)."""
    lines = sp2_path.read_text().splitlines()

    ndir_idx = next(i for i, l in enumerate(lines) if "number of directions" in l)
    n_dirs = int(lines[ndir_idx].split()[0])
    dirs = [float(lines[ndir_idx + 1 + k]) for k in range(n_dirs)]

    factor_idx = next(i for i, l in enumerate(lines) if l.strip() == "FACTOR")
    factor = float(lines[factor_idx + 1])
    matrix = np.array(
        [[float(v) for v in lines[factor_idx + 2 + i].split()] for i in range(len(FREQS))]
    ) * factor

    dir_energy = matrix.sum(axis=0)
    peak_dir = dirs[int(np.argmax(dir_energy))]
    return dirs, peak_dir


class TestSpectraFromSwanDirectionAlignment:
    def test_header_direction_matches_energy_column(self, fake_init):
        bc = BoundaryConditions(init=fake_init)
        bc.spectra_from_swan(
            input_filename="SpecSWAN.out",
            location_points=[(0, 0), (0, -100)],
        )

        sp2_path = (
            Path(fake_init.dict_folders["run"])
            / "bounds_conds" / "point_0" / "spec_time0_point0.sp2"
        )
        assert sp2_path.exists()

        dirs, peak_dir = _read_generated_sp2(sp2_path)

        # All the energy was planted at 85 degN. If the header direction for
        # that column doesn't say 85, XBeach would read the energy under the
        # wrong direction label -- which is exactly what the +270/raw-order
        # bug did (it reported ~101 degN for real data known to be ~78 degN).
        assert peak_dir == pytest.approx(ENERGY_DIR)

    def test_header_has_no_unexplained_offset(self, fake_init):
        bc = BoundaryConditions(init=fake_init)
        bc.spectra_from_swan(
            input_filename="SpecSWAN.out",
            location_points=[(0, 0), (0, -100)],
        )

        sp2_path = (
            Path(fake_init.dict_folders["run"])
            / "bounds_conds" / "point_0" / "spec_time0_point0.sp2"
        )
        dirs, _ = _read_generated_sp2(sp2_path)

        # The header should be the same 36 values as the source file, just
        # re-ordered to ascending (matching the energy matrix) -- not offset
        # by +270 or any other transform. The source declares nautical
        # convention already (SET NAUT), which is what XBeach assumes by
        # default for wbctype=swan.
        assert dirs == pytest.approx(sorted(RAW_DIRS))


class TestSpectraFromSwanLocationsBlock:
    def test_no_leftover_coordinates_leak_into_header(self, fake_init):
        # The fixture's site 0 (LON/LAT) is the *first* of two coordinate
        # lines in the raw file's LONLAT block, so the second site's (LON2/
        # LAT2) coordinate line is the leftover that used to leak through:
        # the old skip-logic used `.isdigit()` to detect and discard
        # unmatched coordinate lines, which never matches a value with a
        # decimal point, so the leftover fell through to the generic
        # "else: write the line" branch instead of being discarded.
        bc = BoundaryConditions(init=fake_init)
        bc.spectra_from_swan(
            input_filename="SpecSWAN.out",
            location_points=[(0, 0), (0, -100)],
        )

        sp2_path = (
            Path(fake_init.dict_folders["run"])
            / "bounds_conds" / "point_0" / "spec_time0_point0.sp2"
        )
        lines = sp2_path.read_text().splitlines()

        loc_idx = next(i for i, l in enumerate(lines) if "number of locations" in l)
        assert lines[loc_idx].split()[0] == "1"
        # Exactly one coordinate line should follow the count, then AFREQ --
        # not a second, unmatched site's coordinate line.
        assert "AFREQ" in lines[loc_idx + 2]
