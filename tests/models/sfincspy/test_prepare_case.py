import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

hydromt_sfincs = pytest.importorskip("hydromt_sfincs")
rasterio = pytest.importorskip("rasterio")

from oceanicospy.models.sfincspy.initializer import Initializer  # noqa: E402
from oceanicospy.models.sfincspy.preprocess.prepare_case import SfincsCaseBuilder  # noqa: E402


# Mirrors the real E1 (Nov 2008) calibration domain/dates, confirmed against
# DANIEL_AGUIRRE/runs/E1/Nov2008_E1_C/sfincs.inp during the SFINCS audit.
E1_GRID_KWARGS = dict(
    x0=423673.095, y0=1390114.015, dx=1, dy=1, mmax=360, nmax=362, rotation=38, epsg=32617,
)
E1_TIME_KWARGS = dict(tref="20081120 110000", tstart="20081120 110000", tstop="20081124 210000")

REAL_TOPOBATHY = Path(
    "D:/04_BACKUP_SERVER/daguirreos_sfincs/sfincs/SFINCSPY/Casos/Caso02/input/new_topobathy_SAI_1m_n.tif"
)
REAL_E1_XBEACH_NC = Path(
    "D:/04_BACKUP_SERVER/daguirreos_sfincs/sfincs/DANIEL_AGUIRRE/data/E1/E1_profile1D_Nov2008.nc"
)


def _write_synthetic_precip_excel(path, start, hours):
    idx = pd.date_range(start, periods=hours, freq="h")
    pd.DataFrame(
        {
            "Date/Time": idx,
            "Precipitacion (mm)": np.abs(np.random.default_rng(0).normal(0.5, 1.0, hours)),
            "Presion Barometrica (hPa)": np.random.default_rng(1).normal(1010, 2, hours),
        }
    ).to_excel(path, index=False)


@pytest.mark.skipif(
    not (REAL_TOPOBATHY.exists() and REAL_E1_XBEACH_NC.exists()),
    reason="Real Los Almendros backup data not available on this machine",
)
class TestSfincsCaseBuilderEndToEnd:
    def test_builds_a_complete_runnable_e1_style_case(self, tmp_path):
        """Rebuilds a case matching the real, already-calibrated E1 (Nov
        2008) event as closely as this integration test reasonably can:
        real topobathy, real domain/dates, real XBeach wave forcing (with
        the bzi fix), synthetic tide and precip (network/file access for
        those two is exercised separately in their own test modules).
        Confirms the full orchestrator wiring - Initializer -> GridMaker ->
        BathyMaker -> BottomFriction -> forcings -> write() - produces a
        complete, internally consistent sfincs.inp with no missing pieces.
        """
        case_root = tmp_path / "CasoE1"
        init = Initializer(str(case_root))
        init.create_folders()

        shutil.copy(REAL_TOPOBATHY, init.dict_folders["input"] + "topobathy.tif")
        catalog_path = init.write_data_catalog(
            {
                "topobathy": {
                    "data_type": "RasterDataset", "driver": "raster",
                    "path": "topobathy.tif", "crs": 32617, "nodata": -3.4028235e38,
                }
            }
        )

        builder = SfincsCaseBuilder(root=init.dict_folders["run"], data_catalog=catalog_path)
        builder.setup_time(**E1_TIME_KWARGS)
        builder.grid.setup_grid(plot=False, **E1_GRID_KWARGS)
        builder.bathy.setup_bathy(zmin_active=-4, zmax_bounds=-2, plot=False)
        builder.friction.setup_uniform(manning_land=0.04, manning_sea=0.02)

        tide_idx = pd.date_range("2008-11-20", periods=200, freq="h")
        tide_df = pd.DataFrame(
            {"depth[m]": 0.5 + 0.15 * np.sin(np.arange(200) / 6)}, index=tide_idx
        )
        builder.waterlevel.from_dataframe(
            tide_df, column_name="depth[m]", x_bnd=423820, y_bnd=1389625,
        )

        builder.waves.from_xbeach(str(REAL_E1_XBEACH_NC), point_index=0)

        precip_xlsx = init.dict_folders["input"] + "precip.xlsx"
        _write_synthetic_precip_excel(precip_xlsx, "2008-11-20", hours=120)
        builder.meteo.from_excel(precip_xlsx)

        builder.write()

        run_dir = Path(init.dict_folders["run"])
        inp_path = run_dir / "sfincs.inp"
        assert inp_path.is_file()

        inp_text = inp_path.read_text()
        # Every forcing called above must show up as a registered file -
        # if the orchestrator dropped a step, the corresponding keyword
        # would simply be missing from sfincs.inp. Note: setup_uniform()
        # writes a spatial manningfile (and removes the scalar
        # manning_land/manning_sea keywords from config) - see
        # BottomFriction.setup_uniform and its own tests for that behaviour.
        for keyword in ["depfile", "mskfile", "manningfile", "bzsfile", "bndfile", "bzifile", "precipfile"]:
            assert keyword in inp_text, f"missing '{keyword}' in sfincs.inp - a preprocessing step didn't wire in"

        # And the raster/mask/forcing files it references must actually exist.
        assert (run_dir / "sfincs.dep").is_file()
        assert (run_dir / "sfincs.msk").is_file()
        assert (run_dir / "sfincs.bzs").is_file()
        assert (run_dir / "sfincs.bzi").is_file()
        assert (run_dir / "sfincs.prcp").is_file()

    def test_manning_and_grid_choices_are_traceable_in_the_written_inp(self, tmp_path):
        """The whole point of not having a monolithic build-everything method
        is that every decision (here: Manning values) is visible in the
        script and ends up verifiably in sfincs.inp - not hidden inside a
        helper that silently picks defaults."""
        case_root = tmp_path / "CasoManningCheck"
        init = Initializer(str(case_root))
        init.create_folders()

        shutil.copy(REAL_TOPOBATHY, init.dict_folders["input"] + "topobathy.tif")
        catalog_path = init.write_data_catalog(
            {
                "topobathy": {
                    "data_type": "RasterDataset", "driver": "raster",
                    "path": "topobathy.tif", "crs": 32617, "nodata": -3.4028235e38,
                }
            }
        )

        builder = SfincsCaseBuilder(root=init.dict_folders["run"], data_catalog=catalog_path)
        builder.setup_time(**E1_TIME_KWARGS)
        builder.grid.setup_grid(plot=False, **E1_GRID_KWARGS)
        builder.bathy.setup_bathy(zmin_active=-4, zmax_bounds=-2, plot=False)
        builder.friction.setup_uniform(manning_land=0.0777, manning_sea=0.0222)

        # setup_uniform() folds manning_land/manning_sea into a spatial
        # manningfile (and removes the scalar keywords from config) - so the
        # traceable check is on model.grid["manning"], not sfincs.inp text.
        dep = builder.model.grid["dep"].values
        man = builder.model.grid["manning"].values
        assert np.allclose(man[dep >= 0], 0.0777)
        assert np.allclose(man[dep < 0], 0.0222)
        assert builder.model.config.get("manningfile") is not None

        builder.write()
        assert (Path(init.dict_folders["run"]) / "sfincs.man").is_file()
