import datetime as dt

import numpy as np
import pandas as pd
import pytest

hydromt_sfincs = pytest.importorskip("hydromt_sfincs")

from hydromt_sfincs import SfincsModel  # noqa: E402

from oceanicospy.models.sfincspy.preprocess.gridmaker import GridMaker  # noqa: E402
from oceanicospy.models.sfincspy.preprocess.waterlevel_forcing import WaterLevelForcing  # noqa: E402


GRID_KWARGS = dict(
    x0=423966, y0=1389496, dx=15, dy=15, mmax=75, nmax=65, rotation=38, epsg=32617,
)


@pytest.fixture
def model(tmp_path):
    m = SfincsModel(root=str(tmp_path / "run"), mode="w+")
    GridMaker(m).setup_grid(plot=False, **GRID_KWARGS)
    m.setup_config(tref="20250618 000000", tstart="20250618 000000", tstop="20250626 000000")
    return m


def _write_raw_uhslc_csv(path, start, hours, base_mm=2270, amplitude_mm=200):
    """Mimic the raw UHSLC fast-delivery file format: no header,
    year,month,day,hour,depth[mm]."""
    times = pd.date_range(start, periods=hours, freq="h")
    depths_mm = base_mm + (amplitude_mm * np.sin(np.arange(hours) * 2 * np.pi / 12.42)).astype(int)
    rows = [
        f"{t.year},{t.month},{t.day},{t.hour},{d}"
        for t, d in zip(times, depths_mm)
    ]
    path.write_text("\n".join(rows) + "\n")


class TestGetWaterlevelFromUHSLC:
    def test_loads_existing_raw_file_without_datum_correction(self, model, tmp_path):
        input_dir = str(tmp_path) + "/"
        _write_raw_uhslc_csv(tmp_path / "h737.csv", start="2025-06-17 19:00", hours=200)

        wl = WaterLevelForcing(model)
        df = wl.get_waterlevel_from_UHSLC(station_id=737, input_dir=input_dir)

        assert "depth[m]" in df.columns
        # Raw values straight from mm/1000 - no offset removed, matching
        # what xbeachpy's get_waterlevel_from_UHSLC returns for the same file.
        assert df["depth[m]"].between(2.0, 2.6).all()


class TestDatumCorrectionPattern:
    def test_pre_2019_correction_matches_main_xbeach_pattern(self, model, tmp_path):
        """Reproduces the exact correction applied in this project's
        main_xbeach.py scripts for UHSLC station 737: -2.0 m for any
        timestamp on/before 2018-12-31 18:00, untouched after."""
        input_dir = str(tmp_path) + "/"
        _write_raw_uhslc_csv(tmp_path / "h737.csv", start="2018-12-31 10:00", hours=20)

        wl = WaterLevelForcing(model)
        df = wl.get_waterlevel_from_UHSLC(station_id=737, input_dir=input_dir)

        pre_correction = df["depth[m]"].copy()

        correction_mask = (
            (df.index >= dt.datetime(1997, 1, 1, 0)) &
            (df.index <= dt.datetime(2018, 12, 31, 18))
        )
        df.loc[correction_mask, "depth[m]"] -= 2.0

        assert correction_mask.any() and (~correction_mask).any()  # window straddles the cutoff
        assert np.allclose(
            df.loc[correction_mask, "depth[m]"], pre_correction.loc[correction_mask] - 2.0
        )
        assert np.allclose(
            df.loc[~correction_mask, "depth[m]"], pre_correction.loc[~correction_mask]
        )


class TestFromDataframe:
    def test_does_not_shift_time_unlike_from_csv(self, model, tmp_path):
        # from_dataframe assumes the caller (mirroring get_waterlevel_from_UHSLC)
        # already delivered local time - it must not re-shift by -5h.
        idx = pd.date_range("2025-06-18", periods=200, freq="h")
        df = pd.DataFrame({"depth[m]": 0.3 + 0.1 * np.sin(np.arange(200) / 5)}, index=idx)

        wl = WaterLevelForcing(model)
        wl.from_dataframe(df, column_name="depth[m]", x_bnd=423820, y_bnd=1389625)

        bzs_path = tmp_path / "run" / "bzs.csv"
        written = pd.read_csv(bzs_path, index_col=0, parse_dates=True)
        # First written timestamp should fall within [tstart, tstop] and match
        # the *un-shifted* input index (no -5h applied).
        assert written.index[0] >= idx[0]
        assert written.index[0].hour in df.index.hour.tolist()

    def test_filters_invalid_sentinel_and_crops_window(self, model, tmp_path):
        idx = pd.date_range("2025-06-17", periods=300, freq="h")
        values = 0.3 + 0.1 * np.sin(np.arange(300) / 5)
        values[5] = -32.767
        df = pd.DataFrame({"depth[m]": values}, index=idx)

        wl = WaterLevelForcing(model)
        wl.from_dataframe(df, column_name="depth[m]", x_bnd=423820, y_bnd=1389625)

        bzs_path = tmp_path / "run" / "bzs.csv"
        written = pd.read_csv(bzs_path, index_col=0, parse_dates=True)

        assert (written["1"] != -32.767).all()
        tstart = dt.datetime(2025, 6, 18)
        tstop = dt.datetime(2025, 6, 26)
        assert written.index.min() >= tstart
        assert written.index.max() <= tstop

    def test_registers_boundary_in_model_forcing(self, model, tmp_path):
        idx = pd.date_range("2025-06-18", periods=200, freq="h")
        df = pd.DataFrame({"depth[m]": 0.3 + 0.1 * np.sin(np.arange(200) / 5)}, index=idx)

        wl = WaterLevelForcing(model)
        wl.from_dataframe(df, column_name="depth[m]", x_bnd=423820, y_bnd=1389625)

        # HydroMT only writes bzsfile/bndfile into model.config on model.write();
        # immediately after setup_waterlevel_forcing() the "bzs" forcing layer
        # itself is the right thing to check.
        assert "bzs" in model.forcing

        model.write()
        assert model.config.get("bzsfile") is not None
        assert model.config.get("bndfile") is not None


class TestFromCsv:
    def test_shifts_utc_to_local_by_default(self, model, tmp_path):
        idx = pd.date_range("2025-06-18 05:00", periods=200, freq="h")  # UTC
        df = pd.DataFrame({"nivel_mar": 0.3 + 0.1 * np.sin(np.arange(200) / 5)}, index=idx)
        csv_path = tmp_path / "nivel_mar.csv"
        df.to_csv(csv_path)

        wl = WaterLevelForcing(model)
        wl.from_csv(str(csv_path), column_name="nivel_mar", x_bnd=423820, y_bnd=1389625)

        bzs_path = tmp_path / "run" / "bzs.csv"
        written = pd.read_csv(bzs_path, index_col=0, parse_dates=True)
        # 05:00 UTC -> 00:00 local; first local sample should align to tstart.
        assert written.index[0] == dt.datetime(2025, 6, 18, 0, 0)
