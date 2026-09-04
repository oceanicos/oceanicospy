from pathlib import Path

import numpy as np
import pandas as pd
import pytest

hydromt_sfincs = pytest.importorskip("hydromt_sfincs")

from hydromt_sfincs import SfincsModel  # noqa: E402

from oceanicospy.models.sfincspy.preprocess.gridmaker import GridMaker  # noqa: E402
from oceanicospy.models.sfincspy.preprocess.meteo_forcing import UniformMetForcingClassic  # noqa: E402


GRID_KWARGS = dict(
    x0=423966, y0=1389496, dx=15, dy=15, mmax=75, nmax=65, rotation=38, epsg=32617,
)

REAL_XLSX_PATH = Path("c:/Users/Daniela/00_OCEANICOS/000_tesis/tesis/data/raw/AG5-0359.xlsx")


@pytest.fixture
def model(tmp_path):
    m = SfincsModel(root=str(tmp_path / "run"), mode="w+")
    GridMaker(m).setup_grid(plot=False, **GRID_KWARGS)
    m.setup_config(tref="20250618 000000", tstart="20250618 060000", tstop="20250618 180000")
    return m


def _write_ag5_style_excel(path, hours=48, start="2025-06-18 00:00", with_pressure=True):
    idx = pd.date_range(start, periods=hours, freq="h")
    data = {
        "Date/Time": idx,
        "Precipitacion (mm)": np.abs(np.random.default_rng(0).normal(0.5, 1.0, hours)),
        "Velocidad Viento (m/s)": np.random.default_rng(1).normal(6, 2, hours),
        "Direccion Viento (°)": np.random.default_rng(2).uniform(0, 360, hours),
    }
    if with_pressure:
        data["Presion Barometrica (hPa)"] = np.random.default_rng(3).normal(1010, 2, hours)
    pd.DataFrame(data).to_excel(path, index=False)


class TestFromExcel:
    def test_writes_precip_by_default(self, model, tmp_path):
        xlsx = tmp_path / "station.xlsx"
        _write_ag5_style_excel(xlsx)

        met = UniformMetForcingClassic(model)
        met.from_excel(str(xlsx))

        assert model.config.get("precipfile") == "sfincs.prcp"
        prcp_path = Path(model.root) / "sfincs.prcp"
        assert prcp_path.is_file()
        written = pd.read_csv(prcp_path, sep=" ", header=None, names=["time", "prcp"])
        assert len(written) == 13  # tstart=06:00 to tstop=18:00 hourly

    def test_does_not_write_wind_by_default(self, model, tmp_path):
        xlsx = tmp_path / "station.xlsx"
        _write_ag5_style_excel(xlsx)

        met = UniformMetForcingClassic(model)
        met.from_excel(str(xlsx))

        assert model.config.get("wndfile") is None
        assert not (Path(model.root) / "sfincs.wnd").exists()

    def test_writes_wind_when_explicitly_requested(self, model, tmp_path):
        xlsx = tmp_path / "station.xlsx"
        _write_ag5_style_excel(xlsx)

        met = UniformMetForcingClassic(model)
        met.from_excel(str(xlsx), write_wind=True)

        assert model.config.get("wndfile") == "sfincs.wnd"
        assert (Path(model.root) / "sfincs.wnd").is_file()

    def test_writes_pressure_when_column_present(self, model, tmp_path):
        xlsx = tmp_path / "station.xlsx"
        _write_ag5_style_excel(xlsx, with_pressure=True)

        met = UniformMetForcingClassic(model)
        met.from_excel(str(xlsx))

        assert model.config.get("patmfile") == "sfincs.patm"
        patm_path = Path(model.root) / "sfincs.patm"
        written = pd.read_csv(patm_path, sep=" ", header=None, names=["time", "patm"])
        # hPa -> Pa conversion: ~1010 hPa should become ~101000 Pa
        assert written["patm"].mean() > 90000

    def test_skips_pressure_when_column_absent(self, model, tmp_path):
        xlsx = tmp_path / "station.xlsx"
        _write_ag5_style_excel(xlsx, with_pressure=False)

        met = UniformMetForcingClassic(model)
        met.from_excel(str(xlsx))

        assert model.config.get("patmfile") is None
        assert not (Path(model.root) / "sfincs.patm").exists()

    def test_raises_when_window_crop_is_empty(self, model, tmp_path):
        xlsx = tmp_path / "station.xlsx"
        _write_ag5_style_excel(xlsx, hours=5, start="2020-01-01 00:00")

        met = UniformMetForcingClassic(model)
        with pytest.raises(RuntimeError):
            met.from_excel(str(xlsx))


@pytest.mark.skipif(not REAL_XLSX_PATH.exists(), reason="Real AG5-0359.xlsx not available on this machine")
class TestFromExcelRealData:
    def test_precip_and_pressure_from_real_station_file(self, tmp_path):
        m = SfincsModel(root=str(tmp_path / "run"), mode="w+")
        GridMaker(m).setup_grid(plot=False, **GRID_KWARGS)
        # A window known to fall within the station's real coverage (~2024-04 to 2025-07).
        m.setup_config(tref="20250618 000000", tstart="20250618 000000", tstop="20250626 230000")

        met = UniformMetForcingClassic(m)
        met.from_excel(str(REAL_XLSX_PATH))

        assert m.config.get("precipfile") == "sfincs.prcp"
        assert m.config.get("patmfile") == "sfincs.patm"
        assert m.config.get("wndfile") is None

        prcp = pd.read_csv(Path(m.root) / "sfincs.prcp", sep=" ", header=None, names=["time", "prcp"])
        assert len(prcp) > 0
        assert (prcp["prcp"] >= 0).all()
