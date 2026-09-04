import shutil

import numpy as np
import pandas as pd
import pytest
import xarray as xr

hydromt_sfincs = pytest.importorskip("hydromt_sfincs")
rasterio = pytest.importorskip("rasterio")

from hydromt_sfincs import SfincsModel  # noqa: E402
from rasterio.transform import from_origin  # noqa: E402

from oceanicospy.models.sfincspy.initializer import Initializer  # noqa: E402
from oceanicospy.models.sfincspy.preprocess.gridmaker import GridMaker  # noqa: E402
from oceanicospy.models.sfincspy.preprocess.bathymaker import BathyMaker  # noqa: E402
from oceanicospy.models.sfincspy.preprocess.wind_forcing import WindForcing  # noqa: E402
import oceanicospy.models.sfincspy.preprocess.wind_forcing as wind_forcing_module  # noqa: E402


GRID_KWARGS = dict(
    x0=423966, y0=1389496, dx=15, dy=15, mmax=75, nmax=65, rotation=38, epsg=32617,
)


def _write_gradient_raster(path, low=-6.0, high=2.0):
    transform = from_origin(423000, 1391000, 1, 1)
    xx, _ = np.meshgrid(np.arange(3000), np.arange(3000))
    data = (low + (high - low) * (xx / 3000)).astype("float32")
    with rasterio.open(
        path, "w", driver="GTiff", height=3000, width=3000, count=1,
        dtype="float32", crs="EPSG:32617", transform=transform, nodata=-9999,
    ) as dst:
        dst.write(data, 1)


def _write_era5_localtime_nc(path, start="2025-06-18 00:00", hours=48, u_mean=5.0, v_mean=3.0):
    """Shaped like a real ERA5 CDS download already run through
    ERA5Downloader.format_to_localtime(): dims (time, latitude, longitude),
    longitude in 0-360 convention, timestamps already in local time."""
    lat = np.linspace(12.9, 12.3, 25)
    lon = np.linspace(278.0, 278.6, 25)  # 0-360 convention, like the real CDS files
    time = pd.date_range(start, periods=hours, freq="h")
    rng = np.random.default_rng(0)
    u10 = rng.normal(u_mean, 0.3, size=(hours, 25, 25)).astype("float32")
    v10 = rng.normal(v_mean, 0.3, size=(hours, 25, 25)).astype("float32")
    ds = xr.Dataset(
        {
            "u10": (("time", "latitude", "longitude"), u10),
            "v10": (("time", "latitude", "longitude"), v10),
        },
        coords={"time": time, "latitude": lat, "longitude": lon},
    )
    ds.to_netcdf(path)


@pytest.fixture
def built_model(tmp_path):
    src_raster = tmp_path / "topobathy.tif"
    _write_gradient_raster(src_raster)

    case_root = tmp_path / "CasoTest"
    init = Initializer(str(case_root))
    init.create_folders()
    shutil.copy(src_raster, init.dict_folders["input"] + "topobathy.tif")

    catalog_path = init.write_data_catalog(
        {
            "topobathy": {
                "data_type": "RasterDataset", "driver": "raster",
                "path": "topobathy.tif", "crs": 32617, "nodata": -9999,
            }
        }
    )

    model = SfincsModel(data_libs=[catalog_path], root=init.dict_folders["run"], mode="w+")
    GridMaker(model).setup_grid(plot=False, **GRID_KWARGS)
    BathyMaker(model, catalog_path).setup_bathy(zmin_active=-4, zmax_bounds=-2, plot=False)
    model.setup_config(tref="20250618 000000", tstart="20250618 060000", tstop="20250618 180000")

    return model, init


class TestFromEra5Nc:
    def test_produces_nonzero_wind_on_rotated_grid(self, built_model, tmp_path):
        model, _ = built_model
        nc_path = tmp_path / "era5_local.nc"
        _write_era5_localtime_nc(nc_path, u_mean=5.0, v_mean=3.0)

        wind = WindForcing(model)
        out_path = wind.from_era5_nc(str(nc_path))

        written = xr.open_dataset(out_path)

        # Regression check: interpolating against model.grid.x/.y (bare cell
        # indices on a rotated grid, not real coordinates) silently produced
        # amu == amv == 0.0 everywhere. A correct fix must diverge from that.
        assert float(written["amu"].mean()) != 0.0
        assert float(written["amv"].mean()) != 0.0
        assert 3.0 < float(written["amu"].mean()) < 7.0  # close to the synthetic u_mean=5.0
        assert 1.0 < float(written["amv"].mean()) < 5.0  # close to the synthetic v_mean=3.0

    def test_output_shape_matches_grid(self, built_model, tmp_path):
        model, _ = built_model
        nc_path = tmp_path / "era5_local.nc"
        _write_era5_localtime_nc(nc_path)

        wind = WindForcing(model)
        out_path = wind.from_era5_nc(str(nc_path))

        written = xr.open_dataset(out_path)
        assert written["amu"].sizes["y"] == GRID_KWARGS["nmax"]
        assert written["amu"].sizes["x"] == GRID_KWARGS["mmax"]

    def test_selects_only_model_time_window(self, built_model, tmp_path):
        model, _ = built_model
        # tstart/tstop are 06:00-18:00 on 2025-06-18 -> 13 hourly steps.
        nc_path = tmp_path / "era5_local.nc"
        _write_era5_localtime_nc(nc_path, start="2025-06-18 00:00", hours=48)

        wind = WindForcing(model)
        out_path = wind.from_era5_nc(str(nc_path))

        written = xr.open_dataset(out_path)
        assert written.sizes["time"] == 13

    def test_raises_keyerror_when_wind_vars_missing(self, built_model, tmp_path):
        model, _ = built_model
        nc_path = tmp_path / "no_wind.nc"
        ds = xr.Dataset(
            {"dummy": (("time",), [1.0, 2.0])},
            coords={"time": pd.date_range("2025-06-18 06:00", periods=2, freq="h")},
        )
        ds.to_netcdf(nc_path)

        wind = WindForcing(model)
        with pytest.raises(KeyError):
            wind.from_era5_nc(str(nc_path))

    def test_registers_netamuamvfile_in_config(self, built_model, tmp_path):
        model, _ = built_model
        nc_path = tmp_path / "era5_local.nc"
        _write_era5_localtime_nc(nc_path)

        wind = WindForcing(model)
        wind.from_era5_nc(str(nc_path), out_filename="wind_era5.nc")

        assert model.config.get("netamuamvfile") == "wind_era5.nc"


class TestGetWindsFromERA5:
    def test_skips_download_and_returns_localtime_path_when_raw_file_exists(
        self, built_model, tmp_path, monkeypatch
    ):
        model, init = built_model
        input_dir = init.dict_folders["input"]

        # Pre-stage the raw file so the "already exists" branch is taken -
        # download_era5_winds must NOT be called (no network in tests).
        (tmp_path / "CasoTest" / "input" / "winds_era5.nc").write_text("placeholder")

        def _fail_if_called(*args, **kwargs):
            raise AssertionError("download_era5_winds should not be called when the raw file already exists")

        monkeypatch.setattr(wind_forcing_module, "download_era5_winds", _fail_if_called)

        wind = WindForcing(model)
        result_path = wind.get_winds_from_ERA5(
            wind_info={
                "lon_ll_corner_wind": -82.0, "lat_ll_corner_wind": 12.3,
                "nx_wind": 25, "ny_wind": 25, "dx_wind": 0.025, "dy_wind": 0.025,
            },
            input_dir=input_dir,
        )

        assert result_path.endswith("winds_era5_localtime.nc")

    def test_download_path_calls_shared_utility_with_format_localtime_true(
        self, built_model, tmp_path, monkeypatch
    ):
        model, init = built_model
        input_dir = init.dict_folders["input"]

        calls = {}

        def _fake_download(wind_info, tstart, tstop, utc_offset_hours, filepath, format_localtime):
            calls["format_localtime"] = format_localtime
            calls["utc_offset_hours"] = utc_offset_hours

        monkeypatch.setattr(wind_forcing_module, "download_era5_winds", _fake_download)

        wind = WindForcing(model)
        wind.get_winds_from_ERA5(
            wind_info={
                "lon_ll_corner_wind": -82.0, "lat_ll_corner_wind": 12.3,
                "nx_wind": 25, "ny_wind": 25, "dx_wind": 0.025, "dy_wind": 0.025,
            },
            input_dir=input_dir,
        )

        # This is the fix the user asked for: SFINCS wind must always be
        # shifted to local time before use, unlike the original wind_forcing.py
        # which read ERA5 straight in UTC and cropped it with local tstart/tstop.
        assert calls["format_localtime"] is True
        assert calls["utc_offset_hours"] == -5
