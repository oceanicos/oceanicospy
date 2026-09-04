from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

hydromt_sfincs = pytest.importorskip("hydromt_sfincs")

from hydromt_sfincs import SfincsModel  # noqa: E402

from oceanicospy.models.sfincspy.preprocess.gridmaker import GridMaker  # noqa: E402
from oceanicospy.models.sfincspy.preprocess.waves_forcing import WavesForcing  # noqa: E402


GRID_KWARGS = dict(
    x0=423966, y0=1389496, dx=15, dy=15, mmax=75, nmax=65, rotation=38, epsg=32617,
)

# Real XBeach output already inspected during the SFINCS audit: contains both
# 'zs' (~0.42-0.90 m) and 'H' (~0.0-0.125 m) at the boundary point.
REAL_NC_PATH = Path(
    "D:/04_BACKUP_SERVER/daguirreos_sfincs/sfincs/DANIEL_AGUIRRE/data/E1/E1_profile1D_Nov2008.nc"
)


@pytest.fixture
def model(tmp_path):
    m = SfincsModel(root=str(tmp_path / "run"), mode="w+")
    GridMaker(m).setup_grid(plot=False, **GRID_KWARGS)
    m.setup_config(tref="20250618 000000", tstart="20250618 060000", tstop="20250618 180000")
    return m


def _write_synthetic_xbeach_nc(path, n_time=48, mean_before=1.0, mean_after=3.0, split=24, amp=0.2):
    """A minimal XBeach-shaped netCDF with 'zs' and 'H', one spatial point,
    where the mean level shifts partway through - used to check that the
    demeaning happens on the *cropped* window, not the whole series."""
    time = np.arange(n_time) * 3600.0  # seconds, hourly
    zs = np.where(
        np.arange(n_time) < split,
        mean_before + amp * np.sin(np.arange(n_time)),
        mean_after + amp * np.sin(np.arange(n_time)),
    ).reshape(n_time, 1, 1)
    H = np.abs(amp * np.sin(np.arange(n_time))).reshape(n_time, 1, 1) + 0.05

    ds = xr.Dataset(
        {
            "zs": (("globaltime", "ny", "nx"), zs),
            "H": (("globaltime", "ny", "nx"), H),
        },
        coords={
            "globaltime": time,
            "globalx": (("ny", "nx"), np.array([[0.0]])),
            "pointx": ("points", np.array([0.0])),
        },
    )
    ds.to_netcdf(path)


class TestFromXbeach:
    def test_raises_keyerror_when_zs_missing(self, model, tmp_path):
        nc_path = tmp_path / "no_zs.nc"
        ds = xr.Dataset(
            {"H": (("globaltime", "ny", "nx"), np.zeros((5, 1, 1)))},
            coords={
                "globaltime": np.arange(5) * 3600.0,
                "globalx": (("ny", "nx"), np.array([[0.0]])),
                "pointx": ("points", np.array([0.0])),
            },
        )
        ds.to_netcdf(nc_path)

        waves = WavesForcing(model)
        with pytest.raises(KeyError):
            waves.from_xbeach(str(nc_path))

    def test_output_oscillates_around_zero_over_cropped_window(self, model, tmp_path):
        # tstart=06:00, tstop=18:00 on 2025-06-18, tref=2025-06-18 00:00 ->
        # cropped window is hours 6-18, entirely in the "mean_after" half
        # (split=24h in) - so this alone doesn't test the crop-before-mean
        # rule yet; the next test does that explicitly with a tighter split.
        nc_path = tmp_path / "xbeach.nc"
        _write_synthetic_xbeach_nc(nc_path, n_time=48, mean_before=1.0, mean_after=3.0, split=24)

        waves = WavesForcing(model)
        out_path = waves.from_xbeach(str(nc_path))

        written = pd.read_csv(out_path, sep=" ", header=None, names=["time", "bzi"])
        assert abs(written["bzi"].mean()) < 1e-6
        # unlike H (always >= 0), bzi must cross zero
        assert (written["bzi"] > 0).any() and (written["bzi"] < 0).any()

    def test_mean_is_computed_on_cropped_window_not_full_series(self, model, tmp_path):
        # Model window is hours 6-18. Put a mean shift exactly at hour 12
        # (mid-window): if the code demeaned over the FULL 48h series first,
        # the cropped output would NOT be centered on zero, since the full
        # series mean is dominated by hours outside [6,18].
        nc_path = tmp_path / "xbeach_shift.nc"
        _write_synthetic_xbeach_nc(nc_path, n_time=48, mean_before=0.0, mean_after=10.0, split=12)

        waves = WavesForcing(model)
        out_path = waves.from_xbeach(str(nc_path))

        written = pd.read_csv(out_path, sep=" ", header=None, names=["time", "bzi"])
        # Within [6,18] the series is a mix of the 0.0-mean part (hours 6-11)
        # and the 10.0-mean part (hours 12-18) - its own mean should still be
        # ~0 because the code demeans using exactly this cropped window.
        assert abs(written["bzi"].mean()) < 1e-6

    def test_registers_bzifile_in_config(self, model, tmp_path):
        nc_path = tmp_path / "xbeach.nc"
        _write_synthetic_xbeach_nc(nc_path)

        waves = WavesForcing(model)
        waves.from_xbeach(str(nc_path), out_filename="sfincs.bzi")

        assert model.config.get("bzifile") == "sfincs.bzi"

    def test_nx_index_selects_arbitrary_profile_point_not_just_pointx(self, model, tmp_path):
        # point_index only ever indexes into ds["pointx"] (the XBeach
        # "point"-type outputs, typically just the runup gauge - a single
        # coordinate) and then finds the nearest match in the *global*
        # spatial array - it cannot target an arbitrary nx position
        # directly. nx_index must bypass pointx entirely and select the
        # nx position directly, so a case can force with e.g. the point
        # where depth crosses a specific isobath, regardless of where the
        # runup gauge happens to sit.
        n_time, nx = 12, 5
        t = np.arange(n_time)
        time = t * 3600.0
        zs = np.zeros((n_time, 1, nx))
        for j in range(nx):
            amplitude = j + 1  # nx=0 -> amp 1, nx=3 -> amp 4, distinct per index
            zs[:, 0, j] = amplitude * np.sin(2 * np.pi * t / n_time)
        ds = xr.Dataset(
            {"zs": (("globaltime", "ny", "nx"), zs)},
            coords={
                "globaltime": time,
                "globalx": (("ny", "nx"), np.array([[0.0, -10.0, -20.0, -30.0, -40.0]])),
                "pointx": ("points", np.array([0.0])),  # nearest to nx_index=0 only
            },
        )
        nc_path = tmp_path / "xbeach_multi.nc"
        ds.to_netcdf(nc_path)

        model.setup_config(tref="20250618 000000", tstart="20250618 000000", tstop="20250618 110000")
        waves = WavesForcing(model)
        out_path = waves.from_xbeach(str(nc_path), nx_index=3)

        written = pd.read_csv(out_path, sep=" ", header=None, names=["time", "bzi"])
        # nx=3 has amplitude 4, nx=0 (what point_index=0 would pick, since
        # pointx=[0.0] is nearest to nx=0) has amplitude 1 - the two are
        # clearly distinguishable, proving nx_index bypassed pointx.
        assert written["bzi"].abs().max() > 3.0

    def test_raises_when_window_crop_is_empty(self, model, tmp_path):
        nc_path = tmp_path / "xbeach.nc"
        # nc timestamps are always tref + globaltime seconds (here: tref, tref+1h).
        # Point [tstart, tstop] a full day later so the crop yields nothing.
        _write_synthetic_xbeach_nc(nc_path, n_time=2)
        model.setup_config(tref="20200101 000000", tstart="20200102 000000", tstop="20200102 010000")

        waves = WavesForcing(model)
        with pytest.raises(RuntimeError):
            waves.from_xbeach(str(nc_path))


@pytest.mark.skipif(not REAL_NC_PATH.exists(), reason="Real XBeach output not available on this machine")
class TestFromXbeachRealData:
    def test_uses_zs_not_h_on_real_e1_output(self, model, tmp_path):
        # E1's real event window: 2008-11-20 11:00 to 2008-11-24 21:00.
        model.setup_config(
            tref="20081120 110000", tstart="20081120 110000", tstop="20081124 210000",
        )

        waves = WavesForcing(model)
        out_path = waves.from_xbeach(str(REAL_NC_PATH), point_index=0)

        written = pd.read_csv(out_path, sep=" ", header=None, names=["time", "bzi"])

        # Regression check: H at this boundary point never exceeds ~0.125 m
        # and is always >= 0 (it's a wave-height magnitude); a correctly
        # demeaned zs anomaly should span a visibly wider, zero-centered range.
        assert (written["bzi"] < 0).any(), "bzi must cross zero - it must not still be H"
        assert abs(written["bzi"].mean()) < 1e-6
