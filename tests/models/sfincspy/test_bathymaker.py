import shutil

import numpy as np
import pytest

hydromt_sfincs = pytest.importorskip("hydromt_sfincs")
rasterio = pytest.importorskip("rasterio")

from hydromt_sfincs import SfincsModel  # noqa: E402
from rasterio.transform import from_origin  # noqa: E402

from oceanicospy.models.sfincspy.initializer import Initializer  # noqa: E402
from oceanicospy.models.sfincspy.preprocess.gridmaker import GridMaker  # noqa: E402
from oceanicospy.models.sfincspy.preprocess.bathymaker import BathyMaker  # noqa: E402


GRID_KWARGS = dict(
    x0=423966, y0=1389496, dx=15, dy=15, mmax=75, nmax=65, rotation=38, epsg=32617,
)


def _write_gradient_raster(path, low=-6.0, high=2.0):
    """North-up 3000x3000 m, 1 m/px raster covering the rotated test grid,
    with an elevation gradient from `low` (west) to `high` (east) so it
    straddles both zmin_active and zmax_bounds thresholds used below."""
    transform = from_origin(423000, 1391000, 1, 1)
    xx, _ = np.meshgrid(np.arange(3000), np.arange(3000))
    data = (low + (high - low) * (xx / 3000)).astype("float32")
    with rasterio.open(
        path, "w", driver="GTiff", height=3000, width=3000, count=1,
        dtype="float32", crs="EPSG:32617", transform=transform, nodata=-9999,
    ) as dst:
        dst.write(data, 1)


@pytest.fixture
def built_case(tmp_path):
    """A case with grid + a single 'topobathy' dataset registered through
    Initializer.write_data_catalog, ready for BathyMaker.setup_bathy()."""
    src_raster = tmp_path / "source_topobathy.tif"
    _write_gradient_raster(src_raster)

    case_root = tmp_path / "CasoTest"
    init = Initializer(str(case_root))
    init.create_folders()
    shutil.copy(src_raster, init.dict_folders["input"] + "topobathy.tif")

    catalog_path = init.write_data_catalog(
        {
            "topobathy": {
                "data_type": "RasterDataset",
                "driver": "raster",
                "path": "topobathy.tif",
                "crs": 32617,
                "nodata": -9999,
            }
        }
    )

    model = SfincsModel(data_libs=[catalog_path], root=init.dict_folders["run"], mode="w+")
    GridMaker(model).setup_grid(plot=False, **GRID_KWARGS)

    return model, catalog_path, init


class TestSetupBathy:
    def test_default_single_dataset_matches_previous_behaviour(self, built_case):
        model, catalog_path, _ = built_case
        bathy = BathyMaker(model, catalog_path)
        bathy.setup_bathy(zmin_active=-4, zmax_bounds=-2, plot=False)

        assert "dep" in model.grid.data_vars
        assert not np.isnan(model.grid["dep"].values).all()

    def test_active_and_boundary_cells_follow_thresholds(self, built_case):
        model, catalog_path, _ = built_case
        bathy = BathyMaker(model, catalog_path)
        bathy.setup_bathy(zmin_active=-4, zmax_bounds=-2, plot=False)

        msk = model.grid["msk"].values
        dep = model.grid["dep"].values

        # msk==1 (active) cells must have elevation >= zmin_active
        active = msk == 1
        assert active.any()
        assert np.nanmin(dep[active]) >= -4

        # msk==2 (waterlevel boundary) cells must have elevation <= zmax_bounds
        boundary = msk == 2
        assert boundary.any()
        assert np.nanmax(dep[boundary]) <= -2

    def test_plot_true_writes_model_and_figure(self, built_case):
        model, catalog_path, init = built_case
        bathy = BathyMaker(model, catalog_path)

        # Regression test: plot=True used to always raise FileNotFoundError
        # because plot_sfincs_raster reads <root>/gis/dep.tif, which HydroMT
        # only writes to disk on model.write() - setup_bathy(plot=True)
        # never called write() first.
        bathy.setup_bathy(zmin_active=-4, zmax_bounds=-2, plot=True)

        from pathlib import Path
        fig_path = Path(init.dict_folders["run"]) / "figures" / "dep.png"
        assert fig_path.is_file()


class TestPlotSfincsRaster:
    def test_raises_if_raster_missing(self, built_case):
        model, catalog_path, _ = built_case
        bathy = BathyMaker(model, catalog_path)
        with pytest.raises(FileNotFoundError):
            bathy.plot_sfincs_raster("dep")

    def test_handles_rotated_grid_without_cartopy(self, built_case):
        model, catalog_path, _ = built_case
        bathy = BathyMaker(model, catalog_path)
        bathy.setup_bathy(zmin_active=-4, zmax_bounds=-2, plot=False)
        model.write()

        # This must not raise even though the grid is rotated (rotation=38)
        # and the written dep.tif therefore has a genuinely rotated affine
        # transform (not north-up) - pcolormesh over per-cell coordinates
        # handles that; imshow(extent=...) would silently distort it.
        out_path = bathy.plot_sfincs_raster("dep", filename="dep_rotated.png")

        from pathlib import Path
        assert Path(out_path).is_file()
