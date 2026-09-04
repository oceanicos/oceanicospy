import shutil

import numpy as np
import pytest

hydromt_sfincs = pytest.importorskip("hydromt_sfincs")
gpd = pytest.importorskip("geopandas")
rasterio = pytest.importorskip("rasterio")

from hydromt_sfincs import SfincsModel  # noqa: E402
from rasterio.transform import from_origin  # noqa: E402
from shapely.geometry import box  # noqa: E402

from oceanicospy.models.sfincspy.preprocess.gridmaker import GridMaker  # noqa: E402
from oceanicospy.models.sfincspy.preprocess.bathymaker import BathyMaker  # noqa: E402
from oceanicospy.models.sfincspy.preprocess.bottom_friction import BottomFriction  # noqa: E402


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


@pytest.fixture
def built_model(tmp_path):
    """A model with grid + dep + mask ready for BottomFriction."""
    src_raster = tmp_path / "topobathy.tif"
    _write_gradient_raster(src_raster)

    catalog_path = tmp_path / "data_catalog.yml"
    catalog_path.write_text(
        "topobathy:\n"
        "  data_type: RasterDataset\n"
        "  driver: raster\n"
        f"  path: {str(src_raster).replace(chr(92), '/')}\n"
        "  crs: 32617\n"
        "  nodata: -9999\n"
    )

    model = SfincsModel(data_libs=[str(catalog_path)], root=str(tmp_path / "run"), mode="w+")
    GridMaker(model).setup_grid(plot=False, **GRID_KWARGS)
    BathyMaker(model, str(catalog_path)).setup_bathy(zmin_active=-4, zmax_bounds=-2, plot=False)
    return model


class TestSetupUniform:
    def test_produces_spatial_manning_map(self, built_model):
        BottomFriction(built_model).setup_uniform(manning_land=0.035, manning_sea=0.02)

        man = built_model.grid["manning"].values
        dep = built_model.grid["dep"].values

        assert np.allclose(man[dep >= 0], 0.035)
        assert np.allclose(man[dep < 0], 0.02)
        assert built_model.config.get("manningfile") == "sfincs.man"


class TestWriteReclassTable:
    def test_writes_expected_csv_format(self, tmp_path):
        out_path = BottomFriction.write_reclass_table(
            {1: 0.035, 2: 0.015}, tmp_path / "reclass.csv"
        )
        content = (tmp_path / "reclass.csv").read_text()
        assert content == ",N\n1,0.035\n2,0.015\n"
        assert out_path == str(tmp_path / "reclass.csv")


class TestRasterizeLandcover:
    def test_later_class_overwrites_earlier_on_overlap(self, tmp_path):
        # A full "green" subcatchment square, and a smaller "impervious"
        # square fully inside it - mirrors cuenca_rejilla-N vs. its _imp
        # subset without requiring an explicit geometric difference.
        green = gpd.GeoDataFrame(geometry=[box(0, 0, 100, 100)], crs="EPSG:32617")
        impervious = gpd.GeoDataFrame(geometry=[box(20, 20, 40, 40)], crs="EPSG:32617")

        out_path = BottomFriction.rasterize_landcover(
            class_polygons={1: [green], 2: [impervious]},
            out_path=tmp_path / "lulc.tif",
            resolution=1,
            bounds=(0, 0, 100, 100),
            crs="EPSG:32617",
        )

        with rasterio.open(out_path) as src:
            data = src.read(1)

        # Sample a point clearly inside impervious (30,30) and one clearly
        # in green-only area (10,10), using rasterio's own indexing.
        r_imp, c_imp = src.index(30, 30)
        r_green, c_green = src.index(10, 10)
        assert data[r_imp, c_imp] == 2
        assert data[r_green, c_green] == 1

    def test_reprojects_mismatched_crs(self, tmp_path):
        # Polygon supplied in a different CRS than the requested output CRS.
        gdf = gpd.GeoDataFrame(geometry=[box(0, 0, 100, 100)], crs="EPSG:32617").to_crs("EPSG:4326")

        out_path = BottomFriction.rasterize_landcover(
            class_polygons={1: [gdf]},
            out_path=tmp_path / "lulc_reproj.tif",
            resolution=1,
            bounds=(0, 0, 100, 100),
            crs="EPSG:32617",
        )

        with rasterio.open(out_path) as src:
            data = src.read(1)
            r, c = src.index(50, 50)
            assert data[r, c] == 1


class TestSetupFromLandcover:
    def test_end_to_end_with_rasterized_classes(self, built_model, tmp_path):
        # Two classes covering only part of the domain footprint used in
        # built_model's dep raster (423000-426000, 1388000-1391000).
        green = gpd.GeoDataFrame(geometry=[box(423600, 1390100, 423900, 1390500)], crs="EPSG:32617")
        impervious = gpd.GeoDataFrame(geometry=[box(423700, 1390200, 423750, 1390250)], crs="EPSG:32617")

        lulc_path = BottomFriction.rasterize_landcover(
            class_polygons={1: [green], 2: [impervious]},
            out_path=tmp_path / "lulc.tif",
            resolution=1,
            bounds=(423000, 1388000, 426000, 1391000),
            crs="EPSG:32617",
        )
        reclass_path = BottomFriction.write_reclass_table(
            BottomFriction.DEFAULT_LANDCOVER_MANNING, tmp_path / "reclass.csv"
        )

        BottomFriction(built_model).setup_from_landcover(
            lulc_path=lulc_path,
            reclass_table=reclass_path,
            manning_land=0.035,
            manning_sea=0.02,
        )

        man = built_model.grid["manning"].values
        # Falls within [min(sea,imp), max(green,land)] given linear resampling
        # at the transition; just confirm the full documented range is used
        # and nothing falls outside it.
        assert np.nanmin(man) >= 0.014
        assert np.nanmax(man) <= 0.036
