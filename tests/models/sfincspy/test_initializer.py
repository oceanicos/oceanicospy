import yaml

from oceanicospy.models.sfincspy.initializer import Initializer


class TestCreateFolders:
    def test_creates_standard_layout(self, tmp_path):
        init = Initializer(str(tmp_path))
        init.create_folders()

        for name in ["input", "pros", "run", "output"]:
            assert (tmp_path / name).is_dir()

    def test_rerun_wipes_run_and_output_but_not_input(self, tmp_path):
        init = Initializer(str(tmp_path))
        init.create_folders()

        (tmp_path / "input" / "keep_me.txt").write_text("static input")
        (tmp_path / "run" / "stale.txt").write_text("previous run leftover")
        (tmp_path / "output" / "stale.nc").write_text("previous output leftover")

        init.create_folders()

        assert (tmp_path / "input" / "keep_me.txt").exists()
        assert not (tmp_path / "run" / "stale.txt").exists()
        assert not (tmp_path / "output" / "stale.nc").exists()


class TestWriteDataCatalog:
    def test_writes_catalog_inside_input_folder(self, tmp_path):
        init = Initializer(str(tmp_path))
        init.create_folders()

        out_path = init.write_data_catalog(
            {
                "topobathy": {
                    "data_type": "RasterDataset",
                    "driver": "raster",
                    "path": "new_topobathy_SAI_1m_n.tif",
                    "crs": 32617,
                    "nodata": -3.4028235e38,
                }
            }
        )

        assert out_path == str(tmp_path / "input" / "data_catalog.yml")

        with open(out_path) as f:
            catalog = yaml.safe_load(f)

        # Bare filename paths must be left untouched (no case-folder baked in),
        # so HydroMT resolves them against this catalog's own directory.
        assert catalog["topobathy"]["path"] == "new_topobathy_SAI_1m_n.tif"
        assert catalog["topobathy"]["crs"] == 32617

    def test_two_cases_never_cross_reference_each_other(self, tmp_path):
        case_a = Initializer(str(tmp_path / "CasoA"))
        case_b = Initializer(str(tmp_path / "CasoB"))
        case_a.create_folders()
        case_b.create_folders()

        entry = {"topobathy": {"data_type": "RasterDataset", "path": "topobathy.tif"}}
        path_a = case_a.write_data_catalog(entry)
        path_b = case_b.write_data_catalog(entry)

        assert path_a != path_b
        assert "CasoA" in path_a
        assert "CasoB" in path_b
