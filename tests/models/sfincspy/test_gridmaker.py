import numpy as np
import pytest

hydromt_sfincs = pytest.importorskip("hydromt_sfincs")
from hydromt_sfincs import SfincsModel  # noqa: E402

from oceanicospy.models.sfincspy.preprocess.gridmaker import GridMaker  # noqa: E402


GRID_KWARGS = dict(
    x0=423966,
    y0=1389496,
    dx=15,
    dy=15,
    mmax=12,  # x-direction, deliberately != nmax to catch axis-swap bugs
    nmax=7,   # y-direction
    rotation=38,
    epsg=32617,
)


@pytest.fixture
def model(tmp_path):
    return SfincsModel(root=str(tmp_path / "model"), mode="w+")


class TestSetupGrid:
    def test_stores_axis_specific_grid_info(self, model):
        grid = GridMaker(model)
        info = grid.setup_grid(plot=False, **GRID_KWARGS)

        assert info["mmax"] == GRID_KWARGS["mmax"]
        assert info["nmax"] == GRID_KWARGS["nmax"]

    def test_writes_expected_grid_shape_to_model(self, model):
        grid = GridMaker(model)
        grid.setup_grid(plot=False, **GRID_KWARGS)

        # HydroMT's own RegularGrid is the ground truth GridMaker.plot_grid
        # must stay consistent with: mmax cells in x, nmax cells in y.
        assert model.reggrid.mmax == GRID_KWARGS["mmax"]
        assert model.reggrid.nmax == GRID_KWARGS["nmax"]

    def test_warns_on_very_large_grid(self, capsys):
        model_stub = type("Stub", (), {})()
        # Only exercise the warning branch, not a full hydromt build,
        # by calling setup_grid with mocked model.setup_grid as a no-op.
        model_stub.setup_grid = lambda **kwargs: None
        grid = GridMaker(model_stub)
        grid.setup_grid(plot=False, mmax=2000, nmax=2000, x0=0, y0=0, dx=1, dy=1)
        captured = capsys.readouterr()
        assert "3 millones" in captured.out


class TestPlotGrid:
    def test_edge_counts_match_axis_specific_dims(self, model):
        grid = GridMaker(model)
        grid.setup_grid(plot=False, **GRID_KWARGS)

        # Recompute the same edges plot_grid uses, without touching matplotlib,
        # to confirm mmax pairs with dx (x) and nmax pairs with dy (y).
        x_edges = grid.grid_info["x0"] + np.arange(grid.grid_info["mmax"] + 1) * grid.grid_info["dx"]
        y_edges = grid.grid_info["y0"] + np.arange(grid.grid_info["nmax"] + 1) * grid.grid_info["dy"]

        assert len(x_edges) == GRID_KWARGS["mmax"] + 1
        assert len(y_edges) == GRID_KWARGS["nmax"] + 1
        assert len(x_edges) != len(y_edges)  # guards against an nmax/mmax swap

    def test_plot_grid_writes_png(self, model):
        grid = GridMaker(model)
        grid.setup_grid(plot=False, **GRID_KWARGS)

        out_path = grid.plot_grid(filename="grid_test.png")

        from pathlib import Path
        assert Path(out_path).is_file()
        assert Path(out_path).name == "grid_test.png"

    def test_plot_grid_without_setup_raises(self, model):
        grid = GridMaker(model)
        with pytest.raises(RuntimeError):
            grid.plot_grid()
