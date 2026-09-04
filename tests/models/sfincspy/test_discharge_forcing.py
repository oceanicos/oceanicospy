import numpy as np
import pandas as pd
import pytest

from oceanicospy.models.sfincspy.preprocess.discharge_forcing import (
    orifice_discharge,
    weir_discharge,
    compute_exchange_discharge,
    DischargeForcing,
)

hydromt_sfincs = pytest.importorskip("hydromt_sfincs")
from hydromt_sfincs import SfincsModel  # noqa: E402

from oceanicospy.models.sfincspy.preprocess.gridmaker import GridMaker  # noqa: E402


GRID_KWARGS = dict(
    x0=423966, y0=1389496, dx=15, dy=15, mmax=75, nmax=65, rotation=38, epsg=32617,
)


# ---------------------------------------------------------------------------
# Física pura (orificio/vertedero) - sin SFINCS, sin SWMM.
# ---------------------------------------------------------------------------
class TestOrificeDischarge:
    def test_positive_head_diff_gives_positive_flow(self):
        q = orifice_discharge(head_diff=0.5, area=0.3, cd=0.6)
        assert q > 0

    def test_negative_head_diff_gives_negative_flow(self):
        q = orifice_discharge(head_diff=-0.5, area=0.3, cd=0.6)
        assert q < 0

    def test_zero_head_diff_gives_zero_flow(self):
        assert orifice_discharge(head_diff=0.0, area=0.3) == 0.0

    def test_matches_manual_formula(self):
        # Q = Cd * A * sqrt(2 g |dh|), same formula the SFINCS manual
        # documents for its own drnfile "culvert" (par1 = Cd*A*sqrt(2g)).
        head_diff, area, cd = 0.8, 0.25, 0.6
        expected = cd * area * np.sqrt(2 * 9.81 * head_diff)
        assert orifice_discharge(head_diff, area, cd) == pytest.approx(expected)

    def test_larger_head_diff_gives_larger_flow(self):
        q_small = orifice_discharge(0.1, area=0.3)
        q_large = orifice_discharge(1.0, area=0.3)
        assert q_large > q_small


class TestWeirDischarge:
    def test_zero_or_negative_head_gives_zero_flow(self):
        assert weir_discharge(head=0.0, length=1.0) == 0.0
        assert weir_discharge(head=-0.1, length=1.0) == 0.0

    def test_positive_head_gives_positive_flow(self):
        assert weir_discharge(head=0.2, length=1.0) > 0

    def test_matches_manual_formula(self):
        head, length, cw = 0.3, 1.5, 1.7
        expected = cw * length * head**1.5
        assert weir_discharge(head, length, cw) == pytest.approx(expected)


class TestComputeExchangeDischarge:
    """Reproduces the regime logic Iber-SWMM uses at the manhole-surface
    connection (Sañudo et al. 2025): weir when the surface is dry/below
    the rim, orifice when both sides are submerged. Sign convention:
    positive = surcharge onto the surface, negative = re-entry into the sewer.
    """

    def test_surcharge_weir_regime_when_surface_dry(self):
        # Node above rim, street still dry (below rim) -> non-submerged weir.
        q = compute_exchange_discharge(
            node_hgl=1.5, rim_elevation=1.2, surface_level=1.0,
            area=0.3, length=1.5,
        )
        assert q > 0
        assert q == pytest.approx(weir_discharge(1.5 - 1.2, 1.5))

    def test_surcharge_orifice_regime_when_surface_also_submerged(self):
        # Both node and street above rim, node still higher -> submerged orifice.
        q = compute_exchange_discharge(
            node_hgl=1.8, rim_elevation=1.2, surface_level=1.4,
            area=0.3, length=1.5,
        )
        assert q > 0
        assert q == pytest.approx(orifice_discharge(1.8 - 1.4, 0.3))

    def test_reentry_when_surface_higher_than_node(self):
        # Street water level exceeds the node's HGL -> flow reverses (drains
        # back into the sewer) - this is the "eventually evacuates" behaviour.
        q = compute_exchange_discharge(
            node_hgl=1.0, rim_elevation=1.2, surface_level=1.4,
            area=0.3, length=1.5,
        )
        assert q < 0

    def test_no_flow_when_node_below_rim_and_surface_dry(self):
        q = compute_exchange_discharge(
            node_hgl=0.8, rim_elevation=1.2, surface_level=0.5,
            area=0.3, length=1.5,
        )
        assert q == 0.0

    def test_full_cycle_surcharge_then_reentry(self):
        """A minimal, SWMM-free simulation of the physical process described
        by the user: the node surcharges, floods the surface, and later -
        once the node level drops enough - the water drains back in."""
        rim = 1.2

        # Fase 1: la calle todavia esta seca y el nodo se satura -> vertedero,
        # el agua se desborda hacia la superficie.
        q_surcharge = compute_exchange_discharge(
            node_hgl=1.6, rim_elevation=rim, surface_level=1.0, area=0.3, length=1.5,
        )
        assert q_surcharge > 0

        # Fase 2: ya se acumulo agua en la calle (nivel superficial ahora por
        # encima de la cota del pozo, producto del desborde anterior) y el
        # nodo bajo lo suficiente -> el agua empieza a regresar a la red.
        q_reentry = compute_exchange_discharge(
            node_hgl=1.0, rim_elevation=rim, surface_level=1.4, area=0.3, length=1.5,
        )
        assert q_reentry < 0


# ---------------------------------------------------------------------------
# DischargeForcing (SFINCS side) - genérico, sin SWMM.
# ---------------------------------------------------------------------------
@pytest.fixture
def model(tmp_path):
    m = SfincsModel(root=str(tmp_path / "run"), mode="w+")
    GridMaker(m).setup_grid(plot=False, **GRID_KWARGS)
    m.setup_config(tref="20250618 000000", tstart="20250618 000000", tstop="20250618 120000")
    return m


class TestDischargeForcing:
    def test_registers_multiple_named_points(self, model):
        idx = pd.date_range("2025-06-18", periods=12, freq="h")
        df = pd.DataFrame(
            {"R1": np.random.default_rng(0).random(12), "R2": np.random.default_rng(1).random(12)},
            index=idx,
        )

        dq = DischargeForcing(model)
        dq.from_dataframe(df, locations={"R1": (423980, 1389510), "R2": (423990, 1389520)})

        assert "dis" in model.forcing
        assert dq.point_ids == {"R1": 1, "R2": 2}

    def test_raises_on_missing_column(self, model):
        idx = pd.date_range("2025-06-18", periods=12, freq="h")
        df = pd.DataFrame({"R1": np.random.default_rng(0).random(12)}, index=idx)

        dq = DischargeForcing(model)
        with pytest.raises(ValueError):
            dq.from_dataframe(df, locations={"R1": (423980, 1389510), "R2": (423990, 1389520)})

    def test_raises_when_window_crop_is_empty(self, model):
        idx = pd.date_range("2020-01-01", periods=12, freq="h")  # outside tstart/tstop
        df = pd.DataFrame({"R1": np.random.default_rng(0).random(12)}, index=idx)

        dq = DischargeForcing(model)
        with pytest.raises(RuntimeError):
            dq.from_dataframe(df, locations={"R1": (423980, 1389510)})

    def test_writes_src_and_dis_files_on_model_write(self, model, tmp_path):
        idx = pd.date_range("2025-06-18", periods=12, freq="h")
        df = pd.DataFrame({"R1": np.random.default_rng(0).random(12)}, index=idx)

        dq = DischargeForcing(model)
        dq.from_dataframe(df, locations={"R1": (423980, 1389510)})
        model.write()

        assert (tmp_path / "run" / "sfincs.src").is_file()
        assert (tmp_path / "run" / "sfincs.dis").is_file()

    def test_end_to_end_without_swmm_using_synthetic_node_series(self, model, tmp_path):
        """Demonstrates the exact workflow the user asked for: build a
        discharge forcing straight from compute_exchange_discharge() output,
        with no pyswmm/SWMM involved at all - synthetic node HGL and surface
        level series standing in for what pyswmm and SFINCS would normally
        provide in the real iterative coupling."""
        rim = 1.2
        idx = pd.date_range("2025-06-18", periods=6, freq="h")
        # Paired (node_hgl, surface_level) per step: dry street while the
        # node surcharges (weir, positive), then the street floods from that
        # surcharge while the node recedes, eventually draining back in
        # (orifice, negative) as the node keeps dropping.
        node_hgl_series =     [0.9, 1.6, 1.6, 1.0, 0.8, 0.8]
        surface_level_series = [1.0, 1.0, 1.3, 1.4, 1.3, 1.2]

        exchange_q = [
            compute_exchange_discharge(h, rim, s, area=0.3, length=1.5)
            for h, s in zip(node_hgl_series, surface_level_series)
        ]
        df = pd.DataFrame({"R1": exchange_q}, index=idx)

        dq = DischargeForcing(model)
        dq.from_dataframe(df, locations={"R1": (423980, 1389510)})
        model.write()

        written = pd.read_csv(
            tmp_path / "run" / "sfincs.dis", sep=r"\s+", header=None, names=["time", "q"]
        )
        # Surcharge (positive) then re-entry (negative) both made it through.
        assert (written["q"] > 0).any()
        assert (written["q"] < 0).any()
