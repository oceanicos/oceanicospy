from pathlib import Path

import pandas as pd
import pytest

pyswmm = pytest.importorskip("pyswmm")

from oceanicospy.models.sfincspy.execution.swmm_coupling import SwmmSurfaceExchange  # noqa: E402


# ---------------------------------------------------------------------------
# A minimal, self-contained SWMM model - never the user's real project
# files. J1 (invert 1.000 m, full_depth 0.800 m -> rim at 1.800 m) drains
# through a small pipe (0.2 m diameter) to a free outfall. A constant
# baseline inflow is set per test to either stay well within capacity or
# overwhelm it and force surcharge.
# ---------------------------------------------------------------------------
def _write_test_swmm_inp(path: Path, baseline_inflow_lps: float) -> None:
    text = f"""[TITLE]
Minimal synthetic model for SwmmSurfaceExchange tests

[OPTIONS]
FLOW_UNITS           LPS
INFILTRATION         CURVE_NUMBER
FLOW_ROUTING         DYNWAVE
START_DATE           06/25/2025
START_TIME           00:00:00
REPORT_START_DATE    06/25/2025
REPORT_START_TIME    00:00:00
END_DATE             06/25/2025
END_TIME             01:00:00
REPORT_STEP          00:05:00
WET_STEP             00:05:00
DRY_STEP             00:05:00
ROUTING_STEP         15
ALLOW_PONDING        YES
INERTIAL_DAMPING     PARTIAL
NORMAL_FLOW_LIMITED  BOTH
MIN_SURFAREA         0
MAX_TRIALS           8
HEAD_TOLERANCE       0.0015

[JUNCTIONS]
;;Name           Elevation  MaxDepth   InitDepth  SurDepth   Aponded
J1               1.000      0.800      0.000      0.000      10.000

[OUTFALLS]
;;Name           Elevation  Type       Stage Data       Gated
OUT1             0.500      FREE                        NO

[CONDUITS]
;;Name           From Node        To Node          Length     Roughness  InOffset   OutOffset  InitFlow   MaxFlow
C1               J1               OUT1             50.000     0.013      0.000      0.000      0.000      0.000

[XSECTIONS]
;;Link           Shape        Geom1            Geom2      Geom3      Geom4      Barrels
C1               CIRCULAR     0.200            0          0          0          1

[INFLOWS]
;;Node   Parameter  TimeSeries  Type  Mfactor  Sfactor  Baseline  Pattern
J1       FLOW       ""          FLOW  1.0      1.0      {baseline_inflow_lps}

[REPORT]
INPUT      NO
CONTROLS   NO
SUBCATCHMENTS ALL
NODES ALL
LINKS ALL

[COORDINATES]
;;Node           X-Coord            Y-Coord
J1               0.000              0.000
OUT1             50.000             0.000

[TAGS]
"""
    path.write_text(text)


NODE_CONFIG = {"J1": {"area": 0.3, "length": 1.5}}


class TestSignAndUnitConventions:
    """These two facts were verified empirically against the real pyswmm/SWMM
    engine before writing SwmmSurfaceExchange (see module docstring) - these
    tests pin that behaviour down so a future pyswmm upgrade can't silently
    flip it under us."""

    def test_negative_generated_inflow_removes_water(self, tmp_path):
        inp = tmp_path / "model.inp"
        _write_test_swmm_inp(inp, baseline_inflow_lps=50)

        from pyswmm import Simulation, Nodes

        with Simulation(str(inp)) as sim:
            j1 = Nodes(sim)["J1"]
            depths = []
            for i, _ in enumerate(sim):
                if i > 10:
                    j1.generated_inflow(-1000)  # large extraction, LPS
                depths.append(j1.depth)
                if i > 30:
                    break

        assert depths[10] > 0  # it had accumulated water from the baseline inflow
        assert depths[-1] == pytest.approx(0.0, abs=1e-6)  # fully drained by extraction


class TestSwmmSurfaceExchange:
    def test_no_surcharge_gives_zero_exchange_throughout(self, tmp_path):
        # Baseline inflow well within the pipe's conveyance capacity.
        inp = tmp_path / "model.inp"
        _write_test_swmm_inp(inp, baseline_inflow_lps=5)

        coupler = SwmmSurfaceExchange(
            swmm_inp_path=str(inp),
            node_config=NODE_CONFIG,
            surface_level_fn=lambda node_id, t0, t1: 0.0,  # dry street throughout
            window_seconds=300,
        )
        result = coupler.run()

        assert set(result.columns) == {"J1", "J1_hgl"}
        assert (result["J1"] == 0.0).all()

    def test_surcharge_produces_positive_exchange_with_dry_surface(self, tmp_path):
        # Baseline inflow (50 LPS) overwhelms the small 0.2 m pipe -> J1
        # should rise above its rim (invert 1.0 + full_depth 0.8 = 1.8 m).
        inp = tmp_path / "model.inp"
        _write_test_swmm_inp(inp, baseline_inflow_lps=50)

        coupler = SwmmSurfaceExchange(
            swmm_inp_path=str(inp),
            node_config=NODE_CONFIG,
            surface_level_fn=lambda node_id, t0, t1: 0.0,  # dry street: weir regime
            window_seconds=300,
        )
        result = coupler.run()

        assert (result["J1"] > 0).any(), "expected at least one surcharge window"

    def test_feedback_actually_relieves_the_node(self, tmp_path):
        """Confirms the loop is genuinely bidirectional: applying the
        computed exchange back into SWMM (via generated_inflow) measurably
        caps J1's HGL closer to the rim, compared to running the identical
        surcharging model with no exchange applied at all."""
        inp_coupled = tmp_path / "model_coupled.inp"
        inp_uncoupled = tmp_path / "model_uncoupled.inp"
        _write_test_swmm_inp(inp_coupled, baseline_inflow_lps=50)
        _write_test_swmm_inp(inp_uncoupled, baseline_inflow_lps=50)

        from pyswmm import Simulation, Nodes

        coupler = SwmmSurfaceExchange(
            swmm_inp_path=str(inp_coupled),
            node_config=NODE_CONFIG,
            surface_level_fn=lambda node_id, t0, t1: 0.0,  # dry street: relief is possible
            window_seconds=300,
        )
        result = coupler.run()
        max_hgl_coupled = result["J1_hgl"].max()

        with Simulation(str(inp_uncoupled)) as sim:
            j1 = Nodes(sim)["J1"]
            max_hgl_uncoupled = 1.0  # J1's own invert elevation as a floor
            for _ in sim:
                max_hgl_uncoupled = max(max_hgl_uncoupled, j1.invert_elevation + j1.depth)

        assert result["J1"].max() > 0, "expected surcharge to actually occur"
        assert max_hgl_coupled < max_hgl_uncoupled, (
            "applying the exchange back into SWMM should relieve the node, "
            "keeping its HGL lower than the same model with no relief"
        )

    def test_returns_dataframe_indexed_by_window_end_time(self, tmp_path):
        inp = tmp_path / "model.inp"
        _write_test_swmm_inp(inp, baseline_inflow_lps=5)

        coupler = SwmmSurfaceExchange(
            swmm_inp_path=str(inp),
            node_config=NODE_CONFIG,
            surface_level_fn=lambda node_id, t0, t1: 0.0,
            window_seconds=300,
        )
        result = coupler.run()

        assert isinstance(result.index, pd.DatetimeIndex)
        # 1-hour simulation (00:00-01:00), 5-minute windows: pyswmm's step
        # loop does not yield a final callback exactly at END_TIME, so this
        # is 11 windows (00:05 through 00:55), not 12 - verified empirically.
        assert len(result) == 11
        assert result.index[0] == pd.Timestamp("2025-06-25 00:05:00")
        assert result.index[-1] == pd.Timestamp("2025-06-25 00:55:00")

    def test_reentry_regime_when_surface_is_already_flooded(self, tmp_path):
        # Small baseline inflow (node stays below rim throughout), but the
        # surface_level_fn simulates a street already flooded from an
        # earlier (unrelated) event - water should drain back into the sewer.
        inp = tmp_path / "model.inp"
        _write_test_swmm_inp(inp, baseline_inflow_lps=2)

        coupler = SwmmSurfaceExchange(
            swmm_inp_path=str(inp),
            node_config=NODE_CONFIG,
            surface_level_fn=lambda node_id, t0, t1: 2.0,  # well above rim (1.8 m)
            window_seconds=300,
        )
        result = coupler.run()

        assert (result["J1"] < 0).any(), "expected re-entry (negative exchange)"
