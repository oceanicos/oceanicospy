import numpy as np
import pytest

from oceanicospy.utils import constants
from oceanicospy.utils.wave_props import angulo_norte, direction, wavelength


class TestWavelength:
    def test_deep_water_limit(self):
        """In very deep water L ≈ g·T²/(2π)."""
        T, h = 10.0, 1000.0
        L_deep = constants.GRAVITY * T**2 / (2 * np.pi)
        assert abs(wavelength(T, h) - L_deep) / L_deep < 0.01

    def test_shallow_water_limit(self):
        """In very shallow water L ≈ T·√(g·h)."""
        T, h = 200.0, 0.5
        L_shallow = T * np.sqrt(constants.GRAVITY * h)
        assert abs(wavelength(T, h) - L_shallow) / L_shallow < 0.05

    def test_positive(self):
        assert wavelength(10.0, 10.0) > 0

    def test_longer_period_longer_wavelength(self):
        assert wavelength(15.0, 20.0) > wavelength(8.0, 20.0)

    def test_deeper_water_longer_wavelength(self):
        assert wavelength(10.0, 100.0) > wavelength(10.0, 5.0)


class TestDirection:
    def test_north(self):
        assert direction(1.0, 0.0) == pytest.approx(0.0, abs=1e-9)

    def test_east(self):
        assert direction(0.0, 1.0) == pytest.approx(90.0, abs=1e-9)

    def test_west(self):
        assert direction(0.0, -1.0) == pytest.approx(-90.0, abs=1e-9)

    def test_south(self):
        d = direction(-1.0, 0.0)
        assert abs(d) == pytest.approx(180.0, abs=1e-9)

    def test_northeast_45(self):
        assert direction(1.0, 1.0) == pytest.approx(45.0, abs=1e-9)

    def test_southwest(self):
        d = direction(-1.0, -1.0)
        assert d == pytest.approx(-135.0, abs=1e-9)

    def test_range(self):
        d = direction(3.0, 2.0)
        assert -180.0 <= d <= 180.0


class TestAnguloNorte:
    def test_q1_x_pos_y_pos(self):
        """x>0, y>0 → angle in [0, 90]."""
        theta = angulo_norte(1, 1)
        assert 0 <= theta <= 90

    def test_q2_x_neg_y_pos(self):
        """x<0, y>0 → angle in [270, 360]."""
        theta = angulo_norte(-1, 1)
        assert 270 <= theta <= 360

    def test_q3_x_neg_y_neg(self):
        """x<0, y<0 → angle in [180, 270]."""
        theta = angulo_norte(-1, -1)
        assert 180 <= theta <= 270

    def test_q4_x_pos_y_neg(self):
        """x>0, y<0 → angle in [90, 180]."""
        theta = angulo_norte(1, -1)
        assert 90 <= theta <= 180

    def test_pure_east(self):
        """x>0, y=0 → 90 degrees."""
        theta = angulo_norte(1, 0)
        assert theta == pytest.approx(90.0, abs=1e-9)

    def test_q1_45_degrees(self):
        """x=y>0 → 45 degrees from north."""
        theta = angulo_norte(1, 1)
        assert theta == pytest.approx(45.0, abs=1e-9)
