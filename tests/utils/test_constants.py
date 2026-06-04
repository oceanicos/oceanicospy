from oceanicospy.utils import constants


def test_gravity():
    assert constants.GRAVITY == 9.8


def test_water_density():
    assert constants.WATER_DENSITY == 1028.1


def test_atm_pressure():
    assert constants.ATM_PRESSURE_BAR == 1.01325


def test_constants_are_positive():
    assert constants.GRAVITY > 0
    assert constants.WATER_DENSITY > 0
    assert constants.ATM_PRESSURE_BAR > 0
