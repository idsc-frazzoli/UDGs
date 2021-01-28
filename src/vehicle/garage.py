from typing import Dict

from vehicle.structures import (
    GokartGeometry,
    SteeringColumn,
    Pacejka,
    WheelGeometry,
    GokartName,
    GokartParams,
    Tire,
)

__all__ = ["KITT", "gokart_pool"]

gokart_pool: Dict[GokartName, GokartParams] = {}

KITT = GokartName("Kitt")

kitt_geometry = GokartGeometry(l=1.19, l1=0.73, w1=.94, w2=1.07)

kitt_steering_column = SteeringColumn(J_steer=0.8875, b_steer=0.1625, k_steer=0.0125)

kitt_front_pacejka = Pacejka(B=9, C=1, D=10)
kitt_rear_pacejka = Pacejka(B=5.2, C=1.1, D=10)

kitt_front_wheel_geometry = WheelGeometry(radius=0.2, width=0.125)  # todo random values to be fixed
kitt_rear_wheel_geometry = WheelGeometry(radius=0.2, width=0.16)  # todo random values to be fixed

kitt = GokartParams(
    geometry=kitt_geometry,
    steering=kitt_steering_column,
    front_tires=Tire(pacejka=kitt_front_pacejka, geometry=kitt_front_wheel_geometry),
    rear_tires=Tire(pacejka=kitt_rear_pacejka, geometry=kitt_rear_wheel_geometry)
)

gokart_pool.update({KITT: kitt})
