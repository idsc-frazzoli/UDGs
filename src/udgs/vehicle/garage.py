from typing import Dict

from udgs.vehicle.structures import (
    CarGeometry,
    WheelGeometry,
    PlayerName,
    CarParams,
)

__all__ = ["VEHICLE1", "vehicles_pool"]

vehicles_pool: Dict[PlayerName, CarParams] = {}

VEHICLE1 = PlayerName("Vehicle1")

car_geometry = CarGeometry(l=1.19, l1=0.73, w1=.94, w2=1.07)

car_front_wheel_geometry = WheelGeometry(radius=0.2, width=0.125)  # todo random values to be fixed
car_rear_wheel_geometry = WheelGeometry(radius=0.2, width=0.16)  # todo random values to be fixed

car = CarParams(
    geometry=car_geometry,
    front_tires=car_front_wheel_geometry,
    rear_tires=car_rear_wheel_geometry
)

vehicles_pool.update({VEHICLE1: car})
