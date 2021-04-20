from typing import Dict

from udgs.vehicle.structures import (
    CarGeometry,
    WheelGeometry,
    PlayerName,
    CarParams,
)

__all__ = ["VEHICLE1", "vehicles_pool"]

vehicles_pool: Dict[PlayerName, CarParams] = {}


car_geometry = CarGeometry(l=1.19, l1=0.73, w1=.94, w2=1.07)
car_front_wheel_geometry = WheelGeometry(radius=0.2, width=0.125)  # todo random values to be fixed
car_rear_wheel_geometry = WheelGeometry(radius=0.2, width=0.16)  # todo random values to be fixed

# this is a bit unsafe as all the cars will refer to the same objects in the fields
green_car = CarParams(
    geometry=car_geometry,
    front_tires=car_front_wheel_geometry,
    rear_tires=car_rear_wheel_geometry,
    color="green"
)
blu_car = CarParams(
    geometry=car_geometry,
    front_tires=car_front_wheel_geometry,
    rear_tires=car_rear_wheel_geometry,
    color="blue"
)
red_car = CarParams(
    geometry=car_geometry,
    front_tires=car_front_wheel_geometry,
    rear_tires=car_rear_wheel_geometry,
    color="red"
)
black_car = CarParams(
    geometry=car_geometry,
    front_tires=car_front_wheel_geometry,
    rear_tires=car_rear_wheel_geometry,
    color="black"
)
VEHICLE1 = PlayerName("Vehicle1")


vehicles_pool.update({VEHICLE1: green_car})
