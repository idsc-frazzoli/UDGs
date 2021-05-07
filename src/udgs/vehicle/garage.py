from typing import Dict, NewType

from udgs.vehicle.structures import (
    CarGeometry,
    WheelGeometry,
    VehicleParams,
)

__all__ = ["VehicleType", "CAR", "vehicles_pool"]

car_geometry = CarGeometry(l=1.19, l1=0.73, w1=.94, w2=1.07)
car_front_wheel_geometry = WheelGeometry(radius=0.2, width=0.125)  # todo random values to be fixed
car_rear_wheel_geometry = WheelGeometry(radius=0.2, width=0.16)  # todo random values to be fixed

# this is a bit unsafe as all the cars will refer to the same objects in the fields
car = VehicleParams(
    geometry=car_geometry,
    front_tires=car_front_wheel_geometry,
    rear_tires=car_rear_wheel_geometry,
)

VehicleType = NewType("VehicleType", str)
CAR = VehicleType("car")
vehicles_pool: Dict[VehicleType, VehicleParams] = {}

vehicles_pool.update({CAR: car})
