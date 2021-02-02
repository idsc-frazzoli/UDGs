from gokartmpcc.human_constraints import dynamics_HC
from dynamics.dynamics import kitt_dynamics
from vehicle import gokart_pool, KITT
from gokartmpcc.human_constraints.driver_config import behaviors_zoo


def test_dynamics():
    velx = 5
    vely = 0
    velrotz = 0
    beta = 0
    ab = 0
    tv = 0

    front_pacejka = gokart_pool[KITT].front_tires.pacejka
    rear_pacejka = gokart_pool[KITT].rear_tires.pacejka
    behavior = behaviors_zoo["medium"].config

    ACCX, ACCY, ACCROTZ = kitt_dynamics(
        VELX=velx,
        VELY=vely,
        VELROTZ=velrotz,
        BETA=beta,
        AB=ab,
        TV=tv,
        B1=front_pacejka.B,
        C1=front_pacejka.C,
        D1=front_pacejka.D,
        B2=rear_pacejka.B,
        C2=rear_pacejka.C,
        D2=rear_pacejka.D,
        Ic=behavior.specificmoi,
    )

    print(ACCX, ACCY, ACCROTZ)
