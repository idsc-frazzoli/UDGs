import plotly.graph_objects as go

from dynamics.dynamics_utils import *
from vehicle import gokart_pool, KITT


def test_dynamics_utils():
    pacejka = gokart_pool[KITT].rear_tires.pacejka
    D2 = pacejka.D
    C2 = pacejka.C
    B2 = pacejka.B

    velx = 5
    vely = 0
    taccx = 0
    reg = 0.5

    slip = simpleslip(VELY=vely, VELX=velx, taccx=taccx, reg=reg, D2=D2)
    # print(slip)

    accy = simpleaccy(VELY=vely, VELX=velx, taccx=taccx, reg=reg, B2=B2, C2=C2, D2=D2)
    # print(accy)


def test_ackerman_map():
    """ beta is the steering angle of the driving wheel, delta the wheels' angle"""
    beta = np.linspace(90, -90, 100, endpoint=True) * np.pi / 180
    delta = ackermann_map(beta)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=beta, y=delta,
                             mode='lines+markers', name="steering map"))
    fig.show()
    fig.write_html("test.html")
