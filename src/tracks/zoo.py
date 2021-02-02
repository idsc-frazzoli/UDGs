import os

from tracks.structures import Track, SplineTrack
from matplotlib import pyplot as plt

path = os.path.dirname(__file__)

winti_001 = Track(
    desc="old by Thomas",
    spline=SplineTrack(
        x=[
            29,
            37,
            45,
            52,
            57,
            61,
            61,
            57,
            50,
            48,
            45,
            46,
            46,
            44,
            40,
            38.2,
            38.2,
            36.4,
            34.2,
            32.2,
            31,
            28,
            25,
            22,
            22,
            26,
            27,
            25,
            21,
            23,
        ],
        y=[
            25,
            23,
            28,
            25,
            25,
            27,
            34,
            36,
            36,
            40,
            46,
            50,
            52,
            54,
            54,
            52,
            49,
            47,
            47,
            49,
            53,
            55,
            55,
            52,
            48,
            44,
            40,
            37,
            32,
            26,
        ],
        radius=[
            3,
            2,
            2,
            3,
            3,
            4,
            4,
            2.6,
            2.4,
            1.7,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3.1,
            3.1,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            2,
            3,
            3,
            3,
        ],
    ),
    background=plt.imread(os.path.join(path, "rieter.png")),
    scale_factor=1/7.5
)

winti_002 = Track(
    desc="Currently in used as of 15.01.2021",
    spline=SplineTrack(
        x=[31.3982,
           34.3467,
           38.8834,
           42.4171,
           46.6534,
           50.0885,
           54.9394,
           56.9185,
           62.5278,
           62.8388,
           60.8294,
           59.4633,
           62.6655,
           63.019,
           60.5657,
           54.793,
           51.622,
           47.5347,
           42.8585,
           42.0865,
           43.7897,
           46.0411,
           47.2262,
           45.6159,
           43.2543,
           40.1219,
           35.8531,
           32.0265,
           28.1481,
           24.1587,
           20.1981,
           15.8017,
           18.7318,
           24.4109,
           27.5437,
           ],
        y=[19.5474,
           18.5505,
           19.1879,
           18.9213,
           19.0295,
           18.9978,
           18.9728,
           19.1531,
           19.6681,
           25.7549,
           28.7396,
           33.4067,
           36.666,
           41.2987,
           45.9278,
           45.7833,
           45.7112,
           45.7716,
           44.937,
           39.7661,
           36.4969,
           35.0977,
           31.1316,
           27.1605,
           24.7234,
           24.7591,
           24.3245,
           24.35,
           24.6296,
           24.4962,
           26.5447,
           23.5585,
           18.0608,
           19.0106,
           18.9692
           ],
        radius=[
            1.6359 / 2,
            3.69128 / 2,
            2.44739 / 2,
            2.94989 / 2,
            2.77792 / 2,
            2.74895 / 2,
            3.00581 / 2,
            2.6302 / 2,
            6.14284 / 2,
            4.27967 / 2,
            2.65155 / 2,
            1.95816 / 2,
            5.2424 / 2,
            5.7855 / 2,
            5.42435 / 2,
            4.06841 / 2,
            4.42409 / 2,
            4.08805 / 2,
            3.82191 / 2,
            1.83155 / 2,
            1.27002 / 2,
            2.16762 / 2,
            2.0162 / 2,
            2.77608 / 2,
            4.47224 / 2,
            2.59586 / 2,
            2.46631 / 2,
            2.28039 / 2,
            2.74581 / 2,
            3.00383 / 2,
            6.1274 / 2,
            4.79043 / 2,
            3.00692 / 2,
            3.39657 / 2,
            2.686 / 2
        ],
    ),
    background=plt.imread(os.path.join(path, "rieter.png")),
    scale_factor=1/7.5
)


straightLineR2L = Track(
    desc="Potential Game Scenario",
    spline=SplineTrack(
        x=[75,
           70,
           65,
           60,
           55,
           50,
           45,
           40,
           35,
           30,
           25,
           20,
           15,
           10,
           5,
           0,
           -5,
           -10,
           -15,
           -20
           ],
        y=[50,
           50,
           50,
           50,
           50,
           50,
           50,
           50,
           50,
           50,
           50,
           50,
           50,
           50,
           50,
           50,
           50,
           50,
           50,
           50
           ],
        radius=[
            3.5,
            3.5,
            3.5,
            3.5,
            3.5,
            3.5,
            3.5,
            3.5,
            3.5,
            3.5,
            3.5,
            3.5,
            3.5,
            3.5,
            3.5,
            3.5,
            3.5,
            3.5,
            3.5,
            3.5
        ],
    ),
    background=plt.imread(os.path.join(path, "road06.png")),
    scale_factor=1/7.5
)