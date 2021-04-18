import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from map.zoo import winti_001


def test_bspline():
    points = winti_001.spline.as_np_array()
    degree = 2

    tck, _ = interpolate.splprep(points.T, k=degree, per=1)
    u = np.linspace(0, 1, 1000, endpoint=True)
    center_x, center_y, width = interpolate.splev(u, tck)
    centerline = np.column_stack([center_x, center_y])
    dx, dy, _ = interpolate.splev(u, tck, der=1, ext=0)
    normal = np.column_stack([-dy, dx])
    normal_unit = normal / np.linalg.norm(normal, axis=1)[:, np.newaxis]

    left_bound = centerline - width[:, np.newaxis] * normal_unit
    right_bound = centerline + width[:, np.newaxis] * normal_unit

    fig, ax = plt.subplots()
    rieter_x, rieter_y = 70, 60
    img = plt.imread("rieter.png")
    ax.imshow(img, cmap=plt.get_cmap("gray"), extent=[0, rieter_x, 0, rieter_y])

    ax.get_kart_plot(centerline[:, 0], centerline[:, 1], ":b", linewidth=1.0, label="Centerline")
    ax.get_kart_plot(left_bound[:, 0], left_bound[:, 1], "-.r", linewidth=1.0, label="Left bound")
    ax.get_kart_plot(right_bound[:, 0], right_bound[:, 1], "--r", linewidth=1.0, label="Right bound")

    plt.legend(loc="best")
    plt.title(f"B-spline curve degree {degree}")
    plt.savefig("test.png")
    # it must be improved https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.BSpline.html#scipy.interpolate.BSpline
