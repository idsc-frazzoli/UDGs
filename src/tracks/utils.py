# from geometry import SE2
import numpy as np
from bspline import *

from tracks import SplineTrack


def spline_progress_from_pose(spline_track: SplineTrack, pose: np.ndarray) -> float:
    # parametrize curve
    n_t = 500
    t = np.linspace(0, len(spline_track.x), n_t, endpoint=True)
    xxyy = []
    for t_val in t:
        xxyy.append(casadiDynamicBSPLINE(t_val, spline_track.as_np_array()[:, 0:2]))
    xxyy = np.asarray(xxyy)

    # find minimum
    eucl_dist = np.linalg.norm(pose[0:2] - xxyy, ord=2, axis=1)
    min_dist = np.amin(eucl_dist)
    min_idx = np.where(eucl_dist == min_dist)
    progress = t[min_idx]
    dx, dy = casadiDynamicBSPLINEforward(progress, spline_track.as_np_array()[:, 0:2])
    progress_ang = float(atan2(dy, dx))
    assert abs(progress_ang - pose[-1]) < np.pi / 2
    r = float(casadiDynamicBSPLINERadius([progress], spline_track.as_np_array()[:, -1:np.newaxis]))
    assert min_dist < r
    return progress
