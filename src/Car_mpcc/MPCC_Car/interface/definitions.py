import numpy
import ctypes

name = "MPCCgokart"
requires_callback = True
lib = "lib/MPCCgokart.dll"
lib_static = "lib/MPCCgokart_static.lib"
c_header = "include/MPCCgokart.h"

# Parameter             | Type    | Scalar type      | Ctypes type    | Numpy type   | Shape     | Len
params = \
[("xinit"               , "dense" , ""               , ctypes.c_double, numpy.float64, (  8,   1),    8),
 ("x0"                  , "dense" , ""               , ctypes.c_double, numpy.float64, (372,   1),  372),
 ("all_parameters"      , "dense" , ""               , ctypes.c_double, numpy.float64, (2015,   1), 2015)]

# Output                | Type    | Scalar type      | Ctypes type    | Numpy type   | Shape     | Len
outputs = \
[("all_var"             , ""      , ""               , ctypes.c_double, numpy.float64,     ( 12,),  372)]

# Info Struct Fields
info = \
[('it', ctypes.c_int),
('it2opt', ctypes.c_int),
('res_eq', ctypes.c_double),
('res_ineq', ctypes.c_double),
('rsnorm', ctypes.c_double),
('rcompnorm', ctypes.c_double),
('pobj', ctypes.c_double),
('dobj', ctypes.c_double),
('dgap', ctypes.c_double),
('rdgap', ctypes.c_double),
('mu', ctypes.c_double),
('mu_aff', ctypes.c_double),
('sigma', ctypes.c_double),
('lsit_aff', ctypes.c_int),
('lsit_cc', ctypes.c_int),
('step_aff', ctypes.c_double),
('step_cc', ctypes.c_double),
('solvetime', ctypes.c_double),
('fevalstime', ctypes.c_double)
]