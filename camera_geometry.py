# Created by Qixun Qu
# quqixun@gmail.com
# 2017/04/29
#


# This script is used to test functions
# in class Triangulation.


import os
import numpy as np
from scipy import io as sio
from triangulation import Triangulation


# Create an instance
tgl = Triangulation()

'''
Test Minimal Solver
'''

# Generate a test case of triangulation
sigma = 0
Ps, us, U_true = tgl.triangulation_test_case(sigma)

# Run the minimal solver to obtain triangulated
# 3D points
U = tgl.minimal_triangulation(Ps, us)

# Display the true points and triangulated points,
# they should be identical when sigma equals to 0
# print("Estimated U:\n", U)
# print("Known U:\n", U_true)

# Check the sign of depths
positive = tgl.check_depth(Ps, U)
# print(positive)

# Calculate the errors between known image points
# and reprojected points
errors = tgl.reprojection_errors(Ps, us, U)
# print(errors)

'''
Refine Triangulation by Gauss-Newton Method
'''

# Load data
Uhat = []

# Extract one example to test the refine function
idx = 1

# Combine two image points into one matrix
us = []

# Combine two camera matrix into one array
Ps = []

# Preceed refination
U = tgl.refine_triangulation(Ps, us, Uhat)

# Plot original 3D points
view = []
tgl.plot_in_views(Uhat, view)

# Plot refined 3D points
view = []
tgl.plot_in_views(U, view)
