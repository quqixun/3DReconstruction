# Created by Qixun Qu
# quqixun@gmail.com
# 2017/04/29
#


# This script is used to test functions
# in class Triangulation.


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
data = sio.loadmat('data/gauss_newton.mat')
u = data['u']
u_tilde = data['u_tilde']
P = data['P']
P_tilde = data['P_tilde']
Uhat = data['Uhat']

# Obtain the number of 3D points
pts_num = Uhat.shape[1]

# Combine two camera matrix into one array
Ps = np.array([[P, P_tilde]])

# Initialize the matrix to store refine points
U = np.zeros((3, pts_num))

for i in range(pts_num):
    # Combine two image points into one matrix
    us = np.hstack((u[:, i:i + 1], u_tilde[:, i:i + 1]))

    # Preceed refination
    U[:, i:i + 1] = tgl.refine_triangulation(Ps, us, Uhat[:, i:i + 1])

# Plot original 3D points
tgl.plot_in_views(Uhat, [6, -92], 2, [6, 2])

# Plot refined 3D points
tgl.plot_in_views(U, [6, -92], 2, [6, 2])
