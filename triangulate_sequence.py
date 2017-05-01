# Created by Qixun Qu
# quqixun@gmail.com
# 2017/04/30
#


import timeit
import numpy as np
import scipy.io as sio
from triangulation import Triangulation


start = timeit.default_timer()

# Load Data
struct = sio.loadmat('data/sequence.mat')
data = struct['triangulation_examples']

pts_num = data.shape[1]

Us = np.zeros((3, pts_num))
nbr_inliers = np.zeros((pts_num, 1))

tgl = Triangulation()

for i in range(pts_num):

    usi = data[0, i][0]
    Psi = tgl.reshape_Ps(data[0, i][1])

    Us[:, i:i + 1], nbr_inliers[i] = \
        tgl.ransac_triangulation(Psi, usi)

    if np.mod(i, 100) == 0:
        print("Step {}".format(i))

Us = Us[:, np.where(nbr_inliers >= 2)[0]]

seconds = timeit.default_timer() - start
print("{} minutes {} seconds".format(np.floor(seconds / 60),
                                     np.mod(seconds, 60)))

tgl.plot_in_views(Us, [-170, 5])
