# Created by Qixun Qu
# quqixun@gmail.com
# 2017/04/29
#


import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class Triangulation():

    def __init__(self):
        return

    def triangulation_test_case(self, sigma):
        '''
        '''

        def small_rotation(max_angle):
            axis = np.random.randn(3, 1)
            axis = axis / np.linalg.norm(axis)
            angle = max_angle * np.random.rand(1)
            s = axis * angle
            R = np.exp(np.array([[0, -s[2], s[1]],
                                 [s[2], 0, -s[0]],
                                 [-s[1], s[0], 0]]))
            return R

        U_true = np.random.rand(3, 1)
        C1 = np.random.rand(3, 1) + np.array([[0], [0], [-4]])
        C2 = np.random.rand(3, 1) + np.array([[0], [0], [-4]])

        temp1 = np.hstack((np.eye(3), -C1))
        temp2 = np.hstack((np.eye(3), -C2))

        K = np.diag(np.array([1000, 1000, 1]))

        Ps = np.zeros((1, 2, 3, 4))
        Ps[0, 0, :, :] = np.dot(np.dot(K, small_rotation(0.1)), temp1)
        Ps[0, 1, :, :] = np.dot(np.dot(K, small_rotation(0.1)), temp2)

        us = np.zeros((3, 2))
        us[:, 0] = np.dot(Ps[0, 0, :, :], np.vstack((U_true, 1))).flatten()
        us[:, 1] = np.dot(Ps[0, 1, :, :], np.vstack((U_true, 1))).flatten()

        us[:, 0] = us[:, 0] / us[2, 0] + sigma * np.random.randn(1, 3)
        us[:, 1] = us[:, 1] / us[2, 1] + sigma * np.random.randn(1, 3)

        return Ps, us[0:2, :], U_true

    def minimal_triangulation(self, Ps, us):
        '''
        '''

        M = np.zeros((5, 5))
        M[0:3, 0:3] = Ps[0, 0, 0:3, 0:3]
        M[3:5, 0:3] = Ps[0, 1, 0:2, 0:3]
        M[0:2, 3] = -us[:, 0]
        M[3:5, 4] = -us[:, 1]
        M[2, 3] = -1

        b = -np.append(Ps[0, 0, :, 3],
                       Ps[0, 1, 0:2, 3]).reshape((-1, 1))

        try:
            theta = np.linalg.lstsq(M, b)[0]
            U = theta[0:3]
        except np.linalg.LinAlgError:
            U = None

        return U

    def check_depth(self, Ps, U):
        '''
        '''

        pts_num = Ps.shape[1]
        U = np.append(U, 1).reshape((-1, 1))

        positive = np.zeros((pts_num, 1))
        for i in range(pts_num):
            _ = np.dot(Ps[0, i, 2, :], U)
            if _ > 0 and (not np.isnan(_)):
                positive[i] = 1

        return positive

    def reprojection_errors(self, Ps, us, U):
        '''
        '''

        positive = self.check_depth(Ps, U)

        pts_num = Ps.shape[1]
        U = np.append(U, 1).reshape((-1, 1))

        errors = np.zeros((pts_num, 1))

        for i in range(pts_num):
            if positive[i] == 0:
                errors[i] = np.inf
            else:
                x_hat = np.dot(Ps[0, i, 0, :], U) / np.dot(Ps[0, i, 2, :], U)
                y_hat = np.dot(Ps[0, i, 1, :], U) / np.dot(Ps[0, i, 2, :], U)
                diff = np.array([x_hat, y_hat]) - us[:, i].reshape((-1, 1))

                errors[i] = np.sqrt(np.sum(np.power(diff, 2)))

        return errors

    def ransac_triangulation(self, Ps, us, threshold=5):
        '''
        '''

        pts_num = Ps.shape[1]

        U = np.zeros((3, 1))
        nbr_inliers = 0

        for i in range(100):
            idx = np.random.randint(pts_num, size=2)
            Ps_pt = Ps[:, idx, :, :]
            us_pt = us[:, idx]

            U_temp = self.minimal_triangulation(Ps_pt, us_pt)
            if np.sum(np.isnan(U_temp) * 1.0) > 0:
                continue

            errors = self.reprojection_errors(Ps, us, U_temp)

            nbr_inliers_temp = len(np.where(errors < threshold)[0])
            if nbr_inliers_temp > nbr_inliers:
                nbr_inliers = nbr_inliers_temp
                U = U_temp

        return U, nbr_inliers

    def reshape_Ps(self, Ps):
        '''
        '''

        Psr = np.zeros((1, Ps.shape[1], 3, 4))

        for i in range(Ps.shape[1]):
            Psr[0, i, :, :] = Ps[0, i]

        return Psr

    def clean_for_plot(self, Us):
        '''
        '''

        minvals = np.percentile(Us, 1, axis=1)
        maxvals = np.percentile(Us, 99, axis=1)

        removed_idx = np.array([]).reshape((1, -1))
        for i in range(Us.shape[0]):
            removed_idx = \
                np.append(removed_idx, np.where(
                          np.logical_or(Us[i, :] > maxvals[i],
                                        Us[i, :] < minvals[i]))[0])

        removed_idx = np.unique(removed_idx.astype(int))

        Us = np.delete(Us, removed_idx, 1)

        return Us, removed_idx

    def plot_in_views(self, Us, views):
        '''
        '''

        Us, _ = self.clean_for_plot(Us)

        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(Us[0, :], Us[1, :], Us[2, :], s=10, c=[0, 0, 0])
        ax.view_init(elev=0, azim=90)
        ax.grid(b=False)
        ax.axis('off')
        plt.show()

        return

    def refine_triangulation(self, Ps, us, Uhat):
        return
