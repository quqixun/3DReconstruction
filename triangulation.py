# Created by Qixun Qu
# quqixun@gmail.com
# 2017/04/29
#


import numpy as np


class Triangulation():

    def __init__(self):
        return

    def triangulation_test_case(self, sigma):
        '''
        '''

        U_true = np.random.rand(3, 1)
        C1 = np.random.rand(3, 1) + np.array([0, 0, -4]).T
        C2 = np.random.rand(3, 1) + np.array([0, 0, -4]).T

        K = np.diag(np.array([2000, 2000, 1]))

        def small_rotation(max_angle):
            axis = np.random.randn(3, 1)
            axis = axis / np.linalg.norm(axis)
            angle = max_angle * np.random.rand(1)
            s = axis * angle
            R = np.exp(np.array([[0, -s[2], s[1]],
                                 [s[2], 0 - s[0]],
                                 [-s[1], s[0], 0]]))
            return R

        return U_true