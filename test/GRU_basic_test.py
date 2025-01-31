import unittest
from unittest import TestCase
import numpy as np
from gru import GRU
from runner import Runner

class Test(TestCase):
    def __init__(self, *args, **kwargs):
        super(Test, self).__init__(*args, **kwargs)
        vocabsize = 3
        hdim = 2
        self.r = GRU(vocabsize, hdim, vocabsize)
        self.r.Vh[0][0] = 0.7
        self.r.Vh[0][1] = 0.3
        self.r.Vh[0][2] = 0.4
        self.r.Vh[1][0] = 0.6
        self.r.Vh[1][1] = 0.9
        self.r.Vh[1][2] = 0.7

        self.r.Uh[0][0] = 0.6
        self.r.Uh[0][1] = 0.4
        self.r.Uh[1][0] = 0.3
        self.r.Uh[1][1] = 0.8

        self.r.Vr[0][0] = 0.2
        self.r.Vr[0][1] = 0.7
        self.r.Vr[0][2] = 0.1
        self.r.Vr[1][0] = 0.9
        self.r.Vr[1][1] = 0.6
        self.r.Vr[1][2] = 0.5

        self.r.Ur[0][0] = 0.1
        self.r.Ur[0][1] = 0.9
        self.r.Ur[1][0] = 0.4
        self.r.Ur[1][1] = 0.6

        self.r.Vz[0][0] = 0.6
        self.r.Vz[0][1] = 0.8
        self.r.Vz[0][2] = 0.9
        self.r.Vz[1][0] = 0.3
        self.r.Vz[1][1] = 0.2
        self.r.Vz[1][2] = 0.7

        self.r.Uz[0][0] = 0.9
        self.r.Uz[0][1] = 0.5
        self.r.Uz[1][0] = 0.9
        self.r.Uz[1][1] = 0.3

        self.r.W[0][0] = 0.6
        self.r.W[0][1] = 0.5
        self.r.W[1][0] = 0.2
        self.r.W[1][1] = 0.6
        self.r.W[2][0] = 0.4
        self.r.W[2][1] = 0.2

        self.p = Runner(self.r)

        self.x = np.array([0, 1, 2, 1, 1, 0, 2])
        self.y, self.s = self.r.predict(self.x)
        self.d_np = np.array([0])

    def test_predicting_y(self):
        y_exp = np.array([[0.3528942, 0.33141165, 0.31569415], [0.36113545, 0.33936667, 0.29949788],
                          [0.36692804, 0.33951603, 0.29355593], [0.37231812, 0.3423382, 0.28534368],
                          [0.37673929, 0.34350021, 0.2797605], [0.38331333, 0.33917522, 0.27751145],
                          [0.38581534, 0.33818498, 0.27599968]])
        s_exp = np.array(
            [[0.21415391, 0.22854546], [0.26690114, 0.44588503], [0.32573672, 0.52650084], [0.36993061, 0.64022728],
             [0.41051185, 0.71839057], [0.49285591, 0.74806453], [0.52161192, 0.76878925], [0., 0.]])

        self.assertTrue(np.isclose(y_exp, self.y, rtol=1e-08, atol=1e-08).all())
        self.assertTrue(np.isclose(s_exp, self.s, rtol=1e-08, atol=1e-08).all())

    def test_binary_prediction_GRU_with_3_steps(self):
        deltaUr_3_exp_np = [[0.00193795, 0.00314452], [0.00195132, 0.00320871]]
        deltaVr_3_exp_np = [[0.00103786, 0.00190269, 0.00172727], [0.00109623, 0.0025203, 0.00126202]]
        deltaUz_3_exp_np = [[-0.00942043, -0.01576331], [-0.00216407, -0.00358799]]
        deltaVz_3_exp_np = [[-0.01069625, -0.00785168, -0.00464374], [-0.00094875, -0.00397553, -0.00080911]]
        deltaUh_3_exp_np = [[0.01726854, 0.02951899], [0.00359722, 0.00635103]]
        deltaVh_3_exp_np = [[0.01217558, 0.03100759, 0.01551138], [0.00432377, 0.00515261, 0.00275538]]
        deltaW_GRU_3_exp_np = [[0.32036604, 0.47217857], [-0.17640132, -0.25999298], [-0.14396472, -0.21218559]]

        self.r.deltaUr.fill(0)
        self.r.deltaVr.fill(0)
        self.r.deltaUz.fill(0)
        self.r.deltaVz.fill(0)
        self.r.deltaUh.fill(0)
        self.r.deltaVh.fill(0)
        self.r.deltaW.fill(0)

        self.r.acc_deltas_bptt_np(self.x, self.d_np, self.y, self.s, 3)

        self.assertTrue(np.isclose(deltaUr_3_exp_np, self.r.deltaUr).all())
        self.assertTrue(np.isclose(deltaVr_3_exp_np, self.r.deltaVr).all())
        self.assertTrue(np.isclose(deltaUz_3_exp_np, self.r.deltaUz).all())
        self.assertTrue(np.isclose(deltaVz_3_exp_np, self.r.deltaVz).all())
        self.assertTrue(np.isclose(deltaUh_3_exp_np, self.r.deltaUh).all())
        self.assertTrue(np.isclose(deltaVh_3_exp_np, self.r.deltaVh).all())
        self.assertTrue(np.isclose(deltaW_GRU_3_exp_np, self.r.deltaW).all())
