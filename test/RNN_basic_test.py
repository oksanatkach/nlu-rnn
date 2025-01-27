import unittest
from unittest import TestCase
import numpy as np
from rnn import RNN
from runner import Runner

class Test(TestCase):
    def __init__(self, *args, **kwargs):
        super(Test, self).__init__(*args, **kwargs)
        vocabsize = 3
        hdim = 2
        # RNN with vocab size 3 and 2 hidden layers
        # Note that, for the binary prediction output vocab size should be 2
        # for test case simplicity, here we will use the same input and vocab size
        self.r = RNN(vocabsize, hdim, vocabsize)
        self.r.V[0][0] = 0.7
        self.r.V[0][1] = 0.3
        self.r.V[0][2] = 0.4
        self.r.V[1][0] = 0.6
        self.r.V[1][1] = 0.9
        self.r.V[1][2] = 0.7

        self.r.W[0][0] = 0.6
        self.r.W[0][1] = 0.5
        self.r.W[1][0] = 0.2
        self.r.W[1][1] = 0.6
        self.r.W[2][0] = 0.4
        self.r.W[2][1] = 0.2

        self.r.U[0][0] = 0.9
        self.r.U[0][1] = 0.8
        self.r.U[1][0] = 0.5
        self.r.U[1][1] = 0.3

        self.p = Runner(self.r)

        self.x = np.array([0, 1, 2, 1, 1, 0, 2])
        self.y, self.s = self.r.predict(self.x)
        self.d = np.array([1, 2, 1, 1, 1, 1, 1])
        self.d_np = np.array([0])

    def test_predicting_y(self):
        y_exp = np.array([[0.39411072, 0.32179748, 0.2840918],
                          [0.4075143, 0.32013043, 0.27235527],
                          [0.41091755, 0.31606385, 0.2730186],
                          [0.41098376, 0.31825833, 0.27075792],
                          [0.41118931, 0.31812307, 0.27068762],
                          [0.41356637, 0.31280332, 0.27363031],
                          [0.41157736, 0.31584609, 0.27257655]])
        s_exp = np.array([[0.66818777, 0.64565631],
                          [0.80500806, 0.80655686],
                          [0.85442692, 0.79322425],
                          [0.84599959, 0.8270955],
                          [0.84852462, 0.82794442],
                          [0.89340731, 0.7811953],
                          [0.86164528, 0.79916155],
                          [0., 0.]])

        self.assertTrue(np.isclose(y_exp, self.y, rtol=1e-08, atol=1e-08).all())
        self.assertTrue(np.isclose(s_exp, self.s, rtol=1e-08, atol=1e-08).all())

    def test_computing_loss_and_mean_loss(self):
        x2 = np.array([1, 1, 0])
        d2 = np.array([1, 0, 2])
        x3 = np.array([1, 1, 2, 1, 2])
        d3 = np.array([1, 2, 1, 2, 1])

        loss_expected = 8.19118156763
        loss2_expected = 3.29724981191
        loss3_expected = 6.01420605985
        mean_loss_expected = 1.16684249596

        loss = self.p.compute_loss(self.x, self.d)
        loss2 = self.p.compute_loss(x2, d2)
        loss3 = self.p.compute_loss(x3, d3)
        mean_loss = self.p.compute_mean_loss([self.x, x2, x3], [self.d, d2, d3])

        self.assertTrue(np.isclose(loss_expected, loss, rtol=1e-08, atol=1e-08))
        self.assertTrue(np.isclose(loss2_expected, loss2, rtol=1e-08, atol=1e-08))
        self.assertTrue(np.isclose(loss3_expected, loss3, rtol=1e-08, atol=1e-08))
        self.assertTrue(np.isclose(mean_loss_expected, mean_loss, rtol=1e-08, atol=1e-08))

    def test_standard_BP(self):
        deltaU_1_exp = np.array([[-0.11298744, -0.107331], [0.07341862, 0.06939134]])
        deltaV_1_exp = np.array([[-0.06851441, -0.05931481, -0.05336094], [0.06079254, 0.0035937, 0.04875759]])
        deltaW_1_exp = np.array([[-2.36320453, -2.24145091], [3.13861959, 2.93420307], [-0.77541506, -0.69275216]])

        self.r.acc_deltas(self.x, self.d, self.y, self.s)
        self.assertTrue(np.isclose(deltaU_1_exp, self.r.deltaU).all())
        self.assertTrue(np.isclose(deltaV_1_exp, self.r.deltaV).all())
        self.assertTrue(np.isclose(deltaW_1_exp, self.r.deltaW).all())

    def test_BPTT_with_3_steps(self):
        deltaU_3_exp = np.array([[-0.12007034, -0.1141893], [0.06377434, 0.06003115]])
        deltaV_3_exp = np.array([[-0.07524721, -0.06495432, -0.05560471], [0.05465826, -0.00306904, 0.04567927]])
        deltaW_3_exp = np.array([[-2.36320453, -2.24145091], [3.13861959, 2.93420307], [-0.77541506, -0.69275216]])

        self.r.deltaU.fill(0)
        self.r.deltaV.fill(0)
        self.r.deltaW.fill(0)

        self.r.acc_deltas_bptt(self.x, self.d, self.y, self.s,3)

        self.assertTrue(np.isclose(deltaU_3_exp, self.r.deltaU).all())
        self.assertTrue(np.isclose(deltaV_3_exp, self.r.deltaV).all())
        self.assertTrue(np.isclose(deltaW_3_exp, self.r.deltaW).all())

    def test_computing_binary_prediction_loss(self):
        np_loss_expected = 0.887758278817
        np_loss = self.p.compute_loss_np(self.x, self.d_np)
        self.assertTrue(np.isclose(np_loss_expected, np_loss, rtol=1e-08, atol=1e-08))

    def test_binary_prediction_BP(self):
        deltaU_1_exp_np = np.array([[0.01926192, 0.01684262], [0.00719671, 0.0062928]])
        deltaV_1_exp_np = np.array([[0., 0., 0.02156006], [0., 0., 0.00805535]])
        deltaW_1_exp_np = np.array([[0.50701159, 0.47024475], [-0.27214729, -0.25241205], [-0.23486429, -0.2178327]])

        self.r.deltaU.fill(0)
        self.r.deltaV.fill(0)
        self.r.deltaW.fill(0)
        self.r.acc_deltas_np(self.x, self.d_np, self.y, self.s)

        self.assertTrue(np.isclose(deltaU_1_exp_np, self.r.deltaU).all())
        self.assertTrue(np.isclose(deltaV_1_exp_np, self.r.deltaV).all())
        self.assertTrue(np.isclose(deltaW_1_exp_np, self.r.deltaW).all())

    def test_binary_prediction_BPTT_with_3_steps(self):
        deltaU_3_exp_np = np.array([[0.0216261, 0.01914693], [0.01044642, 0.00946145]])
        deltaV_3_exp_np = np.array([[0.00223142, 0.00055566, 0.02156006], [0.00336126, 0.00046926, 0.00805535]])
        deltaW_3_exp_np = np.array([[0.50701159, 0.47024475], [-0.27214729, -0.25241205], [-0.23486429, -0.2178327]])

        self.r.deltaU.fill(0)
        self.r.deltaV.fill(0)
        self.r.deltaW.fill(0)

        self.r.acc_deltas_bptt_np(self.x, self.d_np, self.y, self.s, 3)

        self.assertTrue(np.isclose(deltaU_3_exp_np, self.r.deltaU).all())
        self.assertTrue(np.isclose(deltaV_3_exp_np, self.r.deltaV).all())
        self.assertTrue(np.isclose(deltaW_3_exp_np, self.r.deltaW).all())

    def test_compute_accuracy_for_binary_prediction(self):
        acc_expected = 1
        acc = self.p.compute_acc_np(self.x, self.d_np)
        self.assertEquals(acc, acc_expected)
