# coding: utf-8
from rnnmath import *
from model import Model, is_param, is_delta

class RNN(Model):
    '''
    This class implements Recurrent Neural Networks.
    
    You should implement code in the following functions:
        predict				->	predict an output sequence for a given input sequence
        acc_deltas			->	accumulate update weights for the RNNs weight matrices, standard Back Propagation
        acc_deltas_bptt		->	accumulate update weights for the RNNs weight matrices, using Back Propagation Through Time
        acc_deltas_np		->	accumulate update weights for the RNNs weight matrices, standard Back Propagation -- for number predictions
        acc_deltas_bptt_np	->	accumulate update weights for the RNNs weight matrices, using Back Propagation Through Time -- for number predictions

    Do NOT modify any other methods!
    Do NOT change any method signatures!
    '''
    
    def __init__(self, vocab_size, hidden_dims, out_vocab_size):
        '''
        initialize the RNN with random weight matrices.
        
        DO NOT CHANGE THIS
        
        vocab_size		size of vocabulary that is being used
        hidden_dims		number of hidden units
        out_vocab_size	size of the output vocabulary
        '''

        super().__init__(vocab_size, hidden_dims, out_vocab_size)

        # matrices V (input -> hidden), W (hidden -> output), U (hidden -> hidden)
        with is_param():
            self.U = np.random.randn(self.hidden_dims, self.hidden_dims)*np.sqrt(0.1)
            self.V = np.random.randn(self.hidden_dims, self.vocab_size)*np.sqrt(0.1)
            self.W = np.random.randn(self.out_vocab_size, self.hidden_dims)*np.sqrt(0.1)

        # matrices to accumulate weight updates
        with is_delta():
            self.deltaU = np.zeros_like(self.U)
            self.deltaV = np.zeros_like(self.V)
            self.deltaW = np.zeros_like(self.W)

    def predict(self, x):
        '''
        predict an output sequence y for a given input sequence x
        
        x	list of words, as indices, e.g.: [0, 4, 2]
        
        returns	y,s
        y	matrix of probability vectors for each input word
        s	matrix of hidden layers for each input word
        
        '''
        
        # matrix s for hidden states, y for output states, given input x.
        # rows correspond to times t, i.e., input words
        # s has one more row, since we need to look back even at time 0 (s(t=0-1) will just be [0. 0. ....] )

        s = np.zeros((len(x) + 1, self.hidden_dims))
        y = np.zeros((len(x), self.out_vocab_size))

        for t in range(len(x)):
            input_onehot = make_onehot(x[t], self.vocab_size)
            prev_hidden = s[t-1]

            net_in_t = self.V @ input_onehot + self.U @ prev_hidden
            s[t] = sigmoid(net_in_t)

            # prob distribution over the vocab at time step t
            net_out_t = self.W @ s[t]
            y[t] = softmax(net_out_t)

        return y, s
    
    def acc_deltas(self, x, d, y, s):
        '''
        accumulate updates for V, W, U
        standard back propagation
        
        this should not update V, W, U directly. instead, use deltaV, deltaW, deltaU to accumulate updates over time
        
        x	list of words, as indices, e.g.: [0, 4, 2]
        d	list of words, as indices, e.g.: [4, 2, 3]
        y	predicted output layer for x; list of probability vectors, e.g., [[0.3, 0.1, 0.1, 0.5], [0.2, 0.7, 0.05, 0.05] [...]]
            should be part of the return value of predict(x)
        s	predicted hidden layer for x; list of vectors, e.g., [[1.2, -2.3, 5.3, 1.0], [-2.1, -1.1, 0.2, 4.2], [...]]
            should be part of the return value of predict(x)
        
        no return values
        '''

        for t in reversed(range(len(x))):
            y_t = y[t]
            s_t = s[t]
            s_prev = s[t-1]
            x_t = make_onehot(x[t], self.vocab_size)
            d_t = make_onehot(d[t], self.out_vocab_size)

            # wrt W
            delta_out_t = d_t - y_t
            self.deltaW += np.outer(delta_out_t, s_t)

            # wrt V
            delta_in_t = (self.W.T @ delta_out_t) * grad(s_t)
            self.deltaV += np.outer(delta_in_t, x_t)

            # wrt U
            self.deltaU += np.outer(delta_in_t, s_prev)

    def acc_deltas_np(self, x, d, y, s):
        '''
        accumulate updates for V, W, U
        standard back propagation
        
        this should not update V, W, U directly. instead, use deltaV, deltaW, deltaU to accumulate updates over time
        for number prediction task, we do binary prediction, 0 or 1

        x	list of words, as indices, e.g.: [0, 4, 2]
        d	array with one element, as indices, e.g.: [0] or [1]
        y	predicted output layer for x; list of probability vectors, e.g., [[0.3, 0.1, 0.1, 0.5], [0.2, 0.7, 0.05, 0.05] [...]]
            should be part of the return value of predict(x)
        s	predicted hidden layer for x; list of vectors, e.g., [[1.2, -2.3, 5.3, 1.0], [-2.1, -1.1, 0.2, 4.2], [...]]
            should be part of the return value of predict(x)
        
        no return values
        '''
        t = len(x) - 1

        # copied from non-np function
        y_t = y[t]
        s_t = s[t]
        s_prev = s[t - 1]
        x_t = make_onehot(x[t], self.vocab_size)
        # apart from this line
        d_t = make_onehot(d[0], self.out_vocab_size)

        delta_out_t = d_t - y_t
        self.deltaW += np.outer(delta_out_t, s_t)

        delta_in_t = (self.W.T @ delta_out_t) * grad(s_t)
        self.deltaV += np.outer(delta_in_t, x_t)

        self.deltaU += np.outer(delta_in_t, s_prev)

    def acc_deltas_bptt(self, x, d, y, s, steps):
        '''
        accumulate updates for V, W, U
        back propagation through time (BPTT)
        
        this should not update V, W, U directly. instead, use deltaV, deltaW, deltaU to accumulate updates over time
        
        x		list of words, as indices, e.g.: [0, 4, 2]
        d		list of words, as indices, e.g.: [4, 2, 3]
        y		predicted output layer for x; list of probability vectors, e.g., [[0.3, 0.1, 0.1, 0.5], [0.2, 0.7, 0.05, 0.05] [...]]
                should be part of the return value of predict(x)
        s		predicted hidden layer for x; list of vectors, e.g., [[1.2, -2.3, 5.3, 1.0], [-2.1, -1.1, 0.2, 4.2], [...]]
                should be part of the return value of predict(x)
        steps	number of time steps to go back in BPTT
        
        no return values
        '''

        for t in reversed(range(len(x))):
            # wrt W
            y_t = y[t]
            s_t = s[t]
            d_t = make_onehot(d[t], self.out_vocab_size)
            delta_out_t = d_t - y_t
            self.deltaW += np.outer(delta_out_t, s_t)

            delta_in_t = None
            last_step = t-steps-1 if (t-steps-1) >= -1 else -1

            for step in range(t, last_step, -1):
                if step == t:
                    delta_in_t = self.W.T @ delta_out_t * grad(s[step])
                else:
                    delta_in_t = self.U.T @ delta_in_t * grad(s[step])

                # wrt V
                self.deltaV += np.outer(delta_in_t, make_onehot(x[step], self.vocab_size))

                # wrt U
                self.deltaU += np.outer(delta_in_t, s[step-1])


    def acc_deltas_bptt_np(self, x, d, y, s, steps):
        '''
        accumulate updates for V, W, U
        back propagation through time (BPTT)
        
        this should not update V, W, U directly. instead, use deltaV, deltaW, deltaU to accumulate updates over time
        for number prediction task, we do binary prediction, 0 or 1

        x	list of words, as indices, e.g.: [0, 4, 2]
        d	array with one element, as indices, e.g.: [0] or [1]
        y		predicted output layer for x; list of probability vectors, e.g., [[0.3, 0.1, 0.1, 0.5], [0.2, 0.7, 0.05, 0.05] [...]]
                should be part of the return value of predict(x)
        s		predicted hidden layer for x; list of vectors, e.g., [[1.2, -2.3, 5.3, 1.0], [-2.1, -1.1, 0.2, 4.2], [...]]
                should be part of the return value of predict(x)
        steps	number of time steps to go back in BPTT
        
        no return values
        '''
        t = len(x)-1

        y_t = y[t]
        s_t = s[t]
        d_t = make_onehot(d[0], self.out_vocab_size)

        delta_out_t = d_t - y_t
        self.deltaW += np.outer(delta_out_t, s_t)

        delta_in_t = None
        last_step = t-steps-1 if (t-steps-1) >= -1 else -1
        for step in range(t, last_step, -1):
            if step == t:
                delta_in_t = self.W.T @ delta_out_t * grad(s[step])
            else:
                delta_in_t = self.U.T @ delta_in_t * grad(s[step])

            # wrt V
            self.deltaV += np.outer(delta_in_t, make_onehot(x[step], self.vocab_size))

            # wrt U
            self.deltaU += np.outer(delta_in_t, s[step-1])
