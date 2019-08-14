import numpy as np


class HMM:
    def __init__(self, A=None, B=None, pi=None, eps=None):
        """
        A simple hidden Markov model with multinomial emission distribution.

        Parameters
        ----------
        A : numpy array of shape(N, N) or None
            The transition matrix between latent states in the HMM. Index 'i',
           'j' givesthe probability of transitioning from latent state 'i to
            latent state 'j'.
            Default is None.
        B : numpy  array of shape (N, V) or None
            The emission matrix, Entry 'i', 'j' gives the probability of latent
            state i emitting an observation of type 'j'. Default is None.
        pi : numpy array of shape (N, ) or None
            The prior probability of each latent state. If None, use a uniform
            prior over states. Default is None.
        eps : float or None
            Epsilon value to avoid: math 'log(0)' errors. If None, defaults to
            the machine epsilon. Default is None. 
        """
        self.eps = np.finfo(float).eps

        #  transition matrix
        self.A = A

        # emission matrix
        self.B = B

        # prior probability of each latent state
        self.pi = pi
        if self.pi is not None:
            self.pi[self.pi == 0] = self.eps

        # number of latent state types
        self.N = None
        if self.A is not None:
            self.N = self.A.shape[0]
            self.A[self.A == 0] = self.eps

        # number of observation types
        self.V = None
        if self.B is not None:
            self.V = self.B.shape[1]
            self.B[self.B == 0] = self.eps

        # set of training sequences
        self.O = None

        # number of sequences in O
        self.I = None

        # number of observation in each sequence
        self.T = None

    def generate(self, n_steps, latent_state_types, obs_types):
        """
        Sample a sequence from the HMM

        Parameters
        ---------
        n_steps: int
                The length of the generated sequence
        latent_state_types : numpy array of shape '(N, )'
               A collection of labels for the latent states
        obs_types : numpy array of shape '(V,)'
               A collection of labels for the observation

        Returns
        -------
        states : numpy array of shape '(n_steps, )'
              The sampled latent states.
        emissions : numpy array of shape '(n_steps, )'
              The sampled emissions.
        """
        s = np.random.multinomial(1, self.pi).argmax()
        states = [latent_state_types]

        # generate an emission given latent state
        v = np.random.multinomial(1, self.B[s, :]).argmax()
        emissions = [obs_types]

        # sample a latent transition, rinse, and repeat
        for i in range(n_steps - 1):
            s = np.random.multinomial(1, self.A[s, :]).argmax()
            states.append(latent_state_types[s])

            v = np.random.multinomial(1, self.B[s, :]).argmax()
            emissions.append(obs_types[v])

        return np.array(states), np.array(emissions)

    def log_likelihood(self, O):
        """
        Given the HMM parameterized by (A, B, pi)' and an observation
        sequence 'O', compute the marginal likelihood of the observations:
        P(O|A,B,\pi), summing over latent states.

        Parameters
        ----------
        O : numpy array of shape'(1, T)'
            A single set of observation

        Returns
        ---------
        likelihood : float
            The likelihood of the observation 'O' under the HMM
        """
        if O.ndim == 1:
            O = O.reshape(1, -1)

        I, T = O.shape

        if I != 1:
            raise ValueError("likelihood only  accepts a single sequence")

        forward = self._forward(O[0])
        log_likelihood = logsumexp(forward[:, T - 1])
        return log_likelihood

    def decode(self, O):
        """
        Given the HMM parameterized by '(A, B, \pi)' and an observation sequence
        'o_1, \ldots, o_T', compute the most probable sequence of latent state,
        'Q = q_1, \ldots, q_T'

        Parameters
        ---------
        O : np.array of shape '(T, )'
            An observation sequence of length  'T'.

        Returns
        -------
        best_path : list of length 'T'
            The most probable sequence of latent for observations 'O'.
        best_path_prob : float
            The probability of the latent state sequence in 'best_path' under
            the HMM
        """
        eps = self.eps

        if O.ndim == 1:
            O = O.reshape(1, -1)

        # number of observation in each sequence
        T = O.shape[1]

        # number of training sequences
        I = O.shape[0]
        if I != 1:
            raise ValueError("Can only decode a single sequence (O.shape[0] must be 1)")

        # initialize the viterbi and back_pointer matrices
        viterbi = np.zeros((self.N, T))
        back_pointer = np.zeros((self.N, T)).astype(int)

        ot = O[0, 0]
        for s in range(self.N):
            back_pointer[s, 0] = 0
            viterbi[s, 0] = np.log(self.pi[s] + eps) + np.log(self.B[s, ot] + eps)

        for t in range(1, T):
            ot = O[0, t]
            for s in range(self.N):
                seq_probs = [
                    viterbi[s_, t - 1]
                    + np.log(self.A[s_, s] + eps)
                    + np.log(self.B[s, ot] + eps)
                    for s_ in range(self.N)
                ]

                viterbi[s, t] = np.max(seq_probs)
                back_pointer[s, t] = np.argmax(seq_probs)

        best_path_log_prob = viterbi[:, T - 1].max()

        # backtrack through the trellis to get the most likely sequence of
        # latent states
        pointer = viterbi[:, T - 1].argmax()
        best_path = [pointer]
        for t in reversed(range(1, T)):
            pointer = back_pointer[pointer, t]
            best_path.append(pointer)
        best_path = best_path[::-1]
        return best_path, best_path_log_prob

    def _forward(self, Obs):
        """
        Compute the forward probability trellis for an HMM parameterized by
        '(A,B,pi)'
        """
        eps = self.eps
        T = Obs.shape[0]

        # initialize the forward probability matrix
        forward = np.zeros((self.N, T))

        ot = Obs[0]
        for s in range(self.N):
            forward[s, 0] = np.log(self.pi[s] + eps) + np.log(self.B[s, ot] + eps)

        for t in range(1, T):
            ot = Obs[t]
            for s in range(self.N):
                forward[s, t] = logsumexp(
                    [
                        forward[s_, t - 1]
                        + np.log(self.A[s_, s] + eps)
                        + np.log(self.B[s, ot] + eps)
                        for s_ in range(self.N)
                    ]
                )

        return forward

    def _backward(self, Obs):
        """
        Compute the backward probability trellis for an HMM parameterized by
        '(A, B, pi)'
        """
        eps = self.eps
        T = Obs.shape[0]

        # initialize the backward trellis
        backward = np.zeros((self.N, T))

        for s in range(self.N):
            backward[s, T - 1] = 0

        for t in reversed(range(T - 1)):
            ot1 = Obs[t + 1]
            for s in range(self.N):
                backward[s, t] = logsumexp(
                    [
                        np.log(self.A[s, s_] + eps)
                        + np.log(self.B[s_, ot1] + eps)
                        + backward[s_, t + 1]
                        for s_ in range(self.N)
                    ]
                )
        return backward

    def fit(
        self, O, latent_state_types, observation_types, pi=None, tol=1e-5, verbose=False
    ):
        """
        Given an observation sequence 'O' and the set of possible latent states,
        learn the MLE HMM parameters 'A' and 'B'


        Notes
        -----
        Model fitting is done iterativly using the Baum-Welch/Forward-Backward
        algorithm, a special case of the EM algorithm

        Parameters
        ---------
        O : :py:class: 'ndarray<numpy.ndarray>' of shape '(I, T)'
            The set of 'I' training observations, each of length 'T'.
        latent_state_types : list of length  'N'
            The collections of valid latent states.
        observation_types : list of length 'V'
            The collection of valid observation states.
        pi : numpy array of shape '(N, )'
            The prior probability of each latent states. If None, assume each
            latent states is equally likely a prior. Default is None.
        tol : float
            The tolerance value. If the difference in log likelihood between
            two epochs is less than this value, terminate training. Default is
            ie=5.
        verbose : bool
            Print training states after each epoch. Default is True.

        Returns
        --------
        A : numpy array of shape '(N, N)'
            The estimated transition matrix
        B : numpy array of shape '(N, V)'
            The estimated emission matrix
        pi : numpy array of shape '(N, )'
            The estimated prior probabilities of each latent state. 
        """
        if O.ndim == 1:
            O = O.reshape(1, -1)

        # observations
        self.O = O

        # number of training examples (I) and their lengths (T)
        self.I, self.T = self.O.shape

        # number of type of observation
        self.V = len(observation_types)

        # number  of letent state types
        self.N = len(latent_state_types)

        # Uniform initialization of prior over latent states
        self.pi = pi
        if self.pi is None:
            self.pi = np.ones(self.N)
            self.pi = self.pi / self.pi.sum()

        # Uniform initialization of A
        self.A = np.ones((self.N, self.N))
        self.A = self.A / self.A.sum(axis=1)[:, None]

        # Random initializatin of B
        self.B = np.random.rand(self.N, self.V)
        self.B = self.B / self.B.sum(axis=1)[:, None]

        # iterate E and M steps multi convergence criter is met
        step, delta = 0, np.inf
        ll_prev = np.sum([self.log_likelihood(o) for o in self.O])
        while delta > tol:
            gamma, xi, phi = self._Estep()
            self.A, self.B, self.pi = self._Mstep(gamma, xi, phi)
            ll = np.sum([self.log_likelihood(o) for o in self.O])
            delta = ll - ll_prev
            ll_prev = ll
            step += 1

            if verbose:
                fstr = "[Epoch {}] LL : {:.3f} Delta: {:.5f}}"
                print(fstr.format(step, ll_prev, delta))

        return self.A, self.B, self.pi

    def _Estep(self):
        """
        run a single E-step  update for the Baum-Welch/forward-Backward
        algorithm. This step estimates ''xi'' and ''gamma'', the excepted
        state-state transition counts and the expected state-occupancy
        counts respectively
        """
        eps = self.eps

        gamma = np.zeros((self.I, self.N, self.T))
        xi = np.zeros((self.I, self.N, self.N, self.T))
        phi = np.seroz((self.I, self.N))

        for i in range(self.I):
            Obs = self.O[i, :]
            fwd = self._forward(Obs)
            bwd = self._backward(Obs)
            log_likelihood = logsumexp(fwd[:, self.T - 1])

            t = self.T - 1
            for si in range(self.N):
                gamma[i, si, t] = fwd[si, t] + bwd[si, t] - log_likelihood
                phi[i, si] = fwd[si, 0] + bwd[si, 0] - log_likelihood

            for t in range(self.T - 1):
                ot1 = Obs[t + 1]
                for si in range(self.N):
                    gamma[i, si, t] = fwd[si, t] + bwd[si, t] - log_likelihood
                    for sj in range(self.N):
                        xi[i, si, sj, t] = (
                            fwd[si, t]
                            + np.log(self.A[si, sj] + eps)
                            + np.log(self.B[sj, ot1] + eps)
                            + bwd[sj, t + 1]
                            - log_likelihood
                        )

        return gamma, xi, phi

    def _Mstep(self, gamma, xi, phi):
        """
        Run a single M-step update for the Baum-Welch/Forward-Backward
        algorithm

        Parameters
        -----------
        gamma : numpy array of shape '(I, N, T)'
               The estimated state-occupancy count matrix.
        xi : numpy of shape '(I, N, N, T)'
             The estimated state-state transition count matrix.
        phi : numpy array of shape '(I, N)'
             the estimated starting count matrix for each latent state.

        Returns
        ------
        A : numpy array of shape '(N, N)'
            The estimated transiton matrix
        B : numpy array of shape '(N, V)'
            the estimated emission matrix.
        pi : numpy array of shape '(N, )'
            the estimated prior probability for each latent state.
        """
        eps = self.eps

        # initialize the estimated transition (A) and emission (B) matrix
        A = np.zeros((self.N, self.N))
        B = np.zeros((self.N, self.V))
        pi = np.zeros(self.N)

        count_gamma = np.zeros((self.I, self.N, self.V))
        count_xi = np.zeros((self.I, self.N, self.N))

        for i in range(self.I):
            Obs = self.O[i, :]
            for si in range(self.N):
                for vk in range(self.V):
                    if not (Obs == vk).any():
                        # count_gamma[i,si,vk] = -np.inf
                        count_gamma[i, si, vk] = np.log(eps)
                    else:
                        count_gamma[i, si, vk] = logsumexp(gamma[i, si, Obs == vk])

                for sj in range(self.N):
                    count_xi[i, si, sj] = logsumexp(xi[i, si, sj, :])

        pi = logsumexp(phi, axis=0) - np.log(self.I + eps)
        np.testing.assert_almost_equal(np.exp(pi).sum(), 1)

        for si in range(self.N):
            for vk in range(self.V):
                B[si, vk] = logsumexp(count_gamma[:, si, vk]) - logsumexp(
                    count_gamma[:, si, :]
                )

            for sj in range(self.N):
                A[si, sj] = logsumexp(count_xi[:, si, sj]) - logsumexp(
                    count_xi[:, si, :]
                )

            np.testing.assert_almost_equal(np.exp(A[si, :]).sum(), 1)
            np.testing.assert_almost_equal(np.exp(B[si, :]).sum(), 1)

        return np.exp(A), np.exp(B), np.exp(pi)


def logsumexp(log_probs, axis=None):
    _max = np.max(log_probs)
    ds = log_probs - _max
    exp_sum = np.exp(ds).sum(axis=axis)
    return _max + np.log(exp_sum)
