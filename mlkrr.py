"""
Metric Learning for Kernel Regression (MLKR)
"""
import sys
import time
import warnings

import numpy as np
import sklearn as sk
from scipy.optimize import minimize
from scipy.special import logsumexp
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import pairwise_distances

EPS = np.finfo(float).eps


class MLKRR:
    """Metric Learning for Kernel Ridge Regression (MLKRR)

    MLKRR is an algorithm for supervised metric learning, which learns a
    distance function by minimizing the validation error in a KRR.
    This algorithm can also be viewed as a supervised variation of PCA and can be
    used for dimensionality reduction and high dimensional data visualization.

    Parameters
    ----------
    init : string or numpy array, optional (default='auto')
      Initialization of the linear transformation. The identity matrix is used as default.

    tol : float, optional (default=None)
      Convergence tolerance for the optimization.

    max_iter_per_shuffle : int, optional (default=1000)
      Cap on number of conjugate gradient iterations for each shuffling of the data.

    verbose : bool, optional (default=False)
      Whether to print progress messages or not.

    diag : bool, optional (default=False)
      Allows to force the matrix A to be diagonal.

    smothness : float, optional (default=0)
      Induces a constraint to enforce smoothness in the matrix A.

    krr_regularization : float, optional (default=0)
      Regularization on the estimator of KRR.

    sigma : float, optional (default=1)
      Parameter of the gaussian kernel, determines its (initial) width.

    metod : string, optional (default='L-BFGS-B')
      Optimization method used to minimize the loss function.

    test_data: [X, y], optional (default=None)
      Allows to track the accuracy of the KRR on a different test set.

    size_alpha: float, optional (default=0.5)
      Size of the partition used to train the KRR.

    size_A: float, optional (default=0.5)
      Size of the partition used to fit the matrix A.

    shuffle_iterations: int, optional (default=1)
      Number of reshufflings of the data between alpha and A sets.

    Attributes
    ----------
    max_iter_per_shuffle : `int`
      The number of iterations the solver has run for each shuffling of the data.

    components_ : `numpy.ndarray`, shape=(n_components, n_features)
      The learned linear transformation ``A``.

    train_rmse : `list`, shape=(max_iter_per_shuffle * shuffle_iterations)
      Evolution of the root mean squared error of the KRR on the train set.

    test_rmse : `list`, shape=(max_iter_per_shuffle * shuffle_iterations)
      Evolution of the root mean squared error of the KRR on the test set.

    train_mae : `list`, shape=(max_iter_per_shuffle * shuffle_iterations)
      Evolution of the mean absolute error of the KRR on the train set.

    test_mae : `list`, shape=(max_iter_per_shuffle * shuffle_iterations)
      Evolution of the mean absolute error of the KRR on the test set.

    Examples
    --------

    >>> import mlkrr
    >>> from sklearn.datasets import load_iris
    >>> iris_data = load_iris()
    >>> X = iris_data['data']
    >>> Y = iris_data['target']
    >>> mlkr = mlkrr.MLKRR()
    >>> mlkr.fit(X, Y)

    References
    ----------
    .. [1] Tailoring molecular similarity with Metric Learning for Kernel Ridge Regression
            Machine Learning: Science and Technology
    """

    def __init__(
        self,
        init="identity",
        tol=None,
        max_iter_per_shuffle=100,
        verbose=False,
        diag=False,
        smoothness=0,
        krr_regularization=0,
        sigma=1,
        method="L-BFGS-B",
        test_data=None,
        size_alpha=0.5,
        size_A=0.5,
        shuffle_iterations=1,
    ):

        self.smoothness = smoothness
        self.test_data = test_data
        self.diag = diag
        self.init = init
        self.tol = tol
        self.max_iter_per_shuffle = max_iter_per_shuffle
        self.verbose = verbose
        self.krr_regularization = krr_regularization
        self.sigma = sigma
        self.method = method
        self.size_alpha = size_alpha
        self.size_A = size_A
        self.shuffle_iterations = shuffle_iterations

    def fit(self, X, y):
        """
        Fit MLKR model

        Parameters
        ----------
        X : (n x d) array of samples
        y : (n) data labels
        """
        n, d = X.shape

        if self.smoothness != 0:
            self.tmat = np.diag(np.ones(d), k=0) - np.diag(np.ones(d - 1), k=-1)
            self.tmat = self.tmat[:, :-1]
            self.ttpmat = self.tmat @ self.tmat.T

        if self.init == "identity":
            self.init = np.diag(np.ones(shape=[X.shape[1]]))

        self.A = self.init.copy()

        # Measure the total training time
        train_time = time.time()

        self.n_iter_ = 0

        self.train_rmses = []
        self.train_maes = []

        self.test_rmses = []
        self.test_maes = []

        for i in range(self.shuffle_iterations):
            self.shuffle_n_ = i

            self.shuffle_index = i
            print("====================================")
            print("Starting shuffle iteration: ", i)
            print("====================================")
            res = minimize(
                self._loss,
                self.A.ravel(),
                (X, y),
                method=self.method,
                jac=True,
                tol=self.tol,
                options=dict(maxiter=self.max_iter_per_shuffle),
            )

            self.components_ = res.x.reshape(self.A.shape)
            self.A = self.components_

        # Stop timer
        train_time = time.time() - train_time
        if self.verbose:
            # Warn the user if the algorithm did not converge
            if not res.success:
                cls_name = self.__class__.__name__
                warnings.warn(
                    "[{}] MLKR did not converge: {}".format(cls_name, res.message),
                    ConvergenceWarning,
                )
            print("[{}] Training took {:8.2f}s.".format(cls_name, train_time))

        return self

    def _loss(self, flatA, X, y):

        if self.n_iter_ == 0 and self.verbose:
            header_fields = ["Iteration", "Objective Value", "Time(s)"]
            header_fmt = "{:>10} {:>20} {:>10}"
            header = header_fmt.format(*header_fields)
            cls_name = self.__class__.__name__
            print("[{cls}]".format(cls=cls_name))
            print(
                "[{cls}] {header}\n[{cls}] {sep}".format(
                    cls=cls_name, header=header, sep="-" * len(header)
                )
            )

        start_time = time.time()

        A = flatA.reshape((-1, X.shape[1]))
        self.A = A

        indices_X1, indices_X2 = sk.model_selection.train_test_split(
            np.arange(len(X)),
            train_size=self.size_alpha,
            test_size=self.size_A,
            random_state=self.shuffle_index,
        )

        X1 = X[indices_X1]
        X2 = X[indices_X2]

        y1 = y[indices_X1]
        y2 = y[indices_X2]

        Xe = np.dot(X, A.T)
        X1e = Xe[indices_X1]
        X2e = Xe[indices_X2]

        sigma = self.sigma
        kernel_constant = 1 / (1 * np.sqrt(2 * np.pi) * sigma)
        exponent_constant = 1 / (1 * sigma**2)

        dist1 = pairwise_distances(X1e, squared=True, n_jobs=-1)

        kernel1 = kernel_constant * np.exp(-dist1 * exponent_constant)

        n1 = len(X1)
        J = np.linalg.inv(kernel1 + self.krr_regularization * np.eye(n1))
        v = J @ y1
        alphas = v

        intercept = 0

        # yhat = kernel1.dot(alphas) + intercept

        dist2 = pairwise_distances(X2e, X1e, squared=True, n_jobs=-1)

        kernel2 = kernel_constant * np.exp(-dist2 * exponent_constant)

        yhat2 = kernel2 @ alphas + intercept

        ydiff2 = yhat2 - y2

        train_rmse = np.sqrt(np.mean(ydiff2**2))
        train_mae = np.mean(np.abs(ydiff2))

        self.train_rmses.append(train_rmse)
        self.train_maes.append(train_mae)

        print("Train RMSE:", np.round(train_rmse, 3))
        print("Train MAE:", np.round(train_mae, 3))

        if self.test_data != None:
            X_test = self.test_data[0]
            Xt_embedded = np.dot(X_test, A.T)
            distt = pairwise_distances(Xt_embedded, X1e, squared=True, n_jobs=-1)
            kernel_test = kernel_constant * np.exp(-distt * exponent_constant)
            # / (np.sqrt(2 * np.pi) * sigma)

            yhat_test = kernel_test @ alphas + intercept

            y_test = self.test_data[1]

            # ydiff_test = yhat_test - y_test
            ydiff_test = np.array(yhat_test - y_test)

            test_rmse = np.sqrt(np.mean(ydiff_test**2))
            test_mae = np.mean(np.abs(ydiff_test))

            self.test_rmses.append(test_rmse)
            self.test_maes.append(test_mae)

            print("Test RMSE:", np.round(test_rmse, 3))
            print("Test MAE:", np.round(test_mae, 3))

        # print('Train RMSE:', np.round(np.sqrt(np.mean(ydiff ** 2)), 5))
        print(
            "========= shuffle: {},  iteration: {} ==============".format(
                self.shuffle_n_, self.n_iter_
            )
        )

        cost = (ydiff2**2).sum()

        # also compute the gradient
        u = J.T @ kernel2.T @ ydiff2
        W = ydiff2[:, np.newaxis] * kernel2 * alphas

        Q = np.diag(np.sum(W, axis=1))

        R = np.diag(np.sum(W, axis=0))

        t1 = X2.T @ (-W) @ X1
        t2 = X2.T @ Q @ X2
        t3 = X1.T @ R @ X1
        cc = -4 * A * exponent_constant @ (t1 + t1.T + t2 + t3)

        S = kernel1 * u[:, np.newaxis] * v
        T = -S - S.T + np.diag(np.sum(S, axis=0) + np.sum(S, axis=1))

        grad = cc + 4 * exponent_constant * A @ X1.T @ T @ X1

        if self.diag is True:
            grad_diag = np.diag(grad)

            if self.smoothness != 0:
                grad += self.smoothness * 2 * np.diag(np.dot(np.diag(A), self.ttpmat))

                extra = (
                    self.smoothness
                    * np.linalg.norm(np.diag(np.dot(np.diag(A), self.tmat))) ** 2
                )

                cost += extra

            grad = np.diag(grad_diag)

        else:
            if self.smoothness != 0:
                extra = self.smoothness * 2 * A @ self.ttpmat
                grad += extra

                cost += self.smoothness * np.linalg.norm(np.dot(A, self.tmat)) ** 2

        if self.verbose:
            start_time = time.time() - start_time
            values_fmt = "[{cls}] {n_iter:>10} {loss:>20.6e} {start_time:>10.2f}"
            print(
                values_fmt.format(
                    cls=self.__class__.__name__,
                    n_iter=self.n_iter_,
                    loss=cost,
                    start_time=start_time,
                )
            )
            sys.stdout.flush()

        self.n_iter_ += 1
        return cost, grad.ravel()


class MLKR:
    """Metric Learning for Kernel Ridge Regression (MLKRR)"""

    def __init__(
        self,
        init="identity",
        tol=None,
        max_iter=100,
        verbose=False,
        method="L-BFGS-B",
        test_data=None,
    ):

        self.init = init
        self.tol = tol
        self.test_data = test_data
        self.max_iter = max_iter
        self.verbose = verbose
        self.method = method

    def fit(self, X, y):
        """
        Fit MLKR model

        Parameters
        ----------
        X : (n x d) array of samples
        y : (n) data labels
        """
        n, d = X.shape
        cls_name = self.__class__.__name__

        if self.init == "identity":
            self.init = np.diag(np.ones(shape=[X.shape[1]]))

        self.A = self.init.copy()

        # Measure the total training time
        train_time = time.time()

        self.train_rmses = []
        self.train_maes = []

        self.test_rmses = []
        self.test_maes = []

        self.n_iter_ = 0

        res = minimize(
            self._loss,
            self.A.ravel(),
            (X, y),
            method=self.method,
            jac=True,
            tol=self.tol,
            options=dict(maxiter=self.max_iter),
        )

        self.components_ = res.x.reshape(self.A.shape)
        self.A = self.components_

        # Stop timer
        train_time = time.time() - train_time
        if self.verbose:
            # Warn the user if the algorithm did not converge
            if not res.success:
                cls_name = self.__class__.__name__
                warnings.warn(
                    "[{}] MLKR did not converge: {}".format(cls_name, res.message),
                    ConvergenceWarning,
                )
            print("[{}] Training took {:8.2f}s.".format(cls_name, train_time))

        return self

    def _loss(self, flatA, X, y):

        if self.n_iter_ == 0 and self.verbose:
            header_fields = ["Iteration", "Objective Value", "Time(s)"]
            header_fmt = "{:>10} {:>20} {:>10}"
            header = header_fmt.format(*header_fields)
            cls_name = self.__class__.__name__
            print("[{cls}]".format(cls=cls_name))
            print(
                "[{cls}] {header}\n[{cls}] {sep}".format(
                    cls=cls_name, header=header, sep="-" * len(header)
                )
            )

        start_time = time.time()

        A = flatA.reshape((-1, X.shape[1]))
        self.A = A

        X_embedded = np.dot(X, A.T)
        dist = pairwise_distances(X_embedded, squared=True, n_jobs=-1)
        np.fill_diagonal(dist, np.inf)
        softmax = np.exp(-dist - logsumexp(-dist, axis=1)[:, np.newaxis])
        yhat = softmax.dot(y)
        ydiff = yhat - y
        cost = (ydiff**2).sum()

        train_rmse = np.sqrt(np.mean(ydiff**2))
        train_mae = np.mean(np.abs(ydiff))

        self.train_rmses.append(train_rmse)
        self.train_maes.append(train_mae)

        print("Train RMSE:", np.round(train_rmse, 3))
        print("Train MAE:", np.round(train_mae, 3))

        # also compute the gradient

        W = softmax * ydiff[:, np.newaxis] * (y - yhat[:, np.newaxis])
        W_sym = W + W.T
        np.fill_diagonal(W_sym, -W.sum(axis=0))
        grad = 4 * (X_embedded.T.dot(W_sym)).dot(X)

        if self.test_data is not None:
            X_test = self.test_data[0]
            Xt_embedded = np.dot(X_test, A.T)
            distt = pairwise_distances(Xt_embedded, X_embedded, squared=True, n_jobs=-1)

            softmax = np.exp(-distt - logsumexp(-distt, axis=1)[:, np.newaxis])
            yhat_test = softmax.dot(y)

            y_test = self.test_data[1]

            # ydiff_test = yhat_test - y_test
            ydiff_test = np.array(yhat_test - y_test)

            test_rmse = np.sqrt(np.mean(ydiff_test**2))
            test_mae = np.mean(np.abs(ydiff_test))

            self.test_rmses.append(test_rmse)
            self.test_maes.append(test_mae)

            print("Test RMSE:", np.round(test_rmse, 3))
            print("Test MAE:", np.round(test_mae, 3))

        if self.verbose:
            start_time = time.time() - start_time
            values_fmt = "[{cls}] {n_iter:>10} {loss:>20.6e} {start_time:>10.2f}"
            print(
                values_fmt.format(
                    cls=self.__class__.__name__,
                    n_iter=self.n_iter_,
                    loss=cost,
                    start_time=start_time,
                )
            )
            sys.stdout.flush()
        print("===================================================")
        self.n_iter_ += 1
        return cost, grad.ravel()
