"""
Metric Learning for Kernel Regression (MLKR)
"""
import sys
import time
import warnings

import numpy as np
import sklearn as sk
import sklearn.model_selection
from scipy.optimize import minimize
from scipy.special import logsumexp
from sklearn.metrics import pairwise_distances
from sklearn.metrics import pairwise_kernels

from scipy.linalg import lu_factor, lu_solve

import pandas as pd

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

    krr_regularization : float, optional (default=1e-9)
      Regularization on the estimator of KRR.

    sigma : float, optional (default=1)
      Parameter of the gaussian kernel, determines its (initial) width.

    learn_sigma : bool, optional (default=False)
      Learn sigma before every shuffle if True. Sigma is fixed otherwise.

    method : string, optional (default='L-BFGS-B')
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

    A : `numpy.ndarray`, shape=(n_components, n_features)
      The learned linear transformation ``A``.

    sigma : 'float'
      Learned variance.

    train_rmses : `list`, shape=(max_iter_per_shuffle * shuffle_iterations)
      Evolution of the root mean squared error of the KRR on the train set.

    test_rmses : `list`, shape=(max_iter_per_shuffle * shuffle_iterations)
      Evolution of the root mean squared error of the KRR on the test set.

    train_maes : `list`, shape=(max_iter_per_shuffle * shuffle_iterations)
      Evolution of the mean absolute error of the KRR on the train set.

    test_maes : `list`, shape=(max_iter_per_shuffle * shuffle_iterations)
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
        krr_regularization=1e-9,
        sigma=1.0,
        learn_sigma=False,
        method="L-BFGS-B",
        test_data=None,
        size_alpha=0.5,
        size_A=0.5,
        shuffle_iterations=1,
        diag=False
    ):

        self.test_data = test_data
        self.init = init
        self.tol = tol
        self.max_iter_per_shuffle = max_iter_per_shuffle
        self.verbose = verbose
        self.krr_regularization = krr_regularization
        self.sigma = sigma
        self.learn_sigma=learn_sigma
        self.method = method
        self.size_alpha = size_alpha
        self.size_A = size_A
        self.shuffle_iterations = shuffle_iterations
        self.diag = diag

    def fit(self, X, y):
        """
        Fit MLKR model

        Parameters
        ----------
        X : (n x d) array of samples
        y : (n) data labels
        """
        n, d = X.shape
        self.Ashape=d
        assert n==len(y), "The number of samples do not match that of labels."

        if type(self.init)==type('') and self.init == "identity":
            self.init = np.eye(self.Ashape)

        self.A = self.init.copy()
        
        assert len(self.A)==d, "Initial matrix of wrong dimension."

        # Measure the total training time
        train_time = time.time()

        self.n_iter_ = 0

        if self.test_data != None:
            self.train_rmses = []
            self.train_maes = []

            self.test_rmses = []
            self.test_maes = []
        
        for i in range(self.shuffle_iterations):
            self.shuffle_n_ = i
            self.shuffle_index = i

            self.indices_X1, self.indices_X2 = sk.model_selection.train_test_split(
                np.arange(len(X)),
                train_size=self.size_alpha,
                test_size=self.size_A,
                random_state=self.shuffle_index,
            )

            if self.verbose:
                print("====================================")
                print("Starting shuffle iteration: ", i+1)
                print("====================================")
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

            if self.learn_sigma:
                print("Optimizing for sigma. Current sigma:", self.sigma)
                t=time.time()
                res = minimize(
                    self.simpleloss,
                    self.sigma,
                    (self.A, X, y),
                    method=self.method,
                    jac=True,
                    options=dict(maxiter=self.max_iter_per_shuffle),
                    bounds=[(1.0,None)],
                )
                self.sigma=res.x[0]
                print("New sigma:", self.sigma, "(took", np.round(time.time()-t,2), "s)")
            
            res = minimize(
                self._loss,
                self.A.ravel(),
                (X, y),
                method=self.method,
                tol=self.tol,
                jac=True,
                options=dict(maxiter=self.max_iter_per_shuffle),
                callback=self.callback
            )
            self.A = res.x.reshape(self.A.shape)
        
        # Stop timer
        train_time = time.time() - train_time
        if self.verbose:
            cls_name = self.__class__.__name__
            print("[{}] Training took {:8.2f}s.".format(cls_name, train_time))

        return self

    def _loss(self, parms, X, y):
        if self.verbose:
            print(
                "========= shuffle: {},  iteration: {} ==============".format(
                    self.shuffle_n_+1, self.n_iter_+1
                )
            )
        sigma=self.sigma
        reg=self.krr_regularization
        flatA=parms
        start_time = time.time()

        A = flatA.reshape((-1, self.Ashape))
        indices_X1, indices_X2 = self.indices_X1, self.indices_X2
        X1 = X[indices_X1]
        X2 = X[indices_X2]

        y1 = y[indices_X1]
        y2 = y[indices_X2]

        Xe = X@A.T
        X1e = Xe[indices_X1]
        X2e = Xe[indices_X2]
        n1 = len(X1)

        kernel_constant = 1 / (1 * np.sqrt(2 * np.pi) * sigma)
        exponent_constant = 1 / (1 * sigma**2)
        
        n_jobs=-1
        kernel1=pairwise_kernels(X1e, metric='rbf', gamma=exponent_constant, n_jobs=n_jobs)*kernel_constant 
        kernel2=pairwise_kernels(X2e, X1e, metric='rbf', gamma=exponent_constant, n_jobs=n_jobs)*kernel_constant 

        # LU decomposition of H used everytime H^-1 @ b or H^-T @ b is computed
        H=kernel1+reg*np.eye(n1)
        lu, pivot = lu_factor(H, check_finite=False)
        alphas=lu_solve((lu,pivot), y1, check_finite=False)

        intercept = 0

        yhat2 = kernel2 @ alphas + intercept
        ydiff2 = yhat2 - y2
        cost = (ydiff2**2).sum()

        ############## TESTS #############

        self.train_rmse = np.sqrt(np.mean(ydiff2**2))
        self.train_mae = np.mean(np.abs(ydiff2))

        if self.test_data != None:
            X_test = self.test_data[0]
            Xt_embedded = np.dot(X_test, A.T)

            kernel_test=pairwise_kernels(Xt_embedded, X1e, metric='rbf', gamma=exponent_constant, n_jobs=n_jobs)*kernel_constant

            yhat_test = kernel_test @ alphas

            y_test = self.test_data[1]
            
            ydiff_test = np.array(yhat_test - y_test)

            self.test_rmse = np.sqrt(np.mean(ydiff_test**2))
            self.test_mae = np.mean(np.abs(ydiff_test))

        ################# GRADIENTS #################
        # matrix gradient
        u=lu_solve((lu,pivot), kernel2.T@ydiff2, trans=1, check_finite=False)
        W = ydiff2[:, np.newaxis] * kernel2 * alphas
        Q = np.diag(np.sum(W, axis=1))
        R = np.diag(np.sum(W, axis=0))
        S = kernel1 * u[:, np.newaxis] * alphas
        T = -S - S.T + np.diag(np.sum(S, axis=0) + np.sum(S, axis=1))
        s1=X2.T@(-W)@X1
        s2=X2e.T@Q@X2
        s3=X1e.T@(R-T)@X1
        gradA = -4*exponent_constant*(A@(s1+s1.T) +  s2+s3)
       
        if self.diag==True:
            gradA=np.diag(np.diag(gradA))

        ################## VERBOSE ###################
        if self.verbose:
            start_time = time.time() - start_time
            values_fmt = "[{cls}] {n_iter:>10} {loss:>20.6e} {start_time:>10.2f}"
            print(
                values_fmt.format(
                    cls=self.__class__.__name__,
                    n_iter=self.n_iter_+1,
                    loss=cost,
                    start_time=start_time,
                )
            )
            sys.stdout.flush()
        
        return cost, gradA.ravel()
   
    def callback(self,parms):

        if self.test_data != None:
            self.train_rmses.append(self.train_rmse)
            self.train_maes.append(self.train_mae)
            if self.verbose:
                print("Train RMSE:", np.round(self.train_rmse, 5))
                print("Train MAE:", np.round(self.train_mae, 5))
            self.test_rmses.append(self.test_rmse)
            self.test_maes.append(self.test_mae)

            if self.verbose:
                print("Test RMSE:", np.round(self.test_rmse, 5))
            print("Test MAE:", np.round(self.test_mae, 5))
        
        self.n_iter_ += 1
    
    # used for sigma optimization
    def simpleloss(self,sigma,A, X,y):
        indices_X1, indices_X2 = self.indices_X1, self.indices_X2

        X1 = X[indices_X1]
        X2 = X[indices_X2]

        y1 = y[indices_X1]
        y2 = y[indices_X2]

        Xe = np.dot(X, A.T)
        X1e = Xe[indices_X1]
        X2e = Xe[indices_X2]

        kernel_constant = 1 / (1 * np.sqrt(2 * np.pi) * sigma)
        exponent_constant = 1 / (1 * sigma**2)

        dist1 = pairwise_distances(X1e, squared=True, n_jobs=-1)

        kernel1 = kernel_constant * np.exp(-dist1 * exponent_constant)

        n1 = len(X1)
        reg=self.krr_regularization
        H=kernel1+reg*np.eye(n1)
        lu, pivot = lu_factor(H, check_finite=False)
        alphas=lu_solve((lu,pivot), y1, check_finite=False)

        intercept=0
        dist2 = pairwise_distances(X2e, X1e, squared=True, n_jobs=-1)

        kernel2 = kernel_constant * np.exp(-dist2 * exponent_constant)

        yhat2 = kernel2 @ alphas + intercept

        ydiff2 = yhat2 - y2
        cost = (ydiff2**2).sum()
        v=(-kernel2/sigma + 2/sigma**3 * kernel2*dist2)@alphas
        w=(kernel1/sigma - 2/sigma**3 * kernel1*dist1)@alphas
        grads=2*ydiff2@(v+kernel2@lu_solve((lu,pivot),w))
        return cost, grads   
