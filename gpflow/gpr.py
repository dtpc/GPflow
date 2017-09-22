# Copyright 2016 James Hensman, Valentine Svensson, alexggmatthews, fujiisoup
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import absolute_import
import tensorflow as tf
import numpy as np
from .model import GPModel
from .densities import multivariate_normal
from .mean_functions import Zero
from . import likelihoods
from .param import DataHolder
from ._settings import settings
float_type = settings.dtypes.float_type


class GPR(GPModel):
    """
    Gaussian Process Regression.

    This is a vanilla implementation of GP regression with a Gaussian
    likelihood.  Multiple columns of Y are treated independently.

    The log likelihood i this models is sometimes referred to as the 'marginal log likelihood', and is given by

    .. math::

       \\log p(\\mathbf y \\,|\\, \\mathbf f) = \\mathcal N\\left(\\mathbf y\,|\, 0, \\mathbf K + \\sigma_n \\mathbf I\\right)
    """
    def __init__(self, X, Y, kern, mean_function=None, name='name',
                 obj_func='marginal_likelihood'):
        """
        X is a data matrix, size N x D
        Y is a data matrix, size N x R
        kern, mean_function are appropriate GPflow objects
        """
        likelihood = likelihoods.Gaussian()
        X = DataHolder(X, on_shape_change='pass')
        Y = DataHolder(Y, on_shape_change='pass')
        GPModel.__init__(self, X, Y, kern, likelihood, mean_function, name)
        self.num_latent = Y.shape[1]
        valid_obj_funcs = ('marginal_likelihood', 'neg_cv_loss')
        if obj_func not in valid_obj_funcs:
            raise ValueError("obj_func '{}' not in '{}'".format(obj_func, valid_obj_funcs))

        self.obj_func = obj_func

    def build_likelihood(self):
        """
        Construct a tensorflow function to compute the likelihood.

            \log p(Y | theta).

        """
        K = self.kern.K(self.X) + tf.eye(tf.shape(self.X)[0], dtype=float_type) * self.likelihood.variance
        L = tf.cholesky(K)
        m = self.mean_function(self.X)

        return multivariate_normal(self.Y, m, L)

    def build_predict(self, Xnew, full_cov=False):
        fmean, fvar = self.build_predict_xy(self.X, self.Y, Xnew,
                                            full_cov=full_cov)
        return fmean, fvar

    def build_predict_xy(self, X, Y, Xnew, full_cov=False):
        """
        Xnew is a data matrix, point at which we want to predict

        This method computes

            p(F* | Y )

        where F* are points on the GP at Xnew, Y are noisy observations at X.

        """
        Kx = self.kern.K(X, Xnew)
        K = self.kern.K(X) + tf.eye(tf.shape(X)[0], dtype=float_type) * self.likelihood.variance
        L = tf.cholesky(K)
        A = tf.matrix_triangular_solve(L, Kx, lower=True)
        V = tf.matrix_triangular_solve(L, Y - self.mean_function(X))
        fmean = tf.matmul(A, V, transpose_a=True) + self.mean_function(Xnew)
        if full_cov:
            fvar = self.kern.K(Xnew) - tf.matmul(A, A, transpose_a=True)
            shape = tf.stack([1, 1, tf.shape(Y)[1]])
            fvar = tf.tile(tf.expand_dims(fvar, 2), shape)
        else:
            fvar = self.kern.Kdiag(Xnew) - tf.reduce_sum(tf.square(A), 0)
            fvar = tf.tile(tf.reshape(fvar, (-1, 1)), [1, tf.shape(Y)[1]])
        return fmean, fvar

    def neg_cv_loss(self):
        """ Build the Leave-One-Out log predictive probability loss
            function
        """
        print('CV_LOSS')

        def density_i(i, n):
            ind = slice(i, i+n)
            Xi = tf.constant(X[ind, :])
            Yi = tf.constant(Y[ind, :])
            Xjs = tf.constant(np.delete(X, ind, axis=0))
            Yjs = tf.constant(np.delete(Y, ind, axis=0))
            fmean, fvar = self.build_predict_xy(Xjs, Yjs, Xi, full_cov=False)
            logp_i = self.likelihood.predict_density(fmean, fvar, Yi)
            return logp_i

        # k-folds instead of LOO, for speed
        k = 10
        n = self.X.shape[0] // k

        X = self.X.value
        Y = self.Y.value

        with self.tf_mode():
            logps = [density_i(i, n) for i in range(k)]
            f = tf.reduce_sum(logps)
            g, = tf.gradients(f, self._free_vars)

        return f, g