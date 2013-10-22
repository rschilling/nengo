"""
This file contains functions concerned with solving for decoders
or other types of weight matrices.

The idea is that as we develop new methods to find decoders either more
quickly or with different constraints (e.g., L1-norm regularization),
the associated functions will be placed here.
"""

import numpy as np
import scipy as sp
import scipy.linalg

def sample_hypersphere(dimensions, n_samples, rng, surface=False):
    """Generate sample points from a hypersphere.

    Returns float array of sample points: dimensions x n_samples

    """
    samples = rng.randn(n_samples, dimensions)

    # normalize magnitude of sampled points to be of unit length
    norm = np.sum(samples * samples, axis=1)
    samples /= np.sqrt(norm)[:, np.newaxis]

    if surface:
        return samples

    # generate magnitudes for vectors from uniform distribution
    scale = rng.rand(n_samples, 1) ** (1.0 / dimensions)

    # scale sample points
    samples *= scale

    return samples

# def gen_eval_points(n, d, radius=1.0, method='uniform'):
#     """Various methods for randomly generating evaluation points.

#     TODO: this method is currently not used anywhere!
#     """

#     method = method.lower()
#     if method == 'uniform':
#         ### pick points from a Gaussian distribution, then normalize the
#         ### vector lengths so they are uniformly distributed
#         radii = np.random.uniform(low=0, high=1, size=n)
#         points = np.random.normal(size=(n,d))
#         points *= (radii / np.sqrt((points**2).sum(-1)))[:,None]
#     elif method in ['gaussian', 'normal']:
#         ### pick the points from a Gaussian distribution
#         points = np.random.normal(scale=radius, size=(n,d))
#     else:
#         ### TODO: other methods could allow users to specify different radii
#         ### for different dimensions.
#         raise ValueError("Unrecognized evaluation point generation method")

#     return points


# -- James and Terry arrived at this by eyeballing some graphs.
#    Not clear if this should be a constant at all, it
#    may depend on fn being estimated, number of neurons, etc...
DEFAULT_RCOND = 0.01

def lstsq(activities, targets, rng):
    import time
    t0 = time.time()
    weights, res, rank, s = np.linalg.lstsq(
        activities, targets, rcond=DEFAULT_RCOND)
        # activities, targets, rcond=1e-8)
    t1 = time.time()
    print "%s: %0.3f ms" % ('lstsq', 1e3*(t1 - t0))
    return weights.T

def lstsq_noisy(activities, targets, rng):
    sigma = activities.max() * 0.1
    noise = rng.normal(scale=sigma, size=activities.shape)
    activities = activities + noise
    weights, res, rank, s = np.linalg.lstsq(
        activities, targets, rcond=DEFAULT_RCOND)
    return weights.T

def lstsq_noisy_clip(activities, targets, rng):
    sigma = activities.max() * 0.1
    noise = rng.normal(scale=sigma, size=activities.shape)
    activities = (activities + noise).clip(0, np.inf)
    weights, res, rank, s = np.linalg.lstsq(
        activities, targets, rcond=DEFAULT_RCOND)
    return weights.T

def lstsq_noisy_perneuron(activities, targets, rng):
    sigma = activities.max(0) * 0.1
    noise = rng.normal(size=activities.shape) * sigma
    activities = activities + noise
    weights, res, rank, s = np.linalg.lstsq(
        activities, targets, rcond=DEFAULT_RCOND)
    return weights.T

def direct_L2(activities, targets, rng):
    sigma = activities.max() * 0.1
    # return _eigh(activities, targets, sigma).T
    return _cholesky(activities, targets, sigma).T
    # return _svd(activities, targets, sigma).T

def direct_L2_check(activities, targets, rng):
    import time
    sigma = activities.mean() * 0.1

    # solvers = [_cholesky1, _cholesky, _eigh]
    # solvers = [_cholesky, _cholesky1, _eigh]
    solvers = [_cholesky, _eigh, _svd]
    ref_weights = None
    for solver in solvers:
        t0 = time.time()
        weights = solver(activities, targets, sigma)
        t1 = time.time()
        print "%s: %0.3f ms" % (solver.__name__, 1e3*(t1 - t0))
        if ref_weights is None:
            ref_weights = weights
        else:
            assert np.allclose(ref_weights, weights)
    return weights.T

def direct_L2_low(activities, targets, rng):
    sigma = activities.mean() * 0.1
    return _eigh(activities, targets, sigma).T

def direct_L2_perneuron(activities, targets, rng):
    sigma = activities.max(0) * 0.1
    return _cholesky1(activities, targets, sigma).T

def _dropout_mask(shape, drop_rate, rng):
    m, n = shape
    drop_n = int(drop_rate * n)

    mask = np.ones(shape, dtype='bool')
    row_inds = np.arange(n)
    col_count = np.zeros(n, dtype='int32')
    for row in mask:
        p = (col_count.max() - col_count) + 1e-3
        inds = rng.choice(row_inds, size=drop_n, replace=False, p=p/p.sum())
        row[inds] = 0
        col_count[inds] += 1

    return mask

def dropout(activities, targets, rng):
    activ_sum = activities.sum(0)
    # mask = rng.normal(size=activities.shape) > 2
    # activities[mask] = 0

    if 0:
        skip = 2
        drop_rate = 1. / skip

        mask = np.ones(activities.shape, dtype='bool')
        j = 0
        for row in mask:
            row[j::skip] = 0
            j = np.mod(j + 1, skip)
    else:
        drop_rate = 0.1
        mask = _dropout_mask(activities.shape, drop_rate, rng)

    activities = activities * mask
    # weights, res, rank, s = np.linalg.lstsq(
    #     activities, targets, rcond=DEFAULT_RCOND)
    weights = _cholesky1(activities, targets, 0.1*activities.max(0))

    # weights = weights * (1 - drop_rate)

    activ_sum2 = activities.sum(0)
    weights = weights * (activ_sum2 / activ_sum)[:,None]

    return weights.T

def dropout_avg(activities, targets, rng):
    drop_rate = 0.25
    x = []
    for i in xrange(5):
        mask = _dropout_mask(activities.shape, drop_rate, rng)
        masked_acts = activities * mask
        weights, res, rank, s = np.linalg.lstsq(
            masked_acts, targets, rcond=DEFAULT_RCOND)
        weights = weights * (1 - drop_rate)
        x.append(weights)

    # skip = 4
    # drop_rate = 1. / skip
    # x = []
    # for i in xrange(skip):
    #     mask = np.ones(activities.shape, dtype='bool')
    #     j = i
    #     for row in mask:
    #         row[j::skip] = 0
    #         j = np.mod(j + 1, skip)

    #     masked_acts = activities * mask
    #     weights, res, rank, s = np.linalg.lstsq(
    #         masked_acts, targets, rcond=DEFAULT_RCOND)
    #     weights = weights * (1 - drop_rate)
    #     x.append(weights)

    return np.asarray(x).mean(0).T


def _regularizationParameter(sigma, Neval):
    return sigma**2 * Neval

def _eigh(A, b, sigma):
    """
    Solve the given linear system(s) using the eigendecomposition.
    """

    m, n = A.shape
    reglambda = _regularizationParameter(sigma, m)

    transpose = m < n
    if transpose:
        # substitution: x = A'*xbar, G*xbar = b where G = A*A' + lambda*I
        G = np.dot(A, A.T)
    else:
        # multiplication by A': G*x = A'*b where G = A'*A + lambda*I
        G = np.dot(A.T, A)
        b = np.dot(A.T, b)

    e, V = np.linalg.eigh(G)
    eInv = 1. / (e + reglambda)

    x = np.dot(V.T, b)
    x = eInv[:,None] * x if len(b.shape) > 1 else eInv * x
    x = np.dot(V, x)

    if transpose:
        x = np.dot(A.T, x)

    return x

def _cholesky(A, b, sigma):
    """
    Solve the given linear system(s) using the Cholesky decomposition
    """
    # sigma = 0.1 * activities.max()
    # weights = cholesky(activities, targets, sigma=sigma)

    m, n = A.shape
    reglambda = _regularizationParameter(sigma, m)

    transpose = m < n
    if transpose:
        # substitution: x = A'*xbar, G*xbar = b where G = A*A' + lambda*I
        G = np.dot(A, A.T)
        np.fill_diagonal(G, G.diagonal() + reglambda)
    else:
        # multiplication by A': G*x = A'*b where G = A'*A + lambda*I
        G = np.dot(A.T, A)# + reglambda * np.eye(n)
        np.fill_diagonal(G, G.diagonal() + reglambda)
        b = np.dot(A.T, b)

    # L = np.linalg.cholesky(G)
    # L = np.linalg.inv(L.T)
    # x = np.dot(L, np.dot(L.T, b))
    factor = sp.linalg.cho_factor(G, overwrite_a=True, check_finite=False)
    x = sp.linalg.cho_solve(factor, b, check_finite=False)
    if transpose:
        x = np.dot(A.T, x)
    return x

def _cholesky1(A, b, sigma):
    """
    Solve the given linear system(s) using the Cholesky decomposition
    """
    reglambda = _regularizationParameter(sigma, A.shape[0])

    G = np.dot(A.T, A)
    np.fill_diagonal(G, G.diagonal() + reglambda)
    b = np.dot(A.T, b)

    L = np.linalg.cholesky(G)
    L = np.linalg.inv(L.T)
    return np.dot(L, np.dot(L.T, b))

def _svd(A, b, sigma):
    """
    Solve the given linear system(s) using the singular value decomposition.
    """
    reglambda = _regularizationParameter(sigma, A.shape[0])

    U, S, V = np.linalg.svd(A, full_matrices=0)
    Sinv = S / (S**2 + reglambda)
    return np.dot(V.T * Sinv, np.dot(U.T, b))


# def _conjgrad_iters(A, b, x, maxiters=None, atol=1e-6, btol=1e-6, rtol=1e-6):
#     """
#     Perform conjugate gradient iterations
#     """

#     if maxiters is None:
#         maxiters = b.shape[0]

#     r = b - A(x)
#     p = r.copy()
#     rsold = np.dot(r,r)
#     normA = 0.0
#     normb = npl.norm(b)

#     for i in range(maxiters):
#         Ap = A(p)
#         alpha = rsold / np.dot(p,Ap)
#         x += alpha*p
#         r -= alpha*Ap

#         rsnew = np.dot(r,r)
#         beta = rsnew/rsold

#         if np.sqrt(rsnew) < rtol:
#             break

#         if beta < 1e-12: # no perceptible change in p
#             break

#         p = r + beta*p
#         rsold = rsnew

#     # print "normA est: %0.3e" % (normA)
#     return x, i+1

# def conjgrad(A, b, sigma, x0=None, maxiters=None, tol=1e-2):
#     """
#     Solve the given linear system using conjugate gradient
#     """

#     m,n = A.shape
#     D = b.shape[1] if len(b.shape) > 1 else 1
#     damp = m*sigma**2

#     G = lambda x: (np.dot(A.T,np.dot(A,x)) + damp*x)
#     b = np.dot(A.T,b)

#     ### conjugate gradient
#     x = np.zeros((n,D))
#     itns = []

#     for i in range(D):
#         xi = getcol(x0, i).copy() if x0 is not None else x[:,i]
#         bi = getcol(b, i)
#         xi, itn = _conjgrad_iters(G, bi, xi, maxiters=maxiters, rtol=tol*np.sqrt(m))
#         x[:,i] = xi
#         itns.append(itn)

#     if D == 1:
#         return x.ravel(), itns[0]
#     else:
#         return x, itns
