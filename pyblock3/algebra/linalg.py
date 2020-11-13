
"""
Linear algebra functions at module level.
Including some functions not available from numpy.linalg.
"""

from .core import method_alias
import numpy as np

@method_alias('lq')
def lq(a, *args, **kwargs):
    return NotImplemented


@method_alias('qr')
def qr(a, *args, **kwargs):
    return NotImplemented


@method_alias('norm')
def norm(a, *args, **kwargs):
    return NotImplemented


@method_alias('left_svd')
def left_svd(a, *args, **kwargs):
    return NotImplemented


@method_alias('right_svd')
def right_svd(a, *args, **kwargs):
    return NotImplemented


def truncate_svd(l, s, r, *args, **kwargs):
    if l.__class__ != s.__class__:
        return l.__class__.truncate_svd(l, s, r, *args, **kwargs)
    elif r.__class__ != s.__class__:
        return r.__class__.truncate_svd(l, s, r, *args, **kwargs)
    elif hasattr(s.__class__, 'truncate_svd'):
        return s.__class__.truncate_svd(l, s, r, *args, **kwargs)
    else:
        return NotImplemented


@method_alias('canonicalize')
def canonicalize(a, *args, **kwargs):
    return NotImplemented


@method_alias('left_canonicalize')
def left_canonicalize(a, *args, **kwargs):
    return NotImplemented


@method_alias('right_canonicalize')
def right_canonicalize(a, *args, **kwargs):
    return NotImplemented


@method_alias('compress')
def compress(a, *args, **kwargs):
    return NotImplemented

def _olsen_precondition(q, c, ld, diag):
    """Olsen precondition."""
    t = c.copy()
    mask = np.abs(ld - diag) > 1E-12
    t[mask] /= ld - diag[mask]
    numerator = np.dot(t, q)
    denominator = np.dot(c, t)
    q += (-numerator / denominator) * c
    q[mask] /= ld - diag[mask]

def _precondition(r, diag):
    """z = r / diag."""
    z = r.copy()
    if diag is not None:
        mask = np.abs(diag) > 1E-12
        z[mask] /= diag[mask]
    return z

# E.R. Davidson, J. Comput. Phys. 17 (1), 87-94 (1975).
def davidson(a, b, k, max_iter=500, conv_thrd=1E-7, deflation_min_size=2, deflation_max_size=30, iprint=False):
    """
    Davidson diagonalization.

    Args:
        a : Matrix
            The matrix to diagonalize.
        b : list(Vector)
            The initial guesses for eigenvectors.

    Kwargs:
        max_iter : int
            Maximal number of davidson iteration.
        conv_thrd : float
            Convergence threshold for squared norm of eigenvector.
        deflation_min_size : int
            Sub-space size after deflation.
        deflation_max_size : int
            Maximal sub-space size before deflation.
        iprint : bool
            Indicate whether davidson iteration information should be printed.
    
    Returns:
        ld : list(float)
            List of eigenvalues.
        b : list(Vector)
            List of eigenvectors.
    """
    assert len(b) == k
    if deflation_min_size < k:
        deflation_min_size = k
    aa = a.diag() if hasattr(a, "diag") else None
    for i in range(k):
        for j in range(i):
            b[i] += -np.dot(b[j], b[i]) * b[j]
        b[i] /= np.linalg.norm(b[i])
    sigma = [None] * k
    q = b[0]
    l = k
    ck = 0
    msig = 0
    m = l
    xiter = 0
    while xiter < max_iter:
        xiter += 1
        for i in range(msig, m):
            sigma[i] = a @ b[i]
            msig += 1
        atilde = np.zeros((m, m))
        for i in range(m):
            for j in range(i + 1):
                atilde[i, j] = np.dot(b[i], sigma[j])
                atilde[j, i] = atilde[i, j]
        ld, alpha = np.linalg.eigh(atilde)
        # b[1:m] = np.dot(b[:], alpha[:, 1:m])
        tmp = [ib.copy() for ib in b[:m]]
        for j in range(m):
            b[j] *= alpha[j, j]
        for j in range(m):
            for i in range(m):
                if i != j:
                    b[j] += alpha[i, j] * tmp[i]
        # sigma[1:m] = np.dot(sigma[:], alpha[:, 1:m])
        for i in range(m):
            tmp[i] = sigma[i].copy()
        for j in range(m):
            sigma[j] *= alpha[j, j]
        for j in range(m):
            for i in range(m):
                if i != j:
                    sigma[j] += alpha[i, j] * tmp[i]
        for i in range(ck):
            q = sigma[i].copy()
            q += (-ld[i]) * b[i]
            qq = np.dot(q, q)
            if np.sqrt(qq) >= conv_thrd:
                ck = i
                break
        # q = sigma[ck] - b[ck] * ld[ck]
        q = sigma[ck].copy()
        q += (-ld[ck]) * b[ck]
        qq = np.dot(q, q)
        if iprint:
            print("%5d %5d %5d %15.8f %9.2E" % (xiter, m, ck, ld[ck], qq))
        
        if aa is not None:
            _olsen_precondition(q, b[ck], ld[ck], aa)

        if np.sqrt(qq) < conv_thrd:
            ck += 1
            if ck == k:
                break
        else:
            if m >= deflation_max_size:
                m = deflation_min_size
                msig = deflation_min_size
            for j in range(m):
                q += (-np.dot(b[j], q)) * b[j]
            q /= np.linalg.norm(q)
            
            if m >= len(b):
                b.append(None)
                sigma.append(None)
            b[m] = q
            m += 1
        
        if xiter == max_iter:
            raise RuntimeError("Only %d converged!" % ck)
    
    return ld[:ck], b[:ck], xiter

def conjugate_gradient(a, x, b, max_iter=500, conv_thrd=1E-7, iprint=False):
    aa = a.diag() if hasattr(a, "diag") else None
    r = -(a @ x) + b
    p = _precondition(r, aa)
    error = np.dot(p, r)
    if np.sqrt(error) < conv_thrd:
        func = np.dot(x, b)
        if iprint:
            print("%5d %15.8f %9.2E" % (0, func, error))
        return func, x, 1
    old_error = error
    xiter = 0
    while xiter < max_iter:
        xiter += 1
        hp = a @ p
        alpha = old_error / np.dot(p, hp)
        x += alpha * p
        r -= alpha * hp
        z = _precondition(r, aa)
        error = np.dot(z, r)
        func = np.dot(x, b)
        if iprint:
            print("%5d %15.8f %9.2E" % (xiter, func, error))
        if np.sqrt(error) < conv_thrd:
            break
        else:
            beta = error / old_error
            old_error = error
            p[:] = beta * p + z
    if xiter == max_iter:
        raise RuntimeError("Error : linear solver (cg) not converged!")
    return func, x, xiter + 1
