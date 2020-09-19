
#include "tensor_impl.hpp"

int hptt_num_threads = 1;

void tensor_transpose_impl(int ndim, size_t size, const int *perm,
                           const int *shape, const double *a, double *c,
                           const double alpha, const double beta) {
#ifdef _HAS_HPTT
    dTensorTranspose(perm, ndim, alpha, a, shape, nullptr, beta, c, nullptr,
                     hptt_num_threads, 1);
#else
    size_t oldacc[ndim], newacc[ndim];
    oldacc[ndim - 1] = 1;
    for (int i = ndim - 1; i >= 1; i--)
        oldacc[i - 1] = oldacc[i] * shape[perm[i]];
    for (int i = 0; i < ndim; i++)
        newacc[perm[i]] = oldacc[i];
    if (beta == 0)
        for (size_t i = 0; i < size; i++) {
            size_t j = 0, ii = i;
            for (int k = ndim - 1; k >= 0; k--)
                j += (ii % shape[k]) * newacc[k], ii /= shape[k];
            c[j] = alpha * a[i];
        }
    else
        for (size_t i = 0; i < size; i++) {
            size_t j = 0, ii = i;
            for (int k = ndim - 1; k >= 0; k--)
                j += (ii % shape[k]) * newacc[k], ii /= shape[k];
            c[j] = alpha * a[i] + beta * c[j];
        }
#endif
}

void tensordot_impl(const double *a, const int ndima, const ssize_t *na,
                    const double *b, const int ndimb, const ssize_t *nb,
                    const int nctr, const int *idxa, const int *idxb, double *c,
                    const double alpha, const double beta) {
    int outa[ndima - nctr], outb[ndimb - nctr];
    int a_free_dim = 1, b_free_dim = 1, ctr_dim = 1;
    set<int> idxa_set(idxa, idxa + nctr);
    set<int> idxb_set(idxb, idxb + nctr);
    for (int i = 0, ioa = 0; i < ndima; i++)
        if (!idxa_set.count(i))
            outa[ioa] = i, a_free_dim *= na[i], ioa++;
    for (int i = 0, iob = 0; i < ndimb; i++)
        if (!idxb_set.count(i))
            outb[iob] = i, b_free_dim *= nb[i], iob++;
    int trans_a = 0, trans_b = 0;

    int ctr_idx[nctr];
    for (int i = 0; i < nctr; i++)
        ctr_idx[i] = i, ctr_dim *= na[idxa[i]];
    sort(ctr_idx, ctr_idx + nctr,
         [idxa](int a, int b) { return idxa[a] < idxa[b]; });

    // checking whether permute is necessary
    if (idxa[ctr_idx[0]] == 0 && idxa[ctr_idx[nctr - 1]] == nctr - 1)
        trans_a = 1;
    else if (idxa[ctr_idx[0]] == ndima - nctr &&
             idxa[ctr_idx[nctr - 1]] == ndima - 1)
        trans_a = -1;

    if (idxb[ctr_idx[0]] == 0 && idxb[ctr_idx[nctr - 1]] == nctr - 1)
        trans_b = 1;
    else if (idxb[ctr_idx[0]] == ndimb - nctr &&
             idxb[ctr_idx[nctr - 1]] == ndimb - 1)
        trans_b = -1;

    // permute or reshape
    double *new_a = (double *)a, *new_b = (double *)b;
    if (trans_a == 0) {
        vector<int> perm_a(ndima), shape_a(ndima);
        size_t size_a = 1;
        for (int i = 0; i < ndima; i++)
            shape_a[i] = na[i], size_a *= na[i];
        for (int i = 0; i < nctr; i++)
            perm_a[i] = idxa[ctr_idx[i]];
        for (int i = nctr; i < ndima; i++)
            perm_a[i] = outa[i - nctr];
        new_a = new double[size_a];
        tensor_transpose_impl(ndima, size_a, perm_a.data(), shape_a.data(), a,
                              new_a, 1.0, 0.0);
        trans_a = 1;
    }

    if (trans_b == 0) {
        vector<int> perm_b(ndimb), shape_b(ndimb);
        size_t size_b = 1;
        for (int i = 0; i < ndimb; i++)
            shape_b[i] = nb[i], size_b *= nb[i];
        for (int i = 0; i < nctr; i++)
            perm_b[i] = idxb[ctr_idx[i]];
        for (int i = nctr; i < ndimb; i++)
            perm_b[i] = outb[i - nctr];
        new_b = new double[size_b];
        tensor_transpose_impl(ndimb, size_b, perm_b.data(), shape_b.data(), b,
                              new_b, 1.0, 0.0);
        trans_b = 1;
    }

    // n == a-free, m == b-free, k = cont
    // parameter order : m, n, k
    // trans == N -> mat = m x k (fort) k x m (c++)
    // trans == N -> mat = k x n (fort) n x k (c++)
    // matc = m x n (fort) n x m (c++)

    int ldb = trans_b == 1 ? b_free_dim : ctr_dim;
    int lda = trans_a == -1 ? ctr_dim : a_free_dim;
    int ldc = b_free_dim;
    dgemm(trans_b == 1 ? "n" : "t", trans_a == -1 ? "n" : "t", &b_free_dim,
          &a_free_dim, &ctr_dim, &alpha, new_b, &ldb, new_a, &lda, &beta, c,
          &ldc);

    if (new_a != a)
        delete[] new_a;
    if (new_b != b)
        delete[] new_b;
}
