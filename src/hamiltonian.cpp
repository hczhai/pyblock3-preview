
#include "hamiltonian.hpp"
#include "flat_sparse.hpp"
#include <algorithm>
#include <cstring>
#include <iomanip>
#include <numeric>
#include <unordered_map>

inline SZLong from_op(int32_t op, const int32_t *orb_sym, const int32_t m_site,
                      const int32_t m_op) noexcept {
    int n = op / m_op ? -1 : 1;
    int twos = (op % m_site) ^ (op / m_op) ? -1 : 1;
    int pg = orb_sym[(op % m_op) / m_site];
    return SZLong(n, twos, pg);
}

inline size_t op_hash(const int32_t *terms, int n,
                      const int32_t init = 0) noexcept {
    size_t h = (size_t)init;
    for (int i = 0; i < n; i++)
        h ^= (size_t)terms[i] + 0x9E3779B9 + (h << 6) + (h >> 2);
    return h;
}

typedef tuple<py::array_t<uint32_t>, py::array_t<uint32_t>,
              py::array_t<uint32_t>>
    op_skeleton;

void op_matmul(const op_skeleton &ska, const op_skeleton &skb,
               const op_skeleton &skc, const double *pa, const double *pb,
               double *pc) {
    int na = get<0>(ska).shape()[0], nb = get<0>(skb).shape()[0],
        nc = get<0>(skc).shape()[0];
    const uint32_t *pqa = get<0>(ska).data(), *pqb = get<0>(skb).data(),
                   *pqc = get<0>(skc).data();
    const uint32_t *psha = get<1>(ska).data(), *pshb = get<1>(skb).data(),
                   *pshc = get<1>(skc).data();
    const uint32_t *pia = get<2>(ska).data(), *pib = get<2>(skb).data(),
                   *pic = get<2>(skc).data();
    const double scale = 1.0, cfactor = 1.0;
    for (int ic = 0; ic < nc; ic++)
        for (int ia = 0; ia < na; ia++) {
            if (pqa[ia * 2 + 0] != pqc[ic * 2 + 0])
                continue;
            for (int ib = 0; ib < nb; ib++) {
                if (pqb[ib * 2 + 1] != pqc[ic * 2 + 1] ||
                    pqb[ib * 2 + 0] != pqa[ia * 2 + 1])
                    continue;
                int m = psha[ia * 2 + 0], n = pshb[ib * 2 + 1],
                    k = pshb[ib * 2 + 0];
                dgemm("N", "N", &n, &m, &k, &scale, pb + pib[ib], &n,
                      pa + pia[ia], &k, &cfactor, pc + pic[ic], &n);
            }
        }
}

vector<tuple<py::array_t<uint32_t>, py::array_t<uint32_t>, py::array_t<double>,
             py::array_t<uint32_t>>>
build_mpo(py::array_t<int32_t> orb_sym, py::array_t<double> h_values,
          py::array_t<int32_t> h_terms, double cutoff, int max_bond_dim) {
    const int m_site = 2, m_op = 16384;
    int n_sites = (int)orb_sym.shape()[0], n_values = (int)h_values.shape()[0];
    int n_terms = (int)h_terms.shape()[0], term_len = (int)h_terms.shape()[1];
    assert(n_terms == n_values);
    vector<SZLong> left_q = {SZLong(0, 0, 0)};
    unordered_map<uint32_t, uint32_t> info_l, info_r;
    info_l[from_sz(left_q[0])] = 1;
    // terms
    vector<int32_t> term_sorted(n_terms * term_len);
    // length of each term; starting index of each term
    vector<int> term_l(n_terms), term_i(n_terms, 0);
    // index of current terms
    vector<vector<int>> cur_terms(1);
    cur_terms[0].resize(n_terms);
    // multiplying left matrix
    vector<vector<double>> cur_values(1);
    cur_values[0].resize(n_terms);
    const int32_t *pt = h_terms.data(), *porb = orb_sym.data();
    const double *pv = h_values.data();
    const ssize_t hsi = h_terms.strides()[0] / sizeof(uint32_t),
                  hsj = h_terms.strides()[1] / sizeof(uint32_t);
    vector<int> term_site(term_len);
    // pre-processing
    for (int it = 0, ix; it < n_terms; it++) {
        SZLong q(0, 0, 0);
        for (ix = 0; ix < term_len; ix++) {
            int32_t op = pt[it * hsi + ix * hsj];
            if (op == -1)
                break;
            q = q + from_op(op, porb, m_site, m_op);
            term_site[ix] = (op % m_op) / m_site;
            term_sorted[it * term_len + ix] = op;
        }
        if (q != SZLong(0, 0, 0)) {
            cout << "Hamiltonian term #" << it
                 << " has a non-vanishing q: " << q << endl;
            abort();
        }
        term_l[it] = ix;
        cur_terms[0][it] = it;
        int ffactor = 1;
        for (int i = 0; i < ix; i++)
            for (int j = i + 1; j < ix; j++)
                if (term_site[i] > term_site[j])
                    ffactor = -ffactor;
        cur_values[0][it] = pv[it] * ffactor;
        stable_sort(term_sorted.data() + it * term_len,
                    term_sorted.data() + it * term_len + ix,
                    [m_op, m_site](int32_t i, int32_t j) {
                        return (i % m_op) / m_site < (j % m_op) / m_site;
                    });
    }
    vector<tuple<py::array_t<uint32_t>, py::array_t<uint32_t>,
                 py::array_t<double>, py::array_t<uint32_t>>>
        rr(n_sites * 2);
    // do svd from left to right
    // time complexity: O(KDLN(log N))
    // K: n_sites, D: max_bond_dim, L: term_len, N: n_terms
    unordered_map<SZLong, int> q_map;
    vector<unordered_map<size_t, vector<pair<pair<int, int>, int>>>> map_ls;
    vector<unordered_map<size_t, vector<pair<int, int>>>> map_rs;
    vector<vector<pair<pair<int, int>, double>>> mats;
    vector<pair<int, int>> nms;
    vector<int> cur_term_i(n_terms, -1);
    py::array_t<int32_t> perm(vector<ssize_t>{4});
    perm.mutable_data()[0] = 0, perm.mutable_data()[1] = 2,
    perm.mutable_data()[2] = 3, perm.mutable_data()[3] = 1;
    const int32_t *pperm = perm.data();
    for (int ii = 0; ii < n_sites; ii++) {
        cout << "MPO site" << setw(4) << ii << " / " << n_sites << endl;
        q_map.clear();
        map_ls.clear();
        map_rs.clear();
        mats.clear();
        nms.clear();
        info_r.clear();
        unordered_map<uint32_t, uint32_t> basis;
        basis[from_sz(SZLong(0, 0, 0))] = 1;
        basis[from_sz(SZLong(1, 1, porb[ii]))] = 1;
        basis[from_sz(SZLong(1, -1, porb[ii]))] = 1;
        basis[from_sz(SZLong(2, 0, 0))] = 1;
        for (int ip = 0; ip < (int)cur_values.size(); ip++) {
            SZLong qll = left_q[ip];
            for (int ic = 0; ic < (int)cur_terms[ip].size(); ic++) {
                if (abs(cur_values[ip][ic]) < cutoff)
                    continue;
                int it = cur_terms[ip][ic], ik = term_i[it], k = ik,
                    kmax = term_l[it];
                int itt = it * term_len;
                for (; k < kmax && (term_sorted[itt + k] % m_op) / m_site <= ii;
                     k++)
                    ;
                cur_term_i[it] = k;
                size_t hl = op_hash(term_sorted.data() + itt + ik, k - ik, ip);
                size_t hr = op_hash(term_sorted.data() + itt + k, kmax - k);
                SZLong ql = qll;
                for (int i = ik; i < k; i++)
                    ql = ql + from_op(term_sorted[itt + i], porb, m_site, m_op);
                if (q_map.count(ql) == 0) {
                    q_map[ql] = (int)q_map.size();
                    map_ls.emplace_back();
                    map_rs.emplace_back();
                    mats.emplace_back();
                    nms.push_back(make_pair(0, 0));
                }
                int iq = q_map.at(ql), il = -1, ir = -1;
                int &nml = nms[iq].first, &nmr = nms[iq].second;
                auto &mpl = map_ls[iq];
                auto &mpr = map_rs[iq];
                if (mpl.count(hl)) {
                    int iq = 0;
                    auto &vq = mpl.at(hl);
                    for (; iq < vq.size(); iq++) {
                        int vip = vq[iq].first.first, vit = vq[iq].first.second;
                        int vitt = vit * term_len;
                        int vik = term_i[vit], vk = cur_term_i[vit];
                        if (vip == ip && vk - vik == k - ik &&
                            equal(term_sorted.data() + vitt + vik,
                                  term_sorted.data() + vitt + vk,
                                  term_sorted.data() + itt + ik))
                            break;
                    }
                    if (iq == (int)vq.size())
                        vq.push_back(make_pair(make_pair(ip, it), il = nml++));
                    else
                        il = vq[iq].second;
                } else
                    mpl[hl].push_back(make_pair(make_pair(ip, it), il = nml++));
                if (mpr.count(hr)) {
                    int iq = 0;
                    auto &vq = mpr.at(hr);
                    for (; iq < vq.size(); iq++) {
                        int vit = vq[iq].first, vitt = vit * term_len;
                        int vkmax = term_l[vit], vk = cur_term_i[vit];
                        if (vkmax - vk == kmax - k &&
                            equal(term_sorted.data() + vitt + vk,
                                  term_sorted.data() + vitt + vkmax,
                                  term_sorted.data() + itt + k))
                            break;
                    }
                    if (iq == (int)vq.size())
                        vq.push_back(make_pair(it, ir = nmr++));
                    else
                        ir = vq[iq].second;
                } else
                    mpr[hr].push_back(make_pair(it, ir = nmr++));
                mats[iq].push_back(
                    make_pair(make_pair(il, ir), cur_values[ip][ic]));
            }
        }
        vector<array<vector<double>, 3>> svds(q_map.size());
        vector<SZLong> qs(q_map.size());
        int s_kept_total = 0, nr_total = 0;
        for (auto &mq : q_map) {
            int iq = mq.second;
            qs[iq] = mq.first;
            auto &matvs = mats[iq];
            auto &nm = nms[iq];
            int szl = nm.first, szr = nm.second;
            int szm = max_bond_dim == -2
                          ? szl
                          : (max_bond_dim == -3 ? szr : min(szl, szr));
            int lwork = max(szl, szr) * 34, info;
            vector<double> mat((size_t)szl * szr, 0), work(lwork);
            svds[iq][0].resize((size_t)szm * szl);
            svds[iq][1].resize(szm);
            svds[iq][2].resize((size_t)szm * szr);
            int s_kept = 1;
            // NC
            if (max_bond_dim == -2) {
                memset(svds[iq][0].data(), 0,
                       sizeof(double) * svds[iq][0].size());
                memset(svds[iq][2].data(), 0,
                       sizeof(double) * svds[iq][2].size());
                for (auto &lrv : matvs)
                    svds[iq][2][lrv.first.first * szr + lrv.first.second] +=
                        lrv.second;
                for (int i = 0; i < szm; i++)
                    svds[iq][0][i * szm + i] = svds[iq][1][i] = 1;
                s_kept = szm;
            } else if (max_bond_dim == -3) {
                memset(svds[iq][0].data(), 0,
                       sizeof(double) * svds[iq][0].size());
                memset(svds[iq][2].data(), 0,
                       sizeof(double) * svds[iq][2].size());
                for (auto &lrv : matvs)
                    svds[iq][0][lrv.first.first * szr + lrv.first.second] +=
                        lrv.second;
                for (int i = 0; i < szm; i++)
                    svds[iq][2][i * szr + i] = svds[iq][1][i] = 1;
                s_kept = szm;
            } else {
                for (auto &lrv : matvs)
                    mat[lrv.first.first * szr + lrv.first.second] += lrv.second;
                dgesvd("S", "S", &szr, &szl, mat.data(), &szr,
                       svds[iq][1].data(), svds[iq][2].data(), &szr,
                       svds[iq][0].data(), &szm, work.data(), &lwork, &info);
                for (int i = 1; i < szm; i++)
                    if (svds[iq][1][i] > cutoff)
                        s_kept++;
                    else
                        break;
                if (max_bond_dim > 1)
                    s_kept = min(s_kept, max_bond_dim);
                svds[iq][1].resize(s_kept);
            }
            info_r[from_sz(mq.first)] = s_kept;
            s_kept_total += s_kept;
            nr_total += szr;
        }

        // [optional optimization] ip can be inner loop
        // [optional optimization] better set basis as input; cannot remove
        // orb_sym currently just construct basis from orb_sym use skelton and
        // info_l; info_r to build skelton; physical indices at the end
        // skeleton: +-+-; dq = 0
        // skeleton guarentees that right indices are contiguous
        vector<unordered_map<uint32_t, uint32_t>> infos = {info_l, info_r,
                                                           basis, basis};
        auto skl = flat_sparse_tensor_skeleton(infos, "+-+-",
                                               from_sz(SZLong(0, 0, 0)));
        // separate odd and even
        int n_odd = 0, n_total = get<0>(skl).shape()[0];
        ssize_t size_odd = 0;
        vector<bool> skf(n_total, false);
        const uint32_t *psklqs = get<0>(skl).data();
        const uint32_t *psklshs = get<1>(skl).data();
        const uint32_t *psklis = get<2>(skl).data();
        const ssize_t sklqi = get<0>(skl).strides()[0] / sizeof(uint32_t),
                      sklqj = get<0>(skl).strides()[1] / sizeof(uint32_t);
        for (int i = 0; i < n_total; i++)
            if (to_sz(psklqs[i * sklqi + 2 * sklqj]).is_fermion() !=
                to_sz(psklqs[i * sklqi + 3 * sklqj]).is_fermion())
                n_odd++, skf[i] = true, size_odd += psklis[i + 1] - psklis[i];
        int n_even = n_total - n_odd;

        ssize_t size_even = psklis[n_total] - size_odd;
        auto &rodd = rr[ii * 2], &reven = rr[ii * 2 + 1];
        auto &oqs = get<0>(rodd), &oshs = get<1>(rodd), &oi = get<3>(rodd);
        auto &eqs = get<0>(reven), &eshs = get<1>(reven), &ei = get<3>(reven);
        oqs = py::array_t<uint32_t>(vector<ssize_t>{n_odd, 4});
        oshs = py::array_t<uint32_t>(vector<ssize_t>{n_odd, 4});
        oi = py::array_t<uint32_t>(vector<ssize_t>{n_odd + 1});
        eqs = py::array_t<uint32_t>(vector<ssize_t>{n_even, 4});
        eshs = py::array_t<uint32_t>(vector<ssize_t>{n_even, 4});
        auto odata = py::array_t<double>(vector<ssize_t>{size_odd});
        auto edata = py::array_t<double>(vector<ssize_t>{size_even});
        ei = py::array_t<uint32_t>(vector<ssize_t>{n_even + 1});
        uint32_t *poqs = oqs.mutable_data(), *poshs = oshs.mutable_data(),
                 *poi = oi.mutable_data();
        uint32_t *peqs = eqs.mutable_data(), *peshs = eshs.mutable_data(),
                 *pei = ei.mutable_data();
        double *po = odata.mutable_data(), *pe = edata.mutable_data();
        memset(po, 0, sizeof(double) * size_odd);
        memset(pe, 0, sizeof(double) * size_even);
        poi[0] = pei[0] = 0;
        // map<uint64_t (uint32_t << 32 + uint32_t), data index>
        unordered_map<uint64_t, size_t> rdt_map;
        for (int i = 0, iodd = 0, ieven = 0; i < n_total; i++)
            if (skf[i]) {
                for (int j = 0; j < 4; j++) {
                    poqs[iodd * 4 + j] = psklqs[i * sklqi + j * sklqj];
                    poshs[iodd * 4 + j] = psklshs[i * sklqi + j * sklqj];
                }
                poi[iodd + 1] = poi[iodd] + psklis[i + 1] - psklis[i];
                uint64_t pk =
                    ((uint64_t)poqs[iodd * 4 + 0] << 32) | poqs[iodd * 4 + 1];
                if (rdt_map.count(pk) == 0)
                    rdt_map[pk] = poi[iodd];
                iodd++;
            } else {
                for (int j = 0; j < 4; j++) {
                    peqs[ieven * 4 + j] = psklqs[i * sklqi + j * sklqj];
                    peshs[ieven * 4 + j] = psklshs[i * sklqi + j * sklqj];
                }
                pei[ieven + 1] = pei[ieven] + psklis[i + 1] - psklis[i];
                uint64_t pk =
                    ((uint64_t)peqs[ieven * 4 + 0] << 32) | peqs[ieven * 4 + 1];
                if (rdt_map.count(pk) == 0)
                    rdt_map[pk] = pei[ieven];
                ieven++;
            }
        // single term matrix multiplication better use operator mul plan
        // (create one) adding matrices; multiply data; add to operator matrix
        // must be matrices of the same shape been added
        // so first get matrix for all single term -> a vector of length il
        // then just sum data (dgemm)
        // prepare on-site operators
        unordered_map<uint32_t, op_skeleton> sk_map;
        vector<unordered_map<uint32_t, uint32_t>> op_infos = {basis, basis};
        vector<uint32_t> sk_qs = {
            from_sz(SZLong(0, 0, 0)), from_sz(SZLong(1, 1, porb[ii])),
            from_sz(SZLong(1, -1, porb[ii])), from_sz(SZLong(-1, -1, porb[ii])),
            from_sz(SZLong(-1, 1, porb[ii]))};
        for (auto &k : sk_qs)
            sk_map[k] = flat_sparse_tensor_skeleton(op_infos, "+-", k);
        // data for on-site operators
        vector<ssize_t> op_sh(1, 2), op_ish(1, 4);
        unordered_map<uint32_t, vector<double>> dt_map;
        dt_map[sk_qs[0]].resize(4);
        for (int i = 1; i <= 4; i++)
            dt_map[sk_qs[i]].resize(2);
        double *pi = dt_map.at(sk_qs[0]).data();
        double *pca = dt_map.at(sk_qs[1]).data(),
               *pcb = dt_map.at(sk_qs[2]).data();
        double *pda = dt_map.at(sk_qs[3]).data(),
               *pdb = dt_map.at(sk_qs[4]).data();
        pi[0] = pi[1] = pi[2] = pi[3] = 1.0;
        pca[0] = pcb[0] = pca[1] = pda[0] = pdb[0] = pda[1] = 1.0;
        pcb[1] = -1.0, pdb[1] = -1.0;
        const int incx = 1;
        vector<pair<int, int>> ip_idx(cur_values.size());
        ip_idx[0] = make_pair(0, 0);
        for (int ip = 1; ip < (int)cur_values.size(); ip++) {
            if (left_q[ip] == left_q[ip - 1])
                ip_idx[ip] =
                    make_pair(ip_idx[ip - 1].first, ip_idx[ip - 1].second + 1);
            else
                ip_idx[ip] = make_pair(ip_idx[ip - 1].first + 1, 0);
        }
        // sum and multiplication
        for (auto &mq : q_map) {
            int iq = mq.second;
            SZLong q = qs[iq];
            auto &matvs = mats[iq];
            auto &mpl = map_ls[iq];
            auto &nm = nms[iq];
            int rszm = (int)svds[iq][1].size(), szl = nm.first, szr = nm.second;
            int szm = max_bond_dim == -2
                          ? szl
                          : (max_bond_dim == -3 ? szr : min(szl, szr));
            vector<vector<double>> reprs(szl);
            vector<uint32_t> repr_q(szl);
            for (auto &vls : mpl)
                for (auto &vl : vls.second) {
                    int il = vl.second, ip = vl.first.first,
                        it = vl.first.second;
                    int itt = it * term_len;
                    int ik = term_i[it], k = cur_term_i[it];
                    if (ik == k) {
                        reprs[il].resize(4);
                        memcpy(reprs[il].data(), pi, sizeof(double) * 4);
                        repr_q[il] = sk_qs[0];
                    } else {
                        SZLong qi =
                            from_op(term_sorted[itt + ik], porb, m_site, m_op);
                        vector<double> p = dt_map.at(from_sz(qi));
                        for (int i = ik + 1; i < k; i++) {
                            SZLong qx = from_op(term_sorted[itt + i], porb,
                                                m_site, m_op);
                            uint32_t fqk = from_sz(qi + qx), fqx = from_sz(qx),
                                     fqi = from_sz(qi);
                            if (sk_map.count(fqk) == 0)
                                sk_map[fqk] = flat_sparse_tensor_skeleton(
                                    op_infos, "+-", fqk);
                            auto &skt = sk_map.at(fqk);
                            vector<double> pp(
                                get<2>(skt).data()[get<2>(skt).size() - 1], 0);
                            op_matmul(sk_map.at(fqi), sk_map.at(fqx), skt,
                                      p.data(), dt_map.at(fqx).data(),
                                      pp.data());
                            p = pp;
                            qi = qi + qx;
                        }
                        reprs[il] = p;
                        repr_q[il] = from_sz(qi);
                    }
                }
            for (int ir = 0; ir < rszm; ir++) {
                for (auto &vls : mpl)
                    for (auto &vl : vls.second) {
                        int il = vl.second, ip = vl.first.first;
                        int ipp = ip_idx[ip].first, ipr = ip_idx[ip].second;
                        uint64_t ql = from_sz(left_q[ip]), qr = from_sz(q);
                        int npr = (int)info_l.at(ql);
                        double *pr =
                            left_q[ip].is_fermion() == q.is_fermion() ? pe : po;
                        double *term_data = reprs[il].data();
                        int term_size = (int)reprs[il].size();
                        if (term_size == 0)
                            continue;
                        size_t pir = rdt_map.at((ql << 32) | qr);
                        op_skeleton &sk_repr = sk_map.at(repr_q[il]);
                        // singular values multiplies to left
                        double factor =
                            svds[iq][0][il * szm + ir] * svds[iq][1][ir];
                        const int n_blocks = get<0>(sk_repr).shape()[0];
                        const uint32_t *pb = get<2>(sk_repr).data();
                        for (int ib = 0; ib < n_blocks; ib++) {
                            int nb = pb[ib + 1] - pb[ib];
                            daxpy(&nb, &factor, term_data + pb[ib], &incx,
                                  pr + pir + (size_t)pb[ib] * rszm * npr +
                                      nb * ((size_t)rszm * ipr + ir),
                                  &incx);
                        }
                    }
            }
        }
        // transpose
        auto &todata = get<2>(rodd), &tedata = get<2>(reven);
        todata = py::array_t<double>(vector<ssize_t>{size_odd});
        tedata = py::array_t<double>(vector<ssize_t>{size_even});
        flat_sparse_tensor_transpose(oshs, odata, oi, perm, todata);
        flat_sparse_tensor_transpose(eshs, edata, ei, perm, tedata);
        for (int i = 0, iodd = 0, ieven = 0; i < n_total; i++)
            if (skf[i]) {
                for (int j = 0; j < 4; j++) {
                    poqs[iodd * 4 + j] = psklqs[i * sklqi + pperm[j] * sklqj];
                    poshs[iodd * 4 + j] = psklshs[i * sklqi + pperm[j] * sklqj];
                }
                iodd++;
            } else {
                for (int j = 0; j < 4; j++) {
                    peqs[ieven * 4 + j] = psklqs[i * sklqi + pperm[j] * sklqj];
                    peshs[ieven * 4 + j] =
                        psklshs[i * sklqi + pperm[j] * sklqj];
                }
                ieven++;
            }
        // info_l = info_r;
        // assign cur_values
        // assign cur_terms and update term_i
        info_l = info_r;
        vector<vector<double>> new_cur_values(s_kept_total);
        vector<vector<int>> new_cur_terms(s_kept_total);
        int isk = 0;
        left_q.resize(s_kept_total);
        for (auto &mq : q_map) {
            int iq = mq.second;
            SZLong q = qs[iq];
            auto &mpr = map_rs[iq];
            auto &nm = nms[iq];
            int rszm = (int)svds[iq][1].size(), szr = nm.second;
            vector<int> vct(szr);
            for (auto &vrs : mpr)
                for (auto &vr : vrs.second) {
                    vct[vr.second] = vr.first;
                    term_i[vr.first] = cur_term_i[vr.first];
                }
            for (int j = 0; j < rszm; j++) {
                left_q[j + isk] = q;
                new_cur_terms[j + isk] = vct;
                new_cur_values[j + isk] = vector<double>(
                    &svds[iq][2][j * szr], &svds[iq][2][(j + 1) * szr]);
            }
            isk += rszm;
        }
        assert(isk == s_kept_total);
        cur_terms = new_cur_terms;
        cur_values = new_cur_values;
    }
    // end of loop; check last term is identity with cur_values = 1
    assert(cur_values.size() == 1 && cur_values[0].size() == 1);
    assert(cur_values[0][0] == 1.0);
    assert(term_i[cur_terms[0][0]] == term_l[cur_terms[0][0]]);
    return rr;
}
