#!coding=utf-8
import numpy as np
cimport numpy as np
cimport cython

ctypedef np.int32_t itype
ctypedef np.float64_t dtype


@cython.boundscheck(False)
@cython.wraparound(False)
def sum_dG(itype adim, itype bdim, itype cdim, itype kdim, itype mdim,
           np.ndarray[itype, ndim=3] i_abc,
           np.ndarray[itype, ndim=3] j_abc,
           np.ndarray[itype, ndim=1] t_a,
           np.ndarray[dtype, ndim=6] dGdrx_axbkm,
           np.ndarray[dtype, ndim=6] dG_abkmxc,
           np.ndarray[dtype, ndim=6] dGdh_axybkm,
           np.ndarray[dtype, ndim=4] drdrx_abcx,
           np.ndarray[dtype, ndim=3] R_abc):
    """
    Sum up `dGdh_axybkm` and `dGdrx_axbkm`.
    """
    cdef int xdim = 3
    cdef int a, b, c, k, m, x, y, i, j, t_i
    for a in range(adim):
        for b in range(bdim):
            for c in range(cdim):
                i = i_abc[a, b, c]
                j = j_abc[a, b, c]
                t_i = t_a[i]
                if i < 0 or j < 0:
                    continue
                for k in range(kdim):
                    for m in range(mdim):
                        for x in range(xdim):
                            dGdrx_axbkm[t_i, i, x, b, k, m] += \
                                dG_abkmxc[a, b, k, m, x, c]
                            dGdrx_axbkm[t_i, j, x, b, k, m] -= \
                                dG_abkmxc[a, b, k, m, x, c]
                            for y in range(xdim):
                                dGdh_axybkm[i, x, y, b, k, m] += \
                                    dG_abkmxc[a, b, k, m, x, c] * \
                                    drdrx_abcx[a, b, c, y] * R_abc[a, b, c]


@cython.boundscheck(False)
@cython.wraparound(False)
def fill_tensors(np.ndarray[itype, ndim=2] eltmap, 
                 np.ndarray[itype, ndim=1] eltypes,
                 np.ndarray[itype, ndim=1] i_l,
                 np.ndarray[itype, ndim=1] j_l,
                 np.ndarray[itype, ndim=2] neigh,
                 np.ndarray[itype, ndim=1] loc,
                 np.ndarray[dtype, ndim=3] R_abc,
                 np.ndarray[dtype, ndim=1] R_l,
                 np.ndarray[dtype, ndim=4] drdrx_abcx,
                 np.ndarray[dtype, ndim=2] D_lx,
                 np.ndarray[dtype, ndim=4] H_abck,
                 np.ndarray[dtype, ndim=2] H_lk,
                 np.ndarray[dtype, ndim=4] dHdr_abck,
                 np.ndarray[dtype, ndim=2] dH_lk,
                 np.ndarray[itype, ndim=3] i_abc,
                 np.ndarray[itype, ndim=3] j_abc,
                 np.ndarray[itype, ndim=1] t_a,
                 np.ndarray[dtype, ndim=4] M_abcd,
                 np.ndarray[dtype, ndim=5] dMdrx_abcdx,
                 itype max_moment):
    """
    Fill up `R_abc`, `drdrx_abcx`, `H_abck`, `dHdr_abck`, `i_abc`, `j_abc`,
    `t_a`, `M_abcd` and `dMdrx_abcdx`.
    """
    cdef int idx, itype, jtype, a, b, c, k
    cdef int cdim = H_abck.shape[2]
    cdef int kdim = H_abck.shape[3]
    cdef int num_rij = len(i_l)
    cdef double recip, x, y, z, xx, yy, zz, xy, xz, yz
    cdef double xxx, xxy, xxz, xyy, xyz, xzz, yyy, yyz, yzz, zzz
    cdef double xxxx, xxxy, xxxz, xxyy, xxyz, xxzz, xyyy, xyyz, xyzz, xzzz
    cdef double yyyy, yyyz, yyzz, yzzz, zzzz
    cdef double xxxxx, xxxxy, xxxxz, xxxyy, xxxyz, xxxzz, xxyyy, xxyyz, xxyzz
    cdef double xxzzz, xyyyy, xyyyz, xyyzz, xyzzz, xzzzz, yyyyy, yyyyz, yyyzz
    cdef double yyzzz, yzzzz, zzzzz
    cdef double xxxxxx, xxxxxy, xxxxxz, xxxxyy, xxxxyz, xxxxzz, xxxyyy, xxxyyz
    cdef double xxxyzz, xxxzzz, xxyyyy, xxyyyz, xxyyzz, xxyzzz, xxzzzz, xyyyyy
    cdef double xyyyyz, xyyyzz, xyyzzz, xyzzzz, xzzzzz, yyyyyy, yyyyyz, yyyyzz
    cdef double yyyzzz, yyzzzz, yzzzzz, zzzzzz
    for idx in range(num_rij):
        itype = eltypes[i_l[idx]]
        jtype = eltypes[j_l[idx]]
        a = loc[i_l[idx]]
        b = eltmap[itype, jtype]
        c = neigh[a, b]
        if c >= cdim:
            continue
        R_abc[a, b, c] = R_l[idx]
        for k in range(kdim):
            H_abck[a, b, c, k] = H_lk[idx, k]
            dHdr_abck[a, b, c, k] = dH_lk[idx, k]
        i_abc[a, b, c] = a
        j_abc[a, b, c] = loc[j_l[idx]]
        t_a[a] = itype
        M_abcd[a, b, c, 0] = 1.0
        recip = 1.0 / R_l[idx]
        x = D_lx[idx, 0] * recip
        y = D_lx[idx, 1] * recip
        z = D_lx[idx, 2] * recip
        drdrx_abcx[a, b, c, 0] = x
        drdrx_abcx[a, b, c, 1] = y
        drdrx_abcx[a, b, c, 2] = z
        if max_moment > 0:
            xx = x * x
            xy = x * y
            xz = x * z
            yy = y * y
            yz = z * y
            zz = z * z
            M_abcd[a, b, c, 1: 4] = x, y, z
            dMdrx_abcdx[a, b, c, 1, 0] = -recip * (xx - 1)
            dMdrx_abcdx[a, b, c, 1, 1] = -recip * xy
            dMdrx_abcdx[a, b, c, 1, 2] = -recip * xz
            dMdrx_abcdx[a, b, c, 2, 0] = -recip * xy
            dMdrx_abcdx[a, b, c, 2, 1] = -recip * (yy - 1)
            dMdrx_abcdx[a, b, c, 2, 2] = -recip * yz
            dMdrx_abcdx[a, b, c, 3, 0] = -recip * xz
            dMdrx_abcdx[a, b, c, 3, 1] = -recip * yz
            dMdrx_abcdx[a, b, c, 3, 2] = -recip * (zz - 1)
            if max_moment > 1:
                xxx = xx * x
                xxy = xx * y
                xxz = xx * z
                xyy = xy * y
                xyz = xy * z
                xzz = xz * z
                yyy = yy * y
                yyz = yy * z
                yzz = yz * z
                zzz = zz * z
                M_abcd[a, b, c, 4] = xx
                M_abcd[a, b, c, 5] = xy
                M_abcd[a, b, c, 6] = xz
                M_abcd[a, b, c, 7] = yy
                M_abcd[a, b, c, 8] = yz
                M_abcd[a, b, c, 9] = zz
                dMdrx_abcdx[a, b, c, 4, 0] = -recip * (2 * xxx - 2 * x)
                dMdrx_abcdx[a, b, c, 4, 1] = -recip * 2 * xxy
                dMdrx_abcdx[a, b, c, 4, 2] = -recip * 2 * xxz
                dMdrx_abcdx[a, b, c, 5, 0] = -recip * (2 * xxy - y)
                dMdrx_abcdx[a, b, c, 5, 1] = -recip * (2 * xyy - x)
                dMdrx_abcdx[a, b, c, 5, 2] = -recip * 2 * xyz
                dMdrx_abcdx[a, b, c, 6, 0] = -recip * (2 * xxz - z)
                dMdrx_abcdx[a, b, c, 6, 1] = -recip * 2 * xyz
                dMdrx_abcdx[a, b, c, 6, 2] = -recip * (2 * xzz - x)
                dMdrx_abcdx[a, b, c, 7, 0] = -recip * 2 * xyy
                dMdrx_abcdx[a, b, c, 7, 1] = -recip * (2 * yyy - 2 * y)
                dMdrx_abcdx[a, b, c, 7, 2] = -recip * 2 * yyz
                dMdrx_abcdx[a, b, c, 8, 0] = -recip * 2 * xyz
                dMdrx_abcdx[a, b, c, 8, 1] = -recip * (2 * yyz - z)
                dMdrx_abcdx[a, b, c, 8, 2] = -recip * (2 * yzz - y)
                dMdrx_abcdx[a, b, c, 9, 0] = -recip * 2 * xzz
                dMdrx_abcdx[a, b, c, 9, 1] = -recip * 2 * yzz
                dMdrx_abcdx[a, b, c, 9, 2] = -recip * (2 * zzz - 2 * z)
                if max_moment > 2:
                    xxxx = xxx * x
                    xxxy = xxx * y
                    xxxz = xxx * z
                    xxyy = xxy * y
                    xxyz = xxy * z
                    xxzz = xxz * z
                    xyyy = xyy * y
                    xyyz = xyy * z
                    xyzz = xyz * z
                    xzzz = xzz * z
                    yyyy = yyy * y
                    yyyz = yyy * z
                    yyzz = yyz * z
                    yzzz = yzz * z
                    zzzz = zzz * z
                    M_abcd[a, b, c, 10] = xxx
                    M_abcd[a, b, c, 11] = xxy
                    M_abcd[a, b, c, 12] = xxz
                    M_abcd[a, b, c, 13] = xyy
                    M_abcd[a, b, c, 14] = xyz
                    M_abcd[a, b, c, 15] = xzz
                    M_abcd[a, b, c, 16] = yyy
                    M_abcd[a, b, c, 17] = yyz
                    M_abcd[a, b, c, 18] = yzz
                    M_abcd[a, b, c, 19] = zzz
                    dMdrx_abcdx[a, b, c, 10, 0] = -recip * (3 * xxxx - 3 * xx)
                    dMdrx_abcdx[a, b, c, 10, 1] = -recip * 3 * xxxy
                    dMdrx_abcdx[a, b, c, 10, 2] = -recip * 3 * xxxz
                    dMdrx_abcdx[a, b, c, 11, 0] = -recip * (3 * xxxy - 2 * xy)
                    dMdrx_abcdx[a, b, c, 11, 1] = -recip * (3 * xxyy - xx)
                    dMdrx_abcdx[a, b, c, 11, 2] = -recip * 3 * xxyz
                    dMdrx_abcdx[a, b, c, 12, 0] = -recip * (3 * xxxz - 2 * xz)
                    dMdrx_abcdx[a, b, c, 12, 1] = -recip * 3 * xxyz
                    dMdrx_abcdx[a, b, c, 12, 2] = -recip * (3 * xxzz - xx)
                    dMdrx_abcdx[a, b, c, 13, 0] = -recip * (3 * xxyy - yy)
                    dMdrx_abcdx[a, b, c, 13, 1] = -recip * (3 * xyyy - 2 * xy)
                    dMdrx_abcdx[a, b, c, 13, 2] = -recip * 3 * xyyz
                    dMdrx_abcdx[a, b, c, 14, 0] = -recip * (3 * xxyz - yz)
                    dMdrx_abcdx[a, b, c, 14, 1] = -recip * (3 * xyyz - xz)
                    dMdrx_abcdx[a, b, c, 14, 2] = -recip * (3 * xyzz - xy)
                    dMdrx_abcdx[a, b, c, 15, 0] = -recip * (3 * xxzz - zz)
                    dMdrx_abcdx[a, b, c, 15, 1] = -recip * 3 * xyzz
                    dMdrx_abcdx[a, b, c, 15, 2] = -recip * (3 * xzzz - 2 * xz)
                    dMdrx_abcdx[a, b, c, 16, 0] = -recip * 3 * xyyy
                    dMdrx_abcdx[a, b, c, 16, 1] = -recip * (3 * yyyy - 3 * yy)
                    dMdrx_abcdx[a, b, c, 16, 2] = -recip * 3 * yyyz
                    dMdrx_abcdx[a, b, c, 17, 0] = -recip * 3 * xyyz
                    dMdrx_abcdx[a, b, c, 17, 1] = -recip * (3 * yyyz - 2 * yz)
                    dMdrx_abcdx[a, b, c, 17, 2] = -recip * (3 * yyzz - yy)
                    dMdrx_abcdx[a, b, c, 18, 0] = -recip * 3 * xyzz
                    dMdrx_abcdx[a, b, c, 18, 1] = -recip * (3 * yyzz - zz)
                    dMdrx_abcdx[a, b, c, 18, 2] = -recip * (3 * yzzz - 2 * yz)
                    dMdrx_abcdx[a, b, c, 19, 0] = -recip * 3 * xzzz
                    dMdrx_abcdx[a, b, c, 19, 1] = -recip * 3 * yzzz
                    dMdrx_abcdx[a, b, c, 19, 2] = -recip * (3 * zzzz - 3 * zz)
                    if max_moment > 3:
                        xxxxx = xxxx * x
                        xxxxy = xxxx * y
                        xxxxz = xxxx * z
                        xxxyy = xxxy * y
                        xxxyz = xxxy * z
                        xxxzz = xxxz * z
                        xxyyy = xxyy * y
                        xxyyz = xxyy * z
                        xxyzz = xxyz * z
                        xxzzz = xxzz * z
                        xyyyy = xyyy * y
                        xyyyz = xyyy * z
                        xyyzz = xyyz * z
                        xyzzz = xyzz * z
                        xzzzz = xzzz * z
                        yyyyy = yyyy * y
                        yyyyz = yyyy * z
                        yyyzz = yyyz * z
                        yyzzz = yyzz * z
                        yzzzz = yzzz * z
                        zzzzz = zzzz * z
                        M_abcd[a, b, c, 20] = xxxx
                        M_abcd[a, b, c, 21] = xxxy
                        M_abcd[a, b, c, 22] = xxxz
                        M_abcd[a, b, c, 23] = xxyy
                        M_abcd[a, b, c, 24] = xxyz
                        M_abcd[a, b, c, 25] = xxzz
                        M_abcd[a, b, c, 26] = xyyy
                        M_abcd[a, b, c, 27] = xyyz
                        M_abcd[a, b, c, 28] = xyzz
                        M_abcd[a, b, c, 29] = xzzz
                        M_abcd[a, b, c, 30] = yyyy
                        M_abcd[a, b, c, 31] = yyyz
                        M_abcd[a, b, c, 32] = yyzz
                        M_abcd[a, b, c, 33] = yzzz
                        M_abcd[a, b, c, 34] = zzzz
                        dMdrx_abcdx[a, b, c, 20, 0] = -recip * (4 * xxxxx - 4 * xxx)
                        dMdrx_abcdx[a, b, c, 20, 1] = -recip * 4 * xxxxy
                        dMdrx_abcdx[a, b, c, 20, 2] = -recip * 4 * xxxxz
                        dMdrx_abcdx[a, b, c, 21, 0] = -recip * (4 * xxxxy - 3 * xxy)
                        dMdrx_abcdx[a, b, c, 21, 1] = -recip * (4 * xxxyy - xxx)
                        dMdrx_abcdx[a, b, c, 21, 2] = -recip * 4 * xxxyz
                        dMdrx_abcdx[a, b, c, 22, 0] = -recip * (4 * xxxxz - 4 * xxz)
                        dMdrx_abcdx[a, b, c, 22, 1] = -recip * 4 * xxxyz
                        dMdrx_abcdx[a, b, c, 22, 2] = -recip * (4 * xxxzz - xxx)
                        dMdrx_abcdx[a, b, c, 23, 0] = -recip * (4 * xxxyy - 2 * xyy)
                        dMdrx_abcdx[a, b, c, 23, 1] = -recip * (4 * xxyyy - 2 * xxy)
                        dMdrx_abcdx[a, b, c, 23, 2] = -recip * 4 * xxyyz
                        dMdrx_abcdx[a, b, c, 24, 0] = -recip * (4 * xxxyz - 2 * xyz)
                        dMdrx_abcdx[a, b, c, 24, 1] = -recip * (4 * xxyyz - xxz)
                        dMdrx_abcdx[a, b, c, 24, 2] = -recip * (4 * xxyzz - xxy)
                        dMdrx_abcdx[a, b, c, 25, 0] = -recip * (4 * xxxzz - 2 * xzz)
                        dMdrx_abcdx[a, b, c, 25, 1] = -recip * 4 * xxyzz
                        dMdrx_abcdx[a, b, c, 25, 2] = -recip * (4 * xxzzz - 2 * xxz)
                        dMdrx_abcdx[a, b, c, 26, 0] = -recip * (4 * xxyyy - yyy)
                        dMdrx_abcdx[a, b, c, 26, 1] = -recip * (4 * xyyyy - 3 * xyy)
                        dMdrx_abcdx[a, b, c, 26, 2] = -recip * 4 * xyyyz
                        dMdrx_abcdx[a, b, c, 27, 0] = -recip * (4 * xxyyz - yyz)
                        dMdrx_abcdx[a, b, c, 27, 1] = -recip * (4 * xyyyz - 2 * xyz)
                        dMdrx_abcdx[a, b, c, 27, 2] = -recip * (4 * xyyzz - xyy)
                        dMdrx_abcdx[a, b, c, 28, 0] = -recip * (4 * xxyzz - yzz)
                        dMdrx_abcdx[a, b, c, 28, 1] = -recip * (4 * xyyzz - xzz)
                        dMdrx_abcdx[a, b, c, 28, 2] = -recip * (4 * xyzzz - 2 * xyz)
                        dMdrx_abcdx[a, b, c, 29, 0] = -recip * (4 * xxzzz - zzz)
                        dMdrx_abcdx[a, b, c, 29, 1] = -recip * 4 * xyzzz
                        dMdrx_abcdx[a, b, c, 29, 2] = -recip * (4 * xzzzz - 3 * xzz)
                        dMdrx_abcdx[a, b, c, 30, 0] = -recip * 4 * xyyyy
                        dMdrx_abcdx[a, b, c, 30, 1] = -recip * (4 * yyyyy - 4 * yyy)
                        dMdrx_abcdx[a, b, c, 30, 2] = -recip * 4 * yyyyz
                        dMdrx_abcdx[a, b, c, 31, 0] = -recip * 4 * xyyyz
                        dMdrx_abcdx[a, b, c, 31, 1] = -recip * (4 * yyyyz - 3 * yyz)
                        dMdrx_abcdx[a, b, c, 31, 2] = -recip * (4 * yyyzz - yyy)
                        dMdrx_abcdx[a, b, c, 32, 0] = -recip * 4 * xyyzz
                        dMdrx_abcdx[a, b, c, 32, 1] = -recip * (4 * yyyzz - 2 * yzz)
                        dMdrx_abcdx[a, b, c, 32, 2] = -recip * (4 * yyzzz - 2 * yyz)
                        dMdrx_abcdx[a, b, c, 33, 0] = -recip * 4 * xyzzz
                        dMdrx_abcdx[a, b, c, 33, 1] = -recip * (4 * yyzzz - zzz)
                        dMdrx_abcdx[a, b, c, 33, 2] = -recip * (4 * yzzzz - 3 * yzz)
                        dMdrx_abcdx[a, b, c, 34, 0] = -recip * 4 * xzzzz
                        dMdrx_abcdx[a, b, c, 34, 1] = -recip * 4 * yzzzz
                        dMdrx_abcdx[a, b, c, 34, 2] = -recip * (4 * zzzzz - 4 * zzz)
                        if max_moment > 4:
                            xxxxxx = xxxxx * x
                            xxxxxy = xxxxx * y
                            xxxxxz = xxxxx * z
                            xxxxyy = xxxxy * y
                            xxxxyz = xxxxy * z
                            xxxxzz = xxxxz * z
                            xxxyyy = xxxyy * y
                            xxxyyz = xxxyy * z
                            xxxyzz = xxxyz * z
                            xxxzzz = xxxzz * z
                            xxyyyy = xxyyy * y
                            xxyyyz = xxyyy * z
                            xxyyzz = xxyyz * z
                            xxyzzz = xxyzz * z
                            xxzzzz = xxzzz * z
                            xyyyyy = xyyyy * y
                            xyyyyz = xyyyy * z
                            xyyyzz = xyyyz * z
                            xyyzzz = xyyzz * z
                            xyzzzz = xyzzz * z
                            xzzzzz = xzzzz * z
                            yyyyyy = yyyyy * y
                            yyyyyz = yyyyy * z
                            yyyyzz = yyyyz * z
                            yyyzzz = yyyzz * z
                            yyzzzz = yyzzz * z
                            yzzzzz = yzzzz * z
                            zzzzzz = zzzzz * z
                            M_abcd[a, b, c, 35] = xxxxx
                            M_abcd[a, b, c, 36] = xxxxy
                            M_abcd[a, b, c, 37] = xxxxz
                            M_abcd[a, b, c, 38] = xxxyy
                            M_abcd[a, b, c, 39] = xxxyz
                            M_abcd[a, b, c, 40] = xxxzz
                            M_abcd[a, b, c, 41] = xxyyy
                            M_abcd[a, b, c, 42] = xxyyz
                            M_abcd[a, b, c, 43] = xxyzz
                            M_abcd[a, b, c, 44] = xxzzz
                            M_abcd[a, b, c, 45] = xyyyy
                            M_abcd[a, b, c, 46] = xyyyz
                            M_abcd[a, b, c, 47] = xyyzz
                            M_abcd[a, b, c, 48] = xyzzz
                            M_abcd[a, b, c, 49] = xzzzz
                            M_abcd[a, b, c, 50] = yyyyy
                            M_abcd[a, b, c, 51] = yyyyz
                            M_abcd[a, b, c, 52] = yyyzz
                            M_abcd[a, b, c, 53] = yyzzz
                            M_abcd[a, b, c, 54] = yzzzz
                            M_abcd[a, b, c, 55] = zzzzz
                            dMdrx_abcdx[a, b, c, 35, 0] = -recip * (5 * xxxxxx - 5 * xxxx)
                            dMdrx_abcdx[a, b, c, 35, 1] = -recip * 5 * xxxxxy
                            dMdrx_abcdx[a, b, c, 35, 2] = -recip * 5 * xxxxxz
                            dMdrx_abcdx[a, b, c, 36, 0] = -recip * (5 * xxxxxy - 4 * xxxy)
                            dMdrx_abcdx[a, b, c, 36, 1] = -recip * (5 * xxxxyy - xxxx)
                            dMdrx_abcdx[a, b, c, 36, 2] = -recip * 5 * xxxxyz
                            dMdrx_abcdx[a, b, c, 37, 0] = -recip * (5 * xxxxxz - 4 * xxxz)
                            dMdrx_abcdx[a, b, c, 37, 1] = -recip * 5 * xxxxyz
                            dMdrx_abcdx[a, b, c, 37, 2] = -recip * (5 * xxxxzz - xxxx)
                            dMdrx_abcdx[a, b, c, 38, 0] = -recip * (5 * xxxxyy - 3 * xxyy)
                            dMdrx_abcdx[a, b, c, 38, 1] = -recip * (5 * xxxyyy - 2 * xxxy)
                            dMdrx_abcdx[a, b, c, 38, 2] = -recip * 5 * xxxyyz
                            dMdrx_abcdx[a, b, c, 39, 0] = -recip * (5 * xxxxyz - 3 * xxyz)
                            dMdrx_abcdx[a, b, c, 39, 1] = -recip * (5 * xxxyyz - xxxz)
                            dMdrx_abcdx[a, b, c, 39, 2] = -recip * (5 * xxxyzz - xxxy)
                            dMdrx_abcdx[a, b, c, 40, 0] = -recip * (5 * xxxxzz - 3 * xxzz)
                            dMdrx_abcdx[a, b, c, 40, 1] = -recip * 5 * xxxyzz
                            dMdrx_abcdx[a, b, c, 40, 2] = -recip * (5 * xxxzzz - 2 * xxxz)
                            dMdrx_abcdx[a, b, c, 41, 0] = -recip * (5 * xxxyyy - 2 * xyyy)
                            dMdrx_abcdx[a, b, c, 41, 1] = -recip * (5 * xxyyyy - 3 * xxyy)
                            dMdrx_abcdx[a, b, c, 41, 2] = -recip * 5 * xxyyyz
                            dMdrx_abcdx[a, b, c, 42, 0] = -recip * (5 * xxxyyz - 2 * xyyz)
                            dMdrx_abcdx[a, b, c, 42, 1] = -recip * (5 * xxyyyz - 2 * xxyz)
                            dMdrx_abcdx[a, b, c, 42, 2] = -recip * (5 * xxyyzz - xxyy)
                            dMdrx_abcdx[a, b, c, 43, 0] = -recip * (5 * xxxyzz - 2 * xyzz)
                            dMdrx_abcdx[a, b, c, 43, 1] = -recip * (5 * xxyyzz - xxzz)
                            dMdrx_abcdx[a, b, c, 43, 2] = -recip * (5 * xxyzzz - 2 * xxyz)
                            dMdrx_abcdx[a, b, c, 44, 0] = -recip * (5 * xxxzzz - 2 * xzzz)
                            dMdrx_abcdx[a, b, c, 44, 1] = -recip * 5 * xxyzzz
                            dMdrx_abcdx[a, b, c, 44, 2] = -recip * (5 * xxzzzz - 3 * xxzz)
                            dMdrx_abcdx[a, b, c, 45, 0] = -recip * (5 * xxyyyy - yyyy)
                            dMdrx_abcdx[a, b, c, 45, 1] = -recip * (5 * xyyyyy - 4 * xyyy)
                            dMdrx_abcdx[a, b, c, 45, 2] = -recip * 5 * xyyyyz
                            dMdrx_abcdx[a, b, c, 46, 0] = -recip * (5 * xxyyyz - yyyz)
                            dMdrx_abcdx[a, b, c, 46, 1] = -recip * (5 * xyyyyz - 3 * xyyz)
                            dMdrx_abcdx[a, b, c, 46, 2] = -recip * (5 * xyyyzz - xyyy)
                            dMdrx_abcdx[a, b, c, 47, 0] = -recip * (5 * xxyyzz - yyzz)
                            dMdrx_abcdx[a, b, c, 47, 1] = -recip * (5 * xyyyzz - 2 * xyzz)
                            dMdrx_abcdx[a, b, c, 47, 2] = -recip * (5 * xyyzzz - 2 * xyyz)
                            dMdrx_abcdx[a, b, c, 48, 0] = -recip * (5 * xxyzzz - yzzz)
                            dMdrx_abcdx[a, b, c, 48, 1] = -recip * (5 * xyyzzz - xzzz)
                            dMdrx_abcdx[a, b, c, 48, 2] = -recip * (5 * xyzzzz - 3 * xyzz)
                            dMdrx_abcdx[a, b, c, 49, 0] = -recip * (5 * xxzzzz - zzzz)
                            dMdrx_abcdx[a, b, c, 49, 1] = -recip * 5 * xyzzzz
                            dMdrx_abcdx[a, b, c, 49, 2] = -recip * (5 * xzzzzz - 4 * xzzz)
                            dMdrx_abcdx[a, b, c, 50, 0] = -recip * 5 * xyyyyy
                            dMdrx_abcdx[a, b, c, 50, 1] = -recip * (5 * yyyyyy - 5 * yyyy)
                            dMdrx_abcdx[a, b, c, 50, 2] = -recip * 5 * yyyyyz
                            dMdrx_abcdx[a, b, c, 51, 0] = -recip * 5 * xyyyyz
                            dMdrx_abcdx[a, b, c, 51, 1] = -recip * (5 * yyyyyz - 4 * yyyz)
                            dMdrx_abcdx[a, b, c, 51, 2] = -recip * (5 * yyyyzz - yyyy)
                            dMdrx_abcdx[a, b, c, 52, 0] = -recip * 5 * xyyyzz
                            dMdrx_abcdx[a, b, c, 52, 1] = -recip * (5 * yyyyzz - 3 * yyzz)
                            dMdrx_abcdx[a, b, c, 52, 2] = -recip * (5 * yyyzzz - 2 * yyyz)
                            dMdrx_abcdx[a, b, c, 53, 0] = -recip * 5 * xyyzzz
                            dMdrx_abcdx[a, b, c, 53, 1] = -recip * (5 * yyyzzz - 2 * yzzz)
                            dMdrx_abcdx[a, b, c, 53, 2] = -recip * (5 * yyzzzz - 3 * yyzz)
                            dMdrx_abcdx[a, b, c, 54, 0] = -recip * 5 * xyzzzz
                            dMdrx_abcdx[a, b, c, 54, 1] = -recip * (5 * yyzzzz - zzzz)
                            dMdrx_abcdx[a, b, c, 54, 2] = -recip * (5 * yzzzzz - 4 * yzzz)
                            dMdrx_abcdx[a, b, c, 55, 0] = -recip * 5 * xzzzzz
                            dMdrx_abcdx[a, b, c, 55, 1] = -recip * 5 * yzzzzz
                            dMdrx_abcdx[a, b, c, 55, 2] = -recip * (5 * zzzzzz - 5 * zzzz)
        neigh[a, b] += 1


@cython.boundscheck(False)
@cython.wraparound(False)
def setup_tensors(np.ndarray[itype, ndim=2] eltmap, 
                  np.ndarray[itype, ndim=1] eltypes,
                  np.ndarray[itype, ndim=1] i_l,
                  np.ndarray[itype, ndim=1] j_l,
                  np.ndarray[itype, ndim=2] neigh,
                  np.ndarray[itype, ndim=1] loc,
                  np.ndarray[dtype, ndim=3] R_abc,
                  np.ndarray[dtype, ndim=1] R_l,
                  np.ndarray[dtype, ndim=4] drdrx_abcx,
                  np.ndarray[dtype, ndim=2] D_lx,
                  np.ndarray[dtype, ndim=4] H_abck,
                  np.ndarray[dtype, ndim=2] H_lk,
                  np.ndarray[dtype, ndim=4] dHdr_abck,
                  np.ndarray[dtype, ndim=2] dH_lk,
                  np.ndarray[itype, ndim=3] i_abc,
                  np.ndarray[itype, ndim=3] j_abc,
                  np.ndarray[itype, ndim=1] t_a,
                  np.ndarray[dtype, ndim=4] M_abcd,
                  itype max_moment):
    """
    Fill up `R_abc`, `drdrx_abcx`, `H_abck`, `dHdr_abck`, `i_abc`, `j_abc`,
    `t_a`, `M_abcd` and `dMdrx_abcdx`.
    """
    cdef int idx, itype, jtype, a, b, c, k
    cdef int cdim = H_abck.shape[2]
    cdef int kdim = H_abck.shape[3]
    cdef int num_rij = len(i_l)
    cdef double recip, x, y, z, xx, yy, zz, xy, xz, yz
    cdef double xxx, xxy, xxz, xyy, xyz, xzz, yyy, yyz, yzz, zzz
    cdef double xxxx, xxxy, xxxz, xxyy, xxyz, xxzz, xyyy, xyyz, xyzz, xzzz
    cdef double yyyy, yyyz, yyzz, yzzz, zzzz
    cdef double xxxxx, xxxxy, xxxxz, xxxyy, xxxyz, xxxzz, xxyyy, xxyyz, xxyzz
    cdef double xxzzz, xyyyy, xyyyz, xyyzz, xyzzz, xzzzz, yyyyy, yyyyz, yyyzz
    cdef double yyzzz, yzzzz, zzzzz

    for idx in range(num_rij):
        itype = eltypes[i_l[idx]]
        jtype = eltypes[j_l[idx]]
        a = loc[i_l[idx]]
        b = eltmap[itype, jtype]
        c = neigh[a, b]
        if c >= cdim:
            continue
        R_abc[a, b, c] = R_l[idx]
        H_abck[a, b, c] = H_lk[idx]
        dHdr_abck[a, b, c] = dH_lk[idx]
        i_abc[a, b, c] = a
        j_abc[a, b, c] = loc[j_l[idx]]
        t_a[a] = itype
        M_abcd[a, b, c, 0] = 1.0
        recip = 1.0 / R_l[idx]
        x = D_lx[idx, 0] * recip
        y = D_lx[idx, 1] * recip
        z = D_lx[idx, 2] * recip
        drdrx_abcx[a, b, c, 0] = x
        drdrx_abcx[a, b, c, 1] = y
        drdrx_abcx[a, b, c, 2] = z
        if max_moment > 0:
            M_abcd[a, b, c, 1: 4] = x, y, z
            if max_moment > 1:
                xx = x * x
                xy = x * y
                xz = x * z
                yy = y * y
                yz = z * y
                zz = z * z
                M_abcd[a, b, c, 4] = xx
                M_abcd[a, b, c, 5] = xy
                M_abcd[a, b, c, 6] = xz
                M_abcd[a, b, c, 7] = yy
                M_abcd[a, b, c, 8] = yz
                M_abcd[a, b, c, 9] = zz
                if max_moment > 2:
                    xxx = xx * x
                    xxy = xx * y
                    xxz = xx * z
                    xyy = xy * y
                    xyz = xy * z
                    xzz = xz * z
                    yyy = yy * y
                    yyz = yy * z
                    yzz = yz * z
                    zzz = zz * z
                    M_abcd[a, b, c, 10] = xxx
                    M_abcd[a, b, c, 11] = xxy
                    M_abcd[a, b, c, 12] = xxz
                    M_abcd[a, b, c, 13] = xyy
                    M_abcd[a, b, c, 14] = xyz
                    M_abcd[a, b, c, 15] = xzz
                    M_abcd[a, b, c, 16] = yyy
                    M_abcd[a, b, c, 17] = yyz
                    M_abcd[a, b, c, 18] = yzz
                    M_abcd[a, b, c, 19] = zzz
                    if max_moment > 3:
                        xxxx = xxx * x
                        xxxy = xxx * y
                        xxxz = xxx * z
                        xxyy = xxy * y
                        xxyz = xxy * z
                        xxzz = xxz * z
                        xyyy = xyy * y
                        xyyz = xyy * z
                        xyzz = xyz * z
                        xzzz = xzz * z
                        yyyy = yyy * y
                        yyyz = yyy * z
                        yyzz = yyz * z
                        yzzz = yzz * z
                        zzzz = zzz * z
                        M_abcd[a, b, c, 20] = xxxx
                        M_abcd[a, b, c, 21] = xxxy
                        M_abcd[a, b, c, 22] = xxxz
                        M_abcd[a, b, c, 23] = xxyy
                        M_abcd[a, b, c, 24] = xxyz
                        M_abcd[a, b, c, 25] = xxzz
                        M_abcd[a, b, c, 26] = xyyy
                        M_abcd[a, b, c, 27] = xyyz
                        M_abcd[a, b, c, 28] = xyzz
                        M_abcd[a, b, c, 29] = xzzz
                        M_abcd[a, b, c, 30] = yyyy
                        M_abcd[a, b, c, 31] = yyyz
                        M_abcd[a, b, c, 32] = yyzz
                        M_abcd[a, b, c, 33] = yzzz
                        M_abcd[a, b, c, 34] = zzzz
                        if max_moment > 4:
                            xxxxx = xxxx * x
                            xxxxy = xxxx * y
                            xxxxz = xxxx * z
                            xxxyy = xxxy * y
                            xxxyz = xxxy * z
                            xxxzz = xxxz * z
                            xxyyy = xxyy * y
                            xxyyz = xxyy * z
                            xxyzz = xxyz * z
                            xxzzz = xxzz * z
                            xyyyy = xyyy * y
                            xyyyz = xyyy * z
                            xyyzz = xyyz * z
                            xyzzz = xyzz * z
                            xzzzz = xzzz * z
                            yyyyy = yyyy * y
                            yyyyz = yyyy * z
                            yyyzz = yyyz * z
                            yyzzz = yyzz * z
                            yzzzz = yzzz * z
                            zzzzz = zzzz * z
                            M_abcd[a, b, c, 35] = xxxxx
                            M_abcd[a, b, c, 36] = xxxxy
                            M_abcd[a, b, c, 37] = xxxxz
                            M_abcd[a, b, c, 38] = xxxyy
                            M_abcd[a, b, c, 39] = xxxyz
                            M_abcd[a, b, c, 40] = xxxzz
                            M_abcd[a, b, c, 41] = xxyyy
                            M_abcd[a, b, c, 42] = xxyyz
                            M_abcd[a, b, c, 43] = xxyzz
                            M_abcd[a, b, c, 44] = xxzzz
                            M_abcd[a, b, c, 45] = xyyyy
                            M_abcd[a, b, c, 46] = xyyyz
                            M_abcd[a, b, c, 47] = xyyzz
                            M_abcd[a, b, c, 48] = xyzzz
                            M_abcd[a, b, c, 49] = xzzzz
                            M_abcd[a, b, c, 50] = yyyyy
                            M_abcd[a, b, c, 51] = yyyyz
                            M_abcd[a, b, c, 52] = yyyzz
                            M_abcd[a, b, c, 53] = yyzzz
                            M_abcd[a, b, c, 54] = yzzzz
                            M_abcd[a, b, c, 55] = zzzzz
        neigh[a, b] += 1
    

@cython.boundscheck(False)
@cython.wraparound(False)
def calculate_dM_s(np.ndarray[dtype, ndim=2] dM):
    dM[0, 0] = 0.0
    dM[1, 0] = 0.0
    dM[2, 0] = 0.0


@cython.boundscheck(False)
@cython.wraparound(False)
def calculate_dM_p(dtype recip, 
                   np.ndarray[dtype, ndim=1] drdrx, 
                   np.ndarray[dtype, ndim=2] dM):
    cdef dtype x, y, z, xx, xy, xz, yy, yz, zz

    calculate_dM_s(dM)

    x = drdrx[0]
    y = drdrx[1]
    z = drdrx[2]
    xx = x * x
    xy = x * y
    xz = x * z
    yy = y * y
    yz = z * y
    zz = z * z
    dM[0, 1] = -recip * (xx - 1)
    dM[0, 2] = -recip * xy
    dM[0, 3] = -recip * xz
    dM[1, 1] = -recip * xy
    dM[1, 2] = -recip * (yy - 1)
    dM[1, 3] = -recip * yz
    dM[2, 1] = -recip * xz
    dM[2, 2] = -recip * yz
    dM[2, 3] = -recip * (zz - 1)


@cython.boundscheck(False)
@cython.wraparound(False)
def calculate_dM_d(dtype recip, 
                   np.ndarray[dtype, ndim=1] drdrx, 
                   np.ndarray[dtype, ndim=2] dM):
    cdef dtype x, y, z, xx, xy, xz, yy, yz, zz
    cdef dtype xxx, xxy, xxz, xyy, xyz, xzz, yyy, yyz, yzz, zzz
    x = drdrx[0]
    y = drdrx[1]
    z = drdrx[2]

    calculate_dM_p(recip, drdrx, dM)

    xx = x * x
    xy = x * y
    xz = x * z
    yy = y * y
    yz = z * y
    zz = z * z
    xxx = x * xx
    xxy = x * xy
    xxz = x * xz
    xyy = y * xy
    xyz = y * xz
    xzz = z * xz
    yyy = y * yy
    yyz = y * yz
    yzz = z * yz
    zzz = z * zz

    dM[0, 4] = -recip * (2 * xxx - 2 * x)
    dM[0, 5] = -recip * (2 * xxy - y)
    dM[0, 6] = -recip * (2 * xxz - z)
    dM[0, 7] = -recip * 2 * xyy
    dM[0, 8] = -recip * 2 * xyz
    dM[0, 9] = -recip * 2 * xzz
    dM[1, 4] = -recip * 2 * xxy
    dM[1, 5] = -recip * (2 * xyy - x)
    dM[1, 6] = -recip * 2 * xyz
    dM[1, 7] = -recip * (2 * yyy - 2 * y)
    dM[1, 8] = -recip * (2 * yyz - z)
    dM[1, 9] = -recip * 2 * yzz
    dM[2, 4] = -recip * 2 * xxz
    dM[2, 5] = -recip * 2 * xyz
    dM[2, 6] = -recip * (2 * xzz - x)
    dM[2, 7] = -recip * 2 * yyz
    dM[2, 8] = -recip * (2 * yzz - y)
    dM[2, 9] = -recip * (2 * zzz - 2 * z)


@cython.boundscheck(False)
@cython.wraparound(False)
def calculate_dM_f(dtype recip, 
                   np.ndarray[dtype, ndim=1] drdrx, 
                   np.ndarray[dtype, ndim=2] dM):
    cdef dtype x, y, z, xx, xy, xz, yy, yz, zz
    cdef dtype xxx, xxy, xxz, xyy, xyz, xzz, yyy, yyz, yzz, zzz
    cdef dtype xxxx, xxxy, xxxz, xxyy, xxyz, xxzz, xyyy, xyyz, xyzz
    cdef dtype xzzz, yyyy, yyyz, yyzz, yzzz, zzzz
    
    calculate_dM_d(recip, drdrx, dM)

    x = drdrx[0]
    y = drdrx[1]
    z = drdrx[2]

    xx = x * x
    xy = x * y
    xz = x * z
    yy = y * y
    yz = z * y
    zz = z * z
    xxx = x * xx
    xxy = x * xy
    xxz = x * xz
    xyy = y * xy
    xyz = y * xz
    xzz = z * xz
    yyy = y * yy
    yyz = y * yz
    yzz = z * yz
    zzz = z * zz
    xxxx = xxx * x
    xxxy = xxx * y
    xxxz = xxx * z
    xxyy = xxy * y
    xxyz = xxy * z
    xxzz = xxz * z
    xyyy = xyy * y
    xyyz = xyy * z
    xyzz = xyz * z
    xzzz = xzz * z
    yyyy = yyy * y
    yyyz = yyy * z
    yyzz = yyz * z
    yzzz = yzz * z
    zzzz = zzz * z

    dM[0, 10] = -recip * (3 * xxxx - 3 * xx)
    dM[0, 11] = -recip * (3 * xxxy - 2 * xy)
    dM[0, 12] = -recip * (3 * xxxz - 2 * xz)
    dM[0, 13] = -recip * (3 * xxyy - yy)
    dM[0, 14] = -recip * (3 * xxyz - yz)
    dM[0, 15] = -recip * (3 * xxzz - zz)
    dM[0, 16] = -recip * 3 * xyyy
    dM[0, 17] = -recip * 3 * xyyz
    dM[0, 18] = -recip * 3 * xyzz
    dM[0, 19] = -recip * 3 * xzzz

    dM[1, 10] = -recip * 3 * xxxy
    dM[1, 11] = -recip * (3 * xxyy - xx)
    dM[1, 12] = -recip * 3 * xxyz
    dM[1, 13] = -recip * (3 * xyyy - 2 * xy)
    dM[1, 14] = -recip * (3 * xyyz - xz)
    dM[1, 15] = -recip * 3 * xyzz
    dM[1, 16] = -recip * (3 * yyyy - 3 * yy)
    dM[1, 17] = -recip * (3 * yyyz - 2 * yz)
    dM[1, 18] = -recip * (3 * yyzz - zz)
    dM[1, 19] = -recip * 3 * yzzz

    dM[2, 10] = -recip * 3 * xxxz
    dM[2, 11] = -recip * 3 * xxyz
    dM[2, 12] = -recip * (3 * xxzz - xx)
    dM[2, 13] = -recip * 3 * xyyz
    dM[2, 14] = -recip * (3 * xyzz - xy)
    dM[2, 15] = -recip * (3 * xzzz - 2 * xz)
    dM[2, 16] = -recip * 3 * yyyz
    dM[2, 17] = -recip * (3 * yyzz - yy)
    dM[2, 18] = -recip * (3 * yzzz - 2 * yz)
    dM[2, 19] = -recip * (3 * zzzz - 3 * zz)


@cython.boundscheck(False)
@cython.wraparound(False)
def calculate_dM_g(dtype recip, 
                   np.ndarray[dtype, ndim=1] drdrx, 
                   np.ndarray[dtype, ndim=2] dM):
    cdef dtype x, y, z, xx, xy, xz, yy, yz, zz
    cdef dtype xxx, xxy, xxz, xyy, xyz, xzz, yyy, yyz, yzz, zzz
    cdef dtype xxxx, xxxy, xxxz, xxyy, xxyz, xxzz, xyyy, xyyz, xyzz
    cdef dtype xzzz, yyyy, yyyz, yyzz, yzzz, zzzz
    cdef dtype xxxxx, xxxxy, xxxxz, xxxyy, xxxyz, xxxzz, xxyyy, xxyyz, xxyzz
    cdef dtype xxzzz, xyyyy, xyyyz, xyyzz, xyzzz, xzzzz, yyyyy, yyyyz, yyyzz
    cdef dtype yyzzz, yzzzz, zzzzz

    calculate_dM_f(recip, drdrx, dM)

    x = drdrx[0]
    y = drdrx[1]
    z = drdrx[2]

    xx = x * x
    xy = x * y
    xz = x * z
    yy = y * y
    yz = z * y
    zz = z * z
    xxx = x * xx
    xxy = x * xy
    xxz = x * xz
    xyy = y * xy
    xyz = y * xz
    xzz = z * xz
    yyy = y * yy
    yyz = y * yz
    yzz = z * yz
    zzz = z * zz
    xxxx = xxx * x
    xxxy = xxx * y
    xxxz = xxx * z
    xxyy = xxy * y
    xxyz = xxy * z
    xxzz = xxz * z
    xyyy = xyy * y
    xyyz = xyy * z
    xyzz = xyz * z
    xzzz = xzz * z
    yyyy = yyy * y
    yyyz = yyy * z
    yyzz = yyz * z
    yzzz = yzz * z
    zzzz = zzz * z
    xxxxx = xxxx * x
    xxxxy = xxxx * y
    xxxxz = xxxx * z
    xxxyy = xxxy * y
    xxxyz = xxxy * z
    xxxzz = xxxz * z
    xxyyy = xxyy * y
    xxyyz = xxyy * z
    xxyzz = xxyz * z
    xxzzz = xxzz * z
    xyyyy = xyyy * y
    xyyyz = xyyy * z
    xyyzz = xyyz * z
    xyzzz = xyzz * z
    xzzzz = xzzz * z
    yyyyy = yyyy * y
    yyyyz = yyyy * z
    yyyzz = yyyz * z
    yyzzz = yyzz * z
    yzzzz = yzzz * z
    zzzzz = zzzz * z

    dM[0, 20] = -recip * (4 * xxxxx - 4 * xxx)
    dM[0, 21] = -recip * (4 * xxxxy - 3 * xxy)
    dM[0, 22] = -recip * (4 * xxxxz - 3 * xxz)
    dM[0, 23] = -recip * (4 * xxxyy - 2 * xyy)
    dM[0, 24] = -recip * (4 * xxxyz - 2 * xyz)
    dM[0, 25] = -recip * (4 * xxxzz - 2 * xzz)
    dM[0, 26] = -recip * (4 * xxyyy - yyy)
    dM[0, 27] = -recip * (4 * xxyyz - yyz)
    dM[0, 28] = -recip * (4 * xxyzz - yzz)
    dM[0, 29] = -recip * (4 * xxzzz - zzz)
    dM[0, 30] = -recip * 4 * xyyyy
    dM[0, 31] = -recip * 4 * xyyyz
    dM[0, 32] = -recip * 4 * xyyzz
    dM[0, 33] = -recip * 4 * xyzzz
    dM[0, 34] = -recip * 4 * xzzzz

    dM[1, 20] = -recip * 4 * xxxxy
    dM[1, 21] = -recip * (4 * xxxyy - xxx)
    dM[1, 22] = -recip * 4 * xxxyz
    dM[1, 23] = -recip * (4 * xxyyy - 2 * xxy)
    dM[1, 24] = -recip * (4 * xxyyz - xxz)
    dM[1, 25] = -recip * 4 * xxyzz
    dM[1, 26] = -recip * (4 * xyyyy - 3 * xyy)
    dM[1, 27] = -recip * (4 * xyyyz - 2 * xyz)
    dM[1, 28] = -recip * (4 * xyyzz - xzz)
    dM[1, 29] = -recip * 4 * xyzzz
    dM[1, 30] = -recip * (4 * yyyyy - 4 * yyy)
    dM[1, 31] = -recip * (4 * yyyyz - 3 * yyz)
    dM[1, 32] = -recip * (4 * yyyzz - 2 * yzz)
    dM[1, 33] = -recip * (4 * yyzzz - zzz)
    dM[1, 34] = -recip * 4 * yzzzz

    dM[2, 20] = -recip * 4 * xxxxz
    dM[2, 21] = -recip * 4 * xxxyz
    dM[2, 22] = -recip * (4 * xxxzz - xxx)
    dM[2, 23] = -recip * 4 * xxyyz
    dM[2, 24] = -recip * (4 * xxyzz - xxy)
    dM[2, 25] = -recip * (4 * xxzzz - 2 * xxz)
    dM[2, 26] = -recip * 4 * xyyyz
    dM[2, 27] = -recip * (4 * xyyzz - xyy)
    dM[2, 28] = -recip * (4 * xyzzz - 2 * xyz)
    dM[2, 29] = -recip * (4 * xzzzz - 3 * xzz)
    dM[2, 30] = -recip * 4 * yyyyz
    dM[2, 31] = -recip * (4 * yyyzz - yyy)
    dM[2, 32] = -recip * (4 * yyzzz - 2 * yyz)
    dM[2, 33] = -recip * (4 * yzzzz - 3 * yzz)
    dM[2, 34] = -recip * (4 * zzzzz - 4 * zzz)


@cython.boundscheck(False)
@cython.wraparound(False)
def calculate_dM_h(dtype recip, 
                   np.ndarray[dtype, ndim=1] drdrx, 
                   np.ndarray[dtype, ndim=2] dM):
    cdef dtype x, y, z, xx, xy, xz, yy, yz, zz
    cdef dtype xxx, xxy, xxz, xyy, xyz, xzz, yyy, yyz, yzz, zzz
    cdef dtype xxxx, xxxy, xxxz, xxyy, xxyz, xxzz, xyyy, xyyz, xyzz
    cdef dtype xzzz, yyyy, yyyz, yyzz, yzzz, zzzz
    cdef dtype xxxxx, xxxxy, xxxxz, xxxyy, xxxyz, xxxzz, xxyyy, xxyyz, xxyzz
    cdef dtype xxzzz, xyyyy, xyyyz, xyyzz, xyzzz, xzzzz, yyyyy, yyyyz, yyyzz
    cdef dtype yyzzz, yzzzz, zzzzz
    cdef dtype xxxxxx, xxxxxy, xxxxxz, xxxxyy, xxxxyz, xxxxzz, xxxyyy, xxxyyz
    cdef dtype xxxyzz, xxxzzz, xxyyyy, xxyyyz, xxyyzz, xxyzzz, xxzzzz, xyyyyy
    cdef dtype xyyyyz, xyyyzz, xyyzzz, xyzzzz, xzzzzz, yyyyyy, yyyyyz, yyyyzz
    cdef dtype yyyzzz, yyzzzz, yzzzzz, zzzzzz

    calculate_dM_g(recip, drdrx, dM)

    x = drdrx[0]
    y = drdrx[1]
    z = drdrx[2]

    xx = x * x
    xy = x * y
    xz = x * z
    yy = y * y
    yz = z * y
    zz = z * z
    xxx = x * xx
    xxy = x * xy
    xxz = x * xz
    xyy = y * xy
    xyz = y * xz
    xzz = z * xz
    yyy = y * yy
    yyz = y * yz
    yzz = z * yz
    zzz = z * zz
    xxxx = xxx * x
    xxxy = xxx * y
    xxxz = xxx * z
    xxyy = xxy * y
    xxyz = xxy * z
    xxzz = xxz * z
    xyyy = xyy * y
    xyyz = xyy * z
    xyzz = xyz * z
    xzzz = xzz * z
    yyyy = yyy * y
    yyyz = yyy * z
    yyzz = yyz * z
    yzzz = yzz * z
    zzzz = zzz * z
    xxxxx = xxxx * x
    xxxxy = xxxx * y
    xxxxz = xxxx * z
    xxxyy = xxxy * y
    xxxyz = xxxy * z
    xxxzz = xxxz * z
    xxyyy = xxyy * y
    xxyyz = xxyy * z
    xxyzz = xxyz * z
    xxzzz = xxzz * z
    xyyyy = xyyy * y
    xyyyz = xyyy * z
    xyyzz = xyyz * z
    xyzzz = xyzz * z
    xzzzz = xzzz * z
    yyyyy = yyyy * y
    yyyyz = yyyy * z
    yyyzz = yyyz * z
    yyzzz = yyzz * z
    yzzzz = yzzz * z
    zzzzz = zzzz * z
    xxxxxx = xxxxx * x
    xxxxxy = xxxxx * y
    xxxxxz = xxxxx * z
    xxxxyy = xxxxy * y
    xxxxyz = xxxxy * z
    xxxxzz = xxxxz * z
    xxxyyy = xxxxy * y
    xxxyyz = xxxyy * z
    xxxyzz = xxxyz * z
    xxxzzz = xxxzz * z
    xxyyyy = xxyyy * y
    xxyyyz = xxyyy * z
    xxyyzz = xxyyz * z
    xxyzzz = xxyzz * z
    xxzzzz = xxzzz * z
    xyyyyy = xyyyy * y
    xyyyyz = xyyyy * z
    xyyyzz = xyyyz * z
    xyyzzz = xyyzz * z
    xyzzzz = xyzzz * z
    xzzzzz = xzzzz * z
    yyyyyy = yyyyy * y
    yyyyyz = yyyyy * z
    yyyyzz = yyyyz * z
    yyyzzz = yyyzz * z
    yyzzzz = yyzzz * z
    yzzzzz = yzzzz * z
    zzzzzz = zzzzz * z

    dM[0, 35] = -recip * (5 * xxxxxx - 5 * xxxx)
    dM[0, 36] = -recip * (5 * xxxxxy - 4 * xxxy)
    dM[0, 37] = -recip * (5 * xxxxxz - 4 * xxxz)
    dM[0, 38] = -recip * (5 * xxxxyy - 3 * xxyy)
    dM[0, 39] = -recip * (5 * xxxxyz - 3 * xxyz)
    dM[0, 40] = -recip * (5 * xxxxzz - 3 * xxzz)
    dM[0, 41] = -recip * (5 * xxxyyy - 2 * xyyy)
    dM[0, 42] = -recip * (5 * xxxyyz - 2 * xyyz)
    dM[0, 43] = -recip * (5 * xxxyzz - 2 * xyzz)
    dM[0, 44] = -recip * (5 * xxxzzz - 2 * xzzz)
    dM[0, 45] = -recip * (5 * xxyyyy - yyyy)
    dM[0, 46] = -recip * (5 * xxyyyz - yyyz)
    dM[0, 47] = -recip * (5 * xxyyzz - yyzz)
    dM[0, 48] = -recip * (5 * xxyzzz - yzzz)
    dM[0, 49] = -recip * (5 * xxzzzz - zzzz)
    dM[0, 50] = -recip * 5 * xyyyyy
    dM[0, 51] = -recip * 5 * xyyyyz
    dM[0, 52] = -recip * 5 * xyyyzz
    dM[0, 53] = -recip * 5 * xyyzzz
    dM[0, 54] = -recip * 5 * xyzzzz
    dM[0, 55] = -recip * 5 * xzzzzz

    dM[1, 35] = -recip * 5 * xxxxxy
    dM[1, 36] = -recip * (5 * xxxxyy - xxxx)
    dM[1, 37] = -recip * 5 * xxxxyz
    dM[1, 38] = -recip * (5 * xxxyyy - 2 * xxxy)
    dM[1, 39] = -recip * (5 * xxxyyz - xxxz)
    dM[1, 40] = -recip * 5 * xxxyzz
    dM[1, 41] = -recip * (5 * xxyyyy - 3 * xxyy)
    dM[1, 42] = -recip * (5 * xxyyyz - 2 * xxyz)
    dM[1, 43] = -recip * (5 * xxyyzz - xxzz)
    dM[1, 44] = -recip * 5 * xxyzzz
    dM[1, 45] = -recip * (5 * xyyyyy - 4 * xyyy)
    dM[1, 46] = -recip * (5 * xyyyyz - 3 * xyyz)
    dM[1, 47] = -recip * (5 * xyyyzz - 2 * xyzz)
    dM[1, 48] = -recip * (5 * xyyzzz - xzzz)
    dM[1, 49] = -recip * 5 * xyzzzz
    dM[1, 50] = -recip * (5 * yyyyyy - 5 * yyyy)
    dM[1, 51] = -recip * (5 * yyyyyz - 4 * yyyz)
    dM[1, 52] = -recip * (5 * yyyyzz - 3 * yyzz)
    dM[1, 53] = -recip * (5 * yyyzzz - 2 * yzzz)
    dM[1, 54] = -recip * (5 * yyzzzz - zzzz)
    dM[1, 55] = -recip * 5 * yzzzzz

    dM[2, 35] = -recip * 5 * xxxxxz
    dM[2, 36] = -recip * 5 * xxxxyz
    dM[2, 37] = -recip * (5 * xxxxzz - xxxx)
    dM[2, 38] = -recip * 5 * xxxyyz
    dM[2, 39] = -recip * (5 * xxxyzz - xxxy)
    dM[2, 40] = -recip * (5 * xxxzzz - 2 * xxxz)
    dM[2, 41] = -recip * 5 * xxyyyz
    dM[2, 42] = -recip * (5 * xxyyzz - xxyy)
    dM[2, 43] = -recip * (5 * xxyzzz - 2 * xxyz)
    dM[2, 44] = -recip * (5 * xxzzzz - 3 * xxzz)
    dM[2, 45] = -recip * 5 * xyyyyz
    dM[2, 46] = -recip * (5 * xyyyzz - xyyy)
    dM[2, 47] = -recip * (5 * xyyzzz - 2 * xyyz)
    dM[2, 48] = -recip * (5 * xyzzzz - 3 * xyzz)
    dM[2, 49] = -recip * (5 * xzzzzz - 4 * xzzz)
    dM[2, 50] = -recip * 5 * yyyyyz
    dM[2, 51] = -recip * (5 * yyyyzz - yyyy)
    dM[2, 52] = -recip * (5 * yyyzzz - 2 * yyyz)
    dM[2, 53] = -recip * (5 * yyzzzz - 3 * yyzz)
    dM[2, 54] = -recip * (5 * yzzzzz - 4 * yzzz)
    dM[2, 55] = -recip * (5 * zzzzzz - 5 * zzzz)


@cython.boundscheck(False)
@cython.wraparound(False)
def calculate_dM(itype max_moment, 
                 dtype recip, 
                 np.ndarray[dtype, ndim=1] drdrx, 
                 np.ndarray[dtype, ndim=2] dM):
    if max_moment == 0:
        calculate_dM_s(dM)
    elif max_moment == 1:
        calculate_dM_p(recip, drdrx, dM)
    elif max_moment == 2:
        calculate_dM_d(recip, drdrx, dM)
    elif max_moment == 3:
        calculate_dM_f(recip, drdrx, dM)
    elif max_moment == 4:
        calculate_dM_g(recip, drdrx, dM)
    elif max_moment == 5:
        calculate_dM_h(recip, drdrx, dM)
    else:
        raise ValueError("max_moment must be 0, 1, 2, 3, 4 or 5")


@cython.boundscheck(False)
@cython.wraparound(False)
def kernel_F2(itype max_moment, 
              np.ndarray[dtype, ndim=4] V, 
              np.ndarray[dtype, ndim=3] R,
              np.ndarray[dtype, ndim=4] drdrx,
              np.ndarray[itype, ndim=3] mask,
              np.ndarray[dtype, ndim=4] F2):
    cdef int adim, bdim, cdim, ddim
    cdef int a, b, c
    cdef double recip

    adim = V.shape[0]
    bdim = V.shape[1]
    cdim = V.shape[2]
    ddim = V.shape[3]

    cdef np.ndarray[dtype, ndim=2] dM = np.empty((3, ddim), dtype=np.float64)

    for a in range(adim):
        for b in range(bdim):
            for c in range(cdim):
                if mask[a, b, c] == 0:
                    continue
                recip = 1.0 / R[a, b, c]
                calculate_dM(max_moment, recip, drdrx[a, b, c], dM)
                F2[a, b, c] = dM @ V[a, b, c]


@cython.boundscheck(False)
@cython.wraparound(False)
def kernel_F1(np.ndarray[dtype, ndim=4] U, 
              np.ndarray[dtype, ndim=4] dHdr,
              np.ndarray[dtype, ndim=4] drdrx,
              np.ndarray[itype, ndim=3] mask,
              np.ndarray[dtype, ndim=4] F1):
    
    cdef int adim, bdim, cdim
    cdef int a, b, c

    adim = U.shape[0]
    bdim = U.shape[1]
    cdim = U.shape[2]

    for a in range(adim):
        for b in range(bdim):
            for c in range(cdim):
                if mask[a, b, c] == 0:
                    continue
                F1[a, b, c] = np.dot(U[a, b, c], dHdr[a, b, c]) * drdrx[a, b, c]


@cython.boundscheck(False)
@cython.wraparound(False)
def sum_forces(np.ndarray[dtype, ndim=4] F, 
               np.ndarray[itype, ndim=3] i_abc,
               np.ndarray[itype, ndim=3] j_abc,
               np.ndarray[itype, ndim=1] a2i,
               np.ndarray[dtype, ndim=3] R_abc,
               np.ndarray[dtype, ndim=4] dr_abcx,
               np.ndarray[dtype, ndim=2] forces,
               np.ndarray[dtype, ndim=2] virial):
    cdef int adim, bdim, cdim
    cdef int a, b, c, i, j

    adim = F.shape[0]
    bdim = F.shape[1]
    cdim = F.shape[2]
    v = np.zeros(6, dtype=np.float64)

    for a in range(adim):
        for b in range(bdim):
            for c in range(cdim):
                i = i_abc[a, b, c]
                j = j_abc[a, b, c]
                if i < 0 or j < 0:
                    continue
                forces[a2i[i]] += F[a, b, c]
                forces[a2i[j]] -= F[a, b, c]
                v[0] = -dr_abcx[a, b, c, 0] * F[a, b, c, 0] * R_abc[a, b, c]
                v[1] = -dr_abcx[a, b, c, 1] * F[a, b, c, 1] * R_abc[a, b, c]
                v[2] = -dr_abcx[a, b, c, 2] * F[a, b, c, 2] * R_abc[a, b, c]
                v[3] = -dr_abcx[a, b, c, 1] * F[a, b, c, 2] * R_abc[a, b, c]
                v[4] = -dr_abcx[a, b, c, 0] * F[a, b, c, 2] * R_abc[a, b, c]
                v[5] = -dr_abcx[a, b, c, 0] * F[a, b, c, 1] * R_abc[a, b, c]
                virial[a2i[i]] += 0.5 * v
                virial[a2i[j]] += 0.5 * v
                
