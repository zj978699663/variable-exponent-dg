import csv
import os
import tempfile
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy import sparse
from scipy.sparse.linalg import spsolve


ROOT = Path(__file__).resolve().parent
BUILD = ROOT / "build"
REFERENCE = ROOT / "reference"


NS = (2, 4, 8, 16, 32)
METHODS = ("global", "patchwise")


def gauss01(n):
    x, w = leggauss(n)
    return 0.5 * (x + 1.0), 0.5 * w


def tri_duffy(n):
    r, wr = gauss01(n)
    s, ws = gauss01(n)
    pts = []
    wts = []
    for i, ri in enumerate(r):
        for j, sj in enumerate(s):
            pts.append((ri, (1.0 - ri) * sj))
            wts.append(wr[i] * ws[j] * (1.0 - ri))
    return np.array(pts), np.array(wts)


@dataclass
class Mesh:
    xy: np.ndarray
    tri: np.ndarray
    area: np.ndarray
    grads: np.ndarray
    edges: list
    boundary_vertex: np.ndarray
    vertex_incidence: list
    tri_edge_ids: np.ndarray


def make_mesh(n):
    xy = np.array([(i / n, j / n) for j in range(n + 1) for i in range(n + 1)], float)

    def vid(i, j):
        return j * (n + 1) + i

    tris = []
    for j in range(n):
        for i in range(n):
            v00 = vid(i, j)
            v10 = vid(i + 1, j)
            v01 = vid(i, j + 1)
            v11 = vid(i + 1, j + 1)
            tris.append((v00, v10, v11))
            tris.append((v00, v11, v01))
    tri = np.array(tris, int)
    nt = len(tri)
    area = np.zeros(nt)
    grads = np.zeros((nt, 3, 2))
    edge_map = {}
    eids = np.full((nt, 3), -1, int)
    lep = [(0, 1), (1, 2), (2, 0)]
    for k, t in enumerate(tri):
        P = xy[t]
        J = np.column_stack((P[1] - P[0], P[2] - P[0]))
        det = np.linalg.det(J)
        if det <= 0:
            raise ValueError("triangle orientation")
        area[k] = 0.5 * det
        invT = np.linalg.inv(J).T
        grads[k, 1] = invT @ np.array([1.0, 0.0])
        grads[k, 2] = invT @ np.array([0.0, 1.0])
        grads[k, 0] = -grads[k, 1] - grads[k, 2]
        for le, (a, b) in enumerate(lep):
            ga, gb = int(t[a]), int(t[b])
            key = tuple(sorted((ga, gb)))
            pa, pb = xy[ga], xy[gb]
            tang = pb - pa
            L = np.linalg.norm(tang)
            nout = np.array([tang[1], -tang[0]]) / L
            rec = {
                "elem": k,
                "le": le,
                "loc": (a, b),
                "verts": (ga, gb),
                "normal": nout,
                "length": L,
            }
            edge_map.setdefault(key, []).append(rec)
    edges = []
    for eid, (key, sides) in enumerate(edge_map.items()):
        for sd in sides:
            eids[sd["elem"], sd["le"]] = eid
        edges.append(
            {
                "id": eid,
                "key": key,
                "sides": sides,
                "interior": len(sides) == 2,
                "length": sides[0]["length"],
            }
        )
    nv = len(xy)
    bnd = (
        np.isclose(xy[:, 0], 0)
        | np.isclose(xy[:, 0], 1)
        | np.isclose(xy[:, 1], 0)
        | np.isclose(xy[:, 1], 1)
    )
    inc = [[] for _ in range(nv)]
    for k, t in enumerate(tri):
        for li, g in enumerate(t):
            inc[int(g)].append((k, li))
    return Mesh(xy, tri, area, grads, edges, bnd, inc, eids)


def assemble_G(mesh):
    nt = len(mesh.tri)
    nd = 3 * nt
    rows = []
    cols = []
    data = []
    for k in range(nt):
        for li in range(3):
            j = 3 * k + li
            for c in range(2):
                rows.append(2 * k + c)
                cols.append(j)
                data.append(mesh.grads[k, li, c])
    D = sparse.csr_matrix((data, (rows, cols)), shape=(2 * nt, nd))
    rr = []
    cc = []
    dd = []
    for e in mesh.edges:
        L = e["length"]
        if e["interior"]:
            s0, s1 = e["sides"]
            for s, other in ((s0, s1), (s1, s0)):
                k = s["elem"]
                n = s["normal"]
                A = mesh.area[k]
                coef = L / (4 * A)
                for li in s["loc"]:
                    j = 3 * k + li
                    for c in range(2):
                        rr.append(2 * k + c)
                        cc.append(j)
                        dd.append(coef * n[c])
                ko = other["elem"]
                for li in other["loc"]:
                    j = 3 * ko + li
                    for c in range(2):
                        rr.append(2 * k + c)
                        cc.append(j)
                        dd.append(-coef * n[c])
        else:
            s = e["sides"][0]
            k = s["elem"]
            n = s["normal"]
            A = mesh.area[k]
            coef = L / (2 * A)
            for li in s["loc"]:
                j = 3 * k + li
                for c in range(2):
                    rr.append(2 * k + c)
                    cc.append(j)
                    dd.append(coef * n[c])
    R = sparse.csr_matrix((dd, (rr, cc)), shape=(2 * nt, nd))
    return D, R, D - R


def assemble_smoother(mesh):
    nt = len(mesh.tri)
    nd = 3 * nt
    nv = len(mesh.xy)
    int_edges = [e for e in mesh.edges if e["interior"]]
    e_to_row = {e["id"]: nv + i for i, e in enumerate(int_edges)}
    ns = nv + len(int_edges)
    S = sparse.lil_matrix((ns, nd), dtype=float)
    for g in range(nv):
        if mesh.boundary_vertex[g]:
            continue

        k, li = mesh.vertex_incidence[g][0]
        S[g, 3 * k + li] = 1.0
    for e in int_edges:
        row = e_to_row[e["id"]]
        for s in e["sides"]:
            k = s["elem"]
            for li in s["loc"]:
                S[row, 3 * k + li] += 0.25
        a, b = e["key"]
        S[row, :] -= 0.5 * S[a, :]
        S[row, :] -= 0.5 * S[b, :]
    return S.tocsr(), e_to_row, ns


def map_tri_points(mesh, k, refpts):
    P = mesh.xy[mesh.tri[k]]
    return (
        P[0] + refpts[:, 0, None] * (P[1] - P[0]) + refpts[:, 1, None] * (P[2] - P[0])
    )


def smoother_local_basis(mesh, k, refpts, e_to_row):
    xi = refpts[:, 0]
    eta = refpts[:, 1]
    lam = np.column_stack((1 - xi - eta, xi, eta))
    idx = list(map(int, mesh.tri[k]))
    grads = [np.tile(mesh.grads[k, i], (len(refpts), 1)) for i in range(3)]
    lep = [(0, 1), (1, 2), (2, 0)]
    for le, (a, b) in enumerate(lep):
        eid = int(mesh.tri_edge_ids[k, le])
        if eid in e_to_row:
            idx.append(e_to_row[eid])
            gb = 6 * (
                lam[:, a, None] * mesh.grads[k, b] + lam[:, b, None] * mesh.grads[k, a]
            )
            grads.append(gb)
    return np.array(idx, int), np.stack(grads, axis=1)


def pfun_factory(kind):
    if kind == "mixed":
        return lambda x: 1.6 + 0.8 * x[..., 0]
    if kind == "super":
        return lambda x: 2.2 + 0.4 * x[..., 0] + 0.2 * x[..., 1]
    raise ValueError(kind)


def grad_u_exact(x):
    X = x[..., 0]
    Y = x[..., 1]
    return np.stack(((1 - 2 * X) * Y * (1 - Y), (1 - 2 * Y) * X * (1 - X)), axis=-1)


def Avec(p, g):
    n = np.linalg.norm(g, axis=-1)
    fac = np.zeros_like(n)
    mask = n > 0
    fac[mask] = n[mask] ** (p[mask] - 2.0)
    return fac[..., None] * g


def B_env(pmin, pmax, beta, q):
    q = np.asarray(q)
    n = np.linalg.norm(q, axis=-1)
    out = np.zeros_like(q)
    for r in (pmin, pmax):
        a = 0.0 if r <= 2 else beta
        if a == 0:
            fac = np.zeros_like(n)
            m = n > 0
            fac[m] = n[m] ** (r - 2)
        else:
            fac = (a + n) ** (r - 2)
        out += fac[..., None] * q
    return out


class EPDGProblem:
    def __init__(self, n, pkind="mixed", alpha=10.0, qorder=8):
        self.n = n
        self.mesh = make_mesh(n)
        self.pfun = pfun_factory(pkind)
        self.pkind = pkind
        self.alpha = float(alpha)
        self.qorder = qorder
        self.pmin = 1.6 if pkind == "mixed" else 2.2
        self.pmax = 2.4 if pkind == "mixed" else 2.8
        self.D, self.R, self.G = assemble_G(self.mesh)
        self.S, self.e_to_row, self.ns = assemble_smoother(self.mesh)
        self.nd = self.G.shape[1]
        self.nt = len(self.mesh.tri)
        self.refq, self.refw = tri_duffy(qorder)
        self.edgeq, self.edgew = gauss01(qorder)
        self._precompute()

    def _precompute(self):
        m = self.mesh
        self.elem_x = []
        self.elem_w = []
        self.sidx = []
        self.sgr = []
        self.loadS = np.zeros(self.ns)
        for k in range(self.nt):
            x = map_tri_points(m, k, self.refq)
            w = self.refw * (2 * m.area[k])
            idx, gr = smoother_local_basis(m, k, self.refq, self.e_to_row)
            self.elem_x.append(x)
            self.elem_w.append(w)
            self.sidx.append(idx)
            self.sgr.append(gr)
            p = self.pfun(x)
            ge = grad_u_exact(x)
            Ae = Avec(p, ge)
            vals = np.einsum("qi,qmi,q->m", Ae, gr, w)
            self.loadS[idx] += vals
        self.load_dg = np.asarray(self.S.T @ self.loadS).ravel()
        nv = len(m.xy)
        self.int_vertices = np.where(~m.boundary_vertex)[0]
        self.v_to_int = -np.ones(nv, int)
        self.v_to_int[self.int_vertices] = np.arange(len(self.int_vertices))
        self.load_cf = np.zeros(len(self.int_vertices))
        for k in range(self.nt):
            x = self.elem_x[k]
            w = self.elem_w[k]
            p = self.pfun(x)
            ge = grad_u_exact(x)
            Ae = Avec(p, ge)
            for li, g in enumerate(m.tri[k]):
                ii = self.v_to_int[g]
                if ii >= 0:
                    self.load_cf[ii] += np.sum(
                        w * np.einsum("qi,i->q", Ae, m.grads[k, li])
                    )

    def residual(self, U):
        U = np.asarray(U)
        g = (self.G @ U).reshape(self.nt, 2)
        beta = float(np.max(np.linalg.norm(g, axis=1)))
        rS = np.zeros(self.ns)
        for k in range(self.nt):
            X = self.elem_x[k]
            w = self.elem_w[k]
            p = self.pfun(X)
            Ah = Avec(p, np.tile(g[k], (len(p), 1)))
            vals = np.einsum("qi,qmi,q->m", Ah, self.sgr[k], w)
            rS[self.sidx[k]] += vals
        r = np.asarray(self.S.T @ rS).ravel() - self.load_dg
        tq = self.edgeq
        ew = self.edgew
        for e in self.mesh.edges:
            L = e["length"]
            s0 = e["sides"][0]
            k0 = s0["elem"]
            a0, b0 = s0["loc"]
            tr0 = (1 - tq) * U[3 * k0 + a0] + tq * U[3 * k0 + b0]
            n = s0["normal"]
            if e["interior"]:
                s1 = e["sides"][1]
                k1 = s1["elem"]
                a1, b1 = s1["loc"]
                ga, gb = s0["verts"]
                locmap = {
                    int(self.mesh.tri[k1][a1]): a1,
                    int(self.mesh.tri[k1][b1]): b1,
                }
                tr1 = (1 - tq) * U[3 * k1 + locmap[ga]] + tq * U[3 * k1 + locmap[gb]]
                jump = tr0 - tr1
            else:
                s1 = None
                jump = tr0
            qv = (jump / L)[:, None] * n[None, :]
            B = B_env(self.pmin, self.pmax, beta, qv)
            bn = B @ n
            wa = L * ew
            r[3 * k0 + a0] += self.alpha * np.sum(wa * bn * (1 - tq))
            r[3 * k0 + b0] += self.alpha * np.sum(wa * bn * tq)
            if s1 is not None:
                ga, gb = s0["verts"]
                locmap = {
                    int(self.mesh.tri[k1][a1]): a1,
                    int(self.mesh.tri[k1][b1]): b1,
                }
                la = locmap[ga]
                lb = locmap[gb]
                r[3 * k1 + la] -= self.alpha * np.sum(wa * bn * (1 - tq))
                r[3 * k1 + lb] -= self.alpha * np.sum(wa * bn * tq)
        return r

    def cfem_to_dg(self, vals):
        U = np.zeros(self.nd)
        for k, t in enumerate(self.mesh.tri):
            U[3 * k : 3 * k + 3] = vals[t]
        return U

class FastProblem(EPDGProblem):
    def __init__(
        self,
        n,
        kind="mixed",
        alpha=2.0,
        qorder=6,
        local=False,
        constant=None,
        sharp=False,
    ):
        super().__init__(n, kind, alpha, qorder)
        self.local = local
        self.constant = constant
        self.sharp = sharp
        if constant is not None:
            self.pfun = lambda x: np.full(x.shape[:-1], constant)
            self.pmin = self.pmax = constant
        self.X = np.concatenate(self.elem_x)
        self.W = np.concatenate(self.elem_w)
        self.P = self.pfun(self.X)
        self.Q = len(self.refq)
        self.Gx = self.G[0::2].tocsr()
        self.Gy = self.G[1::2].tocsr()

        rows = []
        cols = []
        dx = []
        dy = []
        for k in range(self.nt):
            idx, gr = self.sidx[k], self.sgr[k]
            for qi in range(self.Q):
                for j, si in enumerate(idx):
                    rows.append(k * self.Q + qi)
                    cols.append(si)
                    dx.append(gr[qi, j, 0])
                    dy.append(gr[qi, j, 1])
        shape = (self.nt * self.Q, self.ns)
        self.Vx = (sparse.csr_matrix((dx, (rows, cols)), shape=shape) @ self.S).tocsr()
        self.Vy = (sparse.csr_matrix((dy, (rows, cols)), shape=shape) @ self.S).tocsr()

        rr = []
        cc = []
        dd = []
        self.L = []
        self.patch = []
        pmins = []
        pmaxs = []
        pv = self.pfun(self.mesh.xy)
        for ei, e in enumerate(self.mesh.edges):
            L = e["length"]
            self.L.extend([L] * qorder)
            verts = e["key"]
            ks = set()
            for vertex in verts:
                ks.update(k for k, li in self.mesh.vertex_incidence[vertex])
            self.patch.append(sorted(ks))
            ppatch = pv[self.mesh.tri[list(ks)]]
            pmins.extend([float(ppatch.min())] * qorder)
            pmaxs.extend([float(ppatch.max())] * qorder)
            s0 = e["sides"][0]
            ga, gb = s0["verts"]
            for qi, t in enumerate(self.edgeq):
                row = ei * qorder + qi
                for side, sgn in zip(e["sides"], [1.0, -1.0]):
                    k = side["elem"]
                    mp = {int(v): li for li, v in enumerate(self.mesh.tri[k])}
                    rr.extend([row, row])
                    cc.extend([3 * k + mp[ga], 3 * k + mp[gb]])
                    dd.extend([sgn * (1 - t) / L, sgn * t / L])
        self.J = sparse.csr_matrix((dd, (rr, cc)), shape=(len(self.L), self.nd))
        self.L = np.array(self.L)
        self.WF = self.L**2 * np.tile(self.edgew, len(self.mesh.edges))
        self.Pminus = np.array(pmins) if local else np.full(len(self.L), self.pmin)
        self.Pplus = np.array(pmaxs) if local else np.full(len(self.L), self.pmax)
        self.Gqx = self.Gx[np.repeat(np.arange(self.nt), self.Q)].tocsr()
        self.Gqy = self.Gy[np.repeat(np.arange(self.nt), self.Q)].tocsr()
        ge = grad_u_exact(self.X)
        if sharp:
            x, y = self.X.T
            ge = np.where(
                (y <= x)[:, None],
                np.stack((-6 * y, 6 * (1 - x)), axis=1),
                np.stack((6 * (1 - y), -6 * x), axis=1),
            )
        self.ExactGrad = ge
        nrm = np.linalg.norm(ge, axis=1)
        fac = np.zeros_like(nrm)
        nz = nrm > 0
        fac[nz] = nrm[nz] ** (self.P[nz] - 2)
        if sharp:
            fac = np.ones_like(nrm)
        self.Force = self.Vx.T @ (self.W * fac * ge[:, 0]) + self.Vy.T @ (
            self.W * fac * ge[:, 1]
        )

        ir = []
        ic = []
        for k, t in enumerate(self.mesh.tri):
            for li, v in enumerate(t):
                ii = self.v_to_int[v]
                if ii >= 0:
                    ir.append(3 * k + li)
                    ic.append(ii)
        self.CF = sparse.csr_matrix(
            (np.ones(len(ir)), (ir, ic)), shape=(self.nd, len(self.int_vertices))
        )

    def fields(self, U):
        gx = self.Gx @ U
        gy = self.Gy @ U
        norms = np.hypot(gx, gy)
        if self.local:
            beta = np.repeat([norms[ks].max() for ks in self.patch], self.qorder)
        else:
            beta = np.full(len(self.L), norms.max())
        return np.repeat(gx, self.Q), np.repeat(gy, self.Q), beta

    def residual_fast(self, U, tangent=False):
        gx, gy, beta = self.fields(U)
        ng = np.hypot(gx, gy)
        fac = np.zeros_like(ng)
        mask = ng > 0
        fac[mask] = ng[mask] ** (self.P[mask] - 2)
        res = (
            self.Vx.T @ (self.W * fac * gx)
            + self.Vy.T @ (self.W * fac * gy)
            - self.Force
        )
        t = self.J @ U
        ab = np.abs(t)
        bval = np.zeros_like(t)
        bder = np.zeros_like(t)
        for pp in [self.Pminus, self.Pplus]:
            shift = np.where(pp > 2, beta, 0.0)
            scale = shift + ab
            nz = scale > 0
            c = np.zeros_like(t)
            c[nz] = scale[nz] ** (pp[nz] - 2)
            bval += c * t
            safe = np.maximum(scale, 1e-14)
            bder += safe ** (pp - 2) * (1 + (pp - 2) * ab / safe)
        res += self.alpha * (self.J.T @ (self.WF * bval))
        if not tangent:
            return np.asarray(res)

        nn = np.maximum(ng, 1e-14)
        base = nn ** (self.P - 2)
        extra = (self.P - 2) * nn ** (self.P - 4)
        a = self.W * (base + extra * gx * gx)
        b = self.W * extra * gx * gy
        d = self.W * (base + extra * gy * gy)
        jac = self.Vx.T @ (
            self.Gqx.multiply(a[:, None]) + self.Gqy.multiply(b[:, None])
        )
        jac += self.Vy.T @ (
            self.Gqx.multiply(b[:, None]) + self.Gqy.multiply(d[:, None])
        )
        jac += self.alpha * self.J.T @ self.J.multiply((self.WF * bder)[:, None])
        return np.asarray(res), jac.tocsr()

    def solve_fast(self, start=None, cf=False, tol=1e-10, maxiter=160):
        B = self.CF if cf else sparse.eye(self.nd, format="csr")
        if start is None:
            vals = self.mesh.xy[self.int_vertices]
            if cf:
                x = vals[:, 0] * (1 - vals[:, 0]) * vals[:, 1] * (1 - vals[:, 1])
            else:
                x = self.cfem_to_dg(np.zeros(len(self.mesh.xy)))
        else:
            x = np.array(start, copy=True)
        if not len(x):
            return B @ x, dict(
                success=True, res_inf=0.0, iterations=0
            )
        for it in range(maxiter):
            U = B @ x
            r, j = self.residual_fast(U, True)
            r = np.asarray(B.T @ r)
            norm = float(np.linalg.norm(r, np.inf))
            if norm < tol:
                break
            jac = (B.T @ j @ B).tocsc()
            step = spsolve(jac, -r)
            if not np.all(np.isfinite(step)):
                break
            old = np.linalg.norm(r)
            lam = 1.0
            for ls in range(30):
                xn = x + lam * step
                rn = B.T @ self.residual_fast(B @ xn)
                if np.linalg.norm(rn) < (1 - 1e-4 * lam) * old:
                    break
                lam *= 0.5
            else:
                break
            x = xn
        U = B @ x
        final = float(np.linalg.norm(B.T @ self.residual_fast(U), np.inf))
        return U, dict(
            success=final < tol,
            res_inf=final,
            iterations=it + 1,
        )

    def indicators(self, U, CF=None):
        gx, gy, beta = self.fields(U)
        ng = np.hypot(gx, gy)
        c = np.zeros_like(ng)
        nz = ng > 0
        c[nz] = ng[nz] ** ((self.P[nz] - 2) / 2)
        fh = np.stack((c * gx, c * gy), axis=1)
        ge = self.ExactGrad
        nn = np.linalg.norm(ge, axis=1)
        ce = np.zeros_like(nn)
        mz = nn > 0
        ce[mz] = nn[mz] ** ((self.P[mz] - 2) / 2)
        E = float(np.sum(self.W * np.sum((fh - ce[:, None] * ge) ** 2, axis=1)))
        if CF is None:
            D = float("nan")
        else:
            cx, cy, _ = self.fields(CF)
            cn = np.hypot(cx, cy)
            cc = np.zeros_like(cn)
            mk = cn > 0
            cc[mk] = cn[mk] ** ((self.P[mk] - 2) / 2)
            D = float(
                np.sum(self.W * ((c * gx - cc * cx) ** 2 + (c * gy - cc * cy) ** 2))
            )
        t = np.abs(self.J @ U)
        m = np.zeros_like(t)

        from numpy.polynomial.legendre import leggauss

        zz, ww = leggauss(32)
        zz = (zz + 1) / 2
        ww = ww / 2
        for pp in [self.Pminus, self.Pplus]:
            aa = np.where(pp > 2, beta, 0.0)
            unshifted = aa == 0
            m[unshifted] += t[unshifted] ** pp[unshifted] / pp[unshifted]
            sel = ~unshifted
            if sel.any():
                ss = t[sel, None] * zz
                m[sel] += t[sel] ** 2 * np.sum(
                    ww * (aa[sel, None] + ss) ** (pp[sel, None] - 2) * zz, axis=1
                )
        M = float(self.WF @ m)
        return dict(
            e_nat=E**0.5,
            M=M,
            j_env=M**0.5,
            e_qo=(E + self.alpha * M) ** 0.5,
            D=D,
            beta=float(beta.max()),
        )


def value(cell):
    try:
        return Decimal(cell)
    except InvalidOperation:
        return cell


def write_table(n, header, rows):
    name = f"table{n}.csv"
    actual = [[str(cell) for cell in row] for row in [header, *rows]]
    with (REFERENCE / name).open(encoding="utf-8", newline="") as stream:
        expected = list(csv.reader(stream))
    if len(actual) != len(expected):
        raise ValueError(
            f"{name}: expected {len(expected) - 1} rows, got {len(actual) - 1}"
        )
    for i, (a, b) in enumerate(zip(actual, expected), 1):
        if list(map(value, a)) != list(map(value, b)):
            raise ValueError(f"{name}, line {i}: {a} differs from {b}")
    with (BUILD / name).open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerows(actual)
    print(f"{name}: {len(actual) - 1} rows checked")


def fmt(value):
    return f"{value:.2e}"


def residual(problem, u):
    error = np.linalg.norm(problem.residual_fast(u), np.inf)
    assert error < 1e-10, error


BLUE, ORANGE, TEAL, INK = "#176b91", "#ce6a22", "#16856b", "#30343b"
COLORS = (BLUE, ORANGE, TEAL)
MARKERS = ("o", "s", "^")
LINES = ("-", "--", "-.")
PRESET = dict(
    axis_label_pt=9.5,
    tick_pt=8.5,
    legend_pt=8.0,
    axes_pt=0.60,
    curve_pt=0.90,
    marker_pt=4.0,
    marker_edge_pt=0.80,
    legend_edge_pt=0.45,
    legend_alpha=0.80,
)
RECT = (0.145, 0.16, 0.82, 0.82)
WIDTH = 0.618 * 360 / 72.27
HEIGHT = WIDTH * RECT[2] / (1.5 * RECT[3])


def plotting():
    cache = BUILD / "cache"
    cache.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache / "matplotlib"))
    tempfile.tempdir = str(cache)
    import matplotlib as mpl

    mpl.use("pgf")
    import matplotlib.pyplot as plt

    mpl.rcParams.update(
        {
            "pgf.texsystem": "pdflatex",
            "pgf.rcfonts": False,
            "pgf.preamble": r"\usepackage[T1]{fontenc}\usepackage{amsmath,bm}",
            "text.usetex": True,
            "font.family": "serif",
            "font.size": 9.5,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.0,
            "axes.linewidth": 0.60,
            "lines.linewidth": 0.90,
            "lines.markersize": 4.0,
            "lines.markeredgewidth": 0.80,
            "lines.markerfacecolor": "white",
            "axes.unicode_minus": False,
            "axes.grid": False,
            "savefig.bbox": None,
            "savefig.pad_inches": 0.0,
        }
    )
    return plt


def panel(plt):
    fig = plt.figure(figsize=(WIDTH, HEIGHT))
    ax = fig.add_axes(RECT)
    ax.set_box_aspect(1 / 1.5)
    ax.tick_params(
        which="both",
        direction="in",
        top=True,
        right=True,
        length=2.5,
        width=0.60,
        pad=2,
    )
    ax.tick_params(which="minor", length=1.5, width=0.4)
    ax.xaxis.labelpad = 2
    ax.yaxis.labelpad = 2
    ax.set_xscale("linear")
    ax.set_yscale("linear")
    return fig, ax


def save(plt, fig, name):
    fig.savefig(BUILD / name)
    plt.close(fig)


def legend(ax, **kwargs):
    opts = dict(
        frameon=True,
        facecolor="white",
        framealpha=0.80,
        edgecolor="#b0b0b0",
        handlelength=1.35,
        labelspacing=0.18,
        borderpad=0.25,
        handletextpad=0.4,
        columnspacing=0.65,
    )
    opts.update(kwargs)
    leg = ax.legend(**opts)
    leg.get_frame().set_linewidth(0.45)
    return leg


def line(ax, x, y, index, label, **kwargs):
    return ax.plot(
        x,
        y,
        color=COLORS[index],
        marker=MARKERS[index],
        linestyle=LINES[index],
        label=label,
        **kwargs,
    )[0]
