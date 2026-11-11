from common import (
    BUILD,
    METHODS,
    NS,
    FastProblem,
    fmt,
    map_tri_points,
    residual,
    smoother_local_basis,
    tri_duffy,
    write_table,
)

import numpy as np
from numpy.polynomial.legendre import leggauss


CENTER = np.array([0.5, 0.5])


def exact_grad(x):
    X, Y = x[..., 0], x[..., 1]
    z = x - CENTER
    r = np.linalg.norm(z, axis=-1)
    assert np.all(r > 0), "Quadrature must avoid the singular vertex"
    b = X * (1 - X) * Y * (1 - Y)
    gb = np.stack(((1 - 2 * X) * Y * (1 - Y), (1 - 2 * Y) * X * (1 - X)), axis=-1)
    return (1 - np.sqrt(r))[..., None] * gb - 0.5 * b[..., None] * r[..., None] ** (
        -1.5
    ) * z


def cell_rule(pr, k, order):
    vertices = pr.mesh.xy[pr.mesh.tri[k]]
    dist = np.linalg.norm(vertices - CENTER, axis=1)
    hit = np.flatnonzero(dist < 1e-14)
    if len(hit):
        z, w = leggauss(order)
        z = (z + 1) / 2
        w = w / 2
        t, s = np.meshgrid(z, z, indexing="ij")
        wt, ws = np.meshgrid(w, w, indexing="ij")
        rho = t**4
        li = int(hit[0])
        a = (li + 1) % 3
        b = (li + 2) % 3
        lam = np.zeros((order, order, 3))
        lam[..., li] = 1 - rho
        lam[..., a] = rho * (1 - s)
        lam[..., b] = rho * s
        ref = lam.reshape(-1, 3)[:, 1:]
        weights = (wt * ws * 4 * t**7).ravel() * (2 * pr.mesh.area[k])
    else:
        ref, weights = tri_duffy(order)
        weights = weights * (2 * pr.mesh.area[k])
    return ref, map_tri_points(pr.mesh, k, ref), weights


class RoughProblem(FastProblem):
    def __init__(self, n, local=False, load_order=16, facet_order=20, volume_order=6):
        super().__init__(n, "mixed", alpha=2.0, qorder=volume_order, local=local)
        self.ExactGrad = exact_grad(self.X)

        q0 = self.qorder
        t0, t1 = self.edgeq[0], self.edgeq[-1]
        self.Jleft = (
            (t1 * self.J[::q0] - t0 * self.J[q0 - 1 :: q0]) / (t1 - t0)
        ).tocsr()
        self.Jright = (
            ((1 - t0) * self.J[q0 - 1 :: q0] - (1 - t1) * self.J[::q0]) / (t1 - t0)
        ).tocsr()
        self.pm = self.Pminus[::q0].copy()
        self.pp = self.Pplus[::q0].copy()
        self.set_facet_rule(facet_order)
        self.set_load(load_order)

    def set_facet_rule(self, order):
        t, w = leggauss(order)
        t = (t + 1) / 2
        w = w / 2
        ne = len(self.mesh.edges)
        ix = np.repeat(np.arange(ne), order)
        tt = np.tile(t, ne)
        self.J = (
            self.Jleft[ix].multiply((1 - tt)[:, None])
            + self.Jright[ix].multiply(tt[:, None])
        ).tocsr()
        lengths = np.array([e["length"] for e in self.mesh.edges])
        self.L = np.repeat(lengths, order)
        self.WF = self.L**2 * np.tile(w, ne)
        self.Pminus = np.repeat(self.pm, order)
        self.Pplus = np.repeat(self.pp, order)
        self.qorder = order
        self.edgeq = t
        self.edgew = w

    def set_load(self, order):
        load = np.zeros(self.ns)
        area = 0.0
        energy = 0.0
        count = 0
        for k in range(self.nt):
            ref, x, w = cell_rule(self, k, order)
            idx, gr = smoother_local_basis(self.mesh, k, ref, self.e_to_row)
            g = exact_grad(x)
            ng = np.linalg.norm(g, axis=1)
            p = self.pfun(x)
            A = ng[:, None] ** (p[:, None] - 2) * g
            load[idx] += np.einsum("qi,qmi,q->m", A, gr, w)
            area += w.sum()
            energy += w @ (ng**p)
            count += int(
                np.min(np.linalg.norm(self.mesh.xy[self.mesh.tri[k]] - CENTER, axis=1))
                < 1e-14
            )
        assert abs(area - 1) < 1e-12 and count == 6, (area, count)
        self.Force = np.asarray(self.S.T @ load)
        self.load_order = order
        self.exact_energy = float(energy)

    def rough_indicators(self, U, order=None):
        order = order or self.load_order
        out = super().indicators(U)
        g = np.column_stack((self.Gx @ U, self.Gy @ U))
        E = 0.0
        for k in range(self.nt):
            _, x, w = cell_rule(self, k, order)
            p = self.pfun(x)
            ge = exact_grad(x)
            ng = np.linalg.norm(g[k])
            ne = np.linalg.norm(ge, axis=1)
            fh = np.zeros_like(ge) if ng == 0 else ng ** ((p[:, None] - 2) / 2) * g[k]
            fe = ne[:, None] ** ((p[:, None] - 2) / 2) * ge
            E += w @ np.sum((fh - fe) ** 2, axis=1)
        out.update(
            e_nat=float(np.sqrt(E)),
            e_qo=float(np.sqrt(E + self.alpha * out["M"])),
            weighted_error=float(np.sqrt(E + self.alpha * out["M"])),
        )
        return out


def cusp():
    rows = []
    for n in NS:
        row = [n]
        for method in METHODS:
            pr = RoughProblem(n, local=method == "patchwise")
            xy = pr.mesh.xy[pr.int_vertices]
            x, y = xy.T
            exact = (
                x
                * (1 - x)
                * y
                * (1 - y)
                * (1 - np.sqrt(np.linalg.norm(xy - CENTER, axis=1)))
            )
            cf, info = pr.solve_fast(start=exact, cf=True, tol=1e-12)
            assert info["success"], (n, method, info)
            u, info = pr.solve_fast(start=cf, tol=1e-11, maxiter=240)
            assert info["success"], (n, method, info)
            residual(pr, u)
            values = pr.rough_indicators(u)
            row.extend([fmt(values["e_nat"]), fmt(values["j_env"])])
        rows.append(row)
    write_table(
        2,
        ["n", "e_nat_global", "j_env_global", "e_nat_patchwise", "j_env_patchwise"],
        rows,
    )


def main():
    BUILD.mkdir(exist_ok=True)
    cusp()


if __name__ == "__main__":
    main()
