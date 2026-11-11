import math

from common import (
    BUILD,
    COLORS,
    METHODS,
    FastProblem,
    fmt,
    legend,
    line,
    panel,
    plotting,
    save,
    write_table,
)

import numpy as np
from scipy.optimize import root


def variable_penalty():
    rows = []
    alphas = (2, 10, 100)
    for n in (4, 8, 16):
        values = {}
        for method in METHODS:
            pr = FastProblem(n, local=method == "patchwise")
            cf, info = pr.solve_fast(cf=True, tol=1e-11)
            assert info["success"], (n, method, info)
            u = cf
            for alpha in alphas:
                pr.alpha = alpha
                u, info = pr.solve_fast(u, tol=1e-10)
                assert info["success"], (n, method, alpha, info)
                ind = pr.indicators(u, cf)
                values[method, alpha] = [fmt(ind["D"]), fmt(ind["M"])]
        for alpha in alphas:
            rows.append(
                [n, alpha, *values["global", alpha], *values["patchwise", alpha]]
            )
    write_table(
        3,
        ["n", "alpha", "D_global", "M_global", "D_patchwise", "M_patchwise"],
        rows,
    )


def two_triangles(q, alpha):
    pr = FastProblem(1, constant=q, sharp=True, alpha=alpha, qorder=64)
    scale = (1 + alpha) ** (-1 / (q - 1))
    options = {"xtol": 1e-11, "maxfev": 2000}
    if q < 2:
        pr.load_dg = pr.Force.copy()

        def pack(v):
            a, b = v
            return scale * np.array([a, b, a, a, a, b])

        sol = root(lambda v: pr.residual(pack(v))[[0, 1]], [0.5, 0.2], options=options)
        u = pack(sol.x)
        error = np.linalg.norm(pr.residual(u), np.inf)
    else:
        sol = root(
            lambda v: pr.residual_fast(scale * v), np.full(pr.nd, 0.5), options=options
        )
        u = scale * sol.x
        error = np.linalg.norm(pr.residual_fast(u), np.inf)
    assert error < 1e-9, (q, alpha, error)
    ind = pr.indicators(u, np.zeros(pr.nd))
    return ind["D"], ind["M"]


def constant_penalty():
    data, rows = [], []
    for q in (1.6, 2.0, 2.8):
        D_prev = M_prev = None
        for alpha in (10, 100, 1000, 10000):
            D, M = two_triangles(q, alpha)
            if D_prev is None:
                s_D = s_M = "--"
            else:
                s_D = f"{math.log10(D_prev / D):.2f}"
                s_M = f"{math.log10(M_prev / M):.2f}"
            rows.append([q, alpha, fmt(D), s_D, fmt(M), s_M])
            data.append(dict(q=q, alpha=alpha, D=D))
            D_prev, M_prev = D, M
    write_table(4, ["q", "alpha", "D", "s_D", "M", "s_M"], rows)
    return data


def draw_penalty(rows):
    plt = plotting()
    fig, ax = panel(plt)
    for i, q in enumerate((1.6, 2.0, 2.8)):
        z = sorted(
            (r for r in rows if float(r["q"]) == q), key=lambda r: float(r["alpha"])
        )
        x = np.array([float(r["alpha"]) for r in z])
        y = np.array([float(r["D"]) for r in z])
        rates = np.log(y[:-1] / y[1:]) / np.log(x[1:] / x[:-1])
        line(ax, x[1:], rates, i, rf"$q={q:g}$")
        ax.plot(
            [65, 16000],
            [q / (q - 1)] * 2,
            color=COLORS[i],
            linestyle=":",
            linewidth=0.90,
        )
    ax.set_xscale("log")
    ax.set_yscale("linear")
    ax.set(
        xlim=(65, 16000),
        ylim=(0.95, 2.85),
        xlabel=r"$\alpha$",
        ylabel=r"Decay rate of $D_\alpha$",
        xticks=[100, 1000, 10000],
    )
    legend(ax, loc="lower center", ncol=3)
    save(plt, fig, "fig2.pdf")


def main():
    BUILD.mkdir(exist_ok=True)
    variable_penalty()
    rows = constant_penalty()
    draw_penalty(rows)


if __name__ == "__main__":
    main()
