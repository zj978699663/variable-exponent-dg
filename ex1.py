import math
import numpy as np

from common import (
    BUILD,
    FastProblem,
    INK,
    METHODS,
    NS,
    fmt,
    legend,
    line,
    panel,
    plotting,
    residual,
    save,
    write_table,
)

def smooth():
    rows, convergence, conforming = [], [], []
    for kind in ("mixed", "super"):
        for n in NS:
            row = [kind, n]
            for method in METHODS:
                pr = FastProblem(n, kind, local=method == "patchwise")
                if method == "global":
                    cf, info = pr.solve_fast(cf=True, tol=1e-11)
                    assert info["success"], (kind, n, info)
                    e_cf = pr.indicators(cf)["e_nat"]
                    assert np.max(np.abs(pr.J @ cf)) < 1e-14
                    conforming.append(dict(kind=kind, n=n, e_cf=e_cf))
                u, info = pr.solve_fast(cf, tol=1e-10)
                assert info["success"], (kind, n, method, info)
                residual(pr, u)
                ind = pr.indicators(u, cf)
                row.extend([fmt(ind["e_nat"]), fmt(ind["j_env"])])
                convergence.append(
                    dict(kind=kind, n=n, method=method, e_nat=ind["e_nat"])
                )
            rows.append([*row, fmt(e_cf)])
    write_table(
        1,
        [
            "kind",
            "n",
            "e_nat_global",
            "j_env_global",
            "e_nat_patchwise",
            "j_env_patchwise",
            "e_cf",
        ],
        rows,
    )
    return convergence, conforming


def draw_convergence(rows, cf):
    plt = plotting()
    fig, ax = panel(plt)
    for i, method in enumerate(("global", "patchwise")):
        z = sorted(
            (r for r in rows if r["kind"] == "mixed" and r["method"] == method),
            key=lambda r: float(r["n"]),
        )
        line(
            ax,
            [math.sqrt(2) / float(r["n"]) for r in z],
            [float(r["e_nat"]) for r in z],
            i,
            method.capitalize(),
        )
    z = sorted((r for r in cf if r["kind"] == "mixed"), key=lambda r: int(r["n"]))
    ax.plot(
        [math.sqrt(2) / int(r["n"]) for r in z],
        [float(r["e_cf"]) for r in z],
        color=INK,
        marker="^",
        linestyle="-.",
        label="Conforming P1",
    )
    guide_h = np.array([0.055, 0.22])
    ax.plot(guide_h, 0.07 * guide_h, color="#72767b", linestyle=":", linewidth=0.90)
    ax.text(0.105, 0.0042, r"$O(h)$", fontsize=8.0, color="#666b72")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set(
        xlim=(0.035, 1.05),
        ylim=(0.003, 0.17),
        xlabel=r"$h$",
        ylabel="Natural distance error",
    )
    legend(ax, loc="upper left")
    save(plt, fig, "fig1.pdf")


def main():
    BUILD.mkdir(exist_ok=True)
    rows, cf = smooth()
    draw_convergence(rows, cf)


if __name__ == "__main__":
    main()
