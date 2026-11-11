# Variable-exponent DG experiments

## Scope

This package reproduces Tables 1–4 and Figures 1–2 of *Endpoint envelope discontinuous Galerkin methods for the variable exponent Dirichlet problem*.

## Run

Use Python 3.13.9 and pdfLaTeX with Computer Modern, `amsmath` and `bm`. Run from this directory.

```text
python -m pip install numpy==2.3.5 scipy==1.16.3 matplotlib==3.10.6
python ex1.py
python ex2.py
python ex3.py
```

## Tables, figures and checks

Each command can run on its own. All three solve their problems and write the results to `build/`.

| Paper | Command | Calculation and checks | Output |
|---|---|---|---|
| §8.1 Natural distance convergence | `python ex1.py` | Solve the smooth DG and conforming problems with six-point Gauss rules. Check the residuals and compute the errors. | `table1.csv`, `fig1.pdf` |
| §8.2 An energy solution with a point singularity | `python ex2.py` | Solve the cusp problems on five meshes. Use 6 points per volume coordinate, 20 per edge, and 16 per load and error coordinate. | `table2.csv` |
| §8.3 Penalty limits | `python ex3.py` | Solve the variable-exponent penalty problems and the two-triangle problems. Use 64-point quadrature for the two-triangle cases and calculate the decay rates. | `table3.csv`, `table4.csv`, `fig2.pdf` |

The three scripts use the mesh and solver routines in `common.py`. The table checks use the paper values in `reference/`, which also holds the figure panels.
