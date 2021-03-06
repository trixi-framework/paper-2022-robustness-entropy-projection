# On the entropy projection and the robustness of high order entropy stable discontinuous Galerkin schemes for under-resolved flows

[![License: MIT](https://img.shields.io/badge/License-MIT-success.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6364108.svg)](https://doi.org/10.5281/zenodo.6364108)

This repository contains information and code to reproduce the results presented in the
[article](https://doi.org/10.3389/fphy.2022.898028)
```bibtex
@article{chan2022entropy,
  title={On the entropy projection and the robustness of high order entropy
         stable discontinuous {G}alerkin schemes for under-resolved flows},
  author={Chan, Jesse and Ranocha, Hendrik and Rueda-Ram\'{i}rez, Andr{\'e}s M
          and Gassner, Gregor J and Warburton, Tim},
  journal={Frontiers in Physics},
  year={2022},
  month={07},
  doi={10.3389/fphy.2022.898028},
  volume={10},
  eprint={2203.10238},
  eprinttype={arxiv},
  eprintclass={math.NA}
}
```
(also available [on arXiv.org](https://arxiv.org/abs/2203.10238)).

If you find these results useful, please cite the article mentioned above. If you
use the implementations provided here, please **also** cite this repository as
```bibtex
@misc{chan2022entropyRepro,
  title={Reproducibility repository for
         {O}n the entropy projection and the robustness of high order entropy stable 
         discontinuous {G}alerkin schemes for under-resolved flows},
  author={Chan, Jesse and Ranocha, Hendrik and Rueda-Ram\'{i}rez, Andr{\'e}s M 
          and Gassner, Gregor J and Warburton, Tim},
  year={2022},
  month={02},
  howpublished={\url{https://github.com/trixi-framework/paper-2022-robustness-entropy-projection}},
  doi={10.5281/zenodo.6364108}
}
```


## Abstract

High order entropy stable schemes provide improved robustness for computational simulations of fluid flows. 
However, additional stabilization and positivity preserving limiting can still be required for variable-density 
flows with under-resolved features. We demonstrate numerically that entropy stable DG methods which incorporate 
an "entropy projection" are less likely to require additional limiting to retain positivity for certain types of 
flows. We conclude by investigating potential explanations for this observed improvement in robustness.


## Numerical experiments

The numerical experiments presented in the paper use [Trixi.jl](https://github.com/trixi-framework/Trixi.jl)
and [FLUXO](https://gitlab.com/project-fluxo/fluxo).
To reproduce the numerical experiments using Trixi.jl, you need to install
[Julia](https://julialang.org/).

The subfolders of this repository contain `README.md` files with instructions
to reproduce the numerical experiments, including postprocessing.

The numerical experiments were carried out using Julia v1.7.1.


## Authors

- [Jesse Chan](https://jlchan.github.io) (Rice University, USA)
- [Hendrik Ranocha](https://ranocha.de) (University of Hamburg, Germany)
- [Andr??s M. Rueda-Ram??rez](https://www.mi.uni-koeln.de/NumSim/dr-andres-rueda-ramirez) (University of Cologne, Germany)
- [Gregor Gassner](https://www.mi.uni-koeln.de/NumSim/gregor-gassner) (University of Cologne, Germany)
- [Tim Warburton](https://math.vt.edu/people/faculty/warburton-tim.html) (Virginia Tech)


## Disclaimer

Everything is provided as is and without warranty. Use at your own risk!
