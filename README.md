# CT-MOM.jl: Continuous Time Markov Observation Models in Julia

This repository contains the Julia implementation of Continuous Time Markov Observation Models (CT-MOM) as presented in the paper *"Sampling and filtering with Markov chains"* by Michael A. Kouritzin, published in *Signal Processing* (2024).

## Citation

If you use this code for research or commercial purposes, please cite the following paper:

```bibtex
@article{Kouritzin2024,
  abstract = {A new continuous-time Markov chain rate change formula is proven. This theorem is used to derive existence and uniqueness of novel filtering equations akin to the Duncan–Mortensen–Zakai equation and the Fujisaki–Kallianpur–Kunita equation but for Markov signals with general continuous-time Markov chain observations. The equations in this second theorem have the unique feature of being driven by both the observations and the process counting the observation transitions. A direct method of solving these filtering equations is also derived. Most results apply as special cases to the continuous-time Hidden Markov Models (CTHMM), which have become important in applications like disease progression tracking. The corresponding CTHMM results are stated as corollaries. Finally, application of our general theorems to Markov chain importance sampling, rejection sampling and branching particle filtering algorithms is also explained, and these are illustrated by way of disease tracking simulations.},
  author = {Michael A Kouritzin},
  doi = {https://doi.org/10.1016/j.sigpro.2024.109613},
  issn = {0165-1684},
  journal = {Signal Processing},
  keywords = {Continuous-time hidden Markov model, Disease forecasting, Filtering equations, Importance sampling, Measure change, Rejection sampling, Stochastic analysis},
  pages = {109613},
  title = {Sampling and filtering with Markov chains},
  volume = {225},
  url = {https://www.sciencedirect.com/science/article/pii/S0165168424002329},
  year = {2024},
}
```

## Usage and Licensing
This repository is intended for academic and research purposes. For any use of this code, especially for commercial purposes, you are required to contact me or Professor Michael A. Kouritzin before proceeding.

You can reach out via:

- Email: rkumar5 [at] ualberta.ca, rish030798 [at] gmail.com
- Professor's Email: michaelk [at] ualberta.ca


Acknowledgments
This implementation is part of ongoing research, and contributions are based on the theoretical foundations laid out in the aforementioned paper. The algorithms included here have applications in areas such as disease progression tracking, importance sampling, and rejection sampling.
