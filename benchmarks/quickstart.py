"""Quick start: fit the hierarchical model to a small synthetic population."""
import numpy as np

from hsr.candidates import CandidateSet
from hsr.gibbs import GibbsConfig, run_gibbs
from hsr.likelihood import compute_likelihoods, scale_population
from hsr.simulate import benchmark_population

rng = np.random.default_rng(0)
pop = benchmark_population(rng, n=20, noise_frac=0.05, J=12)          # forms A/B/C, weights .6/.3/.1

cands = CandidateSet.from_json("hsr/data/candidates.json")            # shipped grammar-only set
X, Y, sx, sy = scale_population([d.x for d in pop.datasets], [d.y for d in pop.datasets])
table = compute_likelihoods(cands.skeletons, np.array(X), np.array(Y))  # L[j, e], Laplace and BIC

res = run_gibbs(table.laplace, cands.counts, cands.grammar,
                GibbsConfig(n_sweeps=300, burn_in=100, n_chains=2))
for j, e in enumerate(res.top):
    print(f"dataset {j:2d}: true {pop.true_keys[j]:22s} top {cands.skeletons[e].display:28s} "
          f"p={res.assign_probs[j, e]:.2f}")
print("posterior over number of forms:", res.n_forms_distribution())
