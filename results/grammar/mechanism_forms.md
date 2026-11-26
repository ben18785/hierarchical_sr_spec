# Mechanism-derived forms

Forms of random mechanisms (hsr.mechanisms; generators in its module docstring), one corpus per class (hsr/data/corpus_mech_<class>.json). seen: the form occurs in the training sources. Scores of every prior on these forms: results/grammar/mechanisms.md.

| class | x1 | mechanisms | kept | dropped (by reason) | distinct forms | seen distinct | mechanisms with a seen form |
|---|---|---|---|---|---|---|---|
| enzyme | ligand concentration | 1000 | 1000 | none | 17 | 8 | 932 (93%) |
| compartment | time | 1000 | 947 | oscillatory 45, too_many_consts 8 | 22 | 6 | 818 (86%) |
| temperature | absolute temperature | 1000 | 283 | too_many_terms 413, zero_at_saturation 235, too_many_consts 69 | 8 | 2 | 82 (29%) |

## enzyme

seed 0, at most 8 constants.

| form | mechanisms | share | constants | seen |
|---|---|---|---|---|
| `c0*x1/(c1 + x1)` | 332 | 33.2% | 2 | yes |
| `c0*x1/(c1*x1 + c2 + x1**2)` | 164 | 16.4% | 3 | yes |
| `c0/(c1 + x1)` | 151 | 15.1% | 2 | yes |
| `(c0 + c1*x1)/(c2 + x1)` | 107 | 10.7% | 3 | yes |
| `(c0*x1 + c1)/(c2*x1 + c3 + x1**2)` | 74 | 7.4% | 4 | yes |
| `c0/(c1*x1 + c2 + x1**2)` | 64 | 6.4% | 3 | yes |
| `c0*x1**2/(c1*x1 + c2 + x1**2)` | 38 | 3.8% | 3 | yes |
| `c0*x1/(c1*x1**2 + c2*x1 + c3 + x1**3)` | 19 | 1.9% | 4 | no |
| `(c0*x1 + c1)/(c2*x1**2 + c3*x1 + c4 + x1**3)` | 17 | 1.7% | 5 | no |
| `c0*x1**2/(c1*x1**2 + c2*x1 + c3 + x1**3)` | 15 | 1.5% | 4 | no |
| `c0/(c1*x1**2 + c2*x1 + c3 + x1**3)` | 7 | 0.7% | 4 | no |
| `c0*x1**3/(c1*x1**2 + c2*x1 + c3 + x1**3)` | 5 | 0.5% | 4 | no |
| `(c0 + c1*x1)/(c2 + c3*x1 + c4*x1**2 + c5*x1**3 + x1**4)` | 2 | 0.2% | 6 | no |
| `(c0*x1**2 + c1)/(c2*x1 + c3 + x1**2)` | 2 | 0.2% | 4 | yes |
| `(c0*x1**2 + c1)/(c2*x1**2 + c3*x1 + c4 + x1**3)` | 1 | 0.1% | 5 | no |
| `c0*x1**2/(c1*x1**2 + c2*x1**3 + c3*x1 + c4 + x1**4)` | 1 | 0.1% | 5 | no |
| `(c0*x1**3 + c1)/(c2*x1**2 + c3*x1 + c4 + x1**3)` | 1 | 0.1% | 5 | no |

## compartment

seed 0, at most 8 constants.

| form | mechanisms | share | constants | seen |
|---|---|---|---|---|
| `c0*exp(c1*x1)` | 349 | 36.9% | 2 | yes |
| `c0*exp(c1*x1) + c2` | 240 | 25.3% | 3 | yes |
| `c0*exp(c1*x1) + c2*exp(c3*x1)` | 124 | 13.1% | 4 | yes |
| `c0*exp(c1*x1) + c2*exp(c3*x1) + c4` | 66 | 7.0% | 5 | yes |
| `c0*exp(c1*x1) + c2*exp(c3*x1) + c4*exp(c5*x1)` | 37 | 3.9% | 6 | no |
| `c0 + c1*exp(c2*x1) + c3*exp(c4*x1) + c5*exp(c6*x1)` | 33 | 3.5% | 7 | no |
| `c0*x1 + c1*exp(c2*x1) + c3` | 28 | 3.0% | 4 | yes |
| `c0*exp(c1*x1) + c2*exp(c3*x1) + c4*exp(c5*x1) + c6*exp(c7*x1)` | 13 | 1.4% | 8 | no |
| `c0 + (c1*x1 + c2)*exp(c3*x1)` | 12 | 1.3% | 4 | no |
| `c0*x1*exp(c1*x1)` | 11 | 1.2% | 2 | yes |
| `c0*x1**2*exp(c1*x1)` | 10 | 1.1% | 2 | no |
| `c0 + c1*x1 + c2*exp(c3*x1) + c4*exp(c5*x1)` | 6 | 0.6% | 6 | no |
| `c0 + c1*x1 + c2*exp(c3*x1) + c4*exp(c5*x1) + c6*exp(c7*x1)` | 4 | 0.4% | 8 | no |
| `c0 + (c1*x1**2 + c2*x1 + c3)*exp(c4*x1)` | 3 | 0.3% | 5 | no |
| `c0*exp(c1*x1) + (c2*x1 + c3)*exp(c4*x1)` | 2 | 0.2% | 5 | no |
| `c0 + c1*exp(c2*x1) + (c3 + c4*x1)*exp(c5*x1)` | 2 | 0.2% | 6 | no |
| `c0*x1 + c1 + (c2*x1 + c3)*exp(c4*x1)` | 2 | 0.2% | 5 | no |
| `c0 + c4*exp(c5*x1) + (c1 + c2*x1)*exp(c3*x1)` | 1 | 0.1% | 6 | no |
| `c3*exp(c4*x1) + c5*exp(c6*x1) + (c0 + c1*x1)*exp(c2*x1)` | 1 | 0.1% | 7 | no |
| `c0*exp(c1*x1) + c5*exp(c6*x1) + (c2 + c3*x1)*exp(c4*x1)` | 1 | 0.1% | 7 | no |
| `c0 + c1*x1 + (c2 + c3*x1 + c4*x1**2)*exp(c5*x1)` | 1 | 0.1% | 6 | no |
| `c0*exp(c1*x1) + (c2 + c3*x1 + c4*x1**2)*exp(c5*x1)` | 1 | 0.1% | 6 | no |

## temperature

seed 0, at most 8 constants, at most 6 exponential terms.

| form | mechanisms | share | constants | seen |
|---|---|---|---|---|
| `c3*exp(c0/x1)/(c1*exp(c6/x1) + c2*exp(c7/x1) + c4 + exp(c5/x1))` | 65 | 23.0% | 8 | no |
| `c2*exp(c0/x1)/(c1*exp(c5/x1) + c3 + exp(c4/x1))` | 62 | 21.9% | 6 | no |
| `c0*exp(c3/x1)/(c1 + exp(c2/x1))` | 47 | 16.6% | 4 | yes |
| `c0*exp(c1/x1)` | 35 | 12.4% | 2 | yes |
| `c2*x1*exp(c0/x1)/(c1*exp(c5/x1) + c3 + exp(c4/x1))` | 24 | 8.5% | 6 | no |
| `c0*x1*exp(c1/x1)` | 21 | 7.4% | 2 | no |
| `c3*x1*exp(c0/x1)/(c1*exp(c6/x1) + c2*exp(c7/x1) + c4 + exp(c5/x1))` | 17 | 6.0% | 8 | no |
| `c0*x1*exp(c3/x1)/(c1 + exp(c2/x1))` | 12 | 4.2% | 4 | no |
