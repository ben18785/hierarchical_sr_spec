# Real grouped datasets for `hsr.realdata`

Each CSV has columns `group,x,y`.

| file | groups | x | y | textbook form | source |
|---|---|---|---|---|---|
| indometh.csv | 6 subjects | time (h) | indometacin concentration | bi-exponential | R `datasets::Indometh` |
| theoph.csv | 12 subjects | time (h) | theophylline concentration | first-order absorption/elimination | R `datasets::Theoph` |
| orange.csv | 5 trees | age (days) | trunk circumference | logistic | R `datasets::Orange` |
| dnase.csv | 11 assay runs | DNase concentration | optical density | logistic in log conc (Hill) | R `datasets::DNase` |
| co2.csv | 12 plants | ambient CO2 | CO2 uptake | asymptotic regression with offset | R `datasets::CO2` |
| loblolly.csv | 14 seed sources | age (yr) | height | asymptotic regression | R `datasets::Loblolly` |
| forage_replaced.csv | 133 experiments | prey density | foraging rate | Holling type I/II/III | FoRAGE v2021.2 |

The R datasets are exported by `benchmarks/realdata/nlme_export.R`; their
textbook forms follow Pinheiro & Bates (2000), *Mixed-Effects Models in S and
S-PLUS*.

FoRAGE: Uiterwaal, S. F., Lagerstrom, I. T., Lyon, S. R. & DeLong, J. P. (2022)
FoRAGE database: a compilation of functional responses for consumers and
parasitoids. *Ecology* 103:e3706. Data: KNB, doi:10.5063/DB807S, licence
CC-BY-NC-SA. `benchmarks/realdata/forage_prepare.py` keeps experiments in which
eaten prey were replaced (so the Holling forms apply without depletion), with
at least 8 points and 5 distinct densities; `forage_replaced_meta.csv` gives
each experiment's source and species.
