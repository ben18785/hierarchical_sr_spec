# Export the classic grouped nonlinear datasets that ship with R (package
# datasets) as group,x,y CSVs for hsr.realdata. Run from the repo root:
#   Rscript benchmarks/realdata/nlme_export.R
out <- "hsr/data/real"
w <- function(name, g, x, y) {
  d <- data.frame(group = as.character(g), x = x, y = y)
  d <- d[is.finite(d$x) & is.finite(d$y), ]
  write.csv(d[order(d$group, d$x), ], file.path(out, paste0(name, ".csv")), row.names = FALSE)
}
w("indometh", Indometh$Subject, Indometh$time, Indometh$conc)   # bi-exponential elimination
w("theoph",   Theoph$Subject,   Theoph$Time,   Theoph$conc)     # first-order absorption + elimination
w("orange",   Orange$Tree,      Orange$age,    Orange$circumference)  # logistic growth
w("dnase",    DNase$Run,        DNase$conc,    DNase$density)   # logistic in log(conc), i.e. Hill in conc
w("co2",      CO2$Plant,        CO2$conc,      CO2$uptake)      # asymptotic regression with offset
w("loblolly", Loblolly$Seed,    Loblolly$age,  Loblolly$height) # asymptotic growth
