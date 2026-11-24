# Evaluate every R model function of hsr/data/sources/curve_libraries.csv at
# fixed random parameters on a grid of x, for numerical verification of the
# transcribed formulas.  Usage: Rscript curvelib_eval.R <csv> <out.csv> <rlib>
args <- commandArgs(trailingOnly = TRUE)
if (length(args) >= 3 && nzchar(args[3])) .libPaths(c(path.expand(args[3]), .libPaths()))
suppressMessages({library(nlraa); library(drc)})
tab <- read.table(args[1], sep = "|", header = TRUE, quote = "", stringsAsFactors = FALSE, comment.char = "")
x <- seq(0.3, 4, length.out = 25)
out <- list()
set.seed(1)
for (i in seq_len(nrow(tab))) {
  pn <- strsplit(tab$params[i], ",")[[1]]
  for (draw in 1:3) {
    vals <- round(runif(length(pn), 0.5, 2), 6)
    env <- new.env()
    assign("x", x, envir = env)
    for (k in seq_along(pn)) assign(pn[k], vals[k], envir = env)
    y <- tryCatch(as.numeric(eval(parse(text = tab$r_call[i]), envir = env)), error = function(e) rep(NA, length(x)))
    if (length(y) != length(x)) y <- rep(NA, length(x))
    out[[length(out) + 1]] <- data.frame(func = tab$`function`[i], draw = draw,
                                         params = paste(vals, collapse = ","), x = x, y = y)
  }
}
res <- do.call(rbind, out)
write.csv(res, args[2], row.names = FALSE)
versions <- sapply(c("stats", "nlraa", "drc"), function(p) as.character(packageVersion(p)))
writeLines(paste(names(versions), versions, sep = "=", collapse = ";"), paste0(args[2], ".versions"))
