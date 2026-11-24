# Evaluate an independent implementation of every form of
# hsr/data/sources/bio_catalogue.csv with check "library" or "ode" at fixed
# random parameters on a grid of x, for numerical verification of the
# transcribed formulas (as curvelib_eval.R).
#   library: an R package implementing the model (linpk, drc, frair, FSA);
#   ode:     the model's defining differential equation, integrated with
#            deSolve from the first grid point at which the closed form is
#            positive and not negligible (that point is not compared).
# Usage: Rscript biocat_eval.R <csv> <out.csv> <rlib>
args <- commandArgs(trailingOnly = TRUE)
if (length(args) >= 3 && nzchar(args[3])) .libPaths(c(path.expand(args[3]), .libPaths()))
suppressMessages({library(linpk); library(drc); library(frair); library(FSA); library(deSolve)})

# linear compartmental PK (linpk, matrix exponential) from micro rate constants
pk_lin <- function(x, D, V, k10, k12 = NULL, k21 = NULL, k13 = NULL, k31 = NULL, ka = 0, dur = 0) {
  q <- numeric(0); vp <- numeric(0)
  if (!is.null(k12)) { q <- c(q, k12 * V); vp <- c(vp, k12 * V / k21) }
  if (!is.null(k13)) { q <- c(q, k13 * V); vp <- c(vp, k13 * V / k31) }
  as.numeric(pkprofile(x, cl = k10 * V, vc = V, q = q, vp = vp, ka = ka,
                       dose = list(t.dose = 0, amt = D, dur = dur)))
}

# a drc model's mean function at one parameter vector
drcf <- function(model, x, ...) {
  p <- c(...)
  model$fct(x, matrix(p, nrow = length(x), ncol = length(p), byrow = TRUE))
}

# integrate dy/dt = rhs(t, y) from the first x where f is at least 1e-3 of its
# maximum (starting from a value near 0, e.g. exp(-1e8), only tests the solver)
ode_check <- function(x, f, rhs) {
  fx <- suppressWarnings(f(x))
  ok <- which(is.finite(fx) & fx > 1e-3 * max(fx[is.finite(fx)]))
  out <- rep(NA_real_, length(x))
  if (length(ok) < 2) return(out)
  i0 <- ok[1]
  times <- x[i0:length(x)]
  sol <- ode(y = c(y = fx[i0]), times = times, func = function(t, y, p) list(rhs(t, y)),
             method = "lsoda", rtol = 1e-11, atol = 1e-13)
  out[i0:length(x)] <- sol[, "y"]
  out[i0] <- NA
  out
}

tab <- read.table(args[1], sep = "|", header = TRUE, quote = "", stringsAsFactors = FALSE,
                  comment.char = "", colClasses = "character")
tab <- tab[tab$check %in% c("library", "ode"), ]
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
    y <- tryCatch(as.numeric(eval(parse(text = tab$r_call[i]), envir = env)),
                  error = function(e) { message(tab$id[i], ": ", conditionMessage(e)); rep(NA, length(x)) })
    if (length(y) != length(x)) y <- rep(NA, length(x))
    out[[length(out) + 1]] <- data.frame(func = tab$id[i], draw = draw,
                                         params = paste(vals, collapse = ","), x = x, y = y)
  }
}
res <- do.call(rbind, out)
write.csv(res, args[2], row.names = FALSE)
pk <- c("linpk", "drc", "frair", "FSA", "deSolve", "lamW")
versions <- sapply(pk, function(p) as.character(packageVersion(p)))
writeLines(paste(c("R", names(versions)), c(paste(R.version$major, R.version$minor, sep = "."), versions),
                 sep = "=", collapse = ";"), paste0(args[2], ".versions"))
