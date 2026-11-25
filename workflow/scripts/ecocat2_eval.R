# Evaluate the R implementation of every row of the ecology catalogue
# hsr/data/sources/ecocat2_curves.csv at random parameters on a grid of x,
# for numerical verification of the transcribed formulas (rule
# ecocat2_eval; as rcurves_eval.R, with these additions):
#   * R files of packages that are not installed (not on conda-forge) are
#     read from pinned CRAN source tarballs: only their top-level function
#     definitions (`name <- function(...)`) are evaluated, nothing else.
#   * ranges "x=lo:hi:by" gives an x grid with step `by` (integer grids for
#     probability mass functions); "x=lo:hi" 25 equally spaced points.
#   * r_call may return list(y = ..., x = ..., par = c(name = value, ...)):
#     x replaces the grid and par replaces the drawn values of the named
#     parameters (used where the package fits the parameters itself, e.g.
#     vegan's rank-abundance models fitted to the BCI data, or where a
#     normalising constant of a probability mass function is reported).
# Helpers available to r_call (x, draw and the parameters are bound):
#   mlaw(name, ...)        MortalityLaws::<name>(x, c(...))$hx (hazard)
#   localfun(f, name)      a function defined inside the body of f (e.g. the
#                          JP model inside phytotools::fitJP)
#   ptm(fun, name, p)      localfun(get(fun), name)(p, x)$y
#   bbm(i, ...)            right-hand side of the i-th nls() formula in the
#                          body of plantecophys::fitBB, with variables bound by ...
#   radv(name)             vegan rank-abundance model `name` fitted to a BCI
#                          site (site 10 * draw): list(x = ranks, y = fitted, par = coefficients and J)
# Usage: Rscript ecocat2_eval.R <csv> <out.csv> <pkg1,pkg2,...> <tarball::member.R,...>
args <- commandArgs(trailingOnly = TRUE)
pkgs <- strsplit(args[3], ",")[[1]]
for (p in pkgs) suppressMessages(suppressWarnings(library(p, character.only = TRUE)))
sourced <- character(0)
if (length(args) >= 4 && nzchar(args[4])) {
  for (spec in strsplit(args[4], ",")[[1]]) {
    parts <- strsplit(spec, "::", fixed = TRUE)[[1]]
    td <- tempfile("src")
    untar(parts[1], files = parts[2], exdir = td)
    for (e in parse(file.path(td, parts[2]), keep.source = FALSE)) {
      if (is.call(e) && (identical(e[[1]], as.name("<-")) || identical(e[[1]], as.name("="))) && is.name(e[[2]]) &&
          is.call(e[[3]]) && identical(e[[3]][[1]], as.name("function")))
        eval(e, globalenv())
    }
    sourced <- c(sourced, paste0(basename(parts[1]), "::", parts[2]))
  }
}

mlaw <- function(name, ...) get(name, envir = globalenv())(get("x", envir = parent.frame()), c(...))$hx

find_def <- function(e, name) {
  if (is.call(e)) {
    if ((identical(e[[1]], as.name("<-")) || identical(e[[1]], as.name("="))) && identical(e[[2]], as.name(name)) &&
        is.call(e[[3]]) && identical(e[[3]][[1]], as.name("function")))
      return(e[[3]])
    for (i in seq_along(e)) {
      if (identical(e[[i]], quote(expr = ))) next
      r <- find_def(e[[i]], name)
      if (!is.null(r)) return(r)
    }
  }
  NULL
}

localfun <- function(f, name) eval(find_def(body(f), name), globalenv())

ptm <- function(fun, name, p) localfun(get(fun, envir = globalenv()), name)(p, get("x", envir = parent.frame()))$y

nls_formulas <- function(e, acc = list()) {
  if (is.call(e)) {
    if (identical(e[[1]], as.name("nls"))) return(c(acc, list(e[[2]])))
    for (i in seq_along(e)) {
      if (identical(e[[i]], quote(expr = ))) next
      acc <- nls_formulas(e[[i]], acc)
    }
  }
  acc
}

bbm <- function(i, ...) {
  f <- nls_formulas(body(get("fitBB", envir = globalenv())))[[i]]
  e <- new.env(parent = parent.frame())
  b <- list(...)
  for (n in names(b)) assign(n, b[[n]], envir = e)
  eval(f[[3]], e)
}

radv <- function(name) {
  env <- parent.frame()
  data("BCI", package = "vegan", envir = environment())
  site <- unlist(BCI[10 * get("draw", envir = env), ])
  m <- suppressWarnings(get(name, envir = asNamespace("vegan"))(site))
  fit <- as.numeric(fitted(m))
  list(x = seq_along(fit), y = fit, par = c(coef(m), J = sum(site)))
}

parse_ranges <- function(s) {
  out <- list()
  if (is.na(s) || !nzchar(s)) return(out)
  for (part in strsplit(s, ";")[[1]]) {
    kv <- strsplit(part, "=")[[1]]
    out[[trimws(kv[1])]] <- as.numeric(strsplit(kv[2], ":")[[1]])
  }
  out
}

tab <- read.table(args[1], sep = "|", header = TRUE, quote = "", stringsAsFactors = FALSE,
                  comment.char = "", colClasses = "character")
tab <- tab[nzchar(tab$r_call), ]
out <- list()
set.seed(1)
for (i in seq_len(nrow(tab))) {
  pn <- strsplit(tab$params[i], ",")[[1]]
  rg <- parse_ranges(tab$ranges[i])
  xr <- if (is.null(rg$x)) c(0.3, 4) else rg$x
  x0 <- if (length(xr) == 3) seq(xr[1], xr[2], by = xr[3]) else seq(xr[1], xr[2], length.out = 25)
  for (draw in 1:3) {
    vals <- sapply(pn, function(p) { r <- if (is.null(rg[[p]])) c(0.5, 2) else rg[[p]]; round(runif(1, r[1], r[2]), 6) })
    env <- new.env()
    x <- x0
    assign("x", x, envir = env)
    assign("draw", draw, envir = env)
    for (k in seq_along(pn)) assign(pn[k], vals[[k]], envir = env)
    res <- tryCatch(suppressWarnings(eval(parse(text = tab$r_call[i]), envir = env)),
                    error = function(e) { message(tab$id[i], ": ", conditionMessage(e)); rep(NA, length(x)) })
    if (is.list(res)) {
      if (!is.null(res$x)) x <- res$x
      if (!is.null(res$par)) for (n in intersect(names(res$par), pn)) vals[[n]] <- unname(res$par[[n]])
      res <- res$y
    }
    y <- as.numeric(res)
    if (length(y) != length(x)) y <- rep(NA, length(x))
    out[[length(out) + 1]] <- data.frame(func = tab$id[i], draw = draw,
                                         params = paste(trimws(formatC(unlist(vals), digits = 15, format = "g")), collapse = ","), x = x, y = y)
  }
}
res <- do.call(rbind, out)
write.csv(res, args[2], row.names = FALSE)
versions <- sapply(pkgs, function(p) as.character(packageVersion(p)))
versions <- c(versions, setNames(rep("sourced", length(sourced)), sourced))
writeLines(paste(c("R", names(versions)), c(paste(R.version$major, R.version$minor, sep = "."), versions),
                 sep = "=", collapse = ";"), paste0(args[2], ".versions"))
