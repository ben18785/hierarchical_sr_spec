# Evaluate the R implementation of every row of a curve catalogue
# (hsr/data/sources/thermal_curves.csv, eco_curves.csv) at random parameters
# on a grid of x, for numerical verification of the transcribed formulas
# (as curvelib_eval.R, with per-row ranges).
#   ranges: "x=lo:hi;p=lo:hi;..." overrides the defaults x in [0.3, 4] and
#           every parameter in [0.5, 2] (needed where a function is only
#           defined, or returns a penalty value, outside a domain).
#   r_call: R code evaluated with x and the parameters bound. Helpers:
#     drv(name, i)  right-hand side of devRate::devRateEqList[[name]]$eq (i-th
#                   if a list), with T = x
#     sarx(name)    the `exp` expression of the model list inside the body of
#                   sars::<name>, with A = x
#     nlsm(name, v) right-hand side of the nlsMicrobio formula <name>, with v = x
# Usage: Rscript rcurves_eval.R <csv> <out.csv> <rlib> <pkg1,pkg2,...>
#   An optional 5th argument "tarball::member.R,..." sources R files from
#   package source tarballs (for internal functions of packages that are not
#   installed, e.g. bayesnec's pred_* equations); "tarball" is recorded with
#   its sha256 in the versions file.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) >= 3 && nzchar(args[3])) .libPaths(c(path.expand(args[3]), .libPaths()))
pkgs <- strsplit(args[4], ",")[[1]]
for (p in pkgs) suppressMessages(suppressWarnings(library(p, character.only = TRUE)))
sourced <- character(0)
if (length(args) >= 5 && nzchar(args[5])) {
  for (spec in strsplit(args[5], ",")[[1]]) {
    parts <- strsplit(spec, "::", fixed = TRUE)[[1]]
    td <- tempfile("src")
    untar(parts[1], files = parts[2], exdir = td)
    sys.source(file.path(td, parts[2]), envir = globalenv())
    sourced <- c(sourced, paste0(basename(parts[1]), "::", parts[2]))
  }
}

drv <- function(name, i = 1, env = parent.frame()) {
  eq <- devRate::devRateEqList[[name]]$eq
  if (!inherits(eq, "formula")) eq <- eq[[i]]
  e <- new.env(parent = env)
  assign("T", get("x", envir = env), envir = e)
  eval(eq[[3]], e)
}

find_model <- function(e) {
  if (is.call(e)) {
    if ((identical(e[[1]], as.name("<-")) || identical(e[[1]], as.name("="))) &&
        identical(e[[2]], as.name("model")) && is.call(e[[3]]) && identical(e[[3]][[1]], as.name("list")))
      return(e[[3]])
    for (i in seq_along(e)) {
      if (identical(e[[i]], quote(expr = ))) next
      r <- find_model(e[[i]])
      if (!is.null(r)) return(r)
    }
  }
  NULL
}

sarx <- function(name, env = parent.frame()) {
  m <- find_model(body(getExportedValue("sars", name)))
  e <- new.env(parent = env)
  assign("A", get("x", envir = env), envir = e)
  eval(eval(m$exp)[[1]], e)
}

nlsm <- function(name, v, env = parent.frame()) {
  f <- getExportedValue("nlsMicrobio", name)
  e <- new.env(parent = env)
  assign(v, get("x", envir = env), envir = e)
  eval(f[[3]], e)
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
  x <- seq(xr[1], xr[2], length.out = 25)
  for (draw in 1:3) {
    vals <- sapply(pn, function(p) { r <- if (is.null(rg[[p]])) c(0.5, 2) else rg[[p]]; round(runif(1, r[1], r[2]), 6) })
    env <- new.env()
    assign("x", x, envir = env)
    for (k in seq_along(pn)) assign(pn[k], vals[[k]], envir = env)
    y <- tryCatch(suppressWarnings(as.numeric(eval(parse(text = tab$r_call[i]), envir = env))),
                  error = function(e) { message(tab$id[i], ": ", conditionMessage(e)); rep(NA, length(x)) })
    if (length(y) != length(x)) y <- rep(NA, length(x))
    out[[length(out) + 1]] <- data.frame(func = tab$id[i], draw = draw,
                                         params = paste(vals, collapse = ","), x = x, y = y)
  }
}
res <- do.call(rbind, out)
write.csv(res, args[2], row.names = FALSE)
versions <- sapply(pkgs, function(p) as.character(packageVersion(p)))
versions <- c(versions, setNames(rep("sourced", length(sourced)), sourced))
writeLines(paste(c("R", names(versions)), c(paste(R.version$major, R.version$minor, sep = "."), versions),
                 sep = "=", collapse = ";"), paste0(args[2], ".versions"))
