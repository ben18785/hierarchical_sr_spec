# Formulas of nonlinear regression fits in CRAN package sources (hold-out
# source rnls). Nothing is evaluated: every file is read with R's parser.
#
# Code read per package: R/, tests/, demo/, inst/ (*.R, *.r; vignette code
# purled into inst/doc), vignettes/ (*.R, code chunks of *.Rmd, *.qmd, *.Rnw)
# and the \examples of man/*.Rd (tools::Rd2ex, \dontrun and \donttest kept).
#
# selfStart models are shared across packages (two passes).
#
# Calls found: nls, nlme, gnls, nlsLM, nls2 (also pkg::fn) and selfStart
# definitions. For a fit, the formula argument is taken as written, or
# resolved when it is a name assigned a formula earlier in the same file (or
# formula()/as.formula() of a literal). Parameter names come from start=
# (list/c/data.frame/expand.grid of named values, resolved like formulas),
# nlme fixed= / gnls params= (left-hand sides), lower=/upper= names, and the
# parameter arguments of self-starting models; every other variable of the
# right-hand side is a regressor (data column). Calls on the right-hand side
# of stats' self-starting models (SSlogis, ...) and of functions or
# selfStart models defined in the same package are expanded into closed
# form when the function body is a closed-form expression (assignments
# substituted; .value used where assigned; `if` statements that only stop()
# or warn, and gradient blocks after .value is set, are ignored). I(),
# suppressWarnings() and numerical guards ifelse(is.nan(v), 0, v) are removed. A selfStart definition contributes
# its own model (formula, or function body) with the function arguments that
# are not parameters as regressors.
#
# Usage: Rscript rnls_extract.R <raw_dir> <rnls_packages.tsv> <out.tsv>
# Output: one row per call: package, file, kind, lhs, rhs, params, regressors,
# status ("ok" or the reason it is dropped); <out.tsv>.files.tsv: per package
# the number of code units read and those R could not parse.
args <- commandArgs(trailingOnly = TRUE)
raw <- args[1]
pk <- read.delim(args[2], quote = "", colClasses = "character")
out_path <- args[3]

FITS <- c("nls", "nlme", "gnls", "nlsLM", "nls2")
FORMALS <- list(
  nls = function(formula, data, start, control, algorithm, trace, subset, weights, na.action, model,
                 lower, upper, ...) NULL,
  nlsLM = function(formula, data, start, jac, algorithm, control, lower, upper, trace, subset, weights,
                   na.action, model, ...) NULL,
  nls2 = function(formula, data, start, control, algorithm, trace, weights, ..., all) NULL,
  nlme = function(model, data, fixed, random, groups, start, correlation, weights, subset, method,
                  na.action, naPattern, control, verbose) NULL,
  gnls = function(model, data, params, start, correlation, weights, subset, na.action, naPattern,
                  control, verbose) NULL,
  selfStart = function(model, initial, parameters, template) NULL)

# stats' self-starting models: formals (data arguments first) and closed form
SS <- list(
  SSlogis = list(f = function(input, Asym, xmid, scal) NULL, data = 1,
                 e = quote(Asym / (1 + exp((xmid - input) / scal)))),
  SSfpl = list(f = function(input, A, B, xmid, scal) NULL, data = 1,
               e = quote(A + (B - A) / (1 + exp((xmid - input) / scal)))),
  SSasymp = list(f = function(input, Asym, R0, lrc) NULL, data = 1,
                 e = quote(Asym + (R0 - Asym) * exp(-exp(lrc) * input))),
  SSasympOff = list(f = function(input, Asym, lrc, c0) NULL, data = 1,
                    e = quote(Asym * (1 - exp(-exp(lrc) * (input - c0))))),
  SSasympOrig = list(f = function(input, Asym, lrc) NULL, data = 1,
                     e = quote(Asym * (1 - exp(-exp(lrc) * input)))),
  SSbiexp = list(f = function(input, A1, lrc1, A2, lrc2) NULL, data = 1,
                 e = quote(A1 * exp(-exp(lrc1) * input) + A2 * exp(-exp(lrc2) * input))),
  SSfol = list(f = function(Dose, input, lKe, lKa, lCl) NULL, data = 1:2,
               e = quote(Dose * exp(lKe + lKa - lCl) * (exp(-exp(lKe) * input) - exp(-exp(lKa) * input)) /
                           (exp(lKa) - exp(lKe)))),
  SSgompertz = list(f = function(x, Asym, b2, b3) NULL, data = 1, e = quote(Asym * exp(-b2 * b3^x))),
  SSmicmen = list(f = function(input, Vm, K) NULL, data = 1, e = quote(Vm * input / (K + input))),
  SSweibull = list(f = function(x, Asym, Drop, lrc, pwr) NULL, data = 1,
                   e = quote(Asym - Drop * exp(-exp(lrc) * x^pwr))))

one_line <- function(e) gsub("[\t\r\n]+", " ", paste(trimws(deparse(e, width.cutoff = 500L)), collapse = " "))
is_missing_arg <- function(e) identical(e, quote(expr = ))
fname <- function(e) {
  f <- e[[1]]
  if (is.name(f)) return(as.character(f))
  if (is.call(f) && length(f) == 3 && is.name(f[[1]]) && as.character(f[[1]]) %in% c("::", ":::") &&
      (is.name(f[[3]]) || is.character(f[[3]]))) return(as.character(f[[3]]))
  ""
}

# ---------------------------------------------------------------- code units
code_units <- function(dir) {
  files <- list.files(dir, recursive = TRUE, full.names = TRUE)
  rel <- substring(files, nchar(dir) + 2)
  top <- sub("/.*", "", rel)
  keep <- (top %in% c("R", "tests", "demo", "inst", "vignettes") & grepl("\\.(R|r|Rmd|rmd|qmd|Rnw|rnw)$", rel)) |
    (top == "man" & grepl("\\.Rd$", rel))
  units <- list()
  for (k in which(keep)) {
    f <- files[k]
    r <- rel[k]
    txt <- tryCatch(readLines(f, warn = FALSE, encoding = "UTF-8"), error = function(e) NULL)
    if (is.null(txt)) next
    if (grepl("\\.(Rmd|rmd|qmd)$", r)) {
      st <- grep("^\\s*```+\\s*\\{r", txt)
      en <- grep("^\\s*```+\\s*$", txt)
      for (s in st) { e <- en[en > s][1]; if (!is.na(e) && e > s + 1) units[[length(units) + 1]] <- list(file = r, code = txt[(s + 1):(e - 1)]) }
    } else if (grepl("\\.(Rnw|rnw)$", r)) {
      st <- grep("^\\s*<<.*>>=", txt)
      en <- grep("^\\s*@", txt)
      for (s in st) { e <- en[en > s][1]; if (!is.na(e) && e > s + 1) units[[length(units) + 1]] <- list(file = r, code = txt[(s + 1):(e - 1)]) }
    } else if (grepl("\\.Rd$", r)) {
      tf <- tempfile(fileext = ".R")
      ok <- tryCatch({
        rd <- suppressWarnings(tools::parse_Rd(f))
        tools::Rd2ex(rd, tf, commentDontrun = FALSE, commentDonttest = FALSE)
        TRUE
      }, error = function(e) FALSE)
      if (ok && file.exists(tf)) units[[length(units) + 1]] <- list(file = r, code = readLines(tf, warn = FALSE))
      else if (!ok) units[[length(units) + 1]] <- list(file = r, code = NULL)
    } else {
      units[[length(units) + 1]] <- list(file = r, code = txt)
    }
  }
  units
}

# match.call without `...` arguments (calls inside functions that pass dots on)
safe_match <- function(def, call) {
  keep <- vapply(seq_along(call), function(i) i == 1 || !identical(call[[i]], quote(...)), TRUE)
  match.call(def, call[keep])
}

# ------------------------------------------------------------ traversal
walk <- function(e, visit) {
  visit(e)
  if (is.call(e) || is.expression(e) || is.pairlist(e)) {
    for (i in seq_along(e)) {
      if (is.null(e[[i]]) || is_missing_arg(e[[i]])) next
      if (is.call(e) && i == 1 && !is.call(e[[i]])) next
      walk(e[[i]], visit)
    }
  }
}

# assignments (in traversal order) and target calls of one parsed unit
scan_unit <- function(exprs) {
  asg <- list(); calls <- list(); n <- 0
  visit <- function(e) {
    if (!is.call(e)) return()
    n <<- n + 1
    op <- fname(e)
    if (op %in% c("<-", "=", "<<-") && length(e) == 3 && (is.name(e[[2]]) || is.character(e[[2]]))) {
      asg[[length(asg) + 1]] <<- list(name = as.character(e[[2]]), value = e[[3]], at = n)
    } else if (op == "->" && length(e) == 3 && is.name(e[[3]])) {
      asg[[length(asg) + 1]] <<- list(name = as.character(e[[3]]), value = e[[2]], at = n)
    }
    if (op %in% c(FITS, "selfStart")) calls[[length(calls) + 1]] <<- list(call = e, at = n, fn = op)
  }
  for (x in exprs) walk(x, visit)
  list(asg = asg, calls = calls)
}

resolve_name <- function(nm, at, asg) {
  best <- NULL
  for (a in asg) if (a$name == nm && a$at < at) best <- a$value
  if (is.null(best)) for (a in asg) if (a$name == nm) { best <- a$value; break }
  best
}

as_formula <- function(e, at, asg, depth = 0) {
  if (depth > 4 || is.null(e)) return(NULL)
  if (is.name(e)) return(as_formula(resolve_name(as.character(e), at, asg), at, asg, depth + 1))
  if (is.character(e) && length(e) == 1) {
    p <- tryCatch(parse(text = e, keep.source = FALSE)[[1]], error = function(err) NULL)
    if (is.call(p) && fname(p) == "~") return(p)
    return(NULL)
  }
  if (is.call(e)) {
    op <- fname(e)
    if (op == "~") return(e)
    if (op %in% c("formula", "as.formula", "(") && length(e) >= 2) return(as_formula(e[[2]], at, asg, depth + 1))
  }
  NULL
}

param_names <- function(e, at, asg, depth = 0) {
  if (depth > 4 || is.null(e)) return(character(0))
  if (is.name(e)) return(param_names(resolve_name(as.character(e), at, asg), at, asg, depth + 1))
  if (!is.call(e)) return(character(0))
  op <- fname(e)
  if (op == "~") {                                     # fixed = a + b ~ 1
    return(if (length(e) == 3) all.vars(e[[2]]) else character(0))
  }
  if (op %in% c("list", "c", "data.frame", "expand.grid", "cbind", "rbind", "setNames")) {
    nm <- names(e)
    out <- character(0)
    for (i in seq_along(e)[-1]) {
      a <- e[[i]]
      n <- if (is.null(nm)) "" else nm[i]
      if (n %in% c("fixed", "random")) out <- c(out, param_names(a, at, asg, depth + 1))
      else if (nzchar(n)) out <- c(out, n)
      else if (is.call(a) && fname(a) == "~") out <- c(out, param_names(a, at, asg, depth + 1))
    }
    return(unique(out))
  }
  character(0)
}

is_number <- function(val) {
  if (is.call(val) && fname(val) == "-" && length(val) == 2) val <- val[[2]]
  is.numeric(val) && length(val) == 1
}

# a test made only of is.nan/is.na/is.finite (and !, |, &): a numerical guard
nan_test <- function(t) {
  if (!is.call(t)) return(FALSE)
  op <- fname(t)
  if (op %in% c("is.nan", "is.na", "is.finite", "is.infinite")) return(TRUE)
  if (op %in% c("!", "(")) return(nan_test(t[[2]]))
  if (op %in% c("|", "||", "&", "&&")) return(nan_test(t[[2]]) && nan_test(t[[3]]))
  FALSE
}

# remove I(), suppressWarnings() and numerical guards ifelse(is.nan(v), 0, v) -> v
strip_I <- function(e) {
  if (is.call(e)) {
    if (fname(e) %in% c("I", "(", "suppressWarnings", "as.numeric", "as.vector") && length(e) == 2)
      return(strip_I(e[[2]]))
    if (fname(e) == "ifelse" && length(e) == 4 && nan_test(e[[2]])) {
      a <- e[[3]]; b <- e[[4]]
      if (is.numeric(a) && !is.numeric(b)) return(strip_I(b))
      if (is.numeric(b) && !is.numeric(a)) return(strip_I(a))
    }
    for (i in seq_along(e)[-1]) if (!is_missing_arg(e[[i]])) e[[i]] <- strip_I(e[[i]])
  }
  e
}

# closed form of a function definition `function(args) body`: (expr, args) or NULL
closed_form <- function(fdef) {
  if (!is.call(fdef) || fname(fdef) != "function") return(NULL)
  fargs <- names(fdef[[2]])
  b <- fdef[[3]]
  stmts <- if (is.call(b) && fname(b) == "{") as.list(b)[-1] else list(b)
  env <- list(); last <- NULL; value <- NULL
  sub <- function(x) do.call(substitute, list(x, env))
  for (s in stmts) {
    if (is.call(s) && length(s) != 3 && fname(s) %in% c("<-", "=")) {
      return(NULL)
    } else if (is.call(s) && fname(s) %in% c("<-", "=") && is.name(s[[2]])) {
      v <- sub(s[[3]])
      env[[as.character(s[[2]])]] <- v
      if (as.character(s[[2]]) == ".value") value <- v
      last <- v
    } else if (is.call(s) && fname(s) == "if") {
      txt <- one_line(s)
      # guards that only stop or warn, and (once .value is set) blocks that
      # only compute its gradient attribute, leave the value unchanged
      guard <- grepl("stop\\(|warning\\(|message\\(", txt) && !grepl("<-", txt)
      grad <- !is.null(value) && !grepl("(^|[^(])\\.value <-", txt)
      if (!guard && !grad) return(NULL)
    } else if (is.call(s) && fname(s) %in% c("attr<-", "stopifnot", "on.exit", "message", "warning")) {
      next
    } else if (is.call(s) && fname(s) == "return") {
      last <- if (length(s) >= 2) sub(s[[2]]) else NULL; break
    } else if (is.call(s) && fname(s) == "<-" && is.call(s[[2]])) {
      next                                             # attr(.value, "gradient") <- ...
    } else {
      last <- sub(s)
    }
  }
  e <- if (!is.null(value)) value else last
  if (is.null(e) || is.name(e) && as.character(e) == ".value") return(NULL)
  if (is.name(e) && !is.null(env[[as.character(e)]])) e <- env[[as.character(e)]]
  list(e = e, args = fargs)
}

# replace calls of known models (stats SS models, package functions) by their closed form
expand <- function(e, defs, depth = 0) {
  if (!is.call(e) || depth > 3) return(e)
  op <- fname(e)
  d <- defs[[op]]
  if (!is.null(d) && !(op %in% c("exp", "log", "sqrt"))) {
    mc <- tryCatch(safe_match(d$f, e), error = function(err) NULL)
    if (!is.null(mc)) {
      a <- as.list(mc)[-1]
      for (k in seq_along(a)) if (!is_missing_arg(a[[k]])) a[[k]] <- expand(a[[k]], defs, depth)
      r <- do.call(substitute, list(d$e, a))
      return(expand(r, defs, depth + 1))
    }
  }
  for (i in seq_along(e)[-1]) if (!is_missing_arg(e[[i]])) e[[i]] <- expand(e[[i]], defs, depth)
  e
}

# parameters named in calls of self-starting models (stats or package)
ss_params <- function(e, defs) {
  out <- character(0)
  if (!is.call(e)) return(out)
  d <- defs[[fname(e)]]
  if (!is.null(d) && !is.null(d$pnames)) {
    mc <- tryCatch(safe_match(d$f, e), error = function(err) NULL)
    if (!is.null(mc)) {
      a <- as.list(mc)[-1]
      for (p in intersect(names(a), d$pnames)) if (is.name(a[[p]])) out <- c(out, as.character(a[[p]]))
    }
  }
  for (i in seq_along(e)[-1]) if (!is_missing_arg(e[[i]])) out <- c(out, ss_params(e[[i]], defs))
  unique(out)
}

stats_defs <- lapply(SS, function(s) {
  fa <- names(formals(s$f))
  list(f = s$f, e = s$e, pnames = fa[-s$data])
})

process_package <- function(pkg, tarball, shared = list()) {
  td <- tempfile("pkg"); dir.create(td)
  lst <- tryCatch(untar(tarball, list = TRUE), error = function(e) character(0))
  want <- lst[grepl(paste0("^", pkg, "/(R|tests|demo|inst|vignettes)/.*\\.(R|r|Rmd|rmd|qmd|Rnw|rnw)$"), lst) |
              grepl(paste0("^", pkg, "/man/.*\\.Rd$"), lst)]
  if (length(want)) suppressWarnings(untar(tarball, files = want, exdir = td))
  units <- code_units(file.path(td, pkg))
  unlink(td, recursive = TRUE)
  parsed <- list(); n_fail <- 0
  for (u in units) {
    ex <- if (is.null(u$code)) NULL else tryCatch(parse(text = u$code, keep.source = FALSE), error = function(e) NULL)
    if (is.null(ex)) { n_fail <- n_fail + 1; next }
    parsed[[length(parsed) + 1]] <- list(file = u$file, scan = scan_unit(ex))
  }
  # package-wide definitions: functions with a closed form, and selfStart models
  defs <- stats_defs
  for (nm in names(shared)) if (is.null(defs[[nm]])) defs[[nm]] <- shared[[nm]]
  exported <- list()
  fun_defs <- list()
  for (p in parsed) for (a in p$scan$asg) {
    if (is.call(a$value) && fname(a$value) == "function") {
      cf <- closed_form(a$value)
      if (!is.null(cf) && length(cf$args)) fun_defs[[a$name]] <- cf
    }
  }
  for (nm in names(fun_defs)) if (is.null(defs[[nm]])) {
    cf <- fun_defs[[nm]]
    f <- function() NULL
    formals(f) <- setNames(rep(list(quote(expr = )), length(cf$args)), cf$args)
    defs[[nm]] <- list(f = f, e = cf$e, pnames = NULL)
  }
  rows <- list()
  add <- function(file, kind, lhs, rhs, params, regs, status)
    rows[[length(rows) + 1]] <<- data.frame(package = pkg, file = file, kind = kind, lhs = lhs, rhs = rhs,
                                            params = paste(params, collapse = ","),
                                            regressors = paste(regs, collapse = ","), status = status)
  # selfStart definitions first (they also become expandable models)
  for (p in parsed) for (cl in p$scan$calls) if (cl$fn == "selfStart") {
    mc <- tryCatch(safe_match(FORMALS$selfStart, cl$call), error = function(err) NULL)
    if (is.null(mc) || is.null(mc$model)) { add(p$file, "selfStart", "", "", "", "", "unmatched call"); next }
    pn <- if (!is.null(mc$parameters)) {
      v <- mc$parameters
      if (is.name(v)) v <- resolve_name(as.character(v), cl$at, p$scan$asg)
      tryCatch(as.character(eval(v, baseenv())), error = function(err) character(0))
    } else character(0)
    m <- mc$model
    if (is.name(m)) {
      nm <- as.character(m)
      fd <- NULL
      for (q in parsed) for (a in q$scan$asg) if (a$name == nm && is.call(a$value) && fname(a$value) == "function") fd <- a$value
      m <- fd
    }
    e <- NULL; regs <- character(0); status <- "ok"
    if (is.call(m) && fname(m) == "~") {
      e <- strip_I(m[[length(m)]])
      if (!length(pn)) status <- "no parameter names"
      regs <- setdiff(all.vars(e), c(pn, "pi"))
    } else if (is.call(m) && fname(m) == "function") {
      cf <- closed_form(m)
      if (is.null(cf)) status <- "model function not a closed-form expression"
      else {
        e <- strip_I(expand(cf$e, defs))
        if (!length(pn)) pn <- tail(cf$args, -1)
        regs <- intersect(setdiff(cf$args, pn), all.vars(e))
      }
    } else status <- "model not resolvable"
    if (status == "ok" && !length(regs)) status <- "no regressor"
    # register under the name it is assigned to, for expansion in fits
    for (a in p$scan$asg) if (identical(a$value, cl$call) && !is.null(e) && length(regs) && is.null(SS[[a$name]])) {
      f <- function() NULL
      fa <- c(regs, pn)
      if (is.call(m) && fname(m) == "function") fa <- names(m[[2]])
      formals(f) <- setNames(rep(list(quote(expr = )), length(fa)), fa)
      defs[[a$name]] <- list(f = f, e = e, pnames = pn)
      exported[[a$name]] <- defs[[a$name]]
    }
    add(p$file, "selfStart", "", if (is.null(e)) "" else one_line(e), pn, regs, status)
  }
  for (p in parsed) for (cl in p$scan$calls) if (cl$fn %in% FITS) {
    mc <- tryCatch(safe_match(FORMALS[[cl$fn]], cl$call), error = function(err) NULL)
    if (is.null(mc)) { add(p$file, cl$fn, "", "", "", "", "unmatched call"); next }
    a <- as.list(mc)[-1]
    fe <- if (cl$fn %in% c("nlme", "gnls")) a$model else a$formula
    if (is.null(fe)) { add(p$file, cl$fn, "", "", "", "", "no formula argument"); next }
    fo <- as_formula(fe, cl$at, p$scan$asg)
    if (is.null(fo)) { add(p$file, cl$fn, "", one_line(fe), "", "", "formula not resolvable"); next }
    lhs <- if (length(fo) == 3) one_line(fo[[2]]) else ""
    rhs <- fo[[length(fo)]]
    if (is.call(rhs) && fname(rhs) == "|") rhs <- rhs[[2]]
    rhs <- strip_I(rhs)
    pn <- unique(c(param_names(a$start, cl$at, p$scan$asg), param_names(a$fixed, cl$at, p$scan$asg),
                   param_names(a$params, cl$at, p$scan$asg), param_names(a$lower, cl$at, p$scan$asg),
                   param_names(a$upper, cl$at, p$scan$asg), ss_params(rhs, defs)))
    ex <- tryCatch(strip_I(expand(rhs, defs)), error = function(err) rhs)
    vars <- all.vars(ex)
    pn <- intersect(pn, vars)
    if (!length(pn)) { add(p$file, cl$fn, lhs, one_line(ex), "", "", "no parameter names"); next }
    regs <- setdiff(vars, c(pn, "pi"))
    # names assigned a number earlier in the file are fixed constants, not data columns
    regs <- regs[!vapply(regs, function(v) is_number(resolve_name(v, cl$at, p$scan$asg)), TRUE)]
    if (!length(regs)) { add(p$file, cl$fn, lhs, one_line(ex), pn, "", "no regressor"); next }
    add(p$file, cl$fn, lhs, one_line(ex), pn, regs, "ok")
  }
  list(rows = if (length(rows)) do.call(rbind, rows) else NULL,
       files = data.frame(package = pkg, units = length(units), units_unparsed = n_fail),
       selfstart = exported, has_fits = any(vapply(rows, function(r) r$kind != "selfStart", TRUE)))
}

run <- function(i, shared = list()) {
  tb <- file.path(raw, paste0(pk$package[i], "_", pk$version[i], ".tar.gz"))
  if (!file.exists(tb)) { message("missing ", tb); return(NULL) }
  tryCatch(process_package(pk$package[i], tb, shared),
           error = function(e) { message(pk$package[i], ": ", conditionMessage(e), " in ", one_line(conditionCall(e))); NULL })
}
# pass 1: every package; selfStart models of all packages are collected, so
# that a fit calling another package's model (e.g. nlraa::SSagauss) can be
# expanded in pass 2, which reruns only the packages with fits
res1 <- lapply(seq_len(nrow(pk)), run)
shared <- list()
for (r in res1) if (!is.null(r)) for (nm in names(r$selfstart)) if (is.null(shared[[nm]])) shared[[nm]] <- r$selfstart[[nm]]
all_rows <- list(); all_files <- list()
for (i in seq_len(nrow(pk))) {
  r <- res1[[i]]
  if (is.null(r)) next
  if (r$has_fits && length(shared)) r <- run(i, shared)
  all_rows[[length(all_rows) + 1]] <- r$rows
  all_files[[length(all_files) + 1]] <- r$files
}
res <- do.call(rbind, all_rows)
if (is.null(res)) res <- data.frame(package = character(0), file = character(0), kind = character(0),
                                    lhs = character(0), rhs = character(0), params = character(0),
                                    regressors = character(0), status = character(0))
write.table(res, out_path, sep = "\t", quote = FALSE, row.names = FALSE)
write.table(do.call(rbind, all_files), paste0(out_path, ".files.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
cat(nrow(res), "calls in", length(unique(res$package)), "packages;", sum(res$status == "ok"), "ok\n")
print(table(res$status))
