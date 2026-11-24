# Statements of the model() blocks of the nlmixr2lib model library, read with
# R's parser from the package source (no evaluation, no rxode2 needed).
# One row per assignment: file, kind (assign = `x <- e`, ddt = `d/dt(x) <- e`,
# ini = a parameter defined in ini()), lhs, rhs (deparsed on one line).
# Usage: Rscript nlmixr2lib_extract.R <nlmixr2lib_x.y.z.tar.gz> <out.tsv>
args <- commandArgs(trailingOnly = TRUE)
dir <- tempfile("nlmixr2lib")
dir.create(dir)
untar(args[1], exdir = dir)
files <- sort(list.files(file.path(dir, "nlmixr2lib", "inst", "modeldb"), pattern = "\\.R$",
                         recursive = TRUE, full.names = TRUE))
one_line <- function(e) paste(trimws(deparse(e, width.cutoff = 500L)), collapse = " ")
rows <- list()
add <- function(file, kind, lhs, rhs) rows[[length(rows) + 1]] <<- data.frame(file = file, kind = kind,
                                                                              lhs = lhs, rhs = rhs)
find_call <- function(e, name) {
  if (is.call(e)) {
    if (identical(e[[1]], as.name(name))) return(e)
    for (i in seq_along(e)) {
      if (identical(e[[i]], quote(expr = ))) next
      r <- find_call(e[[i]], name)
      if (!is.null(r)) return(r)
    }
  }
  NULL
}
walk <- function(file, body) {
  stmts <- if (is.call(body) && identical(body[[1]], as.name("{"))) as.list(body)[-1] else list(body)
  for (s in stmts) {
    if (!is.call(s)) next
    op <- as.character(s[[1]])
    if (op %in% c("<-", "=")) {
      lhs <- s[[2]]
      if (is.name(lhs)) {
        add(file, "assign", as.character(lhs), one_line(s[[3]]))
      } else if (is.call(lhs) && identical(lhs[[1]], as.name("/")) && one_line(lhs[[2]]) == "d" &&
                 is.call(lhs[[3]]) && identical(lhs[[3]][[1]], as.name("dt"))) {
        add(file, "ddt", one_line(lhs[[3]][[2]]), one_line(s[[3]]))
      }
    } else if (op == "if") {
      walk(file, s[[3]])
      if (length(s) >= 4) walk(file, s[[4]])
    }
  }
}
for (f in files) {
  rel <- sub(paste0("^", file.path(dir, "nlmixr2lib", "inst", "modeldb"), "/"), "", f)
  ex <- tryCatch(parse(f, keep.source = FALSE), error = function(e) NULL)
  if (is.null(ex)) { add(rel, "error", "", "parse"); next }
  for (e in ex) {
    ini <- find_call(e, "ini")
    mod <- find_call(e, "model")
    if (!is.null(ini)) {
      for (s in as.list(ini[[2]])[-1]) {
        if (is.call(s) && as.character(s[[1]]) %in% c("<-", "=", "~") && is.name(s[[2]]))
          add(rel, "ini", as.character(s[[2]]), "")
      }
    }
    if (!is.null(mod)) walk(rel, mod[[2]])
  }
}
res <- do.call(rbind, rows)
write.table(res, args[2], sep = "\t", quote = FALSE, row.names = FALSE)
cat(length(files), "files,", nrow(res), "rows\n")
