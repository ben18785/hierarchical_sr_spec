# Equations of the allodb allometric-equation table (data/equations.rda in the
# package source), with each equation evaluated by R at a grid of DBH values
# (cm) so that the Python transcription can be checked numerically.
# Usage: Rscript allodb_extract.R <allodb source tarball> <out.tsv>
args <- commandArgs(trailingOnly = TRUE)
dir <- tempfile("allodb")
dir.create(dir)
untar(args[1], exdir = dir)
f <- list.files(dir, pattern = "^equations\\.rda$", recursive = TRUE, full.names = TRUE)
stopifnot(length(f) == 1)
load(f)
grid <- c(5, 10, 20, 40, 80)
y <- sapply(seq_len(nrow(equations)), function(i) {
  if (equations$independent_variable[i] != "DBH") return("")
  v <- tryCatch(sapply(grid, function(d) eval(parse(text = equations$equation_allometry[i]), list(dbh = d))),
                error = function(e) rep(NA, length(grid)))
  paste(format(v, digits = 17), collapse = ";")
})
out <- data.frame(equation_id = equations$equation_id, ref_id = equations$ref_id,
                  independent_variable = equations$independent_variable,
                  dependent_variable = equations$dependent_variable,
                  equation_allometry = gsub("[\t\n]", " ", equations$equation_allometry),
                  dbh_grid = paste(grid, collapse = ";"), y = y)
write.table(out, args[2], sep = "\t", quote = FALSE, row.names = FALSE)
cat(nrow(out), "equations\n")
