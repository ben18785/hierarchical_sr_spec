# Packages of the chosen CRAN task views with their DESCRIPTION fields.
# Usage: Rscript rnls_views.R <Views.rds> <packages.rds> <out.tsv> <View1,View2,...>
# Output columns: view, package, version, license, title, description
# (version empty if the package is not on CRAN on the index date).
args <- commandArgs(trailingOnly = TRUE)
views <- readRDS(args[1])
db <- readRDS(args[2])
db <- db[!duplicated(db[, "Package"]), , drop = FALSE]
rownames(db) <- db[, "Package"]
want <- strsplit(args[4], ",")[[1]]
names(views) <- vapply(views, function(v) v$name, "")
miss <- setdiff(want, names(views))
if (length(miss)) stop("task views not in the index: ", paste(miss, collapse = ", "))
clean <- function(s) { s[is.na(s)] <- ""; gsub("[\t\r\n]+", " ", s) }
field <- function(p, f) { out <- rep("", length(p)); on <- p %in% rownames(db); out[on] <- db[p[on], f]; clean(out) }
rows <- list()
for (w in want) {
  p <- views[[w]]$packagelist$name
  rows[[w]] <- data.frame(view = w, package = p, version = field(p, "Version"), license = field(p, "License"),
                          title = field(p, "Title"), description = field(p, "Description"))
}
res <- do.call(rbind, rows)
write.table(res, args[3], sep = "\t", quote = FALSE, row.names = FALSE)
cat(nrow(res), "view-package rows,", length(unique(res$package)), "packages\n")
