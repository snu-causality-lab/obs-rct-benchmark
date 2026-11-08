# Arm A: the released intRlearner code at the paper's Table 1 and Table 2 settings.
#
# Usage: Rscript --vanilla author_replay.R SOURCE_ROOT OUT_DIR CELL SEED_FROM SEED_TO
# Each replicate follows the README: set.seed(seed), gen_data, then the RCT-only, OBS-only and
# pooled R-learners and the integrative R-learner, in that order. Failures are recorded and
# never retried.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 5L) stop("usage: SOURCE_ROOT OUT_DIR CELL SEED_FROM SEED_TO")
source_root <- normalizePath(args[[1]], mustWork = TRUE)
out_dir <- args[[2]]
cell <- args[[3]]
seed_from <- as.integer(args[[4]])
seed_to <- as.integer(args[[5]])

N <- 1e5
CELLS <- list(
  t1_n200       = list(KPS_data = TRUE,  m = 2000, n = 200,  b = NA,  n0 = NA, cf = 0.75, multidim = FALSE, p = 2,  p0 = NA),
  t1_n500       = list(KPS_data = TRUE,  m = 2000, n = 500,  b = NA,  n0 = NA, cf = 0.75, multidim = FALSE, p = 2,  p0 = NA),
  t1_n1000      = list(KPS_data = TRUE,  m = 2000, n = 1000, b = NA,  n0 = NA, cf = 0.75, multidim = FALSE, p = 2,  p0 = NA),
  t2_J1_b0      = list(KPS_data = FALSE, m = 500,  n = NA,   b = 0,   n0 = 7,  cf = NA,   multidim = NA,    p = 10, p0 = 8),
  t2_J1_b25     = list(KPS_data = FALSE, m = 500,  n = NA,   b = 2.5, n0 = 7,  cf = NA,   multidim = NA,    p = 10, p0 = 8),
  t2_J9_b0      = list(KPS_data = FALSE, m = 500,  n = NA,   b = 0,   n0 = 7,  cf = NA,   multidim = NA,    p = 10, p0 = 0),
  t2_J9_b25     = list(KPS_data = FALSE, m = 500,  n = NA,   b = 2.5, n0 = 7,  cf = NA,   multidim = NA,    p = 10, p0 = 0),
  # The README example itself (J = 8), used only by the preflight environment check.
  readme_J8_b25 = list(KPS_data = FALSE, m = 500,  n = NA,   b = 2.5, n0 = 7,  cf = NA,   multidim = NA,    p = 10, p0 = 1)
)
cc <- CELLS[[cell]]
if (is.null(cc)) stop("unknown cell")

# The README attaches these packages after library(intRlearner); the checkout is sourced instead.
suppressPackageStartupMessages({
  library(glmnet)
  library(mvtnorm)
  library(stringr)
  library(caret)
  library(ncvreg)
})
sources <- file.path(source_root, "R", c("utils.R", "intRlearner.R"))
for (path in sources) source(path)

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
out_csv <- file.path(out_dir, "replicates.csv")
if (file.exists(out_csv)) stop("replicates.csv already exists; use a new directory")
columns <- c("cell", "seed", "status", "n_rct", "n_obs", "rmse_int", "rmse_rct", "rmse_os",
             "rmse_naive", "int_sc_chosen", "int_ga_chosen", "seconds", "n_warnings", "message")
writeLines(paste(columns, collapse = ","), out_csv)
num <- function(x) if (is.null(x) || length(x) != 1L || is.na(x)) "" else sprintf("%.17g", as.numeric(x))
txt <- function(x) gsub("[\r\n,\"]", " ", substr(paste(x, collapse = " | "), 1, 400))
rmse <- function(estimate, tau) sqrt(mean((as.numeric(estimate) - tau)^2))
empty <- list(status = "failure", n_rct = NA, rmse_int = NA, rmse_rct = NA, rmse_os = NA,
              rmse_naive = NA, sc = NA, ga = NA, message = "")

for (seed in seq.int(seed_from, seed_to)) {
  start <- proc.time()[["elapsed"]]
  warned <- character(0)
  result <- tryCatch(withCallingHandlers({
    set.seed(seed)
    dataset <- gen_data(cc$m, N, b = cc$b, p = cc$p, lambda = "lambda.min", n0 = cc$n0, n = cc$n,
                        KPS_data = cc$KPS_data, multidim = cc$multidim, cf = cc$cf, p0 = cc$p0)
    p <- cc$p
    m <- cc$m
    x_rct <- dataset$x_rct
    x_rwe <- dataset$x_rwe
    a_rct <- dataset$a_rct
    a_rwe <- dataset$a_rwe
    y_rct <- dataset$y_rct
    y_rwe <- dataset$y_rwe
    newx_rlearner <- dataset$newx_rlearner
    # gen_data returns tau = 0 in the KPS_data branch; the true CATE is 1 + 2x + cf x^2.
    tau <- if (cc$KPS_data) 1 + 2 * newx_rlearner[, 1] + cc$cf * newx_rlearner[, 1]^2 else dataset$tau
    rlasso_fit_rct <- rlasso(x = x_rct, w = a_rct, y = y_rct, p = p)
    rlasso_est_rct <- predict.rlasso(rlasso_fit_rct, newx_rlearner)
    rlasso_fit_rwe <- rlasso(x = x_rwe, w = a_rwe, y = y_rwe, p = p)
    rlasso_est_rwe <- predict.rlasso(rlasso_fit_rwe, newx_rlearner)
    rlasso_fit_naive <- rlasso(x = rbind(x_rct, x_rwe), w = c(a_rct, a_rwe), y = c(y_rct, y_rwe), p = p)
    rlasso_est_naive <- predict.rlasso(rlasso_fit_naive, newx_rlearner)
    rlasso_fit_cmb <- intrlearner(x = rbind(x_rct, x_rwe), w = c(a_rct, a_rwe),
                                  y = c(y_rct, y_rwe), s = c(rep(1, length(a_rct)), rep(0, m)), p = p)
    rlasso_est_cmb <- predict.rlasso(rlasso_fit_cmb, newx_rlearner)
    values <- c(rmse(rlasso_est_cmb, tau), rmse(rlasso_est_rct, tau),
                rmse(rlasso_est_rwe, tau), rmse(rlasso_est_naive, tau))
    list(status = if (all(is.finite(values))) "success" else "nonfinite", n_rct = length(a_rct),
         rmse_int = values[1], rmse_rct = values[2], rmse_os = values[3], rmse_naive = values[4],
         sc = rlasso_fit_cmb$sc_chosen, ga = rlasso_fit_cmb$ga_chosen, message = "")
  }, warning = function(w) {
    warned <<- c(warned, conditionMessage(w))
    invokeRestart("muffleWarning")
  }), error = function(e) modifyList(empty, list(message = conditionMessage(e))))
  seconds <- proc.time()[["elapsed"]] - start
  message_text <- if (nzchar(result$message)) result$message else if (length(warned)) warned[1] else ""
  line <- c(cell, seed, result$status, num(result$n_rct), cc$m, num(result$rmse_int), num(result$rmse_rct),
            num(result$rmse_os), num(result$rmse_naive), num(result$sc), num(result$ga),
            sprintf("%.3f", seconds), length(warned), txt(message_text))
  cat(paste(line, collapse = ","), "\n", file = out_csv, append = TRUE, sep = "")
  cat(sprintf("cell=%s seed=%d status=%s seconds=%.1f warnings=%d\n", cell, seed, result$status,
              seconds, length(warned)))
}

packages <- c("glmnet", "mvtnorm", "stringr", "caret", "ncvreg")
writeLines(c(
  paste("R", R.version.string),
  sprintf("package %s %s", packages, vapply(packages, function(x) as.character(packageVersion(x)), "")),
  paste("RNGkind", paste(RNGkind(), collapse = " ")),
  paste("libPaths", paste(.libPaths(), collapse = " ")),
  paste("source", sources),
  paste("cell", cell),
  paste("seeds", seed_from, seed_to),
  paste("slurm", Sys.getenv("SLURM_JOB_ID"), Sys.getenv("SLURM_ARRAY_JOB_ID"),
        Sys.getenv("SLURM_ARRAY_TASK_ID"), Sys.getenv("SLURMD_NODENAME"))
), file.path(out_dir, "runtime.txt"))
