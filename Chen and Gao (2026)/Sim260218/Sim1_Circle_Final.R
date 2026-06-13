# =============================================================================
# Sim1: Linear Integral Functional on Unit Circle
# =============================================================================
#
# Produces tables for paper (Tables 1-2) plus rate-optimal K_hat table WITH CI.
#
# Design:
#   X ~ Unif([-2,2]^2),  Y = h0(X) + U,  U ~ N(0,1)
#   h0(x) = x1^2 + 2*sin(x1)*x2           # matches paper definition
#   theta0 = integral_{S^1} h0(x) dH^1(x) = pi
#
# Methods (5 total):
#   (1) Fixed K + Normal CV
#   (2) Adaptive K_tilde (undersmoothed) + Normal CV
#   (3) Fixed K + Bootstrap CV
#   (4) Adaptive K_tilde (undersmoothed) + Bootstrap CV
#   (5) Rate-optimal K_hat (Lepski, no undersmoothing) + Normal CV  <-- NOW WITH CI
#
# Output:
#   Sim1_Circle_Final.RData
#   Sim1_Circle_Final.md
#   Sim1_Table1_AdaptNormal.tex
#   Sim1_Table2_Comparison.tex
#   Sim1_Table_RateOptimal.tex
# =============================================================================

library(splines)
library(MASS)      # for ginv()
library(qrng)     # for sobol()
library(jsonlite)  # for JSON output

# ============================================================
# B-spline basis (matches scipy.interpolate.BSpline)
# ============================================================

bspline_basis_1d <- function(x, J, degree = 3, xmin = -2, xmax = 2) {
  n_interior <- max(J - degree - 1, 0)
  if (n_interior > 0) {
    interior_knots <- seq(xmin, xmax, length.out = n_interior + 2)[-c(1, n_interior + 2)]
  } else {
    interior_knots <- NULL
  }
  boundary_knots <- c(xmin, xmax)
  B <- bs(x, knots = interior_knots, Boundary.knots = boundary_knots,
          degree = degree, intercept = TRUE)
  return(as.matrix(B))
}

tensor_bspline_2d <- function(X, J, degree = 3, xmin = -2, xmax = 2) {
  n <- nrow(X)
  B1 <- bspline_basis_1d(X[, 1], J, degree, xmin, xmax)
  B2 <- bspline_basis_1d(X[, 2], J, degree, xmin, xmax)
  K <- J * J
  B <- matrix(NA, nrow = n, ncol = K)
  col_idx <- 1
  for (j1 in 1:J) {
    for (j2 in 1:J) {
      B[, col_idx] <- B1[, j1] * B2[, j2]
      col_idx <- col_idx + 1
    }
  }
  return(B)
}

# ============================================================
# Core estimation for all K
# ============================================================

compute_estimates_allK <- function(Y, X, int_X, J_list, degree = 3) {
  N <- nrow(X)
  n_K <- length(J_list)
  tau_vec <- rep(NA, n_K)
  sd_vec  <- rep(NA, n_K)
  psi_mat <- matrix(NA, nrow = N, ncol = n_K)

  for (j in 1:n_K) {
    J <- J_list[j]
    Psi_X   <- tensor_bspline_2d(X, J, degree = degree)
    Psi_int <- tensor_bspline_2d(int_X, J, degree = degree)

    Q <- crossprod(Psi_X) / N
    Qinv <- ginv(Q)
    beta_hat <- Qinv %*% (t(Psi_X) %*% Y) / N

    e <- as.numeric(Y - Psi_X %*% beta_hat)
    h_int <- Psi_int %*% beta_hat
    tau_hat <- 2 * pi * mean(h_int)

    L_bK <- 2 * pi * colMeans(Psi_int)
    alpha_riesz <- Qinv %*% L_bK
    v_at_X <- as.numeric(Psi_X %*% alpha_riesz)

    psi <- v_at_X * e
    sd_infl <- sqrt(mean(psi^2))

    tau_vec[j]    <- tau_hat
    sd_vec[j]     <- sd_infl
    psi_mat[, j]  <- psi
  }
  return(list(tau_vec = tau_vec, sd_vec = sd_vec, psi_mat = psi_mat))
}

# ============================================================
# Bootstrap-Lepski selection
# ============================================================

bootstrap_lepski_select <- function(tau_vec, psi_mat, sd_vec, K_list, n,
                                    Bboot = 100, factor = 1.5) {
  m <- length(K_list)
  sqrt_n <- sqrt(n)

  # Pairwise contrast SDs
  sigma <- matrix(NA, nrow = m, ncol = m)
  for (a in 1:(m - 1)) {
    for (bb in (a + 1):m) {
      diff <- psi_mat[, a] - psi_mat[, bb]
      sigma[a, bb] <- max(sqrt(mean(diff^2)), 1e-12)
    }
  }

  # Multiplier bootstrap
  maxT  <- rep(NA, Bboot)
  tstar <- matrix(NA, nrow = Bboot, ncol = m)
  for (b_idx in 1:Bboot) {
    xi <- rnorm(n)
    Z <- as.numeric(crossprod(psi_mat, xi)) / sqrt_n
    tstar[b_idx, ] <- Z / sd_vec
    mx <- 0
    for (a in 1:(m - 1)) {
      for (bb in (a + 1):m) {
        val <- abs((Z[a] - Z[bb]) / sigma[a, bb])
        if (val > mx) mx <- val
      }
    }
    maxT[b_idx] <- mx
  }

  Kmax    <- max(K_list)
  alpha_n <- min(0.5, sqrt(log(Kmax) / Kmax))
  theta   <- quantile(maxT, 1 - alpha_n)

  # Actual contrasts
  Tact <- matrix(NA, nrow = m, ncol = m)
  for (a in 1:(m - 1)) {
    for (bb in (a + 1):m) {
      Tact[a, bb] <- abs(sqrt_n * (tau_vec[a] - tau_vec[bb]) / sigma[a, bb])
    }
  }

  # Lepski: K_hat (rate-optimal)
  idx_hat <- m
  for (a in 1:m) {
    if (a == m) { idx_hat <- a; break }
    max_above <- max(Tact[a, (a + 1):m], na.rm = TRUE)
    if (max_above <= 1.1 * theta) { idx_hat <- a; break }
  }
  K_hat <- K_list[idx_hat]

  # Undersmoothing: K_tilde
  target <- factor * K_hat
  idx_tilde <- m
  for (a in 1:m) {
    if (K_list[a] >= target) { idx_tilde <- a; break }
  }
  K_tilde <- K_list[idx_tilde]

  return(list(K_hat = K_hat, K_tilde = K_tilde,
              idx_hat = idx_hat, idx_tilde = idx_tilde,
              tstar = tstar, theta_star = theta))
}

# ============================================================
# Bootstrap CV for single K
# ============================================================

bootstrap_cv_single <- function(psi, sd_infl, n, Bboot = 100, alpha = 0.05) {
  sqrt_n <- sqrt(n)
  tstar <- rep(NA, Bboot)
  for (b_idx in 1:Bboot) {
    xi <- rnorm(n)
    Z <- sum(xi * psi) / sqrt_n
    tstar[b_idx] <- Z / sd_infl
  }
  return(quantile(abs(tstar), 1 - alpha))
}

# ============================================================
# Summarize one method
# ============================================================

summarize_method <- function(res, true_val) {
  Error   <- res$Est - true_val
  bias    <- mean(Error)
  stdev   <- sd(res$Est)
  rMSE    <- sqrt(mean(Error^2))
  CI      <- res$CI
  CI_cover <- mean(CI[, 1] <= true_val & true_val <= CI[, 2], na.rm = TRUE)
  CI_lower_mean <- mean(CI[, 1], na.rm = TRUE)
  CI_upper_mean <- mean(CI[, 2], na.rm = TRUE)
  CI_length_mean <- CI_upper_mean - CI_lower_mean
  K_bar <- mean(res$K_used)

  return(list(RMSE = rMSE, Bias = bias, SD = stdev,
              Cover = CI_cover,
              CI_L = CI_lower_mean, CI_U = CI_upper_mean,
              CI_Len = CI_length_mean, K_bar = K_bar))
}

# ============================================================
# Output writers
# ============================================================

write_latex_table1 <- function(all_results, N_list, filepath) {
  lines <- c(
    "\\begin{table}[!h]",
    "\\centering",
    "\\small",
    paste0("\\caption{Integral over known manifold: Sieve plug-in estimator with ",
           "bootstrap-Lepski $\\tilde{K}$ selection and normal critical values.}"),
    "\\label{tab:Sim1_Circle}",
    "\\begin{tabular}{lcccccccc}",
    "\\hline\\hline",
    "$n$ & RMSE & Bias & SD & CI$_L$ & CI$_U$ & U--L & Coverage(\\%) & $\\bar{\\tilde{K}}$\\\\",
    "\\hline"
  )
  for (N in N_list) {
    s <- all_results[[as.character(N)]][["AdaptK_NormalCV"]]
    lines <- c(lines, sprintf("%d & %.4f & %.4f & %.4f & %.4f & %.4f & %.4f & %.2f & %.1f\\\\",
                              N, s$RMSE, s$Bias, s$SD, s$CI_L, s$CI_U, s$CI_Len,
                              s$Cover * 100, s$K_bar))
  }
  lines <- c(lines, "\\hline\\hline", "\\end{tabular}", "\\end{table}")
  writeLines(lines, filepath)
}

write_latex_table2 <- function(all_results, N_list, filepath) {
  lines <- c(
    "\\begin{table}[!h]",
    "\\centering",
    "\\small",
    "\\caption{Integral over known manifold: RMSE and CI coverage comparison.}",
    "\\label{tab:Sim1_Circle_Comp}",
    "\\begin{tabular}{lcccccc}",
    "\\hline\\hline",
    " & \\multicolumn{2}{c}{RMSE} & \\multicolumn{4}{c}{CI Coverage (\\%)} \\\\",
    "\\cmidrule(lr){2-3} \\cmidrule(lr){4-7}",
    "$n$ & Fixed & Adapt & Fix+N & Adp+N & Fix+B & Adp+B \\\\",
    "\\hline"
  )
  for (N in N_list) {
    r <- all_results[[as.character(N)]]
    lines <- c(lines, sprintf(
      "%d & %.4f & %.4f & %.1f & %.1f & %.1f & %.1f\\\\",
      N,
      r$FixedK_NormalCV$RMSE, r$AdaptK_NormalCV$RMSE,
      r$FixedK_NormalCV$Cover * 100, r$AdaptK_NormalCV$Cover * 100,
      r$FixedK_BootCV$Cover * 100, r$AdaptK_BootCV$Cover * 100))
  }
  lines <- c(lines, "\\hline\\hline", "\\end{tabular}", "\\end{table}")
  writeLines(lines, filepath)
}

write_latex_table_rateopt <- function(all_results, N_list, filepath) {
  lines <- c(
    "\\begin{table}[!h]",
    "\\centering",
    "\\small",
    paste0("\\caption{Integral over known manifold: Sieve plug-in estimator at ",
           "rate-optimal $\\hat{K}$ selected by bootstrap-Lepski (no undersmoothing), ",
           "with normal critical values.}"),
    "\\label{tab:Sim1_Circle_RateOpt}",
    "\\begin{tabular}{lcccccccc}",
    "\\hline\\hline",
    "$n$ & RMSE & Bias & SD & CI$_L$ & CI$_U$ & U--L & Coverage(\\%) & $\\bar{\\hat{K}}$\\\\",
    "\\hline"
  )
  for (N in N_list) {
    s <- all_results[[as.character(N)]][["RateOptimal_Khat"]]
    lines <- c(lines, sprintf("%d & %.4f & %.4f & %.4f & %.4f & %.4f & %.4f & %.2f & %.1f\\\\",
                              N, s$RMSE, s$Bias, s$SD, s$CI_L, s$CI_U, s$CI_Len,
                              s$Cover * 100, s$K_bar))
  }
  lines <- c(lines, "\\hline\\hline", "\\end{tabular}", "\\end{table}")
  writeLines(lines, filepath)
}

write_latex_table_biasaware <- function(all_results, N_list, filepath) {
  lines <- c(
    "\\begin{table}[!h]",
    "\\centering",
    "\\small",
    paste0("\\caption{Integral over known manifold: Bias-aware CI at rate-optimal ",
           "$\\hat{K}$ with bias buffer critical value.}"),
    "\\label{tab:Sim1_Circle_BiasAware}",
    "\\begin{tabular}{lcccccccc}",
    "\\hline\\hline",
    "$n$ & RMSE & Bias & SD & CI$_L$ & CI$_U$ & U--L & Coverage(\\%) & $\\bar{\\hat{K}}$\\\\",
    "\\hline"
  )
  for (N in N_list) {
    s <- all_results[[as.character(N)]][["BiasAware_Khat"]]
    lines <- c(lines, sprintf("%d & %.4f & %.4f & %.4f & %.4f & %.4f & %.4f & %.2f & %.1f\\\\",
                              N, s$RMSE, s$Bias, s$SD, s$CI_L, s$CI_U, s$CI_Len,
                              s$Cover * 100, s$K_bar))
  }
  lines <- c(lines, "\\hline\\hline", "\\end{tabular}", "\\end{table}")
  writeLines(lines, filepath)
}

write_markdown <- function(all_results, N_list, J_list, K_list, J_fixed, K_fixed, filepath) {
  lines <- c(
    "# Sim1: Linear Integral on Unit Circle --- Results",
    "",
    sprintf("- h0(x) = x1^2 + 2*sin(x1)*x2"),
    sprintf("- theta0 = pi = %.6f", pi),
    sprintf("- J_list = %s,  K_list = %s", paste(J_list, collapse = ", "),
            paste(K_list, collapse = ", ")),
    sprintf("- J_fixed = %d,  K_fixed = %d", J_fixed, K_fixed),
    "",
    "## Table 1: Adaptive K_tilde + Normal CV",
    "",
    "| n | RMSE | Bias | SD | CI_L | CI_U | U-L | Cover(%) | K_bar |",
    "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
  )
  for (N in N_list) {
    s <- all_results[[as.character(N)]][["AdaptK_NormalCV"]]
    lines <- c(lines, sprintf("| %d | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f | %.1f | %.1f |",
                              N, s$RMSE, s$Bias, s$SD, s$CI_L, s$CI_U, s$CI_Len,
                              s$Cover * 100, s$K_bar))
  }
  lines <- c(lines, "",
    "## Table 2: RMSE and Coverage Comparison",
    "",
    "| n | RMSE(Fixed) | RMSE(Adapt) | Cover:Fix+N | Cover:Adp+N | Cover:Fix+B | Cover:Adp+B |",
    "|---:|---:|---:|---:|---:|---:|---:|"
  )
  for (N in N_list) {
    r <- all_results[[as.character(N)]]
    lines <- c(lines, sprintf("| %d | %.4f | %.4f | %.1f | %.1f | %.1f | %.1f |",
                              N, r$FixedK_NormalCV$RMSE, r$AdaptK_NormalCV$RMSE,
                              r$FixedK_NormalCV$Cover * 100, r$AdaptK_NormalCV$Cover * 100,
                              r$FixedK_BootCV$Cover * 100, r$AdaptK_BootCV$Cover * 100))
  }
  lines <- c(lines, "",
    "## Rate-Optimal K_hat + Normal CV",
    "",
    "| n | RMSE | Bias | SD | CI_L | CI_U | U-L | Cover(%) | K_bar_hat |",
    "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
  )
  for (N in N_list) {
    s <- all_results[[as.character(N)]][["RateOptimal_Khat"]]
    lines <- c(lines, sprintf("| %d | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f | %.1f | %.1f |",
                              N, s$RMSE, s$Bias, s$SD, s$CI_L, s$CI_U, s$CI_Len,
                              s$Cover * 100, s$K_bar))
  }
  lines <- c(lines, "",
    "## Bias-Aware K_hat (rate-optimal + bias buffer)",
    "",
    "| n | RMSE | Bias | SD | CI_L | CI_U | U-L | Cover(%) | K_bar_hat |",
    "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
  )
  for (N in N_list) {
    s <- all_results[[as.character(N)]][["BiasAware_Khat"]]
    lines <- c(lines, sprintf("| %d | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f | %.1f | %.1f |",
                              N, s$RMSE, s$Bias, s$SD, s$CI_L, s$CI_U, s$CI_Len,
                              s$Cover * 100, s$K_bar))
  }
  lines <- c(lines, "",
    "## Comparison: Undersmoothing vs Bias-Aware",
    "",
    "| n | US U-L | BA U-L | Diff% | US Cover | BA Cover | K_tilde | K_hat |",
    "|---:|---:|---:|---:|---:|---:|---:|---:|"
  )
  for (N in N_list) {
    us <- all_results[[as.character(N)]][["AdaptK_NormalCV"]]
    ba <- all_results[[as.character(N)]][["BiasAware_Khat"]]
    diff_pct <- (ba$CI_Len - us$CI_Len) / us$CI_Len * 100
    lines <- c(lines, sprintf("| %d | %.4f | %.4f | %+.1f%% | %.1f%% | %.1f%% | %.1f | %.1f |",
                              N, us$CI_Len, ba$CI_Len, diff_pct,
                              us$Cover * 100, ba$Cover * 100,
                              us$K_bar, ba$K_bar))
  }
  writeLines(lines, filepath)
}

# ============================================================
# Main simulation
# ============================================================

run_sim1 <- function(outdir = ".") {

  N_list  <- c(500, 1000, 2000, 4000, 8000)
  B       <- 1000
  M       <- 5000
  J_list  <- c(4, 5, 6, 7, 8, 9, 10)
  J_fixed <- 6
  Bboot   <- 500
  degree  <- 3
  alpha   <- 0.05

  K_list  <- J_list^2
  K_fixed <- J_fixed^2
  idx_fixed <- which(J_list == J_fixed)
  z_alpha <- qnorm(1 - alpha / 2)

  # Sobol integration points on unit circle
  int_points <- sobol(M, d = 1) * 2 * pi
  int_X <- cbind(cos(int_points), sin(int_points))

  methods_ci <- c("FixedK_NormalCV", "AdaptK_NormalCV",
                   "FixedK_BootCV", "AdaptK_BootCV", "BiasAware_Khat")
  methods_all <- c(methods_ci, "RateOptimal_Khat")

  cat("========================================================================\n")
  cat("Sim1: Linear Integral on Unit Circle\n")
  cat("========================================================================\n")
  cat("J_list =", J_list, " => K_list =", K_list, "\n")
  cat("J_fixed =", J_fixed, " => K_fixed =", K_fixed, "\n")
  cat("B =", B, ", M =", M, ", Bboot =", Bboot, "\n\n")

  all_results <- list()
  raw_results <- list()  # per-replication data for post-hoc analysis

  for (N in N_list) {
    cat("Running N =", N, "...", sep = " ")
    flush.console()
    t_start <- Sys.time()

    factor_us <- sqrt(log(N))

    # Storage
    storage <- lapply(methods_all, function(m) {
      list(Est = rep(NA, B), CI = matrix(NA, nrow = B, ncol = 2),
           K_used = rep(NA, B), se = rep(NA, B))
    })
    names(storage) <- methods_all

    set.seed(1234)

    for (b in 1:B) {
      X <- matrix(runif(N * 2, -2, 2), nrow = N, ncol = 2)
      h0_X <- X[, 1]^2 + 2 * sin(X[, 1]) * X[, 2]  # paper's h0
      Y <- h0_X + rnorm(N, 0, 1)

      est_all <- compute_estimates_allK(Y, X, int_X, J_list, degree)
      tau_vec <- est_all$tau_vec
      sd_vec  <- est_all$sd_vec
      psi_mat <- est_all$psi_mat

      lepski <- bootstrap_lepski_select(tau_vec, psi_mat, sd_vec, K_list, N,
                                        Bboot = Bboot, factor = factor_us)
      idx_adapt <- lepski$idx_tilde
      idx_hat   <- lepski$idx_hat

      # Method 1: Fixed K + Normal CV
      tau_1 <- tau_vec[idx_fixed]; se_1 <- sd_vec[idx_fixed] / sqrt(N)
      half_1 <- z_alpha * se_1
      storage$FixedK_NormalCV$Est[b] <- tau_1
      storage$FixedK_NormalCV$CI[b, ] <- c(tau_1 - half_1, tau_1 + half_1)
      storage$FixedK_NormalCV$K_used[b] <- K_fixed
      storage$FixedK_NormalCV$se[b] <- se_1

      # Method 2: Adaptive K_tilde + Normal CV
      tau_2 <- tau_vec[idx_adapt]; se_2 <- sd_vec[idx_adapt] / sqrt(N)
      half_2 <- z_alpha * se_2
      storage$AdaptK_NormalCV$Est[b] <- tau_2
      storage$AdaptK_NormalCV$CI[b, ] <- c(tau_2 - half_2, tau_2 + half_2)
      storage$AdaptK_NormalCV$K_used[b] <- K_list[idx_adapt]
      storage$AdaptK_NormalCV$se[b] <- se_2

      # Method 3: Fixed K + Bootstrap CV
      tau_3 <- tau_vec[idx_fixed]; se_3 <- sd_vec[idx_fixed] / sqrt(N)
      cv_3 <- bootstrap_cv_single(psi_mat[, idx_fixed], sd_vec[idx_fixed],
                                  N, Bboot, alpha)
      half_3 <- cv_3 * se_3
      storage$FixedK_BootCV$Est[b] <- tau_3
      storage$FixedK_BootCV$CI[b, ] <- c(tau_3 - half_3, tau_3 + half_3)
      storage$FixedK_BootCV$K_used[b] <- K_fixed
      storage$FixedK_BootCV$se[b] <- se_3

      # Method 4: Adaptive K_tilde + Bootstrap CV
      tau_4 <- tau_vec[idx_adapt]; se_4 <- sd_vec[idx_adapt] / sqrt(N)
      cv_4 <- quantile(abs(lepski$tstar[, idx_adapt]), 1 - alpha)
      half_4 <- cv_4 * se_4
      storage$AdaptK_BootCV$Est[b] <- tau_4
      storage$AdaptK_BootCV$CI[b, ] <- c(tau_4 - half_4, tau_4 + half_4)
      storage$AdaptK_BootCV$K_used[b] <- K_list[idx_adapt]
      storage$AdaptK_BootCV$se[b] <- se_4

      # Method 5: Rate-optimal K_hat + Normal CV
      tau_5 <- tau_vec[idx_hat]; se_5 <- sd_vec[idx_hat] / sqrt(N)
      half_5 <- z_alpha * se_5
      storage$RateOptimal_Khat$Est[b] <- tau_5
      storage$RateOptimal_Khat$CI[b, ] <- c(tau_5 - half_5, tau_5 + half_5)
      storage$RateOptimal_Khat$K_used[b] <- K_list[idx_hat]
      storage$RateOptimal_Khat$se[b] <- se_5

      # Method 6: Bias-aware K_hat
      tau_6 <- tau_vec[idx_hat]; se_6 <- sd_vec[idx_hat] / sqrt(N)
      J_hat <- J_list[idx_hat]
      A_hat <- log(log(J_hat))
      theta_star <- lepski$theta_star
      crit_6 <- (z_alpha + A_hat * theta_star) * se_6
      storage$BiasAware_Khat$Est[b] <- tau_6
      storage$BiasAware_Khat$CI[b, ] <- c(tau_6 - crit_6, tau_6 + crit_6)
      storage$BiasAware_Khat$K_used[b] <- K_list[idx_hat]
      storage$BiasAware_Khat$se[b] <- se_6
    }

    # Summarize
    summaries <- lapply(methods_all, function(m) summarize_method(storage[[m]], pi))
    names(summaries) <- methods_all
    all_results[[as.character(N)]] <- summaries
    raw_results[[as.character(N)]] <- storage  # save per-replication data

    t_elapsed <- difftime(Sys.time(), t_start, units = "secs")
    cat(" done (", round(t_elapsed, 1), "s)\n")
    cat("  Adapt+Normal: Cover=",
        round(summaries$AdaptK_NormalCV$Cover * 100, 1), "%",
        "  RateOpt+Normal: Cover=",
        round(summaries$RateOptimal_Khat$Cover * 100, 1), "%\n")
  }

  # ===== Write outputs =====
  cat("\nWriting outputs...\n")

  write_latex_table1(all_results, N_list, file.path(outdir, "Sim1_Table1_AdaptNormal.tex"))
  write_latex_table2(all_results, N_list, file.path(outdir, "Sim1_Table2_Comparison.tex"))
  write_latex_table_rateopt(all_results, N_list, file.path(outdir, "Sim1_Table_RateOptimal.tex"))
  write_latex_table_biasaware(all_results, N_list, file.path(outdir, "Sim1_Table_BiasAware.tex"))
  write_markdown(all_results, N_list, J_list, K_list, J_fixed, K_fixed,
                 file.path(outdir, "Sim1_Circle_Final.md"))

  # JSON
  json_out <- lapply(all_results, function(ns) {
    lapply(ns, function(s) lapply(s, function(v) if (is.na(v)) NULL else round(v, 6)))
  })
  writeLines(toJSON(json_out, auto_unbox = TRUE, pretty = TRUE, null = "null"),
             file.path(outdir, "Sim1_Circle_Final.json"))

  # RData (includes raw_results with per-replication Est, CI, K_used, se)
  save(all_results, raw_results, N_list, J_list, K_list, J_fixed, K_fixed, B, M, Bboot,
       file = file.path(outdir, "Sim1_Circle_Final.RData"))

  cat("All outputs saved to:", outdir, "\n")

  # Console summary
  cat("\n")
  cat("========================================================================\n")
  cat("Table 1: Adaptive K_tilde + Normal CV\n")
  cat("------------------------------------------------------------------------\n")
  cat(sprintf("%6s %8s %8s %8s %8s %8s %8s %8s %8s\n",
              "n", "RMSE", "Bias", "SD", "CI_L", "CI_U", "U-L", "Cover%", "K_bar"))
  for (N in N_list) {
    s <- all_results[[as.character(N)]][["AdaptK_NormalCV"]]
    cat(sprintf("%6d %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f %8.1f %8.1f\n",
                N, s$RMSE, s$Bias, s$SD, s$CI_L, s$CI_U, s$CI_Len,
                s$Cover * 100, s$K_bar))
  }

  cat("\nBias-Aware K_hat\n")
  cat("------------------------------------------------------------------------\n")
  cat(sprintf("%6s %8s %8s %8s %8s %8s %8s %8s %8s\n",
              "n", "RMSE", "Bias", "SD", "CI_L", "CI_U", "U-L", "Cover%", "K_bar"))
  for (N in N_list) {
    s <- all_results[[as.character(N)]][["BiasAware_Khat"]]
    cat(sprintf("%6d %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f %8.1f %8.1f\n",
                N, s$RMSE, s$Bias, s$SD, s$CI_L, s$CI_U, s$CI_Len,
                s$Cover * 100, s$K_bar))
  }

  cat("\nRate-Optimal K_hat + Normal CV\n")
  cat("------------------------------------------------------------------------\n")
  cat(sprintf("%6s %8s %8s %8s %8s %8s %8s %8s %8s\n",
              "n", "RMSE", "Bias", "SD", "CI_L", "CI_U", "U-L", "Cover%", "K_bar"))
  for (N in N_list) {
    s <- all_results[[as.character(N)]][["RateOptimal_Khat"]]
    cat(sprintf("%6d %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f %8.1f %8.1f\n",
                N, s$RMSE, s$Bias, s$SD, s$CI_L, s$CI_U, s$CI_Len,
                s$Cover * 100, s$K_bar))
  }

  invisible(all_results)
}

# ============================================================
# Entry point
# ============================================================

if (TRUE) {
  # Determine script directory robustly
  outdir <- tryCatch(dirname(sys.frame(1)$ofile), error = function(e) NULL)
  if (is.null(outdir) || outdir == "") {
    # Fallback: use --file arg from Rscript, or working directory
    args <- commandArgs(trailingOnly = FALSE)
    file_arg <- grep("^--file=", args, value = TRUE)
    if (length(file_arg) > 0) {
      outdir <- dirname(sub("^--file=", "", file_arg))
    } else {
      outdir <- "."
    }
  }
  run_sim1(outdir = outdir)
}
