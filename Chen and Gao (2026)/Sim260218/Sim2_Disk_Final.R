# =============================================================================
# Sim2: Upper Contour Set Integral on Unit Disk
# =============================================================================
#
# Produces tables for paper (Tables 3-4) plus rate-optimal K_hat table WITH CI.
#
# Design:
#   X ~ Unif([-2,2]^2),  Y = h0(X) + U,  U ~ N(0,1)
#   h0(x) = (1 - ||x||^2) * (4 + sin(x1)*x2 + cos(x2))
#   theta0 = V(h0) = int 1{h0(x) >= 0} dx = area(unit disk) = pi
#
# Estimators: Plug-in V(h_hat) and LOO-debiased
# K selection: Fixed K=64 vs Adaptive (Bootstrap-Lepski)
# Critical value: Normal vs Multiplier Bootstrap
#
# Methods (10 total, 8 with CI via undersmoothing + 2 rate-optimal WITH CI):
#   Plug-in:  Fixed+Normal, Adapt+Normal, Fixed+Boot, Adapt+Boot, RateOpt+Normal
#   LOO:      Fixed+Normal, Adapt+Normal, Fixed+Boot, Adapt+Boot, RateOpt+Normal
#
# Output:
#   Sim2_Disk_Final.RData
#   Sim2_Disk_Final.md
#   Sim2_Table3_AdaptNormal.tex
#   Sim2_Table4_Comparison.tex
#   Sim2_Table_RateOptimal.tex
# =============================================================================

library(splines)
library(MASS)
library(qrng)
library(jsonlite)

AREA_X <- 16.0  # area of [-2,2]^2

# ============================================================
# B-spline basis with derivatives
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

bspline_basis_1d_derivs <- function(x, J, degree = 3, xmin = -2, xmax = 2, eps = 1e-5) {
  B <- bspline_basis_1d(x, J, degree, xmin, xmax)
  x_plus  <- pmin(x + eps, xmax - 1e-10)
  x_minus <- pmax(x - eps, xmin + 1e-10)
  B_plus  <- bspline_basis_1d(x_plus, J, degree, xmin, xmax)
  B_minus <- bspline_basis_1d(x_minus, J, degree, xmin, xmax)
  dB  <- (B_plus - B_minus) / (x_plus - x_minus)
  ddB <- (B_plus - 2 * B + B_minus) / (eps^2)
  return(list(B = B, dB = dB, ddB = ddB))
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

tensor_bspline_2d_with_derivs <- function(X, J, degree = 3, xmin = -2, xmax = 2) {
  n <- nrow(X)
  K <- J * J
  d1 <- bspline_basis_1d_derivs(X[, 1], J, degree, xmin, xmax)
  d2 <- bspline_basis_1d_derivs(X[, 2], J, degree, xmin, xmax)

  B   <- matrix(NA, n, K); Bx  <- matrix(NA, n, K); By  <- matrix(NA, n, K)
  Bxx <- matrix(NA, n, K); Bxy <- matrix(NA, n, K); Byy <- matrix(NA, n, K)

  col_idx <- 1
  for (j1 in 1:J) {
    for (j2 in 1:J) {
      B[, col_idx]   <- d1$B[, j1]   * d2$B[, j2]
      Bx[, col_idx]  <- d1$dB[, j1]  * d2$B[, j2]
      By[, col_idx]  <- d1$B[, j1]   * d2$dB[, j2]
      Bxx[, col_idx] <- d1$ddB[, j1] * d2$B[, j2]
      Bxy[, col_idx] <- d1$dB[, j1]  * d2$dB[, j2]
      Byy[, col_idx] <- d1$B[, j1]   * d2$ddB[, j2]
      col_idx <- col_idx + 1
    }
  }
  return(list(B = B, Bx = Bx, By = By, Bxx = Bxx, Bxy = Bxy, Byy = Byy))
}

# ============================================================
# Core estimation: Plug-in + LOO for all K
# ============================================================

compute_estimates_allK_disk <- function(Y, X, mc_pts, mc_derivs_list, J_list,
                                        degree = 3, eps_band = 0.01, min_band = 40) {
  N   <- nrow(X)
  M   <- nrow(mc_pts)
  n_K <- length(J_list)

  tau_plugin_vec <- rep(NA, n_K);  tau_loo_vec <- rep(NA, n_K)
  sd_plugin_vec  <- rep(NA, n_K);  sd_loo_vec  <- rep(NA, n_K)
  psi_plugin_mat <- matrix(NA, N, n_K);  psi_loo_mat <- matrix(NA, N, n_K)

  for (j in 1:n_K) {
    J <- J_list[j]; K <- J * J
    Psi_X <- tensor_bspline_2d(X, J, degree)
    mc_d  <- mc_derivs_list[[j]]
    Psi_mc    <- mc_d$B;  Psi_mc_x  <- mc_d$Bx;  Psi_mc_y  <- mc_d$By
    Psi_mc_xx <- mc_d$Bxx; Psi_mc_xy <- mc_d$Bxy; Psi_mc_yy <- mc_d$Byy

    # Series LS
    Q <- crossprod(Psi_X) / N
    Qinv <- ginv(Q)
    beta_hat <- Qinv %*% (t(Psi_X) %*% Y) / N
    e <- as.numeric(Y - Psi_X %*% beta_hat)

    # LOO residuals
    temp  <- Psi_X %*% Qinv
    lever <- rowSums(temp * Psi_X) / N
    denom <- pmax(1 - lever, 1e-8)
    e_loo <- e / denom

    # Plug-in: V(h_hat)
    h_mc  <- as.numeric(Psi_mc %*% beta_hat)
    V_hat <- AREA_X * mean(h_mc >= 0)

    # Tube for boundary integral
    eps  <- eps_band
    mask <- abs(h_mc) < eps
    while (sum(mask) < min_band) {
      eps <- eps * 1.5
      if (eps > 0.5) break
      mask <- abs(h_mc) < eps
    }

    # DV(h_hat)[b^K] via tube approximation
    n_band <- sum(mask)
    if (n_band == 0) {
      L_bK <- rep(0, K)
    } else {
      L_bK <- (AREA_X / M) * colSums(Psi_mc[mask, , drop = FALSE]) / (2 * eps)
    }

    # Riesz representer & influence functions
    alpha_riesz <- Qinv %*% L_bK
    v_at_X <- as.numeric(Psi_X %*% alpha_riesz)
    psi_plugin <- v_at_X * e
    psi_loo    <- v_at_X * e_loo
    sd_plugin  <- sqrt(mean(psi_plugin^2))
    sd_loo     <- sqrt(mean(psi_loo^2))

    # LOO bias correction: D^2V term
    hx  <- as.numeric(Psi_mc_x  %*% beta_hat)
    hy  <- as.numeric(Psi_mc_y  %*% beta_hat)
    hxx <- as.numeric(Psi_mc_xx %*% beta_hat)
    hxy <- as.numeric(Psi_mc_xy %*% beta_hat)
    hyy <- as.numeric(Psi_mc_yy %*% beta_hat)
    g   <- sqrt(hx^2 + hy^2) + 1e-12

    Vx   <- hx / (g^3)
    Vy   <- hy / (g^3)
    divV <- (hxx + hyy) / (g^3) - 3 * (hx^2 * hxx + 2*hx*hy*hxy + hy^2 * hyy) / (g^5)

    weight <- numeric(M)
    if (n_band > 0) weight[mask] <- (AREA_X / M) * (g[mask] / (2 * eps))

    if (n_band == 0) {
      A <- matrix(0, K, K)
    } else {
      B_band  <- Psi_mc[mask, , drop = FALSE]
      Bx_band <- Psi_mc_x[mask, , drop = FALSE]
      By_band <- Psi_mc_y[mask, , drop = FALSE]
      wdiv <- weight[mask] * divV[mask]
      wVx  <- weight[mask] * Vx[mask]
      wVy  <- weight[mask] * Vy[mask]
      T1 <- t(B_band) %*% (wdiv * B_band)
      Mx <- t(B_band) %*% (wVx * Bx_band)
      My <- t(B_band) %*% (wVy * By_band)
      A  <- -(T1 + Mx + t(Mx) + My + t(My))
    }

    Mmat <- Qinv %*% A %*% Qinv
    S    <- t(Psi_X) %*% (e_loo^2 * Psi_X)
    correction <- sum(diag(Mmat %*% S)) / (2 * N^2)
    tau_loo <- V_hat - correction

    tau_plugin_vec[j] <- V_hat;      tau_loo_vec[j] <- tau_loo
    sd_plugin_vec[j]  <- sd_plugin;  sd_loo_vec[j]  <- sd_loo
    psi_plugin_mat[, j] <- psi_plugin; psi_loo_mat[, j] <- psi_loo
  }

  return(list(tau_plugin_vec = tau_plugin_vec, tau_loo_vec = tau_loo_vec,
              sd_plugin_vec = sd_plugin_vec, sd_loo_vec = sd_loo_vec,
              psi_plugin_mat = psi_plugin_mat, psi_loo_mat = psi_loo_mat))
}

# ============================================================
# Bootstrap-Lepski & Bootstrap CV
# ============================================================

bootstrap_lepski_select <- function(tau_vec, psi_mat, sd_vec, K_list, n,
                                    Bboot = 100, factor = 1.5, alpha = 0.05) {
  m <- length(K_list); sqrt_n <- sqrt(n)
  sigma <- matrix(NA, m, m)
  for (a in 1:(m-1)) for (bb in (a+1):m) {
    sigma[a, bb] <- max(sqrt(mean((psi_mat[, a] - psi_mat[, bb])^2)), 1e-12)
  }
  maxT <- rep(NA, Bboot); tstar <- matrix(NA, Bboot, m)
  for (b_idx in 1:Bboot) {
    xi <- rnorm(n); Z <- as.numeric(crossprod(psi_mat, xi)) / sqrt_n
    tstar[b_idx, ] <- Z / sd_vec
    mx <- 0
    for (a in 1:(m-1)) for (bb in (a+1):m) {
      val <- abs((Z[a] - Z[bb]) / sigma[a, bb])
      if (val > mx) mx <- val
    }
    maxT[b_idx] <- mx
  }
  Kmax <- max(K_list); alpha_n <- min(0.5, sqrt(log(Kmax) / Kmax))
  theta <- quantile(maxT, 1 - alpha_n)
  Tact <- matrix(NA, m, m)
  for (a in 1:(m-1)) for (bb in (a+1):m)
    Tact[a, bb] <- abs(sqrt_n * (tau_vec[a] - tau_vec[bb]) / sigma[a, bb])

  idx_hat <- m
  for (a in 1:m) {
    if (a == m) { idx_hat <- a; break }
    if (max(Tact[a, (a+1):m], na.rm = TRUE) <= 1.1 * theta) { idx_hat <- a; break }
  }
  target <- factor * K_list[idx_hat]; idx_tilde <- m
  for (a in 1:m) if (K_list[a] >= target) { idx_tilde <- a; break }
  cv_boot <- quantile(abs(tstar[, idx_tilde]), 1 - alpha)
  return(list(K_hat = K_list[idx_hat], K_tilde = K_list[idx_tilde],
              idx_hat = idx_hat, idx_tilde = idx_tilde,
              tstar = tstar, cv_boot = cv_boot, theta_star = theta))
}

bootstrap_cv_single <- function(psi, sd_infl, n, Bboot = 100, alpha = 0.05) {
  sqrt_n <- sqrt(n); tstar <- rep(NA, Bboot)
  for (b_idx in 1:Bboot) {
    xi <- rnorm(n); tstar[b_idx] <- sum(xi * psi) / sqrt_n / sd_infl
  }
  return(quantile(abs(tstar), 1 - alpha))
}

# ============================================================
# Summarize
# ============================================================

summarize_method <- function(res, true_val) {
  Error <- res$Est - true_val
  bias  <- mean(Error); stdev <- sd(res$Est); rMSE <- sqrt(mean(Error^2))
  CI <- res$CI
  CI_cover <- mean(CI[, 1] <= true_val & true_val <= CI[, 2], na.rm = TRUE)
  CI_L <- mean(CI[, 1], na.rm = TRUE); CI_U <- mean(CI[, 2], na.rm = TRUE)
  K_bar <- mean(res$K_used)
  return(list(RMSE = rMSE, Bias = bias, SD = stdev,
              Cover = CI_cover, CI_L = CI_L, CI_U = CI_U,
              CI_Len = CI_U - CI_L, K_bar = K_bar))
}

# ============================================================
# Output writers
# ============================================================

write_latex_table3 <- function(all_results, N_list, filepath) {
  lines <- c(
    "\\begin{table}[!h]", "\\centering", "\\small",
    paste0("\\caption{Integral over estimated upper contour set: Sieve estimators ",
           "with bootstrap-Lepski $\\tilde{K}$ selection and normal critical values.}"),
    "\\label{tab:Sim2_Disk}")
  for (panel_info in list(list("A", "Plugin_AdaptK_NormalCV", "Plug-In"),
                          list("B", "LOO_AdaptK_NormalCV", "LOO Debiased"))) {
    pn <- panel_info[[1]]; mk <- panel_info[[2]]; lb <- panel_info[[3]]
    lines <- c(lines,
      "\\vspace{0.3em}",
      sprintf("\\textbf{Panel %s: %s}", pn, lb),
      "\\begin{tabular}{lcccccccc}", "\\hline\\hline",
      "$n$ & RMSE & Bias & SD & CI$_L$ & CI$_U$ & U--L & Coverage(\\%) & $\\bar{\\tilde{K}}$\\\\",
      "\\hline")
    for (N in N_list) {
      s <- all_results[[as.character(N)]][[mk]]
      lines <- c(lines, sprintf("%d & %.4f & %.4f & %.4f & %.4f & %.4f & %.4f & %.2f & %.1f\\\\",
                                N, s$RMSE, s$Bias, s$SD, s$CI_L, s$CI_U, s$CI_Len,
                                s$Cover * 100, s$K_bar))
    }
    lines <- c(lines, "\\hline\\hline", "\\end{tabular}")
  }
  lines <- c(lines, "\\end{table}")
  writeLines(lines, filepath)
}

write_latex_table4 <- function(all_results, N_list, filepath) {
  lines <- c(
    "\\begin{table}[!h]", "\\centering", "\\small",
    "\\caption{Integral over estimated upper contour set: RMSE and CI coverage comparison.}",
    "\\label{tab:Sim2_Disk_Comp}")
  for (panel_info in list(list("A", "Plugin", "Plug-In"),
                          list("B", "LOO", "LOO Debiased"))) {
    pn <- panel_info[[1]]; pf <- panel_info[[2]]; lb <- panel_info[[3]]
    lines <- c(lines,
      "\\vspace{0.3em}",
      sprintf("\\textbf{Panel %s: %s}", pn, lb),
      "\\begin{tabular}{lcccccc}", "\\hline\\hline",
      " & \\multicolumn{2}{c}{RMSE} & \\multicolumn{4}{c}{CI Coverage (\\%)} \\\\",
      "\\cmidrule(lr){2-3} \\cmidrule(lr){4-7}",
      "$n$ & Fixed & Adapt & Fix+N & Adp+N & Fix+B & Adp+B \\\\",
      "\\hline")
    for (N in N_list) {
      r <- all_results[[as.character(N)]]
      lines <- c(lines, sprintf(
        "%d & %.4f & %.4f & %.1f & %.1f & %.1f & %.1f\\\\", N,
        r[[paste0(pf, "_FixedK_NormalCV")]]$RMSE,
        r[[paste0(pf, "_AdaptK_NormalCV")]]$RMSE,
        r[[paste0(pf, "_FixedK_NormalCV")]]$Cover * 100,
        r[[paste0(pf, "_AdaptK_NormalCV")]]$Cover * 100,
        r[[paste0(pf, "_FixedK_BootCV")]]$Cover * 100,
        r[[paste0(pf, "_AdaptK_BootCV")]]$Cover * 100))
    }
    lines <- c(lines, "\\hline\\hline", "\\end{tabular}")
  }
  lines <- c(lines, "\\end{table}")
  writeLines(lines, filepath)
}

write_latex_table_rateopt <- function(all_results, N_list, filepath) {
  lines <- c(
    "\\begin{table}[!h]", "\\centering", "\\small",
    paste0("\\caption{Integral over estimated upper contour set: Sieve estimators ",
           "at rate-optimal $\\hat{K}$ selected by bootstrap-Lepski (no undersmoothing), ",
           "with normal critical values.}"),
    "\\label{tab:Sim2_Disk_RateOpt}")
  for (panel_info in list(list("A", "Plugin_RateOpt", "Plug-In"),
                          list("B", "LOO_RateOpt", "LOO Debiased"))) {
    pn <- panel_info[[1]]; mk <- panel_info[[2]]; lb <- panel_info[[3]]
    lines <- c(lines,
      "\\vspace{0.3em}",
      sprintf("\\textbf{Panel %s: %s}", pn, lb),
      "\\begin{tabular}{lcccccccc}", "\\hline\\hline",
      "$n$ & RMSE & Bias & SD & CI$_L$ & CI$_U$ & U--L & Coverage(\\%) & $\\bar{\\hat{K}}$\\\\",
      "\\hline")
    for (N in N_list) {
      s <- all_results[[as.character(N)]][[mk]]
      lines <- c(lines, sprintf("%d & %.4f & %.4f & %.4f & %.4f & %.4f & %.4f & %.2f & %.1f\\\\",
                                N, s$RMSE, s$Bias, s$SD, s$CI_L, s$CI_U, s$CI_Len,
                                s$Cover * 100, s$K_bar))
    }
    lines <- c(lines, "\\hline\\hline", "\\end{tabular}")
  }
  lines <- c(lines, "\\end{table}")
  writeLines(lines, filepath)
}

write_latex_table_biasaware <- function(all_results, N_list, filepath) {
  lines <- c(
    "\\begin{table}[!h]", "\\centering", "\\small",
    paste0("\\caption{Integral over estimated upper contour set: Bias-aware CI at ",
           "rate-optimal $\\hat{K}$ with bias buffer critical value.}"),
    "\\label{tab:Sim2_Disk_BiasAware}")
  for (panel_info in list(list("A", "Plugin_BiasAware", "Plug-In"),
                          list("B", "LOO_BiasAware", "LOO Debiased"))) {
    pn <- panel_info[[1]]; mk <- panel_info[[2]]; lb <- panel_info[[3]]
    lines <- c(lines,
      "\\vspace{0.3em}",
      sprintf("\\textbf{Panel %s: %s}", pn, lb),
      "\\begin{tabular}{lcccccccc}", "\\hline\\hline",
      "$n$ & RMSE & Bias & SD & CI$_L$ & CI$_U$ & U--L & Coverage(\\%) & $\\bar{\\hat{K}}$\\\\",
      "\\hline")
    for (N in N_list) {
      s <- all_results[[as.character(N)]][[mk]]
      lines <- c(lines, sprintf("%d & %.4f & %.4f & %.4f & %.4f & %.4f & %.4f & %.2f & %.1f\\\\",
                                N, s$RMSE, s$Bias, s$SD, s$CI_L, s$CI_U, s$CI_Len,
                                s$Cover * 100, s$K_bar))
    }
    lines <- c(lines, "\\hline\\hline", "\\end{tabular}")
  }
  lines <- c(lines, "\\end{table}")
  writeLines(lines, filepath)
}

write_markdown <- function(all_results, N_list, J_list, K_list, J_fixed, K_fixed, filepath) {
  md <- c(
    "# Sim2: Upper Contour Integral on Unit Disk --- Results", "",
    "- h0(x) = (1 - ||x||^2)*(4 + sin(x1)*x2 + cos(x2))",
    sprintf("- theta0 = pi = %.6f", pi),
    sprintf("- J_list = %s,  K_list = %s", paste(J_list, collapse = ", "),
            paste(K_list, collapse = ", ")),
    sprintf("- J_fixed = %d,  K_fixed = %d", J_fixed, K_fixed), "",
    "## Table 3: Adaptive K_tilde + Normal CV", "")

  for (panel_info in list(list("Plugin_AdaptK_NormalCV", "Panel A: Plug-In"),
                          list("LOO_AdaptK_NormalCV", "Panel B: LOO Debiased"))) {
    mk <- panel_info[[1]]; lb <- panel_info[[2]]
    md <- c(md, sprintf("### %s", lb), "",
      "| n | RMSE | Bias | SD | CI_L | CI_U | U-L | Cover(%) | K_bar |",
      "|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for (N in N_list) {
      s <- all_results[[as.character(N)]][[mk]]
      md <- c(md, sprintf("| %d | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f | %.1f | %.1f |",
                           N, s$RMSE, s$Bias, s$SD, s$CI_L, s$CI_U, s$CI_Len,
                           s$Cover * 100, s$K_bar))
    }
    md <- c(md, "")
  }

  md <- c(md, "## Rate-Optimal K_hat + Normal CV", "")
  for (panel_info in list(list("Plugin_RateOpt", "A: Plug-In"),
                          list("LOO_RateOpt", "B: LOO Debiased"))) {
    mk <- panel_info[[1]]; lb <- panel_info[[2]]
    md <- c(md, sprintf("### Panel %s", lb), "",
      "| n | RMSE | Bias | SD | CI_L | CI_U | U-L | Cover(%) | K_bar_hat |",
      "|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for (N in N_list) {
      s <- all_results[[as.character(N)]][[mk]]
      md <- c(md, sprintf("| %d | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f | %.1f | %.1f |",
                           N, s$RMSE, s$Bias, s$SD, s$CI_L, s$CI_U, s$CI_Len,
                           s$Cover * 100, s$K_bar))
    }
    md <- c(md, "")
  }

  md <- c(md, "## Bias-Aware K_hat (rate-optimal + bias buffer)", "")
  for (panel_info in list(list("Plugin_BiasAware", "Panel A: Plug-In"),
                          list("LOO_BiasAware", "Panel B: LOO Debiased"))) {
    mk <- panel_info[[1]]; lb <- panel_info[[2]]
    md <- c(md, sprintf("### %s", lb), "",
      "| n | RMSE | Bias | SD | CI_L | CI_U | U-L | Cover(%) | K_bar |",
      "|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for (N in N_list) {
      s <- all_results[[as.character(N)]][[mk]]
      md <- c(md, sprintf("| %d | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f | %.1f | %.1f |",
                           N, s$RMSE, s$Bias, s$SD, s$CI_L, s$CI_U, s$CI_Len,
                           s$Cover * 100, s$K_bar))
    }
    md <- c(md, "")
  }

  md <- c(md, "## Comparison: Undersmoothing vs Bias-Aware", "")
  for (panel_info in list(list("Plugin_AdaptK_NormalCV", "Plugin_BiasAware", "A: Plug-In"),
                          list("LOO_AdaptK_NormalCV", "LOO_BiasAware", "B: LOO Debiased"))) {
    us_key <- panel_info[[1]]; ba_key <- panel_info[[2]]; lb <- panel_info[[3]]
    md <- c(md, sprintf("### Panel %s", lb), "",
      "| n | US U-L | BA U-L | Diff% | US Cover | BA Cover | K_tilde | K_hat |",
      "|---:|---:|---:|---:|---:|---:|---:|---:|")
    for (N in N_list) {
      us <- all_results[[as.character(N)]][[us_key]]
      ba <- all_results[[as.character(N)]][[ba_key]]
      diff_pct <- if (us$CI_Len > 0) (ba$CI_Len - us$CI_Len) / us$CI_Len * 100 else 0
      md <- c(md, sprintf("| %d | %.4f | %.4f | %+.1f%% | %.1f%% | %.1f%% | %.1f | %.1f |",
                           N, us$CI_Len, ba$CI_Len, diff_pct,
                           us$Cover * 100, ba$Cover * 100,
                           us$K_bar, ba$K_bar))
    }
    md <- c(md, "")
  }
  writeLines(md, filepath)
}

# ============================================================
# Main simulation
# ============================================================

run_sim2 <- function(outdir = ".") {

  N_list   <- c(500, 1000, 2000, 4000, 8000)
  B        <- 1000
  M        <- 50000
  J_list   <- c(4, 5, 6, 7, 8, 9, 10)
  J_fixed  <- 8
  Bboot    <- 500
  degree   <- 3
  alpha    <- 0.05
  eps_band <- 0.01

  K_list  <- J_list^2
  K_fixed <- J_fixed^2
  idx_fixed <- which(J_list == J_fixed)
  z_alpha   <- qnorm(1 - alpha / 2)

  # Sobol MC points in [-2,2]^2
  sobol_pts <- sobol(M, d = 2, randomize = "none")
  mc_pts <- -2 + 4 * sobol_pts

  ci_methods_plugin <- c("Plugin_FixedK_NormalCV", "Plugin_AdaptK_NormalCV",
                         "Plugin_FixedK_BootCV",   "Plugin_AdaptK_BootCV",
                         "Plugin_BiasAware")
  ci_methods_loo    <- c("LOO_FixedK_NormalCV", "LOO_AdaptK_NormalCV",
                         "LOO_FixedK_BootCV",   "LOO_AdaptK_BootCV",
                         "LOO_BiasAware")
  rateopt_methods   <- c("Plugin_RateOpt", "LOO_RateOpt")
  all_methods <- c(ci_methods_plugin, ci_methods_loo, rateopt_methods)

  cat("========================================================================\n")
  cat("Sim2: Upper Contour Integral on Unit Disk\n")
  cat("========================================================================\n")
  cat("J_list =", J_list, " => K_list =", K_list, "\n")
  cat("J_fixed =", J_fixed, " => K_fixed =", K_fixed, "\n")
  cat("B =", B, ", M =", M, ", Bboot =", Bboot, "\n\n")

  all_results <- list()
  raw_results <- list()  # per-replication data for post-hoc analysis

  for (N in N_list) {
    cat("Running N =", N, "...\n"); flush.console()
    t_start <- Sys.time()
    factor_us <- sqrt(log(N))

    # Precompute MC basis with derivatives
    cat("  Precomputing MC bases...")
    mc_derivs_list <- lapply(J_list, function(J) {
      tensor_bspline_2d_with_derivs(mc_pts, J, degree)
    })
    cat(" done\n")

    storage <- lapply(all_methods, function(m) {
      list(Est = rep(NA, B), CI = matrix(NA, nrow = B, ncol = 2),
           K_used = rep(NA, B), se = rep(NA, B))
    })
    names(storage) <- all_methods

    set.seed(1234)

    for (b in 1:B) {
      if (b %% 200 == 0) cat("  rep", b, "/", B, "\n")

      X <- matrix(runif(N * 2, -2, 2), nrow = N, ncol = 2)
      Xnorm2 <- X[, 1]^2 + X[, 2]^2
      h0_X <- (1 - Xnorm2) * (4 + sin(X[, 1]) * X[, 2] + cos(X[, 2]))
      Y <- h0_X + rnorm(N, 0, 1)

      est <- compute_estimates_allK_disk(Y, X, mc_pts, mc_derivs_list, J_list,
                                         degree, eps_band)

      # Lepski for plug-in
      lep_pi <- bootstrap_lepski_select(
        est$tau_plugin_vec, est$psi_plugin_mat, est$sd_plugin_vec,
        K_list, N, Bboot, factor_us, alpha)
      idx_ad_pi  <- lep_pi$idx_tilde
      idx_hat_pi <- lep_pi$idx_hat

      # Lepski for LOO
      lep_loo <- bootstrap_lepski_select(
        est$tau_loo_vec, est$psi_loo_mat, est$sd_loo_vec,
        K_list, N, Bboot, factor_us, alpha)
      idx_ad_loo  <- lep_loo$idx_tilde
      idx_hat_loo <- lep_loo$idx_hat

      # --- PLUG-IN CI methods ---
      # 1. Fixed + Normal
      tau <- est$tau_plugin_vec[idx_fixed]; se <- est$sd_plugin_vec[idx_fixed] / sqrt(N)
      half <- z_alpha * se
      storage$Plugin_FixedK_NormalCV$Est[b] <- tau
      storage$Plugin_FixedK_NormalCV$CI[b, ] <- c(tau - half, tau + half)
      storage$Plugin_FixedK_NormalCV$K_used[b] <- K_fixed; storage$Plugin_FixedK_NormalCV$se[b] <- se

      # 2. Adapt + Normal
      tau <- est$tau_plugin_vec[idx_ad_pi]; se <- est$sd_plugin_vec[idx_ad_pi] / sqrt(N)
      half <- z_alpha * se
      storage$Plugin_AdaptK_NormalCV$Est[b] <- tau
      storage$Plugin_AdaptK_NormalCV$CI[b, ] <- c(tau - half, tau + half)
      storage$Plugin_AdaptK_NormalCV$K_used[b] <- K_list[idx_ad_pi]; storage$Plugin_AdaptK_NormalCV$se[b] <- se

      # 3. Fixed + Boot
      tau <- est$tau_plugin_vec[idx_fixed]; se <- est$sd_plugin_vec[idx_fixed] / sqrt(N)
      cv <- bootstrap_cv_single(est$psi_plugin_mat[, idx_fixed], est$sd_plugin_vec[idx_fixed], N, Bboot, alpha)
      half <- cv * se
      storage$Plugin_FixedK_BootCV$Est[b] <- tau
      storage$Plugin_FixedK_BootCV$CI[b, ] <- c(tau - half, tau + half)
      storage$Plugin_FixedK_BootCV$K_used[b] <- K_fixed; storage$Plugin_FixedK_BootCV$se[b] <- se

      # 4. Adapt + Boot
      tau <- est$tau_plugin_vec[idx_ad_pi]; se <- est$sd_plugin_vec[idx_ad_pi] / sqrt(N)
      cv <- quantile(abs(lep_pi$tstar[, idx_ad_pi]), 1 - alpha)
      half <- cv * se
      storage$Plugin_AdaptK_BootCV$Est[b] <- tau
      storage$Plugin_AdaptK_BootCV$CI[b, ] <- c(tau - half, tau + half)
      storage$Plugin_AdaptK_BootCV$K_used[b] <- K_list[idx_ad_pi]; storage$Plugin_AdaptK_BootCV$se[b] <- se

      # --- LOO CI methods ---
      # 5. Fixed + Normal
      tau <- est$tau_loo_vec[idx_fixed]; se <- est$sd_loo_vec[idx_fixed] / sqrt(N)
      half <- z_alpha * se
      storage$LOO_FixedK_NormalCV$Est[b] <- tau
      storage$LOO_FixedK_NormalCV$CI[b, ] <- c(tau - half, tau + half)
      storage$LOO_FixedK_NormalCV$K_used[b] <- K_fixed; storage$LOO_FixedK_NormalCV$se[b] <- se

      # 6. Adapt + Normal
      tau <- est$tau_loo_vec[idx_ad_loo]; se <- est$sd_loo_vec[idx_ad_loo] / sqrt(N)
      half <- z_alpha * se
      storage$LOO_AdaptK_NormalCV$Est[b] <- tau
      storage$LOO_AdaptK_NormalCV$CI[b, ] <- c(tau - half, tau + half)
      storage$LOO_AdaptK_NormalCV$K_used[b] <- K_list[idx_ad_loo]; storage$LOO_AdaptK_NormalCV$se[b] <- se

      # 7. Fixed + Boot
      tau <- est$tau_loo_vec[idx_fixed]; se <- est$sd_loo_vec[idx_fixed] / sqrt(N)
      cv <- bootstrap_cv_single(est$psi_loo_mat[, idx_fixed], est$sd_loo_vec[idx_fixed], N, Bboot, alpha)
      half <- cv * se
      storage$LOO_FixedK_BootCV$Est[b] <- tau
      storage$LOO_FixedK_BootCV$CI[b, ] <- c(tau - half, tau + half)
      storage$LOO_FixedK_BootCV$K_used[b] <- K_fixed; storage$LOO_FixedK_BootCV$se[b] <- se

      # 8. Adapt + Boot
      tau <- est$tau_loo_vec[idx_ad_loo]; se <- est$sd_loo_vec[idx_ad_loo] / sqrt(N)
      cv <- quantile(abs(lep_loo$tstar[, idx_ad_loo]), 1 - alpha)
      half <- cv * se
      storage$LOO_AdaptK_BootCV$Est[b] <- tau
      storage$LOO_AdaptK_BootCV$CI[b, ] <- c(tau - half, tau + half)
      storage$LOO_AdaptK_BootCV$K_used[b] <- K_list[idx_ad_loo]; storage$LOO_AdaptK_BootCV$se[b] <- se

      # --- Rate-optimal K_hat + Normal CV ---
      tau_ro_pi <- est$tau_plugin_vec[idx_hat_pi]; se_ro_pi <- est$sd_plugin_vec[idx_hat_pi] / sqrt(N)
      half_ro_pi <- z_alpha * se_ro_pi
      storage$Plugin_RateOpt$Est[b] <- tau_ro_pi
      storage$Plugin_RateOpt$CI[b, ] <- c(tau_ro_pi - half_ro_pi, tau_ro_pi + half_ro_pi)
      storage$Plugin_RateOpt$K_used[b] <- K_list[idx_hat_pi]; storage$Plugin_RateOpt$se[b] <- se_ro_pi

      tau_ro_loo <- est$tau_loo_vec[idx_hat_loo]; se_ro_loo <- est$sd_loo_vec[idx_hat_loo] / sqrt(N)
      half_ro_loo <- z_alpha * se_ro_loo
      storage$LOO_RateOpt$Est[b] <- tau_ro_loo
      storage$LOO_RateOpt$CI[b, ] <- c(tau_ro_loo - half_ro_loo, tau_ro_loo + half_ro_loo)
      storage$LOO_RateOpt$K_used[b] <- K_list[idx_hat_loo]; storage$LOO_RateOpt$se[b] <- se_ro_loo

      # --- Bias-aware K_hat ---
      # Plugin bias-aware
      tau_ba_pi <- est$tau_plugin_vec[idx_hat_pi]; se_ba_pi <- est$sd_plugin_vec[idx_hat_pi] / sqrt(N)
      J_hat_pi <- J_list[idx_hat_pi]
      A_hat_pi <- log(log(J_hat_pi))
      theta_star_pi <- lep_pi$theta_star
      crit_ba_pi <- (z_alpha + A_hat_pi * theta_star_pi) * se_ba_pi
      storage$Plugin_BiasAware$Est[b] <- tau_ba_pi
      storage$Plugin_BiasAware$CI[b, ] <- c(tau_ba_pi - crit_ba_pi, tau_ba_pi + crit_ba_pi)
      storage$Plugin_BiasAware$K_used[b] <- K_list[idx_hat_pi]; storage$Plugin_BiasAware$se[b] <- se_ba_pi

      # LOO bias-aware
      tau_ba_loo <- est$tau_loo_vec[idx_hat_loo]; se_ba_loo <- est$sd_loo_vec[idx_hat_loo] / sqrt(N)
      J_hat_loo <- J_list[idx_hat_loo]
      A_hat_loo <- log(log(J_hat_loo))
      theta_star_loo <- lep_loo$theta_star
      crit_ba_loo <- (z_alpha + A_hat_loo * theta_star_loo) * se_ba_loo
      storage$LOO_BiasAware$Est[b] <- tau_ba_loo
      storage$LOO_BiasAware$CI[b, ] <- c(tau_ba_loo - crit_ba_loo, tau_ba_loo + crit_ba_loo)
      storage$LOO_BiasAware$K_used[b] <- K_list[idx_hat_loo]; storage$LOO_BiasAware$se[b] <- se_ba_loo
    }

    # Summarize
    summaries <- lapply(all_methods, function(m) summarize_method(storage[[m]], pi))
    names(summaries) <- all_methods
    all_results[[as.character(N)]] <- summaries
    raw_results[[as.character(N)]] <- storage  # save per-replication data

    t_elapsed <- difftime(Sys.time(), t_start, units = "secs")
    cat("  Done (", round(t_elapsed, 1), "s)\n")
    cat("  Plugin Adapt+N:", round(summaries$Plugin_AdaptK_NormalCV$Cover * 100, 1), "%",
        " LOO Adapt+B:", round(summaries$LOO_AdaptK_BootCV$Cover * 100, 1), "%",
        " Plugin RateOpt:", round(summaries$Plugin_RateOpt$Cover * 100, 1), "%",
        " LOO RateOpt:", round(summaries$LOO_RateOpt$Cover * 100, 1), "%\n")
  }

  # ===== Write outputs =====
  cat("\nWriting outputs...\n")
  write_latex_table3(all_results, N_list, file.path(outdir, "Sim2_Table3_AdaptNormal.tex"))
  write_latex_table4(all_results, N_list, file.path(outdir, "Sim2_Table4_Comparison.tex"))
  write_latex_table_rateopt(all_results, N_list, file.path(outdir, "Sim2_Table_RateOptimal.tex"))
  write_latex_table_biasaware(all_results, N_list, file.path(outdir, "Sim2_Table_BiasAware.tex"))
  write_markdown(all_results, N_list, J_list, K_list, J_fixed, K_fixed,
                 file.path(outdir, "Sim2_Disk_Final.md"))

  json_out <- lapply(all_results, function(ns) {
    lapply(ns, function(s) lapply(s, function(v) if (is.na(v)) NULL else round(v, 6)))
  })
  writeLines(toJSON(json_out, auto_unbox = TRUE, pretty = TRUE, null = "null"),
             file.path(outdir, "Sim2_Disk_Final.json"))

  # RData (includes raw_results with per-replication Est, CI, K_used, se)
  save(all_results, raw_results, N_list, J_list, K_list, J_fixed, K_fixed, B, M, Bboot,
       file = file.path(outdir, "Sim2_Disk_Final.RData"))

  cat("All outputs saved to:", outdir, "\n")

  # Console summary
  cat("\n========================================================================\n")
  for (panel_label in c("Adaptive K_tilde + Normal CV", "Rate-Optimal K_hat + Normal CV")) {
    cat("\n", panel_label, "\n", rep("-", 72), "\n", sep = "")
    if (grepl("Rate", panel_label)) {
      keys <- list(c("Plug-In", "Plugin_RateOpt"), c("LOO", "LOO_RateOpt"))
    } else {
      keys <- list(c("Plug-In", "Plugin_AdaptK_NormalCV"), c("LOO", "LOO_AdaptK_NormalCV"))
    }
    for (kk in keys) {
      lb <- kk[1]; mk <- kk[2]; cat("  ", lb, ":\n", sep = "")
      cat(sprintf("  %6s %8s %8s %8s %8s %8s %8s %8s %8s\n",
                  "n", "RMSE", "Bias", "SD", "CI_L", "CI_U", "U-L", "Cover%", "K_bar"))
      for (N in N_list) {
        s <- all_results[[as.character(N)]][[mk]]
        cat(sprintf("  %6d %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f %8.1f %8.1f\n",
                    N, s$RMSE, s$Bias, s$SD, s$CI_L, s$CI_U, s$CI_Len,
                    s$Cover * 100, s$K_bar))
      }
    }
  }
  cat("\n")

  invisible(all_results)
}

# ============================================================
# Entry point
# ============================================================

if (TRUE) {
  # Determine script directory robustly
  outdir <- tryCatch(dirname(sys.frame(1)$ofile), error = function(e) NULL)
  if (is.null(outdir) || outdir == "") {
    args <- commandArgs(trailingOnly = FALSE)
    file_arg <- grep("^--file=", args, value = TRUE)
    if (length(file_arg) > 0) {
      outdir <- dirname(sub("^--file=", "", file_arg))
    } else {
      outdir <- "."
    }
  }
  run_sim2(outdir = outdir)
}
