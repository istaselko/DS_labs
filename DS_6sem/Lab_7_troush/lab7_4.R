#install.packages("tweedie")
library(tweedie)

analyze_ts_residuals <- function(asset_name) {
  filename <- paste0("resid_", asset_name, ".csv")
  data <- read.csv(filename)
  x <- data$resid
  
  # 1. Сдвиг (TS / Tweedie определено для x > 0)
  shift_val <- abs(min(x)) + 0.1
  x_pos <- x + shift_val
  
  # 2. Оценка параметров методом моментов
  p_fixed <- 1.5
  mu_hat <- mean(x_pos)
  s2 <- var(x_pos)
  phi_hat <- s2 / (mu_hat^p_fixed)
  
  cat("=========================================\n")
  cat("Анализ TS для актива:", asset_name, "\n")
  cat("-----------------------------------------\n")
  cat("Параметры TS (Метод моментов):\n")
  cat("Сдвиг =", shift_val, "\n")
  cat("p   =", p_fixed, "\n")
  cat("mu  =", mu_hat, "\n")
  cat("phi =", phi_hat, "\n")
  
  # 3. Критерий Колмогорова-Смирнова
  suppressWarnings({
    ks_result <- ks.test(x_pos, "ptweedie", power=p_fixed, mu=mu_hat, phi=phi_hat)
  })
  
  cat("-----------------------------------------\n")
  cat("Критерий Колмогорова-Смирнова:\n")
  cat("Статистика D =", ks_result$statistic, "\n")
  cat("p-value      =", ks_result$p.value, "\n")
  cat("=========================================\n\n")
}

analyze_ts_residuals("ASML")
analyze_ts_residuals("URA")