#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(tidyr)
  library(ggplot2)
})

parse_args <- function(args) {
  out <- list(
    sample_col = "SampleID",
    score_col = "XBM_WGD_score",
    pred_group_col = "XBM_WGD_pred_group",
    quartile_col = "XBM_WGD_quartile",
    metrics = "",
    covariates = "age_at_diagnosis,sex,anatomical_site,cancer_type",
    true_wgd_col = "",
    tp53_col = ""
  )
  i <- 1
  while (i <= length(args)) {
    key <- sub("^--", "", args[[i]])
    if (i == length(args)) stop("Missing value for argument --", key)
    out[[gsub("-", "_", key)]] <- args[[i + 1]]
    i <- i + 2
  }
  required <- c("input", "out_dir")
  missing <- required[!required %in% names(out)]
  if (length(missing) > 0) stop("Missing required arguments: ", paste(missing, collapse = ", "))
  out
}

split_csv <- function(x) {
  x <- trimws(x)
  if (is.na(x) || x == "") character(0) else trimws(unlist(strsplit(x, ",")))
}

default_metric_candidates <- c(
  "Ploidy", "ploidy",
  "wGII", "weighted_GII",
  "aneuploidy_score", "AneuploidyScore", "cna_burden", "CNAburden",
  "pLOH", "HRD_LOH", "HRD.AI", "HRD_AI", "HRD_TAI", "HRD.LST", "HRD_LST", "HRDsum",
  "CentrosomeAmplification20.score"
)

clean_binary_group <- function(x, positive_name = "positive", negative_name = "negative") {
  y <- tolower(trimws(as.character(x)))
  positive <- y %in% c(
    "1", "yes", "true", "positive", "pos", "mut", "mutant", "mutation", "mutated",
    "wgd", "wgd-positive", "wgd_positive", "xbm-wgd-positive", "xbm_wgd_positive",
    "tp53-mutant", "tp53_mutant", "tp53-mut", "tp53_mut"
  )
  negative <- y %in% c(
    "0", "no", "false", "negative", "neg", "wt", "wildtype", "wild-type",
    "wgd-negative", "wgd_negative", "xbm-wgd-negative", "xbm_wgd_negative",
    "tp53-wildtype", "tp53_wildtype", "tp53-wt", "tp53_wt"
  )
  out <- rep(NA_character_, length(y))
  out[positive] <- positive_name
  out[negative] <- negative_name
  out
}

prepare_data <- function(df, args) {
  if (!args$score_col %in% names(df)) stop("Input table missing score column: ", args$score_col)
  df[[args$score_col]] <- as.numeric(df[[args$score_col]])

  if (!args$pred_group_col %in% names(df)) {
    stop("Input table missing predicted group column: ", args$pred_group_col)
  }
  df$XBM_WGD_pred_group_clean <- clean_binary_group(
    df[[args$pred_group_col]],
    positive_name = "XBM-WGD-positive",
    negative_name = "XBM-WGD-negative"
  )

  if (!args$quartile_col %in% names(df)) {
    df$XBM_WGD_quartile_clean <- cut(
      rank(df[[args$score_col]], ties.method = "first"),
      breaks = quantile(rank(df[[args$score_col]], ties.method = "first"), probs = seq(0, 1, 0.25), na.rm = TRUE),
      include.lowest = TRUE,
      labels = c("Q1", "Q2", "Q3", "Q4")
    )
  } else {
    df$XBM_WGD_quartile_clean <- factor(as.character(df[[args$quartile_col]]), levels = c("Q1", "Q2", "Q3", "Q4"))
  }

  df
}

select_metrics <- function(df, metrics_arg) {
  metrics <- split_csv(metrics_arg)
  if (length(metrics) == 0) metrics <- intersect(default_metric_candidates, names(df))
  missing <- setdiff(metrics, names(df))
  if (length(missing) > 0) stop("Metric columns not found: ", paste(missing, collapse = ", "))
  metrics <- metrics[sapply(metrics, function(x) any(!is.na(suppressWarnings(as.numeric(df[[x]])))))]
  if (length(metrics) == 0) stop("No usable numeric genomic metric columns found.")
  metrics
}

wilcoxon_by_group <- function(df, metrics, group_col, positive, negative) {
  bind_rows(lapply(metrics, function(metric) {
    tmp <- df %>%
      transmute(group = .data[[group_col]], value = as.numeric(.data[[metric]])) %>%
      filter(!is.na(group), !is.na(value), group %in% c(negative, positive))
    if (n_distinct(tmp$group) < 2) {
      return(tibble(Metric = metric, n_negative = NA_integer_, n_positive = NA_integer_,
                    median_negative = NA_real_, median_positive = NA_real_, p_value = NA_real_))
    }
    tibble(
      Metric = metric,
      n_negative = sum(tmp$group == negative),
      n_positive = sum(tmp$group == positive),
      median_negative = median(tmp$value[tmp$group == negative], na.rm = TRUE),
      median_positive = median(tmp$value[tmp$group == positive], na.rm = TRUE),
      p_value = wilcox.test(value ~ group, data = tmp)$p.value
    )
  })) %>%
    mutate(FDR = p.adjust(p_value, method = "BH")) %>%
    arrange(FDR, p_value)
}

kruskal_by_quartile <- function(df, metrics, quartile_col) {
  bind_rows(lapply(metrics, function(metric) {
    tmp <- df %>%
      transmute(quartile = .data[[quartile_col]], value = as.numeric(.data[[metric]])) %>%
      filter(!is.na(quartile), !is.na(value))
    if (n_distinct(tmp$quartile) < 2) {
      return(tibble(Metric = metric, n = nrow(tmp), p_value = NA_real_))
    }
    med <- tmp %>%
      group_by(quartile) %>%
      summarise(median = median(value, na.rm = TRUE), .groups = "drop") %>%
      mutate(name = paste0("median_", quartile)) %>%
      select(name, median) %>%
      tidyr::pivot_wider(names_from = name, values_from = median)
    bind_cols(tibble(Metric = metric, n = nrow(tmp), p_value = kruskal.test(value ~ quartile, data = tmp)$p.value), med)
  })) %>%
    mutate(FDR = p.adjust(p_value, method = "BH")) %>%
    arrange(FDR, p_value)
}

partial_spearman_one <- function(df, score_col, metric, covariates) {
  cols <- c(score_col, metric, covariates)
  cols <- cols[cols %in% names(df)]
  tmp <- df[, cols, drop = FALSE]
  tmp[[score_col]] <- as.numeric(tmp[[score_col]])
  tmp[[metric]] <- as.numeric(tmp[[metric]])
  tmp <- tmp[complete.cases(tmp[, c(score_col, metric), drop = FALSE]), , drop = FALSE]
  if (nrow(tmp) < 5) {
    return(tibble(Metric = metric, n = nrow(tmp), rho = NA_real_, p_value = NA_real_, covariates = paste(covariates, collapse = ",")))
  }

  cov_use <- covariates[covariates %in% names(tmp)]
  cov_use <- cov_use[sapply(cov_use, function(cn) n_distinct(tmp[[cn]][!is.na(tmp[[cn]])]) > 1)]
  keep_cols <- c(score_col, metric, cov_use)
  tmp <- tmp[complete.cases(tmp[, keep_cols, drop = FALSE]), , drop = FALSE]
  if (nrow(tmp) < 5) {
    return(tibble(Metric = metric, n = nrow(tmp), rho = NA_real_, p_value = NA_real_, covariates = paste(cov_use, collapse = ",")))
  }

  tmp$score_rank <- rank(tmp[[score_col]], ties.method = "average")
  tmp$metric_rank <- rank(tmp[[metric]], ties.method = "average")
  for (cn in cov_use) {
    if (is.character(tmp[[cn]])) tmp[[cn]] <- factor(tmp[[cn]])
  }

  if (length(cov_use) == 0) {
    ct <- suppressWarnings(cor.test(tmp$score_rank, tmp$metric_rank, method = "pearson"))
  } else {
    cov_terms <- paste0("`", cov_use, "`")
    form_x <- as.formula(paste("score_rank ~", paste(cov_terms, collapse = " + ")))
    form_y <- as.formula(paste("metric_rank ~", paste(cov_terms, collapse = " + ")))
    rx <- residuals(lm(form_x, data = tmp))
    ry <- residuals(lm(form_y, data = tmp))
    ct <- suppressWarnings(cor.test(rx, ry, method = "pearson"))
  }

  tibble(
    Metric = metric,
    n = nrow(tmp),
    rho = unname(ct$estimate),
    p_value = ct$p.value,
    covariates = paste(cov_use, collapse = ",")
  )
}

partial_spearman_table <- function(df, score_col, metrics, covariates) {
  bind_rows(lapply(metrics, function(metric) partial_spearman_one(df, score_col, metric, covariates))) %>%
    mutate(FDR = p.adjust(p_value, method = "BH")) %>%
    arrange(FDR, p_value)
}

plot_group_boxplots <- function(df, metrics, group_col, out_file, x_label) {
  plot_df <- df %>%
    select(all_of(c(group_col, metrics))) %>%
    pivot_longer(all_of(metrics), names_to = "Metric", values_to = "value") %>%
    mutate(value = as.numeric(value)) %>%
    filter(!is.na(.data[[group_col]]), !is.na(value))
  if (nrow(plot_df) == 0) return(invisible(NULL))
  p <- ggplot(plot_df, aes(x = .data[[group_col]], y = value, fill = .data[[group_col]])) +
    geom_boxplot(outlier.shape = NA, width = 0.6) +
    geom_jitter(width = 0.12, alpha = 0.45, size = 0.7) +
    facet_wrap(~ Metric, scales = "free_y") +
    labs(x = x_label, y = NULL) +
    theme_bw(base_size = 10) +
    theme(legend.position = "none", axis.text.x = element_text(angle = 30, hjust = 1))
  ggsave(out_file, p, width = 11, height = 7)
}

main <- function() {
  args <- parse_args(commandArgs(trailingOnly = TRUE))
  dir.create(args$out_dir, recursive = TRUE, showWarnings = FALSE)

  df <- read_csv(args$input, show_col_types = FALSE) %>%
    prepare_data(args)
  metrics <- select_metrics(df, args$metrics)
  covariates <- intersect(split_csv(args$covariates), names(df))

  pred_tbl <- wilcoxon_by_group(
    df,
    metrics,
    "XBM_WGD_pred_group_clean",
    positive = "XBM-WGD-positive",
    negative = "XBM-WGD-negative"
  )
  write_csv(pred_tbl, file.path(args$out_dir, "predicted_group_wilcoxon_bhfdr.csv"))

  quartile_tbl <- kruskal_by_quartile(df, metrics, "XBM_WGD_quartile_clean")
  write_csv(quartile_tbl, file.path(args$out_dir, "score_quartile_kruskal_bhfdr.csv"))

  partial_tbl <- partial_spearman_table(df, args$score_col, metrics, covariates)
  write_csv(partial_tbl, file.path(args$out_dir, "partial_spearman_bhfdr.csv"))

  plot_group_boxplots(
    df,
    metrics,
    "XBM_WGD_pred_group_clean",
    file.path(args$out_dir, "fig7_predicted_group_boxplots.pdf"),
    "Predicted WGD group"
  )
  plot_group_boxplots(
    df,
    metrics,
    "XBM_WGD_quartile_clean",
    file.path(args$out_dir, "fig7_score_quartile_boxplots.pdf"),
    "XBM-WGD score quartile"
  )

  if (args$true_wgd_col != "" && args$true_wgd_col %in% names(df)) {
    df$true_WGD_group_clean <- clean_binary_group(df[[args$true_wgd_col]], "genomic-WGD-positive", "genomic-WGD-negative")
    true_tbl <- wilcoxon_by_group(
      df,
      metrics,
      "true_WGD_group_clean",
      positive = "genomic-WGD-positive",
      negative = "genomic-WGD-negative"
    )
    write_csv(true_tbl, file.path(args$out_dir, "true_wgd_wilcoxon_bhfdr.csv"))
    plot_group_boxplots(
      df,
      metrics,
      "true_WGD_group_clean",
      file.path(args$out_dir, "supp_true_wgd_group_boxplots.pdf"),
      "Genomic WGD group"
    )
  }

  if (args$tp53_col != "" && args$tp53_col %in% names(df)) {
    df$TP53_group_clean <- clean_binary_group(df[[args$tp53_col]], "TP53-mutant", "TP53-wildtype")
    tp53_df <- df %>%
      transmute(TP53_group = TP53_group_clean, score = .data[[args$score_col]]) %>%
      filter(!is.na(TP53_group), !is.na(score))
    tp53_tbl <- tibble(
      n_wildtype = sum(tp53_df$TP53_group == "TP53-wildtype"),
      n_mutant = sum(tp53_df$TP53_group == "TP53-mutant"),
      median_wildtype = median(tp53_df$score[tp53_df$TP53_group == "TP53-wildtype"], na.rm = TRUE),
      median_mutant = median(tp53_df$score[tp53_df$TP53_group == "TP53-mutant"], na.rm = TRUE),
      p_value = if (n_distinct(tp53_df$TP53_group) == 2) wilcox.test(score ~ TP53_group, data = tp53_df)$p.value else NA_real_
    )
    write_csv(tp53_tbl, file.path(args$out_dir, "tp53_score_wilcoxon.csv"))

    p <- ggplot(tp53_df, aes(x = TP53_group, y = score, fill = TP53_group)) +
      geom_boxplot(outlier.shape = NA, width = 0.6) +
      geom_jitter(width = 0.12, alpha = 0.5, size = 0.8) +
      labs(x = NULL, y = args$score_col) +
      theme_bw(base_size = 11) +
      theme(legend.position = "none")
    ggsave(file.path(args$out_dir, "supp_tp53_score_boxplot.pdf"), p, width = 4.5, height = 4)
  }

  write_lines(metrics, file.path(args$out_dir, "metrics_used.txt"))
  write_lines(covariates, file.path(args$out_dir, "covariates_used.txt"))
  message("Wrote genomic association outputs to ", args$out_dir)
}

main()
