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
    wgd_col = "WGD",
    positive_label = "WGD-positive",
    negative_label = "WGD-negative",
    module_map = NA_character_
  )
  i <- 1
  while (i <= length(args)) {
    key <- sub("^--", "", args[[i]])
    if (i == length(args)) stop("Missing value for argument --", key)
    out[[gsub("-", "_", key)]] <- args[[i + 1]]
    i <- i + 2
  }
  required <- c("composition", "labels", "out_dir")
  missing <- required[!required %in% names(out)]
  if (length(missing) > 0) stop("Missing required arguments: ", paste(missing, collapse = ", "))
  out
}

clean_group <- function(x) {
  y <- tolower(trimws(as.character(x)))
  positive <- y %in% c("1", "wgd", "wgd-positive", "wgd_positive", "positive", "pos", "yes", "true")
  negative <- y %in% c("0", "wgd-negative", "wgd_negative", "negative", "neg", "no", "false")
  out <- rep(NA_character_, length(y))
  out[positive] <- "WGD-positive"
  out[negative] <- "WGD-negative"
  out
}

default_modules <- function() {
  list(
    Tumor_region = c("C7", "C8", "C9", "C5", "C0"),
    Tumor_gland = c("C7", "C8", "C9"),
    Invasive_front = c("C5"),
    Solid_tumor = c("C0"),
    TME_hub = c("C3"),
    Normal_background = c("C1", "C2", "C4", "C6"),
    Invasion_TME = c("C3", "C5", "C0")
  )
}

read_module_map <- function(path) {
  if (is.na(path) || path == "") return(default_modules())
  df <- read_csv(path, show_col_types = FALSE)
  if (!all(c("Module", "Clusters") %in% names(df))) {
    stop("Module map must contain columns Module and Clusters.")
  }
  modules <- strsplit(df$Clusters, "[;,| ]+")
  names(modules) <- df$Module
  modules
}

ensure_cluster_columns <- function(df) {
  if (!"n_valid_tiles" %in% names(df)) {
    stop("Composition table must contain n_valid_tiles.")
  }
  count_cols <- grep("^C[0-9]+_count$", names(df), value = TRUE)
  if (length(count_cols) == 0) stop("Composition table must contain C*_count columns.")
  clusters <- sub("_count$", "", count_cols)
  for (cluster in clusters) {
    prop_col <- paste0(cluster, "_prop")
    count_col <- paste0(cluster, "_count")
    if (!prop_col %in% names(df)) {
      df[[prop_col]] <- df[[count_col]] / df$n_valid_tiles
    }
  }
  df
}

cluster_long_tables <- function(df, sample_col) {
  count_long <- df %>%
    select(all_of(sample_col), WGD_group, n_valid_tiles, matches("^C[0-9]+_count$")) %>%
    pivot_longer(matches("^C[0-9]+_count$"), names_to = "Cluster", values_to = "Cluster_count") %>%
    mutate(Cluster = sub("_count$", "", Cluster),
           Cluster_prop = Cluster_count / n_valid_tiles)

  prop_long <- count_long %>%
    select(all_of(sample_col), WGD_group, n_valid_tiles, Cluster, Cluster_prop, Cluster_count)

  list(count = count_long, prop = prop_long)
}

cluster_wilcoxon <- function(cluster_long) {
  cluster_long %>%
    group_by(Cluster) %>%
    summarise(
      n_negative = sum(WGD_group == "WGD-negative" & !is.na(Cluster_prop)),
      n_positive = sum(WGD_group == "WGD-positive" & !is.na(Cluster_prop)),
      mean_negative = mean(Cluster_prop[WGD_group == "WGD-negative"], na.rm = TRUE),
      mean_positive = mean(Cluster_prop[WGD_group == "WGD-positive"], na.rm = TRUE),
      p_value = {
        x <- Cluster_prop[WGD_group == "WGD-negative"]
        y <- Cluster_prop[WGD_group == "WGD-positive"]
        if (length(na.omit(x)) > 0 && length(na.omit(y)) > 0) {
          wilcox.test(x, y)$p.value
        } else {
          NA_real_
        }
      },
      .groups = "drop"
    ) %>%
    mutate(FDR = p.adjust(p_value, method = "BH"))
}

make_module_tables <- function(df, modules, sample_col) {
  count_cols <- grep("^C[0-9]+_count$", names(df), value = TRUE)
  available <- sub("_count$", "", count_cols)
  rows <- list()

  for (module_name in names(modules)) {
    clusters <- intersect(modules[[module_name]], available)
    if (length(clusters) == 0) next
    module_count <- rowSums(df[paste0(clusters, "_count")], na.rm = TRUE)
    tmp <- tibble(
      SampleID_tmp = df[[sample_col]],
      WGD_group = df$WGD_group,
      Module = module_name,
      Module_count = as.integer(module_count),
      Non_module_count = as.integer(df$n_valid_tiles - module_count),
      n_valid_tiles = df$n_valid_tiles,
      Module_prop = module_count / df$n_valid_tiles,
      Clusters = paste(clusters, collapse = ";")
    )
    names(tmp)[names(tmp) == "SampleID_tmp"] <- sample_col
    rows[[module_name]] <- tmp
  }
  bind_rows(rows)
}

fit_module_glm <- function(module_df) {
  module_df %>%
    group_by(Module, Clusters) %>%
    group_modify(function(dat, key) {
      dat <- dat %>%
        filter(!is.na(WGD_group), !is.na(Module_count), !is.na(Non_module_count)) %>%
        mutate(WGD_group = factor(WGD_group, levels = c("WGD-negative", "WGD-positive")))
      if (nrow(dat) < 3 || n_distinct(dat$WGD_group) < 2) {
        return(tibble(
          beta = NA_real_, se = NA_real_, OR = NA_real_,
          CI_low = NA_real_, CI_high = NA_real_, p_value = NA_real_,
          mean_negative = NA_real_, mean_positive = NA_real_
        ))
      }
      fit <- glm(cbind(Module_count, Non_module_count) ~ WGD_group,
                 family = quasibinomial(), data = dat)
      coef_tab <- summary(fit)$coefficients
      row_name <- grep("^WGD_group", rownames(coef_tab), value = TRUE)[1]
      beta <- unname(coef_tab[row_name, "Estimate"])
      se <- unname(coef_tab[row_name, "Std. Error"])
      tibble(
        beta = beta,
        se = se,
        OR = exp(beta),
        CI_low = exp(beta - 1.96 * se),
        CI_high = exp(beta + 1.96 * se),
        p_value = unname(coef_tab[row_name, "Pr(>|t|)"]),
        mean_negative = mean(dat$Module_prop[dat$WGD_group == "WGD-negative"], na.rm = TRUE),
        mean_positive = mean(dat$Module_prop[dat$WGD_group == "WGD-positive"], na.rm = TRUE)
      )
    }) %>%
    ungroup() %>%
    mutate(FDR = p.adjust(p_value, method = "BH")) %>%
    arrange(FDR, p_value)
}

plot_module_forest <- function(glm_df, out_file) {
  plot_df <- glm_df %>% filter(!is.na(OR), !is.na(CI_low), !is.na(CI_high))
  if (nrow(plot_df) == 0) return(invisible(NULL))
  p <- ggplot(plot_df, aes(x = OR, y = reorder(Module, OR))) +
    geom_vline(xintercept = 1, linetype = 2, color = "grey50") +
    geom_errorbarh(aes(xmin = CI_low, xmax = CI_high), height = 0.2) +
    geom_point(size = 2) +
    scale_x_log10() +
    labs(x = "Odds ratio for WGD-positive", y = NULL) +
    theme_bw(base_size = 11)
  ggsave(out_file, p, width = 6, height = 4)
}

plot_module_boxplot <- function(module_df, out_file) {
  p <- ggplot(module_df, aes(x = WGD_group, y = Module_prop, fill = WGD_group)) +
    geom_boxplot(outlier.shape = NA, width = 0.6) +
    geom_jitter(width = 0.12, alpha = 0.45, size = 0.8) +
    facet_wrap(~ Module, scales = "free_y") +
    labs(x = NULL, y = "Module tile proportion") +
    theme_bw(base_size = 10) +
    theme(legend.position = "none", axis.text.x = element_text(angle = 30, hjust = 1))
  ggsave(out_file, p, width = 10, height = 7)
}

main <- function() {
  args <- parse_args(commandArgs(trailingOnly = TRUE))
  dir.create(args$out_dir, recursive = TRUE, showWarnings = FALSE)

  composition <- read_csv(args$composition, show_col_types = FALSE) %>%
    ensure_cluster_columns()
  labels <- read_csv(args$labels, show_col_types = FALSE)

  if (!args$sample_col %in% names(composition)) stop("Composition table missing sample column.")
  if (!args$sample_col %in% names(labels)) stop("Labels table missing sample column.")
  if (!args$wgd_col %in% names(labels)) stop("Labels table missing WGD column.")

  labels_clean <- labels %>%
    transmute(
      SampleID_join = .data[[args$sample_col]],
      WGD_group = clean_group(.data[[args$wgd_col]])
    )
  names(labels_clean)[names(labels_clean) == "SampleID_join"] <- args$sample_col

  dat <- composition %>%
    left_join(labels_clean, by = args$sample_col) %>%
    filter(!is.na(WGD_group)) %>%
    mutate(WGD_group = factor(WGD_group, levels = c("WGD-negative", "WGD-positive")))

  if (nrow(dat) == 0) stop("No samples with usable WGD group labels.")

  long_tables <- cluster_long_tables(dat, args$sample_col)
  write_csv(long_tables$prop, file.path(args$out_dir, "cluster_prop_long.csv"))
  write_csv(long_tables$count, file.path(args$out_dir, "cluster_count_long.csv"))
  write_csv(cluster_wilcoxon(long_tables$prop), file.path(args$out_dir, "cluster_wilcoxon_tests.csv"))

  modules <- read_module_map(args$module_map)
  module_df <- make_module_tables(dat, modules, args$sample_col)
  glm_df <- fit_module_glm(module_df)

  write_csv(module_df, file.path(args$out_dir, "module_prop_long.csv"))
  write_csv(module_df, file.path(args$out_dir, "module_count_long.csv"))
  write_csv(glm_df, file.path(args$out_dir, "module_quasibinomial_glm.csv"))

  plot_module_forest(glm_df, file.path(args$out_dir, "module_forest_plot.pdf"))
  plot_module_boxplot(module_df, file.path(args$out_dir, "module_prop_boxplot.pdf"))

  message("Wrote module analysis outputs to ", args$out_dir)
}

main()

