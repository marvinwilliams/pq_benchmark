library(ggplot2)
library(dplyr)
library(stringr)
library(tidyr)

input <- file("stdin")

results <- read.csv(input, stringsAsFactors = FALSE)
pattern <- "^BM_(\\w+)<(\\w+)<(.+)>>/(\\d+)$"

parsed_data <- results %>%
  filter(!str_detect(name, "_BigO$|_RMS$")) %>%
  extract(
    col = name, 
    into = c("workload", "pq_type", "data_type", "size"),
    regex = pattern,
    remove = FALSE,
    convert = TRUE
    ) %>%
  mutate(
         # detect if workload ends in "Small"
    small = ifelse(str_detect(workload, "Small"), "Small", "Large"),
    # remove "Small" from workload name
    workload = str_replace(workload, "Small", "")
    ) %>%
  select(workload, small, pq_type, data_type, size, real_time)

parsed_data$pq_type <- factor(parsed_data$pq_type, levels = c(
                                                              "std_pq",
                                                              "mq_pq",
                                                              "merge_heap",
                                                              "merge_heap_avx2",
                                                              "radix_heap",
                                                              "quick_heap",
                                                              "quick_heap_avx2",
                                                              "quick_heap_avx2_skip_popcnt",
                                                              "quick_heap_avx2_no_unroll",
                                                              "quick_heap_avx2_with_lut",
                                                              "quick_heap_avx2_with_lut_no_skip_popcnt",
                                                              "quick_heap_avx2_no_equal",
                                                              "boost_4_ary_heap",
                                                              "boost_8_ary_heap",
                                                              "boost_16_ary_heap",
                                                              "fib_heap",
                                                              "pairing_heap"
                                                              ))

plot <- ggplot(parsed_data, aes(x = size, y = real_time/size, color = pq_type, shape = pq_type)) +
  geom_line(alpha = 0.8) +
  geom_point(size = 2.5) +
  facet_grid(rows = vars(small), cols = vars(workload), scales = "free_y") +
  scale_x_log10(breaks = unique(parsed_data$size)) +
  scale_y_log10() +
  labs(
    title = "Priority Queue Benchmark",
    x = "Number of Iterations",
    y = "Time/(ns/iteration)",
    color = "PQ Type",
    shape = "PQ Type"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    legend.position = "bottom",
    axis.text.x = element_text(angle = 45, hjust = 1),
    plot.title = element_text(face = "bold")
  )

ggsave("plot.pdf", plot, width = 16, height = 8, dpi = 300)
