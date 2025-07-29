library(sf)
library(dplyr)
library(ggplot2)

# load function to rotate Hawaii, Alaska, PR into view
source("shift_states.r")

## prep shapefile
## ____________________________________________________________________________

crs_lambert <- "+proj=laea +lat_0=45 +lon_0=-100 +x_0=0 +y_0=0 +a=6370997 +b=6370997 +units=m +no_defs"


# download shapefile from https://www.census.gov/geographies/mapping-files/time-series/geo/carto-boundary-file.html
# exact file location: https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_us_state_20m.zip

zip_file_path <- "cb_2018_us_state_20m.zip"
tmp_dir <- tempdir()
unzip(zip_file_path, exdir = tmp_dir)

shapefile_path <- file.path(tmp_dir, "cb_2018_us_state_20m.shp")
shapefile_data <- st_read(shapefile_path) |>
    filter(!STATEFP %in% c("66", "69", "78", "60")) |>
    st_transform(crs = crs_lambert)

usa <- shift_states(shapefile_data)

## plot feature importance
## ____________________________________________________________________________
imp <- read.csv('updated_model/csv/feature_importance_by_state.csv')

# get top feature by state
imp_grp <- imp |>
    arrange(state, desc(Importance)) |>
    group_by(state) |>
    slice_head(n = 1)

plot_data <- usa |>
    left_join(imp_grp, by = c("NAME" = "state"))

# generate plot
ggplot() +
    geom_sf(data = plot_data, aes(fill = Feature_Name), color = 'black') +
    scale_fill_brewer(palette = "Paired", name = "Feature_Name") +
    theme(
      axis.text = element_blank(),  
      axis.ticks = element_blank(), 
      axis.title = element_blank(), 
      panel.grid = element_blank(),
      panel.background = element_blank(), 
      legend.position = "bottom",   
      legend.direction = "vertical",
    ) +
    ggtitle("Top Feature Importance (Local Models)")

## save to local file
ggsave("updated_model/plots/map_top_local_features2.png", width = 8, height = 6, dpi = 300)


## plot accuracy comparison map
## ____________________________________________________________________________
perf <- read.csv('updated_model/csv/model_performance_by_state.csv') |>
    mutate(diff = local_auc - global_auc) |>
    select(state, global_auc, local_auc, diff, rbo)

perf_plot_data <- usa |>
    left_join(perf, by = c("NAME" = "state"))

ggplot() +
  geom_sf(data = perf_plot_data, aes(fill = diff), color = 'black') +
  scale_fill_gradient2(
    low = "red",      # for negative values
    mid = "white",    # for zero
    high = "blue",    # for positive values
    midpoint = 0,
    name = "AUC\nImprovement"
  ) +
  theme(
    axis.text = element_blank(),  
    axis.ticks = element_blank(), 
    axis.title = element_blank(), 
    panel.grid = element_blank(),
    panel.background = element_blank(), 
    legend.position = "bottom",   
    # legend.direction = "vertical"
  ) +
  ggtitle("AUC Improvement From Local Models")

ggsave("updated_model/plots/map_auc_improvement.png", width = 8, height = 6, dpi = 300)

## plot rank-biased overlap rbo
## ____________________________________________________________________________
rbo <- read.csv('updated_model/csv/model_performance_by_state.csv')

rbo_plot_data <- usa |>
    left_join(perf, by = c("NAME" = "state"))

ggplot() +
  geom_sf(data = rbo_plot_data, aes(fill = rbo), color = 'black') +
  scale_fill_gradient(
    low = "white", 
    high = "blue", 
    name = "Rank-Biased\nOverlap"
  )  +
  theme(
    axis.text = element_blank(),  
    axis.ticks = element_blank(), 
    axis.title = element_blank(), 
    panel.grid = element_blank(),
    panel.background = element_blank(), 
    legend.position = "bottom",   
    # legend.direction = "vertical"
  ) +
  ggtitle("Ranked-Biased Overlap Feature Importance")

ggsave("updated_model/plots/map_rbo.png", width = 8, height = 6, dpi = 300)
