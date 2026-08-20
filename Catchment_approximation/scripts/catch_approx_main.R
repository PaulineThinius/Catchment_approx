### main pipeline that sources all other required scrips for catchment approximations based on the circular catchment approximation approach

# Libraries
# Core / general utilities
library(rlang)
library(glue)
library(here)
library(lubridate)
library(units)

# Data manipulation
library(dplyr)
library(tidyr)
library(purrr)
library(readr)
library(tidyverse)

# Spatial vector data
library(sf)

# Raster / spatial gridded data
library(terra)
library(raster)
library(stars)
library(exactextractr)

# NetCDF handling
library(ncdf4)

# GIS / hydrology / terrain tools
library(whitebox)

# Visualization
library(ggplot2)
library(tmap)
library(cols4all)


rm(list = ls())
cat("\014")

#-------------------------------------------------------------------------------
# check if working directory is set correctly (e.g. start R from within the project directory)
here::i_am("Catchment_approximation/scripts/catch_approx_main.R")

path_buff_in <- here::here("input_data")
path_buff_func <- here::here("Catchment_approximation", "scripts", "functions")
path_buff_approx_temp <- here::here("Catchment_approximation", "temp")
path_buff_approx_out <- here::here("Catchment_approximation", "output_data")
path_ALPAKAS <- here::here("input_data", "AlpAKaS_dataset")
path_project_files <- here::here("input_data", "project_files")


# Load functions
source(file.path(path_buff_func, "hydrometeo_valid_hydro_years.R"))
source(file.path(path_buff_func, "compute_recharge_era5.R"))
source(file.path(path_buff_func, "iterative_catchment_approximations.R"))
source(file.path(path_buff_func, "compute_topographic_catchment.R"))

#-------------------------------------------------------------------------------

# read selected ALPAKAS IDs
selected_ids <- read.csv(
  file.path(path_project_files, "alpakas_ids.csv"), header=FALSE,col.names = "ALPAKAS_ID"
)


# read metadata
station_meta <- read.csv(
  file.path(path_ALPAKAS, "ALPAKAS_station_meta.csv"),
  fileEncoding = "Windows-1252"
)


# keep only selected stations
station_meta <- station_meta %>%
  filter(ALPAKAS_ID %in% selected_ids$ALPAKAS_ID)



### determine hydrological years with valid basis of data for both discharge and meteorological time series
hydrometeo_valid_hy(station_meta = station_meta, path_ALPAKAS = path_ALPAKAS)

### compute catchment approximations based on buffer approach
# Compute recharge per hydrological year
if (file.exists(file.path(path_buff_approx_temp, "R_annual_1951_2024.nc"))){
  recharge_nc<- nc_open(file.path(path_buff_approx_temp, "R_annual_1951_2024.nc"))
} else{
  recharge_nc = compute_recharge_nc(
    path_buff_in = path_buff_in, 
    out_dir = path_buff_approx_temp,
    include_soil = TRUE, 
    include_snow = TRUE, 
    include_runoff = FALSE)
}


# Compute buffers iteratively using topography (and tracer tests)
catchment_approx_gdf = compute_iterative_buffers(
  recharge_nc  = recharge_nc,
  station_meta = station_meta,
  path_buff_approx_temp = path_buff_approx_temp, 
  path_buff_in = path_buff_in
  )

# save catchment approximations
st_write(catchment_approx_gdf, file.path(path_buff_approx_out, "./catchment_delineations/catchment_approx.geojson"),driver = "GeoJSON", append=FALSE)
