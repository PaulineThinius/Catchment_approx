### Iterative recharge area size computation
# main function returns sf object with one circular catchment geometry per ALPAKAS_ID


#-------------------------------------------------------------------------------   

# --- Functions ---

# Function to compute mean recharge for one cell
get_mean_recharge_all <- function(
    x, y, hydro_years, spring_names, recharge_array, 
    recharge_coords, lat_values, lon_values, 
    hydro_years_available, circle_geom=NULL
) {
  
  # Create spatial point
  point <- sf::st_sfc(sf::st_point(c(x, y)), crs = 4326)
  distances <- sf::st_distance(point, recharge_coords)
  ordered_indices <- order(as.numeric(distances))[1:10]
  
  lon_len <- length(lon_values)
  lat_len <- length(lat_values)
  
  # Map requested years to available years
  year_indices <- match(hydro_years, hydro_years_available)
  year_indices <- year_indices[!is.na(year_indices)]
  
  if (length(year_indices) < 1) {
    return(list(mean_recharge_mm_per_year = NA_real_))
  }
  
  # Loop over closest cells
  for (i in seq_along(ordered_indices)) {
    closest_index <- ordered_indices[i]
    
    lon_index <- ((closest_index - 1) %% lon_len) + 1
    lat_index <- ((closest_index - 1) %/% lon_len) + 1
    
    # Make sure indices are within array bounds
    if (lon_index > dim(recharge_array)[1] || lat_index > dim(recharge_array)[2]) next
    
    recharge_values <- sapply(year_indices, function(year_index) {
      if (year_index > dim(recharge_array)[3]) return(NA_real_)
      val <- recharge_array[lon_index, lat_index, year_index]
      if (is.null(val) || is.na(val)) NA_real_ else val
    })
    

    # If all values are NA, raise an error
    if (all(is.na(recharge_values))) {
      cell_lon <- lon_values[lon_index]
      cell_lat <- lat_values[lat_index]
      stop(sprintf(
        "All recharge values are NA for cell at lon=%.5f, lat=%.5f. Cannot proceed.",
        cell_lon, cell_lat
      ))
    }
    

    # If all values are negative, move to next closest cell
    if (mean(recharge_values, na.rm = TRUE) < 0) { #if (all(recharge_values < 0, na.rm = TRUE))
      cell_lon <- lon_values[lon_index]
      cell_lat <- lat_values[lat_index]
      
      message(sprintf(
        "Skipping cell at lon=%.5f, lat=%.5f (mean recharge < 0). Moving to the next closest cell.",
        cell_lon, cell_lat
      ))
      
      next
    }
    
    # Otherwise, calculate the mean
    mean_val <- mean(recharge_values, na.rm = TRUE)
    return(list(mean_recharge_mm_per_year = mean_val))
    
  }
  
  # If all cells failed
  return(list(mean_recharge_mm_per_year = NA_real_))
}


# Function  to compute aggregated mean recharge over a polygon
get_mean_recharge_buffer <- function(
    x,y,hydro_years,spring_names, recharge_array, recharge_coords, 
    lat_values, lon_values, hydro_years_available, circle_geom
) {

  # Map requested years
  year_indices <- match(hydro_years, hydro_years_available)
  year_indices <- year_indices[!is.na(year_indices)]
  if (length(year_indices) < 1) return(list(ERA5_all = NA_real_))
  
  lon_res <- mean(diff(lon_values))
  lat_res <- mean(diff(lat_values))

  r <- brick(aperm(recharge_array, c(2,1,3)),  
             xmn = min(lon_values) - lon_res / 2,
             xmx = max(lon_values) + lon_res / 2,
             ymn = min(lat_values) + lat_res / 2,
             ymx = max(lat_values) - lat_res / 2,
             crs = CRS("+init=EPSG:4326")
  )

  # Subset raster to requested years 
  r_subset <- raster::subset(r, year_indices) 
  
  # Check CRS 
  st_crs(circle_geom) 
  crs(r_subset) 
  
  # Transform buffer to raster CRS if needed 
  if (st_crs(circle_geom) != st_crs(r_subset)) {
    circle_geom <- st_transform(circle_geom, st_crs(r_subset)) 
  }
  
  # Exact extraction
  yearly_means <- exactextractr::exact_extract(r_subset, circle_geom, 'mean')
  yearly_means_vec <- as.numeric(yearly_means)
  
  if (all(is.na(yearly_means_vec))) return(list(mean_recharge_mm_per_year = NA_real_))
  
  mean_val <- mean(yearly_means_vec, na.rm = TRUE)
  
  return(list(mean_recharge_mm_per_year = mean_val))
}


# Wrapper Function to compute recharge results
compute_recharge_results <- function(
    springs_sf, recharge_array, recharge_coords,
    lat_values, lon_values, hydro_years_available, circle_approx_sf = NULL, mode = "cell"
) {
  
  # Decide which mean function to use
  if (mode == "cell") {
    mean_function <- get_mean_recharge_all
  } else if (mode == "circle_approx") {
    mean_function <- get_mean_recharge_buffer
  }
  
  # Transform springs_sf to EPSG:4326
  springs_sf_4326 <- st_transform(springs_sf, crs = 4326)
  
  pmap_dfr(
    list(
      x = st_coordinates(springs_sf_4326)[,1],
      y = st_coordinates(springs_sf_4326)[,2],
      hydro_years = springs_sf_4326$hydrological_years,
      spring_names = springs_sf_4326$spring_name,
      local_station_ID = springs_sf_4326$local_station_ID,
      ALPAKAS_ID = springs_sf_4326$ALPAKAS_ID
    ),
    function(x, y, hydro_years, spring_names, local_station_ID, ALPAKAS_ID) {
      
      # If circle_approx_sf exists, get the circle geometry for the current spring
      circle_geom <- NULL
      if (!is.null(circle_approx_sf)) {
        circle_geom <- circle_approx_sf$geometry[circle_approx_sf$local_station_ID == local_station_ID]
        if (length(circle_geom) == 0) circle_geom <- NULL
      }
      
      mean_function(
        x = x,
        y = y,
        hydro_years = hydro_years,
        spring_names = spring_names,
        recharge_array = recharge_array,
        recharge_coords = recharge_coords,
        lat_values = lat_values,
        lon_values = lon_values,
        hydro_years_available = hydro_years_available,
        circle_geom = circle_geom
      ) %>%
        as_tibble() %>%
        mutate(
          spring_name = spring_names,
          local_station_ID = local_station_ID,
          ALPAKAS_ID = ALPAKAS_ID
        )
    }
  )
}


# Function to create shifted circles (shifted along direction of topographic catchment) and save as geojson
create_circular_approx <- function(
    station_sf, topographic_catch_sf,  shift_list, iter, out_dir
) { 
  
  # Filter topographic_catch
  topographic_catch_sf <- topographic_catch_sf %>%
    rename(ALPAKAS_ID = AlpAKaS_ID) %>%
    filter(ALPAKAS_ID %in% station_sf$ALPAKAS_ID) %>%
    arrange(match(ALPAKAS_ID, station_sf$ALPAKAS_ID))
  
  # station coords
  geom_coords <- st_coordinates(station_sf$geometry)
  
  # Calculate the centroid of topographic catchment
  centroids <- st_centroid(st_geometry(topographic_catch_sf))
  centroid_sf <- st_as_sf(data.frame(geometry = centroids), crs = st_crs(topographic_catch_sf))
  centroid_coords <- st_coordinates(centroids)
  
  # Initialize direction vector matrix
  direction_vectors <- matrix(NA, nrow = nrow(station_sf), ncol = 2)
  
  for (i in seq_len(nrow(station_sf))) {

    station_point <- geom_coords[i, ]
    target_point <- centroid_coords[i, ]
    
    # Compute vector: target - station
    vec <- target_point - station_point
    # normalize to unit vector
    vec <- vec / sqrt(sum(vec^2))
    
    direction_vectors[i, ] <- vec
  }
  
  
  unit_vectors <- apply(direction_vectors, 1, function(vec) {
    vec / sqrt(sum(vec^2))
  })
  
  unit_vectors <- t(unit_vectors)
  
  
  
  ## Create a shifted circular catchment approximation
  for (shift in shift_list){
    # Calculate the new centroid of the circle using radius * shift
    new_points <- pmap(list(
      geom = split(st_coordinates(station_sf$geometry), row(st_coordinates(station_sf$geometry))),
      unit_vec = split(unit_vectors, row(unit_vectors)),
      radius = station_sf$radius_km*1000
    ), function(geom, unit_vec, radius) {
      st_point(c(geom[1] + unit_vec[1] * radius * shift, geom[2] + unit_vec[2] * radius * shift))
    })
    new_points_sf <- st_sfc(new_points, crs = st_crs(station_sf))
    
    # Create a circular buffer around the new point with radius r
    circular_buffers <- mapply(function(new_point, radius) {
      st_buffer(st_sfc(new_point, crs = st_crs(station_sf)), dist = radius*1000)
    }, new_points, station_sf$radius_km)
    
    
    
    # Save shifted buffers as a shapefile (EPSG:3035)
    valid_years_str <- map_chr(
      station_sf$hydrological_years, 
      ~ paste(.x, collapse = ", ")
    )
    # Base attributes
    sf_attrs <- list(
      country_code = station_sf$country_code,
      ALPAKAS_ID = station_sf$ALPAKAS_ID,
      local_station_ID = station_sf$local_station_ID,
      spring_name = station_sf$spring_name,
      mean_recharge_mm_per_year = round(station_sf$mean_recharge_mm_per_year, 2),
      mean_discharge_l_per_s = round(station_sf$mean_discharge_l_per_s, 2),
      mean_discharge_m3_per_year = round(station_sf$mean_discharge_m3_per_year, 2),
      estimated_recharge_area_km2 = station_sf$estimated_recharge_area_km2,
      radius_km = round(station_sf$radius_km, 2),
      valid_years = valid_years_str,
      iter=station_sf$iter,
      geometry = st_sfc(circular_buffers, crs = st_crs(station_sf))
    )
    
    
    

    
    # Create sf object
    output_gdf <- do.call(st_sf, sf_attrs)
    today <- format(Sys.Date(), "%Y%m%d")
    target_dir <- file.path(out_dir, "circle_approx_data")
    if (!dir.exists(target_dir)) {
      dir.create(target_dir, recursive = TRUE)
    }
    

    filename <- glue("approx_shift",shift, "_iter",iter, "_",today,".geojson" )
    
    
    
    st_write(output_gdf, file.path(target_dir,filename), driver = "GeoJSON",append=FALSE, quiet=TRUE, delete_dsn = TRUE)
    return(output_gdf)
  }
}



# Function to update states and check convergence for iteration process
check_convergence <- function(
    station_sf,
    station_df_old,
    area_history,
    active_ids_current,
    i,
    tol
) {
  
  result <- list(
    station_sf = station_sf,
    active_ids = unique(station_sf$ALPAKAS_ID),
    oscillating_ids = character(),
    area_history = area_history
  )
  
  # ---- 1. UPDATE HISTORY (relevant for oscillation check)
  
  id_map <- setNames(seq_len(nrow(area_history)), area_history$ALPAKAS_ID)
  
  for (idx in seq_len(nrow(station_sf))) {
    
    id <- as.character(station_sf$ALPAKAS_ID[idx])
    area_val <- station_sf$estimated_recharge_area_km2[idx]
    
    hist_idx <- id_map[[id]]
    
    if (is.null(hist_idx)) {
      
      area_history <- dplyr::bind_rows(
        area_history,
        tibble::tibble(
          ALPAKAS_ID = id,
          history = list(area_val)
        )
      )
      
      id_map[[id]] <- nrow(area_history)
      
    } else {
      
      area_history$history[[hist_idx]] <- c(
        area_history$history[[hist_idx]],
        area_val
      )
    }
  }
  
  result$area_history <- area_history
  
  # Add the current iter to each ALPAKAS_ID
  station_sf$iter[station_sf$ALPAKAS_ID %in% active_ids_current] <- i

  ## Skip convergence logic only for i == 0
  if (i == 0) return(result)
  
  # ---- 2. Compare current vs previous ----
  station_current <- station_sf %>%
    st_drop_geometry() %>%
    dplyr::select(
      ALPAKAS_ID,
      area_new = estimated_recharge_area_km2
    )
  
  station_previous <- station_df_old %>%
    dplyr::select(
      ALPAKAS_ID,
      area_old = estimated_recharge_area_km2
    )
  
  df_compare <- station_current %>%
    left_join(station_previous, by = "ALPAKAS_ID") %>%
    mutate(
      pct_change = ifelse(
        is.na(area_old) | area_old == 0,
        NA_real_,
        abs(100 * (area_new - area_old) / area_old)
      )
    )
  
  # ---- 3. Active IDs ----
  active_ids <- df_compare %>%
    filter(!is.na(pct_change) & pct_change > tol & area_new >= 0) %>%
    pull(ALPAKAS_ID)
  
  # ---- 4. Negative values ----
  negative_ids <- df_compare %>%
    filter(area_new < 0) %>%
    pull(ALPAKAS_ID)
  
  if (length(negative_ids) > 0) {
    
    update_data <- station_df_old %>%
      filter(ALPAKAS_ID %in% negative_ids)
    
    idx <- which(station_sf$ALPAKAS_ID %in% negative_ids)
    attr_cols <- setdiff(names(update_data), "ALPAKAS_ID")
    
    station_sf[idx, attr_cols] <- update_data[, attr_cols]
    
    active_ids <- setdiff(active_ids, negative_ids)
  }
  
  # ---- 5. Oscillations ----
  oscillating_ids <- character()
  
  if (i >= 3) {
    
    is_oscillating <- function(x, n = 4) {
      if (length(x) < n) return(FALSE)
      last_vals <- tail(x, n)
      u <- unique(last_vals)
      length(u) == 2 && all(last_vals %in% u)
    }
    
    oscillating_ids <- area_history %>%
      filter(map_lgl(history, is_oscillating)) %>%
      pull(ALPAKAS_ID) %>%
      intersect(active_ids)
    
    for (id in oscillating_ids) {
      
      id_char <- as.character(id)
      hist_idx <- which(as.character(area_history$ALPAKAS_ID) == id_char)
      
      if (length(hist_idx) != 1) next
      
      frozen_val <- tail(area_history$history[[hist_idx]], 1)
      
      station_sf <- station_sf %>%
        mutate(
          estimated_recharge_area_km2 = ifelse(
            ALPAKAS_ID == id_char,
            frozen_val,
            estimated_recharge_area_km2
          )
        )
      
      message("Oscillating ID frozen: ", id_char, " -> ", frozen_val)
    }
    
    active_ids <- setdiff(active_ids, oscillating_ids)
  }
  
  # ---- 6. Output ----
  result$station_sf <- station_sf
  result$active_ids <- active_ids
  result$oscillating_ids <- oscillating_ids
  
  return(result)
}



#-------------------------------------------------------------------------------   

# --- Main function ---

compute_iterative_buffers <- function(
    recharge_nc,
    station_meta,
    path_buff_approx_temp, 
    path_buff_in
){
  
  ## Load data
  # Hydrological years
  hydro_years_path <- file.path(path_buff_approx_temp,"hydrometeo_valid_hydro_years_era5land.RDS")
  hydro_years_data <- readRDS(hydro_years_path) %>%
    distinct(ALPAKAS_ID, hydro_year, mean_discharge) %>%
    mutate(hydro_year = as.numeric(hydro_year)) %>%
    group_by(ALPAKAS_ID) %>%
    summarise(
      hydrological_years = list(hydro_year),
      .groups = "drop"
    )
  
  # Mean discharge
  mean_discharge_path <- file.path(path_buff_approx_temp,"hydrometeo_valid_mean_discharge_era5land.RDS")
  discharge_data <- readRDS(mean_discharge_path)
  discharge_data <- discharge_data %>%
    rename(mean_discharge_l_per_s = q_mean)
  
  # Station metadata merge
  station_sf <- sf::st_as_sf(
    station_meta,
    coords = c("station_lon", "station_lat"),
    crs = 4326,
    remove = FALSE
  )
  station_sf <- station_sf %>%
    dplyr::left_join(hydro_years_data, by = "ALPAKAS_ID") %>%
    dplyr::left_join(discharge_data, by = "ALPAKAS_ID")
  
  # Recharge from NetCDF
  recharge       <- ncvar_get(recharge_nc, "R")
  lat            <- ncvar_get(recharge_nc, "latitude")
  lon            <- ncvar_get(recharge_nc, "longitude")
  hydro_years_available <- recharge_nc$dim$hydrological_year$vals
  
  # Build spatial grid
  recharge_nc_sf <- expand.grid(lon = lon, lat = lat) %>%
    st_as_sf(coords = c("lon","lat"), crs = 4326)
  
  nc_close(recharge_nc)
  
  
  
  
  ## Iteration Loop
  # Initialize iteration parameters
  circle_approx_sf <- NULL
  tol <- 5 # percent area difference between iterations
  i <- 0 # initialize counter
  recharge_mode <- "cell" # start with recharge from one cell
  active_ids <- station_sf$ALPAKAS_ID # start with all IDs
  
  # Initialize history before iterations (for oscillation check)
  area_history <- tibble(
    ALPAKAS_ID = station_sf$ALPAKAS_ID,
    history = vector("list", nrow(station_sf))
  )
  
  
  ## Start iteration
  while (length(active_ids) > 0){
    
    # Process report
    cat("\n", strrep("=", 40))
    message(sprintf(
      "\n\nIteration %d: %d springs still active.\nALPAKAS_IDs:\n %s",
      i,
      length(active_ids),
      paste(active_ids, collapse = ", ")
    ))
    
    # subset active springs
    springs_active <- station_sf %>%
      filter(ALPAKAS_ID %in% active_ids)
    
    # Recharge from ERA5-Land estimate
    message("\nExtracting mean recharge from ERA5-LAND based recharge grid...")
    recharge_results <- compute_recharge_results(
      springs_active,
      recharge, recharge_nc_sf, lat, lon, hydro_years_available,
      circle_approx_sf = circle_approx_sf,
      mode = recharge_mode
    )
    
    # Update recharge values
    if (i ==0) {
      
      station_sf <- station_sf %>%
        left_join(
          recharge_results,
          by = c("local_station_ID", "spring_name", "ALPAKAS_ID")
        )
      
    } else {
      
      station_sf <- station_sf %>%
        left_join(
          recharge_results,
          by = c("local_station_ID", "spring_name", "ALPAKAS_ID"),
          suffix = c("", "_new")
        ) %>%
        mutate(
          mean_recharge_mm_per_year = ifelse(
            ALPAKAS_ID %in% active_ids,
            mean_recharge_mm_per_year_new,
            mean_recharge_mm_per_year
          )
        ) %>%
        dplyr::select(-ends_with("_new"))
      
    }
    
    # set CRS
    station_sf <- st_transform(station_sf, crs = 3035)
    
    # Estimate recharge area: A = Q/R
    station_sf <- station_sf %>%
      mutate(
        recharge_m_per_year = mean_recharge_mm_per_year / 1000,
        mean_discharge_m3_per_year = mean_discharge_l_per_s * 60 * 60 * 24 * 365 * 0.001,
        estimated_recharge_area_km2_raw = mean_discharge_m3_per_year / recharge_m_per_year / (1000 * 1000),
        estimated_recharge_area_km2 = ifelse(
          estimated_recharge_area_km2_raw < 1,
          ceiling(estimated_recharge_area_km2_raw * 10) / 10,
          round(estimated_recharge_area_km2_raw, 0)
        ),
        radius_km = sqrt(ifelse(estimated_recharge_area_km2 >= 0,
                                       estimated_recharge_area_km2,
                                       NA_real_) / pi),
        valid_years = lengths(hydrological_years)
      )
  
    
    
    # One time initialization actions
    if (i==0){

      station_sf$iter <- 0
      
      # Compute topographic catchment once
      message("\nDelineating topographic catchments from Copernicus-30 DEM...")
      topo_path <- file.path(path_buff_approx_temp,"all_topographic_catchments.geojson")
      if (file.exists(topo_path))
      {
        topographic_catch_sf <- st_read(topo_path, quiet=TRUE) 
      }else{
        topographic_catch_sf <- compute_topographic_catchments(
          station_sf,
          path_buff_in,
          path_buff_approx_temp
        )
      }

    }
    

    ## Convergence check and update state
    res <- check_convergence(station_sf, station_df_old, area_history, active_ids, i, tol)
    
    station_sf <- res$station_sf
    active_ids <- res$active_ids
    area_history <- res$area_history

    if (length(active_ids) > 0) {
      
      message(
        "\nIteration ", i,
        ": springs still above tolerance (", tol, "%):"
      )
      
      history_wide <- area_history %>% 
        filter(ALPAKAS_ID %in% active_ids) %>%               
        unnest(history) %>%                               
        group_by(ALPAKAS_ID) %>% 
        mutate(iter = row_number() - 1) %>%                
        ungroup() %>% 
        rename(area = history) %>%                          
        pivot_wider(                                        
          id_cols       = ALPAKAS_ID,
          names_from    = iter,
          values_from   = area,
          names_prefix  = "area_iter_",
          values_fill   = list(area = NA_real_)   
        ) %>% 
        arrange(ALPAKAS_ID)   
      
      print(history_wide)
      
    } else {
      message("\nAll springs converged below ", tol, "%")
    }
    
    
    
    ## Compute shifted circular catchment approximations
    # create shifted circular catchment approximations for each spring
    message("\nCreating circular shifted catchment approximations based on recharge area estimate...")
    circle_approx_sf <- create_circular_approx(
      station_sf,
      topographic_catch_sf,
      shift_list = c(1),
      iter = i,
      out_dir=path_buff_approx_temp
    )
    
    # save the recharge area estimate from the current iteration
    station_df_old <- station_sf%>%
      st_drop_geometry()
    
    # Update workflow state
    recharge_mode <- "circle_approx" 
    i <- i + 1
    
  }
  
  # Return the final catchment approximations
  return(circle_approx_sf)
}



