# -*- coding: utf-8 -*-
"""
Preprocessing data from the ALPAKAS-dataset (https://doi.org/10.5281/zenodo.19329693 ) as input for the global LSTM.
The reference catchments (expert-based) can directly be extracted from the ALPAKAS dataset, the aggregated static and
 meteorological catchment aggregates can be derived according to the aggregation code provided along the ALPAKAS dataset.


"""


import pandas as pd
import os
import numpy as np
from pathlib import Path
from scipy.optimize import curve_fit
from sklearn.preprocessing import LabelEncoder
from functools import reduce
import geopandas as gpd

# ------------------------------------------------------------
# Filepaths
# ------------------------------------------------------------

base_dir = r"../../input_data/ALPAKAS/catchment_aggregates" # path to ALPAKAS dataset
base_dir_discharge = "../../input_data/ALPAKAS/"

static_variables = ["elev_mean", "elev_min", "elev_max", "slope_mean", "flat_area_perc", "steep_area_perc", "northn_mean", "eastn_mean", # topography (DEM)
                    "crop_perc", "grass_perc", "shrub_perc", "dwood_perc", "mix_wood_perc", "ewood_perc", "wetland_perc",
                    "loose_rock_perc","rock_perc","urban_perc", # Land Use (CORINE)
                    "climate_zone", # Beck et al Köppen Geiger
                    "porous_high_prod_perc","porous_low_mod_prod_perc","fissured_high_prod_perc",
                    "fissured_low_mod_prod_perc","local_aquif_perc","non_aquif_perc", #IHME
                    "station_lon","station_lat"]


alpakas_ids = pd.read_csv(    "../../input_data/project_files/alpakas_ids.csv", header=None)[0].tolist()

meta_data = pd.read_csv(os.path.join(base_dir_discharge,"ALPAKAS_station_meta.csv"), encoding="cp1252" )

meta_data = meta_data[meta_data["ALPAKAS_ID"].isin(alpakas_ids)]

datasplit_df = pd.read_csv("../../input_data/project_files/datasplit.csv")



# ------------------------------------------------------------
# Functions
# ------------------------------------------------------------
def create_area_df_dict(mode_list, basepath, crs_proj=3035):
    """
    Create a dictionary of area DataFrames for different modes.

    Parameters
    ----------
    mode_list : list of str
        List of modes (used to identify GeoJSON files)
    basepath : str
        Directory containing GeoJSON files
    crs_proj : int or str, optional
        Projected CRS for area calculation (default: EPSG:3035)

    Returns
    -------
    dict
        {mode: DataFrame with ['AlpAKaS_ID', 'area']}
    """

    area_df_dict = {}

    for mode in mode_list:

        # --- build file path ---
        filepath = os.path.join(basepath, f"catchment_{mode}/catchment_delineations/catchment_{mode}.geojson")

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        # --- read GeoJSON ---
        gdf = gpd.read_file(filepath)
        print(gdf.columns)
        # --- ensure ID column exists ---
        if "AlpAKaS_ID" not in gdf.columns:
            raise ValueError(f"'ALPAKAS_ID' missing in {filepath}")

        # --- project for correct area ---
        gdf = gdf.to_crs(crs_proj)

        # --- compute area ---
        gdf["area"] = gdf.geometry.area

        # --- store minimal DataFrame ---
        area_df = gdf[["AlpAKaS_ID", "area"]].copy()

        # --- ensure consistent dtype ---
        area_df["AlpAKaS_ID"] = area_df["AlpAKaS_ID"].astype(str)
        area_df = area_df.rename(columns={"AlpAKaS_ID": "ALPAKAS_ID"})

        area_df_dict[mode] = area_df

    return area_df_dict

def build_static_attributes(base_dir, mode):
    """
    Reads all CSVs in {base_dir}/{mode}/static_attributes,
    merges them on ALPAKAS_ID, saves attr_all.csv, and returns final DF.
    """

    folder = os.path.join(base_dir, f"catchment_{mode}", "static_attributes")

    if not os.path.exists(folder):
        raise FileNotFoundError(f"Folder not found: {folder}")

    files = [
        f for f in os.listdir(folder)
        if f.endswith(".csv")
    ]

    if len(files) == 0:
        raise ValueError(f"No CSV files found in {folder}")

    dfs = []

    for f in files:
        path = os.path.join(folder, f)
        df = pd.read_csv(path)

        if "ALPAKAS_ID" not in df.columns:
            raise ValueError(f"'ALPAKAS_ID' missing in {path}")

        df["ALPAKAS_ID"] = df["ALPAKAS_ID"].astype(str)
        dfs.append(df)

    # merge all on ALPAKAS_ID
    merged_df = reduce(
        lambda left, right: pd.merge(left, right, on="ALPAKAS_ID", how="outer"),
        dfs
    )

    # save result

    return merged_df



def compute_Tsin(
    df,
    start_test_date,
    date_col="date",
    temp_col="mean_temperature"
):
    """
    Fit a sinusoidal temperature model on training data and
    generate values for the full dataframe timeline.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing a date column and temperature column.
    start_test_date : str or datetime
        Cutoff date separating training and test data.
    date_col : str
        Name of the column containing dates.
    temp_col : str
        Name of the column containing mean temperature values.

    Returns
    -------
    tsin_full : pd.Series
        Sinusoidal temperature values for the entire dataframe.
    popt : np.ndarray
        Optimized sinusoidal parameters.
    """

    df = df.copy()

    # Create continuous time index (in days)
    x_full = np.arange(len(df))

    # Training mask
    cutoff = pd.to_datetime(start_test_date)
    train_mask = df.index < cutoff

    if not train_mask.any():
        print(df)
        print(df.index.min())
        raise ValueError("No training data before start_test_date.")

    # Training data
    y_train = df.loc[train_mask, temp_col].astype(float).values
    x_train = x_full[train_mask]

    # Sinusoidal model
    def sinusoidal_function(x, a, b, c, d):
        return a * np.sin(b * x + c) + d

    # Initial parameter estimates
    mean_temp = np.nanmean(y_train)
    temp_range = np.nanmax(y_train) - np.nanmin(y_train)

    initial_guesses = [
        temp_range / 2,           # amplitude
        2 * np.pi / 365.25,       # annual seasonality (daily data)
        0.0,                      # phase
        mean_temp                 # offset
    ]

    # Fit on training data only
    popt, _ = curve_fit(
        sinusoidal_function,
        x_train,
        y_train,
        p0=initial_guesses,
        maxfev=10_000
    )

    # Evaluate on full timeline
    tsin_full = sinusoidal_function(x_full, *popt)

    tsin_full = pd.Series(
        tsin_full,
        index=df.index,
        name="Tsin"
    )
    print(popt)
    return tsin_full



# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

modes_list = ["expert", "approx"] #
dynamic_variables = [
    "date",
    "precipitation_nat",
    "temperature_mean_nat",
    "temperature_min_nat",
    "temperature_max_nat",
]

static_output_dir = Path("../preprocessed_data/static")
dynamic_output_dir = Path("../preprocessed_data/dynamic")

static_output_dir.mkdir(parents=True, exist_ok=True)
dynamic_output_dir.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Prepare catchment-area data
# ------------------------------------------------------------

area_df_dict = create_area_df_dict(modes_list, base_dir, crs_proj=3035)


# ------------------------------------------------------------
# Process each catchment representation
# ------------------------------------------------------------

for mode in modes_list:
    print(f"\nMode: {mode}")

    # ========================================================
    # Static attributes
    # ========================================================

    static_vars_df = build_static_attributes(base_dir, mode)

    static_vars_df = static_vars_df.loc[
        static_vars_df["ALPAKAS_ID"].isin(alpakas_ids)
    ].copy()

    # Add station coordinates
    static_vars_df = static_vars_df.merge(
        meta_data[["ALPAKAS_ID", "station_lon", "station_lat"]],
        on="ALPAKAS_ID",
        how="left"
    )

    existing_stations = set(static_vars_df["ALPAKAS_ID"])
    print(f"Number of available stations: {len(existing_stations)}")

    # Select static variables
    static_vars_df = static_vars_df[static_variables + ["ALPAKAS_ID"]].copy()

    # Aggregate land-cover variables
    static_vars_df["rock_total_perc"] = static_vars_df["rock_perc"] + static_vars_df["loose_rock_perc"]
    static_vars_df["wood_total_perc"] = (
        static_vars_df["ewood_perc"] + static_vars_df["mix_wood_perc"] + static_vars_df["dwood_perc"]
    )
    static_vars_df["shrub_and_grass_perc"] = static_vars_df["shrub_perc"] + static_vars_df["grass_perc"]

    cols_to_drop = [
        "rock_perc", "loose_rock_perc",
        "ewood_perc", "mix_wood_perc", "dwood_perc",
        "shrub_perc", "grass_perc"
    ]
    static_vars_df = static_vars_df.drop(columns=cols_to_drop)

    # Put ID first
    ordered_cols = ["ALPAKAS_ID"] + [c for c in static_vars_df.columns if c != "ALPAKAS_ID"]
    static_vars_df = static_vars_df[ordered_cols]

    # Encode climate zone
    label_encoder = LabelEncoder()
    static_vars_df["climate_zone"] = label_encoder.fit_transform(
        static_vars_df["climate_zone"].astype(str)
    )

    # Add catchment-area attributes
    static_vars_df = static_vars_df.merge(
        area_df_dict[mode],
        on="ALPAKAS_ID",
        how="left"
    )

    static_vars_df.to_csv(
        static_output_dir / f"{mode}_stat_attrs.csv",
        index=False
    )

    # ========================================================
    # Random static attributes
    # ========================================================

    random_static_vars_df = static_vars_df.copy()
    cols_to_randomize = [c for c in random_static_vars_df.columns if c != "ALPAKAS_ID"]

    rng = np.random.default_rng(42)

    random_static_vars_df[cols_to_randomize] = rng.uniform(
        -1, 1,
        size=(len(random_static_vars_df), len(cols_to_randomize))
    )

    random_static_vars_df.to_csv(
        static_output_dir / f"{mode}_random_attrs.csv",
        index=False
    )

    # ========================================================
    # Dynamic time series
    # ========================================================

    for station in sorted(existing_stations):
        print(f"Processing station: {station}")

        # Test start date
        split_match = datasplit_df.loc[
            datasplit_df["ALPAKAS_ID"] == station,
            "testing_start"
        ]

        if split_match.empty:
            print(f"No testing start date for {station}; skipping.")
            continue

        test_start_date = pd.to_datetime(split_match.iloc[0])

        # ----------------------------------------------------
        # Meteorological data
        # ----------------------------------------------------

        meteo_path = (
            Path(base_dir)
            / f"catchment_{mode}"
            / "meteorological_time_series"
            / "daily"
            / f"AlpAKaS_meteo_daily_{station}.csv"
        )

        if not meteo_path.exists():
            print(f"No meteorological file for {station}; skipping.")
            continue

        station_df = pd.read_csv(meteo_path, usecols=dynamic_variables)

        if station_df.empty:
            print(f"Meteorological file for {station} is empty; skipping.")
            continue

        station_df["date"] = pd.to_datetime(station_df["date"])

        temp_mean_col = "temperature_mean_nat"
        temp_min_col = "temperature_min_nat"
        temp_max_col = "temperature_max_nat"

        # Fill mean temperature from min/max if necessary
        if temp_mean_col not in station_df.columns or station_df[temp_mean_col].isna().all():
            station_df[temp_mean_col] = (
                station_df[temp_max_col] + station_df[temp_min_col]
            ) / 2

        # ----------------------------------------------------
        # Discharge data
        # ----------------------------------------------------

        discharge_path = (
            Path(base_dir_discharge)
            / "discharge_time_series"
            / "daily"
            / f"AlpAKaS_discharge_daily_{station}.csv"
        )

        if not discharge_path.exists():
            print(f"No discharge file for {station}; skipping.")
            continue

        discharge_df = pd.read_csv(
            discharge_path,
            index_col="date",
            parse_dates=["date"],
            usecols=[
                "date", "discharge", "qc_flag", "dq_issue",
                "cpd_segment_id", "temp_res_dom", "cpd_segment_main"
            ],
            encoding="latin1"
        )

        # ----------------------------------------------------
        # Discharge quality control
        # ----------------------------------------------------

        discharge_df = discharge_df.loc[discharge_df["qc_flag"] == False].copy()

        if discharge_df["cpd_segment_id"].notna().any():
            print(f"{station}: multiple breakpoint segments present.")

        # Keep main breakpoint segment or periods without detected breakpoints
        discharge_df = discharge_df.loc[
            (discharge_df["cpd_segment_main"] == True)
            | discharge_df["cpd_segment_id"].isna()
        ]

        # Remove periods with known data-quality issues
        discharge_df = discharge_df.loc[discharge_df["dq_issue"].isna()]

        if (discharge_df["temp_res_dom"] != "daily").any():
            print(f"{station}: non-daily discharge records found.")

        # Keep daily records only
        discharge_df = discharge_df.loc[
            discharge_df["temp_res_dom"] == "daily"
        ].copy()

        if discharge_df.empty:
            print(f"No valid discharge data for {station}; skipping.")
            continue

        # ----------------------------------------------------
        # Create continuous daily discharge series
        # ----------------------------------------------------

        discharge_df = discharge_df.sort_index()
        discharge_df = discharge_df.loc[
            ~discharge_df.index.duplicated(keep="first")
        ]

        all_dates = pd.date_range(
            start=discharge_df.index.min(),
            end=discharge_df.index.max(),
            freq="D"
        )

        discharge_df = discharge_df.reindex(all_dates)
        discharge_df.index.name = "date"

        discharge_df = discharge_df.rename(columns={"discharge": "QobsS"})
        discharge_df["QobsS"] = pd.to_numeric(
            discharge_df["QobsS"],
            errors="coerce"
        ) / 1000

        discharge_df = discharge_df[["QobsS"]].reset_index()

        # ----------------------------------------------------
        # Merge meteorology and discharge
        # ----------------------------------------------------

        station_df = station_df.merge(
            discharge_df,
            on="date",
            how="left"
        )

        station_df = station_df.loc[
            :,
            ~station_df.columns.str.contains("Unnamed")
        ]

        # ----------------------------------------------------
        # Remove trailing data after last valid period
        # ----------------------------------------------------

        last_valid_index = station_df.dropna().last_valid_index()

        if last_valid_index is None:
            print(f"No complete observations for {station}; skipping.")
            continue

        station_df = station_df.loc[:last_valid_index].copy()

        # ----------------------------------------------------
        # Keep up to six years of meteorological history before
        # the first available discharge observation
        # ----------------------------------------------------

        first_q_idx = station_df["QobsS"].first_valid_index()

        if first_q_idx is not None:
            first_q_date = station_df.loc[first_q_idx, "date"]
            cutoff_date = first_q_date - pd.DateOffset(years=6)
            station_df = station_df.loc[station_df["date"] >= cutoff_date].copy()

        # Remove leading rows without mean-temperature input
        first_temp_idx = station_df[temp_mean_col].first_valid_index()

        if first_temp_idx is not None:
            station_df = station_df.loc[first_temp_idx:].copy()

        # ----------------------------------------------------
        # Final preparation
        # ----------------------------------------------------

        station_df["date"] = pd.to_datetime(station_df["date"])
        station_df = station_df.set_index("date").sort_index()

        station_df["Tsin"] = compute_Tsin(
            station_df,
            test_start_date,
            "date",
            temp_mean_col
        )

        station_df["ALPAKAS_ID"] = station

        # Save
        station_df.to_csv(
            dynamic_output_dir / f"{mode}_{station}_dynamic.csv"
        )