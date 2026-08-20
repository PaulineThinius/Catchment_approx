# -*- coding: utf-8 -*-
"""
@author: Pauline Thinius
"""

#%% paths and packages

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

import copy


#--- Functions ---
def get_percentiles(df, value_col):
    """
    Return:
    - 10th percentile (cal + val)
    - 90th percentile (cal + val)
    - Mean (cal only)
    - Interquartile range (Q75 - Q25) over entire timespan

    Parameters
    ----------
    df : pd.DataFrame (DatetimeIndex expected)
    value_col : str

    Returns
    -------
    q10, q90, q_mean_cal, iqr_all : float
    """


    if value_col not in df.columns:
        raise KeyError(f"Column '{value_col}' not found in DataFrame")

    # ---- Percentiles on all ----
    df_all = df.loc[df["split"].isin(["train", "stop","test"]), value_col].dropna()

    # ---- Mean on train/stop only ----
    df_train = df.loc[df["split"].isin(["train", "stop"]), value_col].dropna()


    # 10th and 90th percentiles
    q10 = df_all.quantile(0.10)
    q90 = df_all.quantile(0.90)

    # Mean (train and stop)
    q_mean_train = df_train.mean() if not df_train.empty else pd.NA

    # IQR (entire timespan)
    if df_all.empty:
        iqr_all = pd.NA
    else:
        q25 = df_all.quantile(0.25)
        q75 = df_all.quantile(0.75)
        iqr_all = q75 - q25

    return q10, q90, q_mean_train, iqr_all

def compute_metrics(obs, sim, q_mean, iqr):
    """
    Compute performance metrics between observed and simulated values.

    Metrics
    -------
    NSE
        Nash-Sutcliffe Efficiency.
    NSE_ref
        NSE using q_mean as the reference value.
    Bias
        Mean simulation error.
    rBias
        Bias normalized by the mean observed value.
    nBias
        Bias normalized by the interquartile range.
    RMSE
        Root Mean Squared Error.
    rRMSE
        RMSE normalized by the mean observed value.
    nRMSE
        RMSE normalized by the interquartile range.
    sMAPE
        Symmetric Mean Absolute Percentage Error [%].
    KGE
        Original Kling-Gupta Efficiency (Gupta et al., 2009).
    KGE_prime
        Modified Kling-Gupta Efficiency, KGE' (Kling et al., 2012).
    KGEnp
        Non-parametric KGE following Pool et al. (2018).
    VE
        Volumetric Efficiency (Criss and Winston, 2008).

    Parameters
    ----------
    obs : array-like
        Observed values.
    sim : array-like
        Simulated values.
    q_mean : float
        Reference mean used for NSE_ref.
    iqr : float
        Interquartile range used to normalize Bias and RMSE.

    Returns
    -------
    dict
        Dictionary containing all calculated metrics.
    """

    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)

    # ------------------------------------------------------------
    # Remove pairs containing NaN or infinite values
    # ------------------------------------------------------------
    mask = (
        np.isfinite(obs) &
        np.isfinite(sim)
    )

    obs = obs[mask]
    sim = sim[mask]

    metric_names = [
        "NSE", "NSE_ref",
        "Bias", "rBias", "nBias",
        "RMSE", "rRMSE", "nRMSE",
        "sMAPE",
        "KGE", "KGE_prime",
        "alpha", "beta", "gamma",
        "KGEnp", "alpha_np", "beta_np", "r_np",
        "VE"
    ]

    # Not enough valid observations
    if len(obs) < 2:
        return {name: np.nan for name in metric_names}

    mean_obs = np.mean(obs)
    mean_sim = np.mean(sim)

    std_obs = np.std(obs)
    std_sim = np.std(sim)

    # ------------------------------------------------------------
    # Bias
    # ------------------------------------------------------------
    bias = np.mean(sim - obs)

    rbias = (
        bias / mean_obs
        if mean_obs != 0
        else np.nan
    )

    nbias = (
        bias / iqr
        if iqr != 0
        else np.nan
    )

    # ------------------------------------------------------------
    # RMSE
    # ------------------------------------------------------------
    rmse = np.sqrt(
        np.mean((sim - obs) ** 2)
    )

    rrmse = (
        rmse / mean_obs
        if mean_obs != 0
        else np.nan
    )

    nrmse = (
        rmse / iqr
        if iqr != 0
        else np.nan
    )

    # ------------------------------------------------------------
    # sMAPE
    # ------------------------------------------------------------
    denominator = np.abs(obs) + np.abs(sim)

    valid_smape = denominator != 0

    if np.any(valid_smape):
        smape = (
            np.mean(
                2
                * np.abs(sim[valid_smape] - obs[valid_smape])
                / denominator[valid_smape]
            )
            * 100
        )
    else:
        smape = np.nan

    # ------------------------------------------------------------
    # NSE
    # Nash and Sutcliffe (1970)
    # DOI: 10.1016/0022-1694(70)90255-6
    # ------------------------------------------------------------
    nse_denominator = np.sum(
        (obs - mean_obs) ** 2
    )

    nse = (
        1 - np.sum((sim - obs) ** 2) / nse_denominator
        if nse_denominator != 0
        else np.nan
    )

    nse_ref_denominator = np.sum(
        (obs - q_mean) ** 2
    )

    nse_ref = (
        1 - np.sum((sim - obs) ** 2) / nse_ref_denominator
        if nse_ref_denominator != 0
        else np.nan
    )

    # ------------------------------------------------------------
    # Classical KGE
    # Gupta et al. (2009)
    # DOI: 10.1016/j.jhydrol.2009.08.003
    #
    # KGE = 1 - sqrt(
    #     (r - 1)^2 +
    #     (alpha - 1)^2 +
    #     (beta - 1)^2
    # )
    # ------------------------------------------------------------
    if std_obs != 0 and mean_obs != 0:
        r = np.corrcoef(obs, sim)[0, 1]
        alpha = std_sim / std_obs
        beta = mean_sim / mean_obs

        kge = 1 - np.sqrt(
            (r - 1) ** 2
            + (alpha - 1) ** 2
            + (beta - 1) ** 2
        )
    else:
        r = np.nan
        alpha = np.nan
        beta = np.nan
        kge = np.nan

    # ------------------------------------------------------------
    # Modified KGE (KGE')
    # Kling et al. (2012)
    # DOI: 10.1016/j.jhydrol.2012.01.011
    #
    # gamma = CV_sim / CV_obs
    # ------------------------------------------------------------
    if (
        mean_obs != 0
        and mean_sim != 0
        and std_obs != 0
    ):
        gamma = (
            (std_sim / mean_sim)
            / (std_obs / mean_obs)
        )

        kge_prime = 1 - np.sqrt(
            (r - 1) ** 2
            + (gamma - 1) ** 2
            + (beta - 1) ** 2
        )
    else:
        gamma = np.nan
        kge_prime = np.nan

    # ------------------------------------------------------------
    # Non-parametric KGE (KGEnp)
    # Pool et al. (2018)
    # DOI: 10.1080/02626667.2018.1552002
    # ------------------------------------------------------------
    if (
        len(obs) > 1
        and not np.all(sim == sim[0])
        and not np.all(obs == obs[0])
    ):
        r_np, _ = spearmanr(sim, obs)
    else:
        r_np = np.nan

    if mean_obs != 0 and mean_sim != 0:
        fdc_sim = np.sort(
            sim / (mean_sim * len(sim))
        )

        fdc_obs = np.sort(
            obs / (mean_obs * len(obs))
        )

        alpha_np = (
            1
            - 0.5
            * np.sum(
                np.abs(fdc_sim - fdc_obs)
            )
        )

        beta_np = mean_sim / mean_obs

        if np.isfinite(r_np):
            kgenp = 1 - np.sqrt(
                (alpha_np - 1) ** 2
                + (beta_np - 1) ** 2
                + (r_np - 1) ** 2
            )
        else:
            kgenp = np.nan

    else:
        alpha_np = np.nan
        beta_np = np.nan
        kgenp = np.nan

    # ------------------------------------------------------------
    # Volumetric Efficiency
    # Criss and Winston (2008)
    # DOI: 10.1002/hyp.7072
    # ------------------------------------------------------------
    ve_denominator = np.sum(
        np.abs(obs)
    )

    VE = (
        1
        - np.sum(np.abs(obs - sim))
        / ve_denominator
        if ve_denominator != 0
        else np.nan
    )

    # ------------------------------------------------------------
    # Return metrics
    # ------------------------------------------------------------
    return {
        "NSE": nse,
        "NSE_ref": nse_ref,

        "Bias": bias,
        "rBias": rbias,
        "nBias": nbias,

        "RMSE": rmse,
        "rRMSE": rrmse,
        "nRMSE": nrmse,

        "sMAPE": smape,

        "KGE": kge,
        "KGE_prime": kge_prime,
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,

        "KGEnp": kgenp,
        "alpha_np": alpha_np,
        "beta_np": beta_np,
        "r_np": r_np,

        "VE": VE
    }

def evaluate_single_id(
    alpakas_id,
    model_type,
    catch_rep,
    df,
    p10,
    p90,
    q_mean,
    iqr,
    static_type=None
):
    """
    Evaluate one ID over the test period and selected flow/seasonal subsets.
    """

    df = df.copy()

    # ------------------------------------------------------------
    # Restrict evaluation to test period
    # ------------------------------------------------------------
    df = df.loc[df["split"] == "test"].copy()

    # ------------------------------------------------------------
    # Define evaluation subsets
    # ------------------------------------------------------------
    subsets = {
        "testing": df,

        # Flow subsets
        "Q10": df[df["obs"] <= p10], # low flow
        "Q90": df[df["obs"] >= p90], # high flow

        # Snow-related period
        "snow": df[df.index.month.isin([4, 5, 6])],

        # Seasons
        "spring": df[df.index.month.isin([3, 4, 5])],
        "summer": df[df.index.month.isin([6, 7, 8])],
        "autumn": df[df.index.month.isin([9, 10, 11])],
        "winter": df[df.index.month.isin([12, 1, 2])],

        # Individual months
        "Jan": df[df.index.month == 1],
        "Feb": df[df.index.month == 2],
        "Mar": df[df.index.month == 3],
        "Apr": df[df.index.month == 4],
        "May": df[df.index.month == 5],
        "Jun": df[df.index.month == 6],
        "Jul": df[df.index.month == 7],
        "Aug": df[df.index.month == 8],
        "Sep": df[df.index.month == 9],
        "Oct": df[df.index.month == 10],
        "Nov": df[df.index.month == 11],
        "Dec": df[df.index.month == 12],
    }

    results = []

    # ------------------------------------------------------------
    # Compute metrics for each subset
    # ------------------------------------------------------------
    for period_name, df_sub in subsets.items():

        if df_sub.empty:
            metrics = {
                name: np.nan
                for name in [
                    "NSE", "NSE_ref",
                    "Bias", "rBias", "nBias",
                    "RMSE", "rRMSE", "nRMSE",
                    "sMAPE",
                    "KGE", "KGE_prime",
                    "alpha", "beta", "gamma",
                    "KGEnp", "alpha_np", "beta_np", "r_np",
                    "VE"
                ]
            }

        else:
            metrics = compute_metrics(
                obs=df_sub["obs"],
                sim=df_sub["sim"],
                q_mean=q_mean,
                iqr=iqr
            )

        results.append({
            "alpakas_id": alpakas_id,
            "model_type": model_type,
            "catchment_representation": catch_rep,
            "static_type": static_type,
            "phase": period_name,
            **metrics
        })

    return pd.DataFrame(results)

def compute_climatological_parde(df, alpakas_id, obs_col="Obs"):
    """
    Compute long-term climatological Pardé coefficients and derived features.

    Assumes the DataFrame has the datetime index.

    Returns a single row DataFrame with P1..P12, Pmax, Pmin, IS, months,
    sin/cos encoding, and alpakas_id.
    """

    df = df.copy()


    # Ensure index is datetime
    if not np.issubdtype(df.index.dtype, np.datetime64):
        df.index = pd.to_datetime(df.index)

    # Filter for hydrological years with >= 80% coverage per year
    df["hydro_year"] = df.index.year
    df.loc[df.index.month >= 10, "hydro_year"] += 1

    # Only keep years with >= 80% daily data
    valid_years = df.groupby("hydro_year")["obs"].count()
    valid_years = valid_years[valid_years >= 0.8 * 365].index
    df = df[df["hydro_year"].isin(valid_years)]

    mask = df["obs"].isna() | (df["obs"] == "")

    df.loc[mask, obs_col] = np.nan



    # Compute long-term monthly mean (across all years)
    monthly_mean = df[obs_col].resample("ME").mean()


    climatology =  monthly_mean.groupby(monthly_mean.index.month).mean()

    Q_annual = df[obs_col].mean()

    P = climatology / Q_annual

    # Pmax, Pmin
    Pmax = P.max()
    Pmin = P.min()
    Pmax_month = P.idxmax()
    Pmin_month = P.idxmin()

    # Integrated seasonal deviation (IS)
    IS = (np.abs(P - 1).sum() / 12) * 100  # %

    # Sin/cos encoding for circular months
    def month_to_sincos(m):
        theta = 2 * np.pi * (m - 1) / 12
        return np.sin(theta), np.cos(theta)

    Pmax_sin, Pmax_cos = month_to_sincos(Pmax_month)
    Pmin_sin, Pmin_cos = month_to_sincos(Pmin_month)

    row = {
        "Pmax": Pmax,
        "IS": IS,
        "Pmax_sin": Pmax_sin,
        "Pmax_cos": Pmax_cos,
        "alpakas_id": alpakas_id
    }

    # Add P1..P12
    for m in range(1, 13):
        row[f"P{m}"] = P.get(m, np.nan)

    features_df = pd.DataFrame([row])

    return features_df

def cluster_parde_springs(all_parde_df, features, n_clusters=4, approaches=["kmeans", "hierarchical"],
                          save_dir="../plots", save_csv="../output_data/parde_coefficent_cluster.csv"):
    """
    Cluster springs based on Pardé features using KMeans and/or Hierarchical clustering.

    Parameters
    ----------
    all_parde_df : pd.DataFrame
        DataFrame with Pardé features (one row per alpakas_id).
    features : list
        List of column names to use for clustering.
    n_clusters : int
        Number of clusters to form.
    approaches : list of str
        Clustering methods to apply: "kmeans" and/or "hierarchical".
    save_dir : str
        Directory to save cluster plots.
    save_csv : str
        Path to save final DataFrame with cluster labels.
    """

    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.dirname(save_csv), exist_ok=True)

    # Extract features and scale
    X = all_parde_df[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    results_df = all_parde_df.copy()
    print(results_df)

    if "kmeans" in [a.lower() for a in approaches]:

        # --------------------------------------------------
        # Find optimal number of clusters (k = 3–5)
        # --------------------------------------------------
        sil_scores = {}

        for k in range(3, 6):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
            labels = kmeans.fit_predict(X_scaled)

            score = silhouette_score(X_scaled, labels)
            sil_scores[k] = score

            print(f"k = {k} / {len(X_scaled)} → silhouette = {score:.4f}")

        best_k = max(sil_scores, key=sil_scores.get)
        print(f"\nSelected best k = {best_k}")

        # --------------------------------------------------
        # Final KMeans with best k
        # --------------------------------------------------
        kmeans = KMeans(n_clusters=best_k, random_state=42, n_init="auto")
        labels = kmeans.fit_predict(X_scaled)
        results_df["cluster_kmeans"] = labels

        # --------------------------------------------------
        # Plot Pardé curves per cluster
        # --------------------------------------------------
        clusters = sorted(results_df["cluster_kmeans"].unique())

        fig, axes = plt.subplots(
            1, best_k,
            figsize=(4 * best_k, 4),
            sharey=True
        )

        axes = np.atleast_1d(axes)

        for i, c in enumerate(clusters):
            ax = axes[i]
            subset = results_df[results_df["cluster_kmeans"] == c]

            ax.plot(
                range(1, 13),
                subset[[f"P{m}" for m in range(1, 13)]].T,
                alpha=0.5
            )

            ax.set_title(f"KMeans Cluster {c}")
            ax.set_xlabel("Month")

            if i == 0:
                ax.set_ylabel("Pardé coefficient")

            ax.set_xticks(range(1, 13))

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "cluster_aclag_kmeans.png"))
        plt.close()

    if "hierarchical" in [a.lower() for a in approaches]:
        # Hierarchical clustering
        Z = linkage(X_scaled, method="ward")
        print(Z)
        # Plot dendrogram
        plt.figure(figsize=(12, 6))
        dendrogram(Z, labels=results_df["alpakas_id"].values, leaf_rotation=90)
        plt.title("Hierarchical Clustering Dendrogram")
        plt.xlabel("ALPAKAS ID")
        plt.ylabel("Distance")
        plt.tight_layout()
        plt.show()


        # Cut tree to get cluster labels
        labels_hier = fcluster(Z, n_clusters, criterion="maxclust") - 1  # zero-based
        results_df["cluster_hierarchical"] = labels_hier

        # Plot Pardé curves per hierarchical cluster
        clusters = sorted(results_df["cluster_hierarchical"].unique())
        fig, axes = plt.subplots(1, n_clusters, figsize=(4 * n_clusters, 4), sharey=True)
        for i, c in enumerate(clusters):
            ax = axes[i] if n_clusters > 1 else axes
            subset = results_df[results_df["cluster_hierarchical"] == c]
            ax.plot(range(1, 13), subset[[f"P{m}" for m in range(1, 13)]].T, alpha=0.5)
            ax.set_title(f"Hier Cluster {c}")
            ax.set_xlabel("Month")
            if i == 0:
                ax.set_ylabel("Pardé coefficient")
            ax.set_xticks(range(1, 13))
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "cluster_hierarchical.png"))
        print("absolute path:", os.path.abspath(save_dir))
        plt.close()

    # Save the DataFrame with cluster labels
    results_df.to_csv(save_csv, index=False)

    print(f"Clustering completed. Results saved to {save_csv}")

    return



# ------------------------------------------------------------
# Evaluate all IDs and model configurations
# ------------------------------------------------------------

alpakas_ids = pd.read_csv('../../input_data/project_files/alpakas_ids.csv', header=None)[0].tolist()
catch_representations = [ "expert","point", "approx"]
static_feature_types = ["env", "rand","timeseries"]

all_scores = []
all_parde_coefficients = []

model_type ="LSTM"
for alpakas_id in alpakas_ids:

    # --------------------------------------------------------
    # Load one reference dataset for ID-specific quantities
    # that do not depend on model configuration
    # --------------------------------------------------------
    reference_path = (
        f"../output_data/{static_feature_types[0]}/"
        f"{catch_representations[0]}/"
        f"{alpakas_id}_median_all.csv"
    )

    df_ref = pd.read_csv(
        reference_path,
        sep=";",
        parse_dates=["date"],
        index_col="date"
    )

    # Quantities that only need to be calculated once per ID
    p10, p90, q_mean, iqr = get_percentiles(
        df_ref,
        "obs"
    )

    parde_coefficients = compute_climatological_parde(
        df_ref,
        alpakas_id,
        obs_col="obs"
    )

    all_parde_coefficients.append(parde_coefficients)

    # --------------------------------------------------------
    # Evaluate every model configuration for this ID
    # --------------------------------------------------------
    for catch_rep in catch_representations:
        for static_type in static_feature_types:

            file_path = (
                f"../output_data/{static_type}/"
                f"{catch_rep}/"
                f"{alpakas_id}_median_all.csv"
            )

            if not os.path.exists(file_path):
                continue

            df = pd.read_csv(
                file_path,
                sep=";",
                parse_dates=["date"],
                index_col="date"
            )

            single_scores_df = evaluate_single_id(
                alpakas_id=alpakas_id,
                model_type=model_type,
                catch_rep=catch_rep,
                df=df,
                p10=p10,
                p90=p90,
                q_mean=q_mean,
                iqr=iqr,
                static_type=static_type
            )

            all_scores.append(single_scores_df)


# ------------------------------------------------------------
# Combine model evaluation scores
# ------------------------------------------------------------
all_scores_df = pd.concat(
    all_scores,
    ignore_index=True
)

all_scores_df.to_csv("../output_data/all_scores_median.csv")

# ------------------------------------------------------------
# Combine Pardé coefficients
# ------------------------------------------------------------
all_parde_df = pd.concat(
    all_parde_coefficients,
    ignore_index=True
)


# ------------------------------------------------------------
# Cluster IDs according to flow seasonality
# ------------------------------------------------------------
parde_cols = (
    ["Pmax", "IS", "Pmax_sin", "Pmax_cos"]
    + [f"P{m}" for m in range(1, 13)]
)

cluster_parde_springs(
    all_parde_df,
    parde_cols,
    n_clusters=4,
    approaches=["hierarchical"]
)