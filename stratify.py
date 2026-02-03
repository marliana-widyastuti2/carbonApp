### open libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from scipy.ndimage import generic_filter
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import math
import random
from shapely.geometry import Point
import seaborn as sns
from matplotlib.colors import rgb2hex
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from skimage.util import view_as_windows
from scipy.spatial import distance_matrix
from scipy.spatial import cKDTree
import geopandas as gpd

random.seed(42)
warnings.filterwarnings("ignore")

"""
This module contains functions for stratifying datasets based on its spatial distribution.
"""
def open_CSV(filename):
    dataset = pd.read_csv(filename)
    col = ['X', 'Y', 'Val', 'Var']
    dataset = dataset.set_axis(col, axis=1) 
    dataset['id'] = dataset.index                                              
    return dataset

def estimate_pixel_area(df):
    # Get unique sorted X and Y
    x_unique = sorted(df['X'].unique())
    y_unique = sorted(df['Y'].unique())

    # Estimate resolution
    dx = min([x2 - x1 for x1, x2 in zip(x_unique, x_unique[1:])])
    dy = min([y2 - y1 for y1, y2 in zip(y_unique, y_unique[1:])])

    pixel_area = dx * dy  # in square meters (if coordinates are in meters)
    total_area = len(df) * pixel_area / 10000
    return total_area, pixel_area

## Plotting function
def plot_continuous_data_fig(dataset, column_to_plot, plot_title="Value", figsize=(15, 8),
                             gdf=None, raster_crs=None):
    grid = dataset.pivot(index="Y", columns="X", values=column_to_plot).sort_index(ascending=False)
    raster = grid.to_numpy(dtype=float)

    x_coords = grid.columns.values
    y_coords = grid.index.values

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        raster, cmap="viridis", origin="upper",
        extent=[x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()],
        aspect="equal"  # important
    )
    fig.colorbar(im, ax=ax)

    # overlay shapefile if provided
    if gdf is not None:
        gdf_plot = gdf.copy()
        if raster_crs is not None and gdf_plot.crs != raster_crs:
            gdf_plot = gdf_plot.to_crs(raster_crs)

        gdf_plot.boundary.plot(ax=ax, edgecolor="red", linewidth=2)

    ax.set_title(plot_title)
    fig.tight_layout()
    return fig


def plot_stratum_grid_fig(
    dataset,
    column_of_strata,
    sampling_points=None,
    figsize=(15, 8),
    plot_title="Stratification",
    gdf=None,                 # <-- NEW
    raster_crs=None,          # <-- NEW
):
    grid = dataset.pivot(index="Y", columns="X", values=column_of_strata).sort_index(ascending=False)
    raster = grid.to_numpy(dtype=float)

    x_coords = grid.columns.values
    y_coords = grid.index.values

    stratum_labels = sorted(dataset[column_of_strata].dropna().unique())
    stratum_to_color = {val: plt.get_cmap("tab10")(i % 10) for i, val in enumerate(stratum_labels)}

    colors = [stratum_to_color[val] for val in stratum_labels]
    cmap = mcolors.ListedColormap(colors)
    bounds = [val - 0.5 for val in stratum_labels] + [stratum_labels[-1] + 0.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(
        raster, cmap=cmap, norm=norm, origin="upper",
        extent=[x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()],
        aspect="equal"  # <-- match continuous plot
    )

    # overlay shapefile boundary (same as continuous plot)
    if gdf is not None:
        gdf_plot = gdf.copy()
        if raster_crs is not None and gdf_plot.crs != raster_crs:
            gdf_plot = gdf_plot.to_crs(raster_crs)
        gdf_plot.boundary.plot(ax=ax, edgecolor="red", linewidth=2, zorder=6)

    # sample points
    if sampling_points is not None and len(sampling_points) > 0:
        ax.scatter(
            sampling_points["X"], sampling_points["Y"],
            color="black", s=10, label="Sample Points", zorder=7
        )

    # legend for strata (and optional sample points)
    handles = [mpatches.Patch(color=stratum_to_color[val], label=f"Stratum {val}")
               for val in stratum_labels]
    if sampling_points is not None and len(sampling_points) > 0:
        handles.append(mpatches.Patch(color="black", label="Sample Points"))

    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5), borderaxespad=0.)
    ax.set_title(plot_title)
    fig.tight_layout()
    return fig

def overall_mean_variance(dataset):
    N = len(dataset)
    values = dataset['Val']
    overall_mean = np.mean(values)

    # Calculate the overall variance
    variance_within = np.sum(dataset['Var'])  # Sum of individual variances
    variance_between = np.sum((values - overall_mean) ** 2)  # Variance due to mean difference
    overall_variance = (variance_within + variance_between) / N

    return overall_mean, overall_variance

def cumrootfreq(data, n_strata=5):
    n_bins = 500  # number of bins of the histogram of the data.

    bin_edges = np.linspace(min(data), max(data), n_bins + 1)
    frequencies, _ = np.histogram(data, bins=bin_edges)
    bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2

    root = np.sqrt(frequencies)
    cumulative_root = np.cumsum(root)

    cumulative_final = cumulative_root.max()
    temp = pd.DataFrame({'cs' : bin_midpoints, 'cr' : cumulative_root})
    min_data = [data.min()]
    max_data = [data.max()]

    ratio = cumulative_final/n_strata
    bou = [cumulative_root.min()]
    n = 1
    while n < n_strata:
        mBound = ratio*n
        bou.append(mBound)
        n=n+1
    bou.append(cumulative_final)

    temp['Bins'] = pd.cut(temp['cr'], bins=bou, include_lowest=True)
    csBou = temp.groupby('Bins').max()['cs'].to_list()
    final_boundary = min_data + csBou[:-1] + max_data

    stratum = pd.DataFrame(data)
    stratum['bins'] = pd.cut(data, bins=final_boundary,  include_lowest=True)
    stratum = stratum.groupby('bins').ngroup()
    return stratum, final_boundary

def generate_simulated_data(dataset, n_sim=100):
    np.random.seed(42)
    sim = []
    def simVar(mea, var, n): 
        std_dev = np.sqrt(var)
        samples = np.random.normal(mea, std_dev, n) 
        return samples

    for _, row in dataset.iterrows():
        onesim = simVar(row['Val'],row['Var'],n_sim)
        sim.append(onesim)

    sim = np.vstack(sim)
    sim = pd.DataFrame(sim, columns=[f'sim_{i+1}' for i in range(sim.shape[1])]) 
    return sim

def stratify_mode(dataframe, n_strata=5, n_sim=100):
    sim = generate_simulated_data(dataframe, n_sim)

    def stratify_sim_dataset(sim, n_strata=5):
        stratified_sim = pd.DataFrame()
        for col in sim.columns:
            stratified_sim[col], x = cumrootfreq(sim[col], n_strata)
        return stratified_sim
    
    stratified_sim = stratify_sim_dataset(sim, n_strata)

    def most_frequent(row):
        return row.mode().iloc[0]
    
    mode = stratified_sim.apply(most_frequent, axis=1)
    return mode

def majority_filter(values):
    valid = values[~np.isnan(values)]
    if valid.size == 0:
        return np.nan
    values, counts = np.unique(valid, return_counts=True)
    most_frequent_value = values[np.argmax(counts)]
    if np.isnan(most_frequent_value) and len(values) > 1:
        second_most_frequent_value = values[np.argsort(counts)[-2]]
        return second_most_frequent_value
    return most_frequent_value  # Avoid nan_policy for older scipy

def stratify_major2D(dataframe, kernel_size=3, n_strata=5, n_sim=100):
    copyDF = dataframe.copy()
    copyDF["mode"] = stratify_mode(copyDF, n_strata, n_sim)

    grid = copyDF.pivot(index="Y", columns="X", values="mode").sort_index(ascending=False)
    raster = grid.to_numpy(dtype=float)

    filtered = generic_filter(raster, majority_filter, size=kernel_size)

    filtered_df = pd.DataFrame(filtered, index=grid.index, columns=grid.columns)
    filtered_long = filtered_df.stack().reset_index()
    filtered_long.columns = ["Y", "X", "major"]

    oriCoord = copyDF[["X", "Y", "Val", "Var"]]
    majorDF = pd.merge(oriCoord, filtered_long, on=["X", "Y"], how="left")
    return majorDF["major"].astype("int32")

def smooth_strata_majority(
    dataframe,
    strata_col,
    kernel_size=3,
):
    df = dataframe.copy()

    # pivot to raster
    grid = (
        df.pivot(index="Y", columns="X", values=strata_col)
          .sort_index(ascending=False)
    )
    raster = grid.to_numpy(dtype=float)

    # apply majority filter
    filtered = generic_filter(raster, majority_filter, size=kernel_size)

    # back to long format
    filtered_df = pd.DataFrame(filtered, index=grid.index, columns=grid.columns)
    filtered_long = filtered_df.stack().reset_index()
    filtered_long.columns = ["Y", "X", strata_col]

    # merge back
    out = df.drop(columns=[strata_col]).merge(
        filtered_long, on=["X", "Y"], how="left"
    )

    out[strata_col] = out[strata_col].astype("int32")
    return out

def _grid_from_df(df, col):
    grid = df.pivot(index="Y", columns="X", values=col).sort_index(ascending=False)
    return grid, grid.to_numpy()

def _df_from_grid(grid, arr, colname):
    out = pd.DataFrame(arr, index=grid.index, columns=grid.columns)
    long = out.stack().reset_index()
    long.columns = ["Y", "X", colname]
    return long

def merge_small_strata_by_area_adjacency(df, strata_col, min_share=0.08, max_iter=20):
    """
    Merge strata whose total area share < min_share into the dominant adjacent stratum (4-neighborhood).
    Assumes strata labels are non-negative integers (0,1,2,... or 1,2,...). NaNs allowed.
    """
    out = df.copy()

    for _ in range(max_iter):
        grid, arr = _grid_from_df(out, strata_col)
        A = arr.astype(float)  # keep NaNs if any

        valid = ~np.isnan(A)
        vals = A[valid].astype(np.int32)

        unique, counts = np.unique(vals, return_counts=True)
        total = counts.sum()
        shares = {int(u): c / total for u, c in zip(unique, counts)}

        small = [k for k, s in shares.items() if s < min_share]
        if not small:
            break

        A2 = A.copy()

        for lab in small:
            mask = valid & (A2.astype(np.int32) == int(lab))
            if mask.sum() == 0:
                continue

            neigh = []

            for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                shifted = np.full(A2.shape, np.nan, dtype=float)

                y0_src = max(0, -dy)
                y1_src = A2.shape[0] - max(0, dy)
                x0_src = max(0, -dx)
                x1_src = A2.shape[1] - max(0, dx)

                y0_dst = max(0, dy)
                y1_dst = A2.shape[0] - max(0, -dy)
                x0_dst = max(0, dx)
                x1_dst = A2.shape[1] - max(0, -dx)

                shifted[y0_dst:y1_dst, x0_dst:x1_dst] = A2[y0_src:y1_src, x0_src:x1_src]

                # boundary pixels of this lab that touch a different (valid) neighbor
                b = mask & ~np.isnan(shifted) & (shifted.astype(np.int32) != int(lab))
                if b.any():
                    neigh.append(shifted[b].astype(np.int32))

            if len(neigh) == 0:
                continue

            neigh_labels = np.concatenate(neigh).astype(np.int32)

            # >>> critical fix: remove any negative labels (and anything weird) before bincount
            neigh_labels = neigh_labels[neigh_labels >= 0]
            if neigh_labels.size == 0:
                continue

            new_lab = int(np.bincount(neigh_labels).argmax())
            A2[mask] = new_lab

        merged_long = _df_from_grid(grid, A2, strata_col)
        out = out.drop(columns=[strata_col]).merge(merged_long, on=["X", "Y"], how="left")
        out[strata_col] = out[strata_col].astype("int32")

    return out


def smooth_and_merge_loop(
    df,
    strata_col="strata",
    kernel_size=3,
    n_iter=3,
    min_share=0.08,
    merge_after_iter=2,   # start merging from this iteration (2 is a good default)
):
    out = df.copy()

    for i in range(1, n_iter + 1):
        prev = out[strata_col].copy()

        # 1) smoothing
        out = smooth_strata_majority(out, strata_col, kernel_size=kernel_size)

        # 2) merging small strata (optionally delayed)
        if (min_share is not None) and (i >= merge_after_iter):
            out = merge_small_strata_by_area_adjacency(out, strata_col, min_share=min_share)

        # diagnostics (optional but super useful)
        changed = (prev != out[strata_col]).mean() * 100
        n_strata = out[strata_col].nunique(dropna=True)
        print(f"iter {i}: changed={changed:.2f}% | n_strata={n_strata}")

        # stop if very stable
        if changed < 1.0:
            break

    return out


def calculate_stratum(dataset, strata_column, n_samples=40):
    ## dataset should contain 'Val', 'Var', and the 'defined Strata'
    N = len(dataset)
    group = dataset[strata_column].value_counts().reset_index()
    group.columns = ['h', 'Nh']
    group['Wh'] = group['Nh']/N
    Yh = []
    Sh = []

    for i in group.index.to_list():
        h = group['h'].unique()[i]
        subdf = dataset[dataset[strata_column]==h]
        mean_pop, var_pop = overall_mean_variance(subdf)
        Yh.append(mean_pop) 
        Sh.append(np.sqrt(var_pop))
    group['Yh']=Yh
    group['Sh']=Sh
    group['WhSh'] = group['Wh']*group['Sh']
    
    sumWhSh= sum(group['WhSh'])
    nh = round(n_samples*group['WhSh']/sumWhSh).values
    group['nh'] = np.int32(nh)

    groupNonZero = group[group['nh'] != 0]      ## in case there is stratum with 0 samples, exclude it.
    Wh2 = pow(groupNonZero['Wh'].values,2)
    Sh2 = pow(groupNonZero['Sh'].values,2)
    onef = 1-(groupNonZero['nh']/groupNonZero['Nh'])
    Varh = sum(Wh2*Sh2*onef/groupNonZero['nh'])
    return group[['h', 'nh']], Varh

def where_to_sample_in_stratum(
    stratum_dataset,
    min_distance=0,
    nh=5,
    max_tries=100,
    geom_boundary=None,
    edge_buffer=0,
):
    df = stratum_dataset.copy()

    # --- Normalize boundary to a single shapely geometry ---
    if geom_boundary is None:
        safe_geom = None
    else:
        # If user passed GeoDataFrame/GeoSeries, dissolve to shapely geometry
        if isinstance(geom_boundary, (gpd.GeoDataFrame, gpd.GeoSeries)):
            geom_boundary = geom_boundary.unary_union

        # If someone passed a pandas Series, try dissolving its values
        if hasattr(geom_boundary, "values") and not hasattr(geom_boundary, "geom_type"):
            # likely a Series of geometries
            geom_boundary = gpd.GeoSeries(list(geom_boundary)).unary_union

        safe_geom = geom_boundary

        if edge_buffer and edge_buffer > 0:
            safe_geom = safe_geom.buffer(-edge_buffer)

            # IMPORTANT: safe_geom is shapely now, so is_empty is a bool
            if safe_geom.is_empty:
                raise ValueError(
                    f"edge_buffer={edge_buffer} too large: boundary becomes empty after inward buffer."
                )

    # --- prefilter candidates to safe area (inside & away from edge) ---
    if safe_geom is not None:
        inside_mask = df.apply(
            lambda r: Point(r["X"], r["Y"]).covered_by(safe_geom),
            axis=1
        )
        df = df[inside_mask].copy()
        if df.empty:
            raise ValueError("No candidate pixels remain after applying boundary + edge buffer.")

    sampled_points = []

    for _ in range(nh):
        for _ in range(max_tries):
            idx = np.random.choice(df["id"].values)
            samp = df[df["id"] == idx].iloc[0]
            x, y = samp["X"], samp["Y"]
            new_point = Point(x, y)

            if all(new_point.distance(Point(px, py)) >= min_distance for px, py in sampled_points):
                sampled_points.append([x, y])
                break
        else:
            raise RuntimeError(
                "Failed to generate a valid point (try smaller min_distance/edge_buffer or increase max_tries)."
            )

    sampled_points = pd.DataFrame(sampled_points, columns=["X", "Y"])
    sampled_points = sampled_points.merge(df, on=["X", "Y"], how="left")
    return sampled_points

def _min_pairwise_distance_xy(df):
    coords = df[['X', 'Y']].to_numpy()
    if len(coords) < 2:
        return float('inf')
    tree = cKDTree(coords)
    distances, _ = tree.query(coords, k=2)  # k=2 to get the nearest neighbor (excluding self)
    min_distance = np.min(distances[:, 1])  # Exclude the zero distance to self
    return min_distance

def choose_global_minimum_samples_by_fixed_H(
    data,
    defH=7,
    area=None,
    nh_min=3,
    aimed_Svar=0.001,
    minDistance=50,
    kernel_size=3,
    n_sim=100,
    max_attempts=100,
    packing_factor=0.5,
    geom_boundary=None,
    edge_buffer=0,
):
    if area is None or area <= 0:
        raise ValueError("area must be provided in hectares and > 0.")
    if defH < 3:
        raise ValueError("defH must be >= 3.")

    H = defH  # fixed H

    # Conservative-ish physical cap (still approximate)
    n_max = int(max(1, packing_factor * area * 10000 / (minDistance * minDistance)))

    # ✅ fixed H => minimum is nh_min * H
    n_min = nh_min * H

    # ✅ only compute strata for this H once
    temp_df = data.copy()
    temp_df["strata"] = stratify_major2D(
        dataframe=temp_df,
        kernel_size=kernel_size,
        n_strata=H,
        n_sim=n_sim,
    )
    # # smooth multiple times
    temp_df = smooth_and_merge_loop(
        temp_df,
        strata_col="strata",
        kernel_size=3,      # boss wants 3 or 5 → start with 3
        n_iter=3,           # usually 2–3 is enough
        min_share=0.08,     # 8% threshold
        merge_after_iter=2  # merge from iteration 2
    )

    strata_ids = sorted(temp_df["strata"].dropna().unique().tolist())
    strata_groups = {
        h: temp_df[temp_df["strata"] == h].reset_index(drop=True)
        for h in strata_ids
    }

    # --- Search: n first (global minimum n for this fixed H) ---
    for n_samples in range(n_min, n_max + 1):

        g, Svar = calculate_stratum(
            temp_df, strata_column="strata", n_samples=n_samples
        )
        g = g.sort_values("h")

        nh = g["nh"].to_list()
        stratum = g["h"].to_list()

        # constraints
        if Svar > aimed_Svar:
            continue
        if any(v < nh_min for v in nh):
            continue
        if len(stratum) < 3:
            continue

        samp_df = None
        for _ in range(max_attempts):
            parts = []
            ok = True

            for h, nh_i in zip(stratum, nh):
                subdf = strata_groups.get(h)
                if subdf is None or len(subdf) == 0:
                    ok = False
                    break
                try:
                    samp_h = where_to_sample_in_stratum(
                        subdf,
                        min_distance=minDistance,
                        nh=nh_i,
                        edge_buffer=edge_buffer,
                        geom_boundary=geom_boundary,
                    )
                except RuntimeError:
                    ok = False
                    break
                parts.append(samp_h)

            if not ok:
                continue

            samp_try = pd.concat(parts, ignore_index=True)
            min_dist = _min_pairwise_distance_xy(samp_try)

            if min_dist >= minDistance:
                samp_df = samp_try
                break

        if samp_df is None:
            continue

        H = len(strata_ids)

        # ✅ return immediately when first feasible n is found
        return {
            "n_strata": int(H),
            "n_samples": int(n_samples),
            "sampling_variance": float(Svar),
            "alloc_df": g,          # allocation table
            "strata_df": temp_df,   # per-pixel strata map
            "samp_df": samp_df,
            "min_dist": float(min_dist),
        }

    # ✅ nothing feasible
    return None

def choose_best_by_lowest_svar_across_H(
    data,
    H_min=3,
    H_max=6,
    area=None,
    nh_min=3,
    aimed_Svar=0.001,
    minDistance=50,
    kernel_size=3,
    n_sim=100,
    max_attempts=100,
    geom_boundary=None,
    edge_buffer=0,
    smooth_kwargs=None,
):
    """
    Try H in [H_min..H_max], find a feasible design for each H (using your existing logic),
    then return the feasible design with the lowest sampling_variance.
    """
    smooth_kwargs = smooth_kwargs or {}

    candidates = []

    for H in range(H_min, H_max + 1):
        # --- call your per-H finder (whatever you use internally) ---
        # If your code currently only exposes choose_global_minimum_samples_by_fixed_H,
        # then you can call it with defH=H and treat that as "try this H".
        res = choose_global_minimum_samples_by_fixed_H(
            data=data,
            defH=H,
            area=area,
            nh_min=nh_min,
            aimed_Svar=aimed_Svar,
            minDistance=minDistance,
            kernel_size=kernel_size,
            n_sim=n_sim,
            max_attempts=max_attempts,
            geom_boundary=geom_boundary,
            edge_buffer=edge_buffer,
        )

        if res is None:
            continue
        candidates.append(res)

    if not candidates:
        return None

    # Choose the lowest sampling variance; tie-breakers are optional
    candidates.sort(key=lambda d: (d["sampling_variance"], d["n_samples"], d["n_strata"]))
    return candidates[0]
