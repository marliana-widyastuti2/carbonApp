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

def create3D(dataframe, n_strata, n_sim):
    temp_df2 = dataframe.copy()
    sim = generate_simulated_data(temp_df2, n_sim)

    stratified_sim = pd.DataFrame()
    for col in sim.columns:
        stratified_sim[col], _ = cumrootfreq(sim[col], n_strata)

    majorDF = temp_df2.merge(stratified_sim, left_index=True, right_index=True)

    list_rasters = []
    sim_cols = list(stratified_sim.columns)  # ONLY sim_* columns
    for co in sim_cols:
        grid = majorDF.pivot(index="Y", columns="X", values=co).sort_index(ascending=False)
        list_rasters.append(grid.to_numpy(dtype=float))

    arr = np.stack(list_rasters, axis=2)  # (rows, cols, n_sim)
    return arr

def major3D(arr, size=3):
    rows, cols, _ = arr.shape

    padded = np.pad(arr, ((1, 1), (1, 1), (0, 0)), mode='constant', constant_values=-9999)
    padded = np.where(padded == -9999, np.nan, padded).astype(float)

    windows = view_as_windows(padded, (size, size, arr.shape[2])) 
    windows_reshaped = windows.reshape(rows, cols, -1) 

    def nan_mode(data):
        out = np.full(data.shape[:2], np.nan)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                vals, counts = np.unique(data[i, j][~np.isnan(data[i, j])], return_counts=True)
                if counts.size > 0:
                    out[i, j] = vals[np.argmax(counts)]
        return out

    majority = nan_mode(windows_reshaped)
    return majority

def stratify_major3D(dataframe, size=3, n_strata=5, n_sim=100):
    temp_df1 = dataframe.copy()

    array3D = create3D(dataframe, n_strata, n_sim)

    grid = temp_df1.pivot(index='Y', columns='X', values='Val').sort_index(ascending=False)
    raster = grid.to_numpy(dtype=float)

    arrMajor = major3D(array3D, size=size)

    arrMajor[np.isnan(raster)]=np.nan
    filtered_df = pd.DataFrame(
        arrMajor,
        index=grid.index,     # Y coordinates
        columns=grid.columns  # X coordinates
    )

    # Convert to long format
    majorDF = filtered_df.stack().reset_index()
    majorDF.columns = ['Y', 'X', 'major']

    mergedDF = pd.merge(temp_df1, majorDF, on=['X', 'Y'], how='left')
    return mergedDF['major']

# Horgan's algorithm
def geometric(data, n_strata=5):
    temp = pd.DataFrame({'Val':data})
    min = temp['Val'].min()
    max = temp['Val'].max()

    bou = []
    for i in range(n_strata+1):
        # print(i)
        cr = math.pow(max/min, 1/n_strata)
        # print(cr)
        r= math.pow(cr, i)
        k=r*min
        # print(k)
        bou.append(k)

    temp['Bins'] = pd.cut(temp['Val'], bins=bou,  include_lowest=True)
    return temp.groupby('Bins').ngroup()

def stratify_random_search(dataframe, n_strata=5, c_precision=0.005, iteration = 100):
    ## define total number of dataset
    N = dataframe['Val'].count()

    ## Sorted the dataset based on variable value, then extract the index into a new column 'index'.
    dfSorted = dataframe.copy().sort_values(by='Val')  
    dfSorted.reset_index(drop=True, inplace=True)
    dfSorted.reset_index(inplace=True)

    ## Define initial Stratification using Cum-Root-Freq method
    dfSorted['crf'], crf_boundary = cumrootfreq(dfSorted['Val'], n_strata) 

    ## some inner-functions
    def find_max_indices_of_boundary_groups(series):
        max_indices = [0]
        for idx in range(1, len(series)):
            if series[idx] != series[idx - 1]:
                max_indices.append(idx - 1)
        max_indices.append(len(series) - 1)
        return max_indices

    ## Using the index boundary to calculate initial of n(a) and nh for each strata
    def stratifyBound(dataframe, column, boundary):
        df = dataframe.copy()
        df['Bins'] = pd.cut(df[column], bins=boundary, include_lowest=True)
        df['Strata'] = df.groupby('Bins').ngroup()
        return df

    def calc_strata(dataset, n_strata, c_precision):
        df = dataset.copy()
        ds = df.groupby('Strata').count().iloc[:,1].reset_index()
        ds.columns = ['Strata', 'Nh']
        ds['Wh'] = ds['Nh']/N

        Ybar = df['Val'].mean()

        YhBar = df.groupby('Strata')['Val'].mean().reset_index()
        YhBar.columns = ['Strata', 'YhBar']
        ds = pd.merge(ds, YhBar, on='Strata', how='left')
        Sh = df.groupby('Strata')['Val'].std().reset_index()
        Sh.columns = ['Strata', 'Sh']
        ds = pd.merge(ds, Sh, on='Strata', how='left')

        L = n_strata-1                 ### the substract is only adjustment in which python always starts from 0 
        NL = ds['Nh'][L] 

        otherL = ds[~(ds['Strata'] == L)]
        otherL['Sh2'] = otherL['Sh'].pow(2)
        otherL['WhSh'] = otherL['Wh']*otherL['Sh']
        otherL['WhSh2'] = otherL['Wh']*otherL['Sh2']

        a = pow(sum(otherL['WhSh']),2)
        b = pow(Ybar, 2)*pow(c_precision, 2)+(sum(otherL['WhSh2'])/N)
        n = np.round(NL+(a/b))

        ds['WhSh'] = ds['Wh']*ds['Sh']
        
        ds['sumWhSh'] = sum(ds['WhSh'])
        nh = round(n*ds['WhSh']/(ds['sumWhSh']-ds['WhSh'])).values

        ds['nh'] = nh
        return nh, n
    
    ## convert the boundaries into its according index value
    bound = find_max_indices_of_boundary_groups(dfSorted['crf'])
    init_stratified = stratifyBound(dfSorted, 'index', bound)
    nh, n = calc_strata(init_stratified, n_strata, c_precision)

    ## 
    r=0
    while r < iteration:
        # print(r)
        randG = random.choice(range(1, n_strata)) ## randomly choose one boundary to be optimised
        randBou = bound[randG]
        popG=int(N/100)
        jran = range(-popG, popG,1)
        filtered_jran = [num for num in jran if num != 0]

        ## get random integer to transfer a 'p' number of data between two strata divided by 'randG' boundary.
        p = random.choice(filtered_jran) 

        newRandBou = randBou+p
        tempBound = bound.copy()
        tempBound[randG]=newRandBou

        def is_increasing(lst):
            return all(x < y for x, y in zip(lst, lst[1:]))
        def count_differences(lst):
            return [y - x for x, y in zip(lst, lst[1:])]

        if is_increasing(tempBound) == True:
            Nh = count_differences(tempBound)
            if all(numb >= 2 for numb in Nh):
                # print('Nh condition satisfied')
                
                ## Using the initial boundary, calculate initial value of n(a) and nh each strata
                stratDF = stratifyBound(dfSorted, 'index', tempBound)
                # print(stratDF)
                up_nh, up_n = calc_strata(stratDF, n_strata, c_precision)
                # print(up_nhL_1)
                if all(numb >= 2 for numb in up_nh) & (up_n < n):
                    # print('nh and condition n(a) satisfied')
                    n = up_n.copy()
                    bound = tempBound.copy()
                          
        r += 1

    ## Take the final boundary to get the final stratum area (optimal result)
    finalStratDF = pd.DataFrame(stratifyBound(dfSorted, 'index', bound))

    finalStratDF = finalStratDF[['X', 'Y', 'Strata']]
    finalStratDF = pd.merge(dataframe, finalStratDF, on=['X', 'Y'], how='left')
    return finalStratDF['Strata']

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

def choose_global_minimum_samples_by_n(
    data,
    H_max=7,
    area=None,
    nh_min=3,
    aimed_Svar=0.001,
    minDistance=50,
    kernel_size=3,
    n_sim=100,
    max_attempts=100,
    prefer_lower_H=True,
    packing_factor=0.5,   # more conservative than 1.0
    geom_boundary=None, 
    edge_buffer=0,
):
    """
    Guaranteed global minimum n_samples:
    - Loop n_samples from smallest to largest
    - For each n_samples, try all H
    - Return immediately when at least one feasible design exists for that n_samples
      (pick best among H using tie rules).
    """
    if area is None or area <= 0:
        raise ValueError("area must be provided in hectares and > 0.")
    if H_max < 3:
        raise ValueError("H_max must be >= 3.")

    # Conservative-ish physical cap (still approximate)
    n_max = int(max(1, packing_factor * area * 10000 / (minDistance * minDistance)))

    # Smallest possible n overall occurs at H=3 (since you require H>=3)
    n_min = nh_min * 3

    # --- Precompute strata maps for each H once (big speed-up) ---
    strata_cache = {}
    for H in range(3, H_max + 1):
        temp_df = data.copy()
        temp_df["strata"] = stratify_major2D(
            dataframe=temp_df,
            kernel_size=kernel_size,
            n_strata=H,
            n_sim=n_sim,
        )
        # Cache grouped pixels per stratum id for faster sampling later
        strata_ids = sorted(temp_df["strata"].dropna().unique().tolist())
        groups = {h: temp_df[temp_df["strata"] == h].reset_index(drop=True) for h in strata_ids}
        strata_cache[H] = (temp_df, groups)

    # --- Global search: n first, then H ---
    for n_samples in range(n_min, n_max + 1):
        feasible_designs = []

        for H in range(3, H_max + 1):
            # cannot allocate < nh_min per stratum
            if n_samples < nh_min * H:
                continue

            temp_df, strata_groups = strata_cache[H]

            g, Svar = calculate_stratum(temp_df, strata_column="strata", n_samples=n_samples)
            g = g.sort_values("h")

            nh = g["nh"].to_list()
            stratum = g["h"].to_list()

            # statistical + minimum-per-stratum constraints
            if Svar > aimed_Svar:
                continue
            if any(v < nh_min for v in nh):
                continue

            # Try to place points satisfying minDistance
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
                        samp_h = where_to_sample_in_stratum(subdf, min_distance=minDistance, 
                                                            nh=nh_i, edge_buffer=edge_buffer,
                                                            geom_boundary=geom_boundary)
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

            feasible_designs.append({
                "n_strata": int(H),
                "n_samples": int(n_samples),
                "sampling_variance": float(Svar),
                "nh": nh,
                "val_stratum": stratum,
                "strata_df": temp_df,   # keep
                "samp_df": samp_df,     # keep
                "min_dist": float(_min_pairwise_distance_xy(samp_df)),
            })

        # ✅ Early stop at the FIRST n_samples that yields any feasible design
        if feasible_designs:
            # choose best among H for this minimal n_samples
            feasible_designs.sort(
                key=lambda d: (
                    d["n_samples"],                              # (all equal here)
                    d["n_strata"] if prefer_lower_H else -d["n_strata"],
                    d["sampling_variance"],                      # smaller is better
                )
            )
            return feasible_designs[0]

    return None
