import rasterio
import numpy as np
import pandas as pd
from rasterio.transform import from_origin
from rasterio.features import shapes

import geopandas as gpd
from shapely.geometry import shape

from rasterstats import zonal_stats
from pyproj import CRS

DIR_RASTER = "data/raster/"
DST_CRS = CRS.from_epsg(32755)

def df_to_raster(dataFrame, EPSG = 32755):
    # Ensure the DataFrame has the required columns
    if not all(col in dataFrame.columns for col in ['X', 'Y', 'strata']):
        raise ValueError("DataFrame must contain 'X', 'Y', and 'strata' columns.")
    
    # df has columns: X, Y, value
    x_unique = np.sort(dataFrame["X"].unique())
    y_unique = np.sort(dataFrame["Y"].unique())[::-1]   # descending for top-to-bottom raster

    res_x = np.min(np.diff(x_unique))
    res_y = np.min(np.diff(np.sort(dataFrame["Y"].unique())))

    grid = dataFrame.pivot(index="Y", columns="X", values="strata").sort_index(ascending=False)
    raster = grid.to_numpy()

    # IMPORTANT: X and Y are assumed to be cell centers
    xmin = x_unique.min() - res_x / 2
    ymax = y_unique.max() + res_y / 2

    transform = from_origin(xmin, ymax, res_x, res_y)

    with rasterio.open(
        f"{DIR_RASTER}strata.tif",
        "w",
        driver="GTiff",
        height=raster.shape[0],
        width=raster.shape[1],
        count=1,
        dtype=raster.dtype,
        crs=f"EPSG:{EPSG}",   # replace with your CRS
        transform=transform,
    ) as dst:
        dst.write(raster, 1)

def get_mean_x_h(farm_name=None):
    with rasterio.open("data/raster/strata.tif") as src:
        image = src.read(1)
        mask = image != src.nodata
        transform = src.transform
        crs = src.crs

    # polygonize
    geoms = [
        {"geometry": shape(geom), "strata": value}
        for geom, value in shapes(image, mask=mask, transform=transform)
    ]

    gdf = gpd.GeoDataFrame(geoms, crs=crs)

    # dissolve by strata
    gdf_diss = gdf.dissolve(by="strata").reset_index()
    gdf_diss["A_h"] = gdf_diss.geometry.area

    SOC_RAS_path = f"OUT_SOC/{farm_name}/{farm_name}_SOC_PRED_MOSAIC_CLIPPED.tif"

    # --- Make sure CRS matches raster CRS ---
    with rasterio.open(SOC_RAS_path) as src:
        raster_crs = src.crs
        nodata = src.nodata

    strata_gdf = gdf_diss.to_crs(raster_crs)

    # --- Calculate mean raster value for each polygon ---
    stats = zonal_stats(
        strata_gdf,
        SOC_RAS_path,
        stats=["mean"],
        nodata=nodata,
        geojson_out=False
    )

    # Add result back to GeoDataFrame
    strata_gdf["mean_x_h"] = [s["mean"] for s in stats]
    return strata_gdf

def sample_raster_to_gdf(gdf, raster_path, out_col):
    with rasterio.open(raster_path) as src:
        gdf_tmp = gdf.to_crs(src.crs)
        coords = [(geom.x, geom.y) for geom in gdf_tmp.geometry]
        vals = [v[0] for v in src.sample(coords)]

        if src.nodata is not None:
            vals = [np.nan if v == src.nodata else v for v in vals]

    gdf[out_col] = vals
    return gdf

def calc_est_difference(FARM_NAME = None, strata_df=None, samp_df=None):
    strata_gdf = get_mean_x_h(FARM_NAME)
    Nh = strata_df.groupby("strata").size().reset_index(name="N_h")
    strata_gdf = strata_gdf.merge(Nh, on="strata", how="left")

    samp = samp_df.copy()
    samp_gdf = gpd.GeoDataFrame(
        samp,
        geometry=gpd.points_from_xy(
            samp["X"],
            samp["Y"]
        ),
        crs=DST_CRS
    )
    samp_gdf

    samp_gdf = sample_raster_to_gdf(samp_gdf, "/home/marliana/shared_folder/CarbonApp/data/SOC_AU/SOC_0_100_stock_tonclipped_30m.tif", "y_h_s")
    samp_gdf = sample_raster_to_gdf(samp_gdf, f"OUT_SOC/{FARM_NAME}/{FARM_NAME}_SOC_PRED_MOSAIC_CLIPPED.tif", "x_h_s")
    
    strata = strata_gdf['strata'].unique()

    Y_diff = []
    T_diff = []
    var_Y = []
    var_T = []

    for s in strata:
        print(f"Stratum {s}:")

        row = strata_gdf[strata_gdf["strata"] == s].iloc[0]
        samp_in_stratum = samp_gdf[samp_gdf["strata"] == s].copy()

        A_h_m2 = row["A_h"]
        A_h = A_h_m2 / 10000  # ha
        W_h = A_h_m2 / strata_gdf["A_h"].sum()

        N_h = row["N_h"]                 # total population units in stratum h
        n_h = len(samp_in_stratum)       # sampled units in stratum h
        f_h = n_h / N_h

        print(f"  Area (A_h): {A_h:.2f} ha")
        print(f"  Weight (W_h): {W_h:.4f}")
        print(f"  Sample fraction (f_h): {f_h:.4f}")

        mean_y_h_s = samp_in_stratum["y_h_s"].mean()
        mean_x_h_s = samp_in_stratum["x_h_s"].mean()
        mean_x_h = row["mean_x_h"]

        print(f"  Mean SOC_w (sample points): {mean_y_h_s:.2f} ton/ha")
        print(f"  Mean SOC_RS (sample points): {mean_x_h_s:.2f} ton/ha")
        print(f"  Mean SOC_RS (stratum {s}): {mean_x_h:.2f} ton/ha")

        mean_Y_h_diff = mean_y_h_s + (mean_x_h - mean_x_h_s)
        print(f"  Adjusted mean SOC for stratum {s}: {mean_Y_h_diff:.2f} ton/ha")

        Wh_Yh_diff = W_h * mean_Y_h_diff
        print(f"  Weighted contribution to overall mean for stratum {s}: {Wh_Yh_diff:.4f}")
        Y_diff.append(Wh_Yh_diff)

        T_hat_h = A_h * mean_Y_h_diff
        print(f"  Estimated total SOC for stratum {s}: {T_hat_h:.4f} ton")
        T_diff.append(T_hat_h)

        samp_in_stratum["e_h_s"] = samp_in_stratum["y_h_s"] - samp_in_stratum["x_h_s"]
        var_e_h = np.var(samp_in_stratum["e_h_s"], ddof=1)

        var_Y_diff_h = (1 - f_h) * var_e_h / n_h * W_h**2
        print(f"  Variance of Y_diff for stratum {s}: {var_Y_diff_h:.4f} (ton/ha)^2")
        var_Y.append(var_Y_diff_h)

        var_T_diff_h = (1 - f_h) * var_e_h / n_h * A_h**2
        print(f"  Variance of T_diff for stratum {s}: {var_T_diff_h:.4f} (ton)^2")
        var_T.append(var_T_diff_h)
        
    sum_Y_diff = np.sum(Y_diff)
    sum_T_diff = np.sum(T_diff)
    sum_var_Y = np.sum(var_Y)
    sum_var_T = np.sum(var_T)
    return sum_Y_diff, sum_var_Y, sum_T_diff, sum_var_T

