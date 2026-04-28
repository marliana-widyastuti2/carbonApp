import rasterio
import numpy as np
import pandas as pd
from rasterio.transform import from_origin
from rasterio.features import shapes

import geopandas as gpd
from shapely.geometry import shape

from rasterstats import zonal_stats
from pyproj import CRS
from pyproj import Transformer

from pathlib import Path
from rasterio.warp import reproject, Resampling

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

def resample_SOC(FARM_NAME = None):
    src_path = Path(f"/home/marliana/shared_folder/CarbonApp/OUT_SOC/{FARM_NAME}/{FARM_NAME}_SOC_PRED_MOSAIC_CLIPPED.tif")
    ref_path = Path("/home/marliana/shared_folder/CarbonApp/data/raster/strata.tif")
    out_dir = Path("/home/marliana/shared_folder/CarbonApp/OUT_SOC/resampled/")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / src_path.name  # keep same raster name

    # =========================================================
    # Resample to match reference raster
    # =========================================================
    with rasterio.open(src_path) as src, rasterio.open(ref_path) as ref:
        # Copy source metadata first
        out_meta = src.meta.copy()

        # Update to match reference raster grid
        out_meta.update({
            "crs": ref.crs,
            "transform": ref.transform,
            "width": ref.width,
            "height": ref.height,
            "compress": "lzw"
        })

        with rasterio.open(out_path, "w", **out_meta) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=ref.transform,
                    dst_crs=ref.crs,
                    resampling=Resampling.bilinear
                )

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

def format_the_csv(csv_path, x_col = None, y_col = None, CRS = None, SOC_col = None):
    df = pd.read_csv(csv_path)

    if SOC_col is not None:
        df = df.rename(columns={SOC_col: "y_h_s"})

    if x_col is not None and y_col is not None:
        df = df.rename(columns={x_col: "Lon", y_col: "Lat"})

    if CRS is not None:
        if CRS != DST_CRS:
            transformer = Transformer.from_crs(CRS, DST_CRS, always_xy=True)
            df["X"], df["Y"] = transformer.transform(df["Lon"].to_numpy(), df["Lat"].to_numpy())
        else:
            df = df.rename(columns={"Lon": "X", "Lat": "Y"})

        return df
    return df

def calc_est_regression(FARM_NAME = None, strata_df=None, samp_df=None, Farm_area_ha = None):
    strata_gdf = get_mean_x_h(FARM_NAME)
    Nh = strata_df.groupby("strata").size().reset_index(name="N_h")
    strata_gdf = strata_gdf.merge(Nh, on="strata", how="left")

    df_to_raster(strata_df) ## Convert strata_df to strata.tif for further analysis

    samp = samp_df.copy()
    samp_gdf = gpd.GeoDataFrame(
        samp,
        geometry=gpd.points_from_xy(
            samp["X"],
            samp["Y"]
        ),
        crs=DST_CRS
    )

    samp_gdf = sample_raster_to_gdf(samp_gdf, "/home/marliana/shared_folder/CarbonApp/data/raster/strata.tif", "strata")
    # samp_gdf = sample_raster_to_gdf(samp_gdf, f"OUT_SOC/{FARM_NAME}/{FARM_NAME}_SOC_PRED_MOSAIC_CLIPPED.tif", "x_h_s")
    samp_gdf = sample_raster_to_gdf(samp_gdf, f"OUT_SOC/resampled/{FARM_NAME}_SOC_PRED_MOSAIC_CLIPPED.tif", "x_h_s")
    
    strata = strata_gdf['strata'].unique()

    results = []

    for s in strata:
        row = strata_gdf[strata_gdf["strata"] == s].iloc[0]
        samp_in_stratum = samp_gdf[samp_gdf["strata"] == s].copy()

        A_h = row["A_h"] / 10000  # convert m^2 to ha
        A_total = strata_gdf["A_h"].sum()/10000 # convert m^2 to ha
        prop_A = A_h / A_total
        adj_A_h = prop_A * Farm_area_ha  # ha

        N_h = row["N_h"]                 # total population units in stratum h
        W_h = N_h / strata_gdf["N_h"].sum()
        n_h = len(samp_in_stratum)       # sampled units in stratum h
        f_h = n_h / N_h
    
        mean_y_h_s = samp_in_stratum["y_h_s"].mean()
        mean_x_h_s = samp_in_stratum["x_h_s"].mean()
        mean_x_h = row["mean_x_h"]

        S2_y_h_s = np.var(samp_in_stratum["y_h_s"], ddof=1)
        S2_x_h_s = np.var(samp_in_stratum["x_h_s"], ddof=1)
        Sxy_h_s = np.cov(samp_in_stratum["y_h_s"], samp_in_stratum["x_h_s"], ddof=1)[0, 1]

        a = (W_h**2) * (1-f_h)/n_h
        ah = a / (n_h - 1)
        bh_top = np.sum((samp_in_stratum["y_h_s"] - mean_y_h_s) * (samp_in_stratum["x_h_s"] - mean_x_h_s)) * ah
        bh_bottom = np.sum((samp_in_stratum["x_h_s"] - mean_x_h_s)**2) * ah

        # save everything into dictionary
        results.append({
            "strata": s,
            "A_h_ha": A_h,
            "A_total_ha": A_total,
            "prop_A": prop_A,
            "adj_A_h_ha": adj_A_h,
            "N_h": N_h,
            "W_h": W_h,
            "n_h": n_h,
            "f_h": f_h,
            "mean_y_h_s": mean_y_h_s,
            "mean_x_h_s": mean_x_h_s,
            "mean_x_h": mean_x_h,
            "S2_y_h_s": S2_y_h_s,
            "S2_x_h_s": S2_x_h_s,
            "Sxy_h_s": Sxy_h_s,
            "a": a,
            "ah": ah,
            "bh_top": bh_top,
            "bh_bottom": bh_bottom
        })

    results_df = pd.DataFrame(results)
    B = np.sum(results_df["bh_top"]) / np.sum(results_df["bh_bottom"])

    Y_y = np.sum(results_df["W_h"] * results_df["mean_y_h_s"])
    Y_x = np.sum(results_df["W_h"] * results_df["mean_x_h_s"])
    X_bar = np.mean(results_df["mean_x_h"])
    Y_reg = Y_y + B * (X_bar - Y_x)

    var_Y_reg = np.sum(results_df["a"] * (results_df["S2_y_h_s"] + B**2 * results_df["S2_x_h_s"] - 2*B*results_df["Sxy_h_s"]))

    T_reg = Y_reg * Farm_area_ha
    var_T_reg = np.sum((results_df['adj_A_h_ha']**2) * (1- results_df['f_h']) / (results_df['n_h']) * (results_df["S2_y_h_s"] + B**2 * results_df["S2_x_h_s"] - 2*B*results_df["Sxy_h_s"]))

    return B, Y_reg, var_Y_reg, T_reg, var_T_reg

def calc_est_stratified(FARM_NAME = None, strata_df=None, samp_df=None, Farm_area_ha = None):
    strata_gdf = get_mean_x_h(FARM_NAME)
    Nh = strata_df.groupby("strata").size().reset_index(name="N_h")
    strata_gdf = strata_gdf.merge(Nh, on="strata", how="left")

    df_to_raster(strata_df) ## Convert strata_df to strata.tif for further analysis

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
    samp_gdf = sample_raster_to_gdf(samp_gdf, "/home/marliana/shared_folder/CarbonApp/data/raster/strata.tif", "strata")

    strata = strata_gdf['strata'].unique()

    Y_hat = []
    T_hat = []
    var_Y = []
    var_T = []

    for s in strata:
        row = strata_gdf[strata_gdf["strata"] == s].iloc[0]
        samp_in_stratum = samp_gdf[samp_gdf["strata"] == s].copy()

        A_h = row["A_h"] / 10000  # convert m^2 to ha
        A_total = strata_gdf["A_h"].sum()/10000 # convert m^2 to ha
        prop_A = A_h / A_total
        adj_A_h = prop_A * Farm_area_ha  # ha

        N_h = row["N_h"]                 # total population units in stratum h
        W_h = N_h / strata_gdf["N_h"].sum()
        n_h = len(samp_in_stratum)       # sampled units in stratum h
        f_h = n_h / N_h

        mean_y_h_s = samp_in_stratum["y_h_s"].mean()
        Wh_Yh_diff = W_h * mean_y_h_s
        Y_hat.append(Wh_Yh_diff)

        T_hat_h = adj_A_h * mean_y_h_s
        T_hat.append(T_hat_h)

        var_y_h = np.var(samp_in_stratum["y_h_s"], ddof=1)

        var_Y_h = W_h**2 / n_h * var_y_h
        var_Y.append(var_Y_h)

        var_T_h = adj_A_h**2 / n_h * var_y_h
        var_T.append(var_T_h)    

    sum_Y_hat = np.sum(Y_hat)
    sum_T_hat = np.sum(T_hat)
    sum_var_Y = np.sum(var_Y)
    sum_var_T = np.sum(var_T)
    return sum_Y_hat, sum_var_Y, sum_T_hat, sum_var_T
