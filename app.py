from datetime import datetime
import os
import tempfile
from turtle import home, pd
import zipfile
from pathlib import Path

import streamlit as st
import geopandas as gpd
import rasterio
from rasterio.mask import mask

import utils
import stratify
import soc_rs
import estimateStock

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pyproj import CRS
from pyproj import Transformer

transformer = Transformer.from_crs(
    "EPSG:32755",
    "EPSG:4326",
    always_xy=True
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
DST_CRS = CRS.from_epsg(32755)

def clear_results():
    keys_to_clear = [
        "results_ready",
        "results_optimal_n",
        "results_best",
        "results_strata_df",
        "results_samp_df",
        "results_fig_sampling",
        "results_strata_csv_bytes",
        "results_samp_csv_bytes",
        "field_results_ready",
        "Y_reg",
        "var_Y_reg",
        "T_reg",
        "var_T_reg",
        "Y_str",
        "var_Y_str",
        "T_str",
        "var_T_str",
    ]

    for k in keys_to_clear:
        st.session_state.pop(k, None)

st.set_page_config(page_title="SamplingApp", layout="centered")
st.title("Optimised Sampling Design")

# --- Vector upload ---
uploaded = st.file_uploader(
    "Upload Farm boundary (ZIP Shapefile / GeoJSON / KML / KMZ)",
    type=["zip", "geojson", "json", "kml", "kmz"]
)
FARM_NAME = None

if uploaded is not None:
    file_signature = (uploaded.name, uploaded.size)

    if st.session_state.get("last_uploaded") != file_signature:
        clear_results()
        st.session_state["last_uploaded"] = file_signature
        st.info("New farm boundary uploaded. Click **Generate sampling design** to run the analysis.")

    FARM_NAME = Path(uploaded.name).stem

# --- Output options ---
out_dir = OUTPUT_DIR

# col1, col2 = st.columns(2)
crop_to_geom = True #col1.checkbox("Crop to geometry bounds", value=True)
all_touched =  True #st.checkbox("All touched (more inclusive edges)", value=False)

run_btn = st.button("Generate sampling design")

def read_vector_upload(uploaded_file) -> gpd.GeoDataFrame:
    name = uploaded_file.name.lower()

    with tempfile.TemporaryDirectory() as tmpdir:
        in_path = os.path.join(tmpdir, uploaded_file.name)
        with open(in_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # ---- Shapefile ZIP ----
        if name.endswith(".zip"):
            with zipfile.ZipFile(in_path, "r") as z:
                z.extractall(tmpdir)

            shp_files = [os.path.join(tmpdir, p) for p in os.listdir(tmpdir) if p.lower().endswith(".shp")]
            if not shp_files:
                raise ValueError("ZIP does not contain a .shp file.")
            gdf = gpd.read_file(shp_files[0])

        # ---- KMZ (zip containing KML) ----
        elif name.endswith(".kmz"):
            with zipfile.ZipFile(in_path, "r") as z:
                kml_candidates = [p for p in z.namelist() if p.lower().endswith(".kml")]
                if not kml_candidates:
                    raise ValueError("KMZ does not contain a .kml file.")
                # pick the first KML (common case: doc.kml)
                kml_name = kml_candidates[0]
                z.extract(kml_name, tmpdir)
                kml_path = os.path.join(tmpdir, kml_name)

            # Some GDAL builds need the KML driver specified
            try:
                gdf = gpd.read_file(kml_path, driver="KML")
            except TypeError:
                gdf = gpd.read_file(kml_path)

        # ---- KML ----
        elif name.endswith(".kml"):
            try:
                gdf = gpd.read_file(in_path, driver="KML")
            except TypeError:
                gdf = gpd.read_file(in_path)

        # ---- GeoJSON / JSON ----
        elif name.endswith(".geojson") or name.endswith(".json"):
            gdf = gpd.read_file(in_path)

        else:
            raise ValueError("Unsupported file type.")
        
    # save gdf into shapefile
    out_dir = BASE_DIR / "shapefiles"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{FARM_NAME}.shp"
    gdf.to_file(out_path, driver="ESRI Shapefile")

    if not out_path.exists():
        raise RuntimeError(f"Shapefile write failed: {out_path}")
    
    return gdf

def _calculate_area_ha() -> float:
    gdf = read_vector_upload(uploaded)

    if gdf.empty:
        raise ValueError("Vector file has no features.")
    if gdf.crs is None:
        raise ValueError("Vector CRS missing (.prj missing).")

    if gdf.crs != DST_CRS:
        gdf = gdf.to_crs(DST_CRS)

    geom = gdf.geometry.unary_union
    area_m2 = geom.area
    area_ha = area_m2 / 10_000.0

    return area_ha

def _clip_raster(
    path_raster_in: str,
    path_raster_out: str,
    crop: bool,
    all_touched: bool,
    buffer_m: float = 0.0,           # <-- NEW
) -> None:
    gdf = read_vector_upload(uploaded)
    if gdf.empty:
        raise ValueError("Vector file has no features.")

    if gdf.crs is None:
        raise ValueError("Shapefile CRS missing (.prj missing).")

    with rasterio.open(path_raster_in) as src:
        if src.crs is None:
            raise ValueError("Raster CRS missing.")

        # reproject vector to raster CRS (so buffer is in the same units as raster CRS)
        if gdf.crs != src.crs:
            gdf = gdf.to_crs(src.crs)

        # dissolve to one geometry
        geom = gdf.geometry.unary_union

        # apply buffer (in CRS units; for EPSG:32755 this is meters)
        if buffer_m and buffer_m != 0:
            geom = geom.buffer(buffer_m)

        out_image, out_transform = mask(
            src,
            [geom],
            crop=crop,
            all_touched=all_touched,
            nodata=src.nodata,
            filled=True
        )

        out_meta = src.meta.copy()
        out_meta.update({
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })

        # ensure nodata exists
        if out_meta.get("nodata", None) is None:
            if "float" in str(out_meta["dtype"]).lower():
                out_meta["nodata"] = -9999.0
            else:
                out_meta["nodata"] = 0

        with rasterio.open(path_raster_out, "w", **out_meta) as dst:
            dst.write(out_image)

def smart_format(x, fixed_decimals=4, sci_threshold=1e-3):
    if x == 0:
        return "0"
    if abs(x) < sci_threshold:
        return f"{x:.2e}"
    return f"{x:.{fixed_decimals}f}"

# --- RUN BLOCK: compute and store results ---
if run_btn:
    # clear old results first
    st.session_state.pop("results_ready", None)
    st.session_state.pop("optimal_n", None)
    st.session_state.pop("results", None)
    if uploaded is None:
        st.error("Please upload a file.")
    else:
        try:
            out_dir_path = Path(out_dir).expanduser()
            out_path = out_dir_path

            with tempfile.TemporaryDirectory() as tmpdir:

                ## area calc (not used in sampling, just informative)
                area_ha = _calculate_area_ha()
                st.metric(f"Area of Farm (ha)", smart_format(area_ha, fixed_decimals=2, sci_threshold=1e-3))

                # clip rasters
                meanSOC_in = DATA_DIR / "SOC_AU/SOC_0_100_stock_tonclipped_30m.tif"
                meanSOC_out = OUTPUT_DIR / "clipped_SOC_mean.tif"
                _clip_raster(str(meanSOC_in), str(meanSOC_out),
                             crop=crop_to_geom, all_touched=all_touched, buffer_m=15)

                varSOC_in = DATA_DIR / "SOC_AU/SOC_0_100_variance_tonclipped_30m.tif"
                varSOC_out = OUTPUT_DIR / "clipped_SOC_var.tif"
                _clip_raster(str(varSOC_in), str(varSOC_out),
                             crop=crop_to_geom, all_touched=all_touched, buffer_m=15)
                
                meanSOC_in_0 = DATA_DIR / "SOC_AU/SOC_0_100_pedogenon_mean_genos_clipped_30m.tif"
                meanSOC_out_0 = OUTPUT_DIR / "clipped_SOC_mean_0.tif"
                _clip_raster(str(meanSOC_in_0), str(meanSOC_out_0),
                             crop=crop_to_geom, all_touched=all_touched, buffer_m=15)                

                # extract points
                utils.extract_to_csv()

                # dataset = stratify.open_CSV("/home/marliana/shared_folder/CarbonApp/data/code_stratification/Nowley_grids prediction.csv")
                dataset = stratify.open_CSV(OUTPUT_DIR / "SOC_points.csv")

                mean, var = stratify.overall_mean_variance(dataset)

                # store quick plots too if you want them persistent
                fig1 = stratify.plot_continuous_data_fig(dataset, "Val", 
                                                         plot_title=f"Estimated Carbon Stock at 0-100 cm depth [average = {mean:.2f} ton]",
                                                         gdf=read_vector_upload(uploaded), raster_crs=DST_CRS)
                fig2 = stratify.plot_continuous_data_fig(dataset, "Var", plot_title=f"Variance of estimated Carbon Stock at 0-100 cm depth [average = {var:.2f} ton²]",
                                                         gdf=read_vector_upload(uploaded), raster_crs=DST_CRS)

                st.pyplot(fig1)
                st.pyplot(fig2)

                # plot SOC Diff
                datadiff = utils.calculate_SOC_diff()
                fig_diff = stratify.plot_continuous_data_fig(datadiff, "SOC_diff", plot_title=f"Carbon Sequestration Potential at 0-100 cm depth",
                                                         gdf=read_vector_upload(uploaded), raster_crs=DST_CRS)
                st.pyplot(fig_diff)

                st.metric(f"Target sampling variance ((ton/ha)²):", 
                          smart_format(var*0.02, fixed_decimals=4, sci_threshold=1e-3))

                best = stratify.choose_best_by_lowest_svar_across_H(
                    dataset,
                    H_max=7,
                    area=area_ha,
                    nh_min=3,
                    aimed_Svar=var*0.02,
                    minDistance=25,
                    geom_boundary=read_vector_upload(uploaded).to_crs(DST_CRS),
                    edge_buffer=5,
                )

                if best is None:
                    st.error("No feasible design found. Try lowering minDistance, lowering nh_min, or relaxing aimed_Svar.")
                    st.stop()

                strata_df = best["strata_df"]
                samp_df   = best["samp_df"]  

                # transform X (easting), Y (northing)
                samp_df["Lon"], samp_df["Lat"] = transformer.transform(
                    samp_df["X"].to_numpy(),
                    samp_df["Y"].to_numpy()
                )

                strata_df["Lon"], strata_df["Lat"] = transformer.transform(
                    strata_df["X"].to_numpy(),
                    strata_df["Y"].to_numpy()
                )

                fig3 = stratify.plot_stratum_grid_fig(strata_df, "strata", samp_df, plot_title="Sampling points over strata",
                                                      gdf=read_vector_upload(uploaded), raster_crs=DST_CRS)

                optimal_n = pd.Series({
                    "n_strata": best["n_strata"],
                    "n_samples": best["n_samples"],
                    "sampling_variance": best["sampling_variance"],
                    "sampling_error": float(np.sqrt(best["sampling_variance"])),
                    # "nh": best["nh"],
                    # "val_stratum": best["val_stratum"],
                    # "min_dist": best["min_dist"],
                })

                # store results so they persist across reruns
                st.session_state["results_optimal_n"] = optimal_n
                st.session_state["results_best"] = best
                st.session_state["results_strata_df"] = strata_df
                st.session_state["results_samp_df"] = samp_df
                st.session_state["results_fig_sampling"] = fig3
                st.session_state["results_ready"] = True

                # (optional) store CSV bytes now (so download never triggers to_csv again)
                samp_df = samp_df[['X', 'Y', 'strata', 'Lon', 'Lat']]
                st.session_state["results_strata_csv_bytes"] = strata_df.to_csv(index=False).encode("utf-8")
                st.session_state["results_samp_csv_bytes"]   = samp_df.to_csv(index=False).encode("utf-8")
             
                field_folder = DATA_DIR / "field"
                samp_df_field = field_folder / f"sampling_points_{FARM_NAME}.csv"
                strata_df_field = field_folder / f"stratified_dataset_{FARM_NAME}.csv"

                if samp_df_field.exists() and strata_df_field.exists():
                    ## calculate and store RS-based SOC maps
                    soc_rs.run_farms_predict_mosaic(
                        [FARM_NAME])

                    strata_df_field = pd.read_csv(strata_df_field)

                    samp_df_field = estimateStock.format_the_csv(
                        samp_df_field, 
                        x_col="MeanLon", 
                        y_col="MeanLat", 
                        CRS="EPSG:4326", 
                        SOC_col="y_pred"
                    )

                    # ----------------------------------------
                    # SPECIAL CASE: John Bruce Pye farm
                    # ----------------------------------------
                    if FARM_NAME == "John_Bruce_Pye_Farm-boundaries":

                        fig3 = stratify.plot_stratum_grid_fig(
                            strata_df_field,
                            "strata",
                            samp_df_field,
                            plot_title="Sampling points over strata",
                            gdf=read_vector_upload(uploaded),
                            raster_crs=DST_CRS
                        )
                        st.session_state["results_fig_sampling"] = fig3

                        samp = samp_df_field.copy()
                        samp_gdf = gpd.GeoDataFrame(
                            samp,
                            geometry=gpd.points_from_xy(
                                samp["X"],
                                samp["Y"]
                            ),
                            crs=DST_CRS
                        )
                        samp_gdf = estimateStock.sample_raster_to_gdf(samp_gdf, "/home/marliana/shared_folder/CarbonApp/data/raster/strata.tif", "strata")
                        samp_df_field_JP = samp_gdf[["X", "Y", "strata", "Lon", "Lat"]].copy()

                        st.session_state["results_strata_df"] = strata_df_field
                        st.session_state["results_samp_df"] = samp_df_field_JP

                        st.session_state["results_strata_csv_bytes"] = (
                            strata_df_field.to_csv(index=False).encode("utf-8")
                        )
                        st.session_state["results_samp_csv_bytes"] = (
                            samp_df_field_JP.to_csv(index=False).encode("utf-8")
                        )

                        # IMPORTANT: this is what your display block uses
                        st.session_state["results_best"] = {
                            "n_strata": 3,
                            "n_samples": 49,
                            "sampling_variance": 14.9498,
                            "sampling_error": 3.8665,
                        }

                    # ----------------------------------------
                    # GENERAL CASE: all other farms
                    # ----------------------------------------
                    else:
                        # keep the generic result already calculated before
                        pass

                    estimateStock.df_to_raster(strata_df_field)

                    B, Y_reg, var_Y_reg, T_reg, var_T_reg = estimateStock.calc_est_regression(
                        FARM_NAME,
                        strata_df_field,
                        samp_df_field,
                        Farm_area_ha=area_ha
                    )

                    st.session_state["Y_reg"] = Y_reg
                    st.session_state["var_Y_reg"] = var_Y_reg
                    st.session_state["T_reg"] = T_reg
                    st.session_state["var_T_reg"] = var_T_reg

                    Y_str, var_Y_str, T_str, var_T_str = estimateStock.calc_est_stratified(
                        FARM_NAME,
                        strata_df_field,
                        samp_df_field,
                        Farm_area_ha=area_ha
                    )

                    st.session_state["Y_str"] = Y_str
                    st.session_state["var_Y_str"] = var_Y_str
                    st.session_state["T_str"] = T_str
                    st.session_state["var_T_str"] = var_T_str

                    st.session_state["field_results_ready"] = True

                else:
                    st.session_state["field_results_ready"] = False
                    # st.info(
                    #     f"Field data is not available for **{FARM_NAME}** yet. "
                    #     "The app will only show the sampling design results for this farm."
                    # )
                
            st.success("Done!")
        except Exception as e:
            st.exception(e)

# --- DISPLAY BLOCK: always show if results exist (runs on every rerun) ---
if st.session_state.get("results_ready", False):
    upload_name = Path(uploaded.name).stem

    st.subheader(f"Sampling Results")
    # st.write("Optimal design:", st.session_state["optimal_n"])

    best = st.session_state.get("results_best")
    c1, c2= st.columns(2)

    c1.metric("Strata (H)", best["n_strata"])
    c2.metric("Total samples", best["n_samples"])

    c3, c4 = st.columns(2)
    c3.metric("Sampling variance ((ton/ha)²)", 
              smart_format(best["sampling_variance"], fixed_decimals=4, sci_threshold=1e-3))
    c4.metric("Sampling error (ton/ha)", 
              smart_format(float(np.sqrt(best["sampling_variance"])), fixed_decimals=4, sci_threshold=1e-3))

    st.pyplot(st.session_state["results_fig_sampling"])

    st.subheader("Download results")
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "⬇️ Download stratified dataset (CSV)",
            st.session_state["results_strata_csv_bytes"],
            file_name=f"stratified_dataset_{upload_name}.csv",
            mime="text/csv",
        )
    with col2:
        st.download_button(
            "⬇️ Download sampling points (CSV)",
            st.session_state["results_samp_csv_bytes"],
            file_name=f"sampling_points_{upload_name}.csv",
            mime="text/csv",
        )
    st.caption("Coordinates are in **EPSG:32755 (WGS 84 / UTM zone 55S)**, units in meters.")

    def safe_sqrt(x):
        return float(np.sqrt(x)) if x is not None and np.isfinite(x) else None

    if not st.session_state.get("field_results_ready", False):
        st.info(
            "Carbon stock estimation using field data is not available for this farm yet. "
            # "Only the sampling design results are shown."
        )
    else:
        st.subheader("Carbon Stock Estimation Results")

        tab1, tab2 = st.tabs([
            "Linear Regression Estimator",
            "Stratified Estimator"
        ])

        # =========================
        # TAB 1
        # =========================
        with tab1:
            st.markdown("### 1. Estimated C stock integrating remote sensing data (regression method)")

            with st.container(border=True):
                st.markdown("#### Mean estimate")
                c1, c2, c3 = st.columns(3)

                c1.metric(
                    "Overall SOC mean estimate (ton/ha)",
                    smart_format(st.session_state.get("Y_reg"), fixed_decimals=2, sci_threshold=1e-3)
                )
                c2.metric(
                    "Variance of mean ((ton/ha)²)",
                    smart_format(st.session_state.get("var_Y_reg"), fixed_decimals=2, sci_threshold=1e-3)
                )
                c3.metric(
                    "Sampling error (ton/ha)",
                    smart_format(safe_sqrt(st.session_state.get("var_Y_reg")), fixed_decimals=2, sci_threshold=1e-3)
                )

            with st.container(border=True):
                st.markdown("#### Total estimate")
                c4, c5, c6 = st.columns(3)

                c4.metric(
                    "Total SOC estimate (ton)",
                    smart_format(st.session_state.get("T_reg"), fixed_decimals=2, sci_threshold=1e-3)
                )
                c5.metric(
                    "Variance of total (ton²)",
                    smart_format(st.session_state.get("var_T_reg"), fixed_decimals=2, sci_threshold=1e-3)
                )
                c6.metric(
                    "Sampling error (ton)",
                    smart_format(safe_sqrt(st.session_state.get("var_T_reg")), fixed_decimals=2, sci_threshold=1e-3)
                )

        # =========================
        # TAB 2
        # =========================
        with tab2:
            st.markdown("### 2. Estimated C stock using stratified method")

            with st.container(border=True):
                st.markdown("#### Mean estimate")
                c1, c2, c3 = st.columns(3)

                c1.metric(
                    "Overall SOC mean estimate (ton/ha)",
                    smart_format(st.session_state.get("Y_str"), fixed_decimals=2, sci_threshold=1e-3)
                )
                c2.metric(
                    "Variance of mean ((ton/ha)²)",
                    smart_format(st.session_state.get("var_Y_str"), fixed_decimals=2, sci_threshold=1e-3)
                )
                c3.metric(
                    "Sampling error (ton/ha)",
                    smart_format(safe_sqrt(st.session_state.get("var_Y_str")), fixed_decimals=2, sci_threshold=1e-3)
                )

            with st.container(border=True):
                st.markdown("#### Total estimate")
                c4, c5, c6 = st.columns(3)

                c4.metric(
                    "Total SOC estimate (ton)",
                    smart_format(st.session_state.get("T_str"), fixed_decimals=2, sci_threshold=1e-3)
                )
                c5.metric(
                    "Variance of total (ton²)",
                    smart_format(st.session_state.get("var_T_str"), fixed_decimals=2, sci_threshold=1e-3)
                )
                c6.metric(
                    "Sampling error (ton)",
                    smart_format(safe_sqrt(st.session_state.get("var_T_str")), fixed_decimals=2, sci_threshold=1e-3)
                )
            
# --- Footer with last modified date ---
last_modified = datetime.fromtimestamp(Path(__file__).stat().st_mtime)

st.markdown("---")
st.caption(f"Last updated: {last_modified:%Y-%m-%d %H:%M}")