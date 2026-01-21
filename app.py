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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


st.set_page_config(page_title="Lorem Ipsum", layout="centered")
st.title("Lorem Ipsum - Sampling")

# --- Raster selection ---
raster_path = Path.home() / "shared_folder/CarbonApp/data/raster/SOC.tif"

# --- Vector upload ---
shp_zip = st.file_uploader("Shapefile (ZIP)", type=["zip"])

# --- Output options ---
out_dir = Path.home() / "shared_folder/CarbonApp/output/"

# col1, col2 = st.columns(2)
crop_to_geom = True #col1.checkbox("Crop to geometry bounds", value=True)
all_touched =  True #st.checkbox("All touched (more inclusive edges)", value=False)

run_btn = st.button("Run")

def _extract_zip_to_dir(zip_bytes: bytes, out_dir: str) -> Path:
    zpath = Path(out_dir) / "shape.zip"
    zpath.write_bytes(zip_bytes)

    with zipfile.ZipFile(zpath, "r") as zf:
        zf.extractall(out_dir)

    shp_files = list(Path(out_dir).glob("**/*.shp"))
    if not shp_files:
        raise ValueError("No .shp found in the ZIP. Ensure the zip contains .shp/.shx/.dbf/.prj.")
    return shp_files[0]

def _clip_raster(path_raster_in: str, vector_path: str, path_raster_out: str, crop: bool, all_touched: bool) -> None:
    gdf = gpd.read_file(vector_path)
    if gdf.empty:
        raise ValueError("Vector file has no features.")

    # dissolve to one geometry
    geom = gdf.geometry.unary_union

    with rasterio.open(path_raster_in) as src:
        if gdf.crs is None:
            raise ValueError("Shapefile CRS missing (.prj missing).")
        if src.crs is None:
            raise ValueError("Raster CRS missing.")
        if gdf.crs != src.crs:
            gdf2 = gdf.to_crs(src.crs)
            geom = gdf2.geometry.unary_union

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

        if out_meta.get("nodata", None) is None:
            if "float" in str(out_meta["dtype"]).lower():
                out_meta["nodata"] = -9999.0
            else:
                out_meta["nodata"] = 0

        with rasterio.open(path_raster_out, "w", **out_meta) as dst:
            dst.write(out_image)

def clear_results():
    for k in [
        "results_ready",
        "optimal_n",
        "strata_df",
        "samp_df",
        "fig_sampling",
        "fig_soc_mean",
        "fig_soc_var",
        "strata_csv_bytes",
        "samp_csv_bytes",
    ]:
        st.session_state.pop(k, None)


# --- RUN BLOCK: compute and store results ---
if run_btn:
    clear_results()
    if shp_zip is None:
        st.error("Please upload a zipped shapefile.")
    else:
        try:
            out_dir_path = Path(out_dir).expanduser()
            out_path = out_dir_path

            with tempfile.TemporaryDirectory() as tmpdir:
                shp_path = _extract_zip_to_dir(shp_zip.read(), tmpdir)

                # clip rasters
                meanSOC_in = Path.home() / "shared_folder/CarbonApp/data/SOC_AU/SOC_000_005_EV_32755.tif"
                meanSOC_out = Path.home() / "shared_folder/CarbonApp/output/clipped_SOC_mean.tif"
                _clip_raster(str(meanSOC_in), str(shp_path), str(meanSOC_out),
                             crop=crop_to_geom, all_touched=all_touched)

                varSOC_in = Path.home() / "shared_folder/CarbonApp/data/SOC_AU/SOC_000_005_VAR_32755.tif"
                varSOC_out = Path.home() / "shared_folder/CarbonApp/output/clipped_SOC_var.tif"
                _clip_raster(str(varSOC_in), str(shp_path), str(varSOC_out),
                             crop=crop_to_geom, all_touched=all_touched)

                # extract points
                utils.extract_to_csv()

                # dataset = stratify.open_CSV("/home/marliana/shared_folder/CarbonApp/data/code_stratification/Nowley_grids prediction.csv")
                dataset = stratify.open_CSV("/home/marliana/shared_folder/CarbonApp/output/SOC_points.csv")

                # store quick plots too if you want them persistent
                fig1 = stratify.plot_continuous_data_fig(dataset, "Val", plot_title="SOC mean")
                st.pyplot(fig1)
                fig2 = stratify.plot_continuous_data_fig(dataset, "Var", plot_title="SOC variance")
                st.pyplot(fig2)

                mean, var = stratify.overall_mean_variance(dataset)
                st.write(f"Overall mean SOC: {mean:.2f}")
                st.write(f"Overall SOC variance: {var:.2f}")
                st.write(f"Aimed sampling variance: {(var*0.02):.4f}")

                best = stratify.choose_global_minimum_samples(
                    dataset,
                    H_max=7,
                    nh_min=3,
                    aimed_Svar=var*0.02,
                    minDistance=50,
                )

                if best is None:
                    st.error("No feasible design found. Try lowering minDistance, lowering nh_min, or relaxing aimed_Svar.")
                    st.stop()

                strata_df = best["strata_df"]
                samp_df   = best["samp_df"]       

                fig3 = stratify.plot_stratum_grid_fig(strata_df, "strata", samp_df, plot_title="Sampling points over strata")

                optimal_n = pd.Series({
                    "n_strata": best["n_strata"],
                    "n_samples": best["n_samples"],
                    "sampling_variance": best["sampling_variance"],
                    "sampling_error": float(np.sqrt(best["sampling_variance"])),
                    "nh": best["nh"],
                    "val_stratum": best["val_stratum"],
                    "min_dist": best["min_dist"],
                })

                # store results so they persist across reruns
                st.session_state["optimal_n"] = optimal_n
                st.session_state["strata_df"] = strata_df
                st.session_state["samp_df"] = samp_df
                st.session_state["fig_sampling"] = fig3
                st.session_state["results_ready"] = True

                # (optional) store CSV bytes now (so download never triggers to_csv again)
                st.session_state["strata_csv_bytes"] = strata_df.to_csv(index=False).encode("utf-8")
                st.session_state["samp_csv_bytes"]   = samp_df.to_csv(index=False).encode("utf-8")

            st.success("Done!")
        except Exception as e:
            st.exception(e)

# --- DISPLAY BLOCK: always show if results exist (runs on every rerun) ---
if st.session_state.get("results_ready", False):

    st.subheader("Sampling result")
    st.write("Optimal design:", st.session_state["optimal_n"])
    st.session_state["fig_sampling"]

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "Download stratified_dataset.csv",
            st.session_state["strata_csv_bytes"],
            file_name="stratified_dataset.csv",
            mime="text/csv",
        )
    with col2:
        st.download_button(
            "Download sampling_points.csv",
            st.session_state["samp_csv_bytes"],
            file_name="sampling_points.csv",
            mime="text/csv",
        )
