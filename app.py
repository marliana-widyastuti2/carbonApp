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

from pyproj import CRS

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"


st.set_page_config(page_title="SamplingApp", layout="centered")
st.title("Optimised Sampling Design")

# --- Vector upload ---
uploaded = st.file_uploader(
    "Upload Farm boundary (ZIP Shapefile / GeoJSON / KML / KMZ)",
    type=["zip", "geojson", "json", "kml", "kmz"]
)


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
            return gpd.read_file(shp_files[0])

        # ---- KMZ (zip containing KML) ----
        if name.endswith(".kmz"):
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
                return gpd.read_file(kml_path, driver="KML")
            except TypeError:
                return gpd.read_file(kml_path)

        # ---- KML ----
        if name.endswith(".kml"):
            try:
                return gpd.read_file(in_path, driver="KML")
            except TypeError:
                return gpd.read_file(in_path)

        # ---- GeoJSON / JSON ----
        if name.endswith(".geojson") or name.endswith(".json"):
            return gpd.read_file(in_path)

        raise ValueError("Unsupported file type.")

def _calculate_area_ha() -> float:
    gdf = read_vector_upload(uploaded)
    DST_CRS = CRS.from_epsg(32755)
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

def _clip_raster(path_raster_in: str, path_raster_out: str, crop: bool, all_touched: bool) -> None:
    gdf = read_vector_upload(uploaded)
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
    if uploaded is None:
        st.error("Please upload a file.")
    else:
        try:
            out_dir_path = Path(out_dir).expanduser()
            out_path = out_dir_path

            with tempfile.TemporaryDirectory() as tmpdir:

                ## area calc (not used in sampling, just informative)
                area_ha = _calculate_area_ha()
                st.metric(f"Area of Farm (ha)", f"{area_ha:.2f}")

                # clip rasters
                meanSOC_in = DATA_DIR / "SOC_AU/SOC_000_005_EV_32755.tif"
                meanSOC_out = OUTPUT_DIR / "clipped_SOC_mean.tif"
                _clip_raster(str(meanSOC_in), str(meanSOC_out),
                             crop=crop_to_geom, all_touched=all_touched)

                varSOC_in = DATA_DIR / "SOC_AU/SOC_000_005_VAR_32755.tif"
                varSOC_out = OUTPUT_DIR / "clipped_SOC_var.tif"
                _clip_raster(str(varSOC_in), str(varSOC_out),
                             crop=crop_to_geom, all_touched=all_touched)

                # extract points
                utils.extract_to_csv()

                # dataset = stratify.open_CSV("/home/marliana/shared_folder/CarbonApp/data/code_stratification/Nowley_grids prediction.csv")
                dataset = stratify.open_CSV(OUTPUT_DIR / "SOC_points.csv")

                mean, var = stratify.overall_mean_variance(dataset)

                # store quick plots too if you want them persistent
                fig1 = stratify.plot_continuous_data_fig(dataset, "Val", plot_title=f"Estimated SOC at 0-5 cm depth [average = {mean:.2f}%]")
                fig2 = stratify.plot_continuous_data_fig(dataset, "Var", plot_title=f"Variance of estimated SOC at 0-5 cm depth [average = {var:.2f}%²]")

                st.pyplot(fig1)
                st.pyplot(fig2)

                st.metric(f"Target sampling variance (%²):", f"{var*0.02:.4f}")

                best = stratify.choose_global_minimum_samples_with_fallback(
                    dataset,
                    H_max=6,
                    nh_min_start=3,
                    aimed_Svar=var*0.02,
                    minDistance=50,
                )

                if best is None:
                    st.error("No feasible design found. Try lowering minDistance, lowering nh_min, or relaxing aimed_Svar.")
                    st.stop()

                strata_df = best["strata_df"]
                samp_df   = best["samp_df"] 
                # Extra info for Streamlit UI:
                used_nh_min = best["used_nh_min"]      

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
                st.session_state["best"] = best
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
    # st.write("Optimal design:", st.session_state["optimal_n"])

    best = st.session_state.get("best")
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Strata (H)", best["n_strata"])
    c2.metric("Total samples", best["n_samples"])
    c3.metric("Sampling variance (%²)", f'{best["sampling_variance"]:.4f}')
    c4.metric("Sampling error (%)", f'{float(np.sqrt(best["sampling_variance"])):.3f}')

    st.session_state["fig_sampling"]


    st.subheader("Download results")
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "⬇️ Download stratified dataset (CSV)",
            st.session_state["strata_csv_bytes"],
            file_name="stratified_dataset.csv",
            mime="text/csv",
        )
    with col2:
        st.download_button(
            "⬇️ Download sampling points (CSV)",
            st.session_state["samp_csv_bytes"],
            file_name="sampling_points.csv",
            mime="text/csv",
        )
    st.caption("Coordinates are in **EPSG:32755 (WGS 84 / UTM zone 55S)**, units in meters.")
