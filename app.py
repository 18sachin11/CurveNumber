import streamlit as st
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.io import MemoryFile
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import os
from io import BytesIO
import pandas as pd

st.set_page_config(page_title="Runoff Time-Series (SCS-CN) App", layout="wide")

st.title("Runoff Time-Series & Maximum Runoff Maps using SCS Curve Number")

st.write(
    """
    This app performs **all analysis for a single year**:
    
    1. Reads **SOIL** (OpenLandMap USDA texture) and **LULC** (ESA CCI) rasters,
    2. Computes **Curve Number (CN-II)** and **Retention S-II (mm)**,
    3. Reads a **daily rainfall CSV for one year**,
    4. Uses 5-day antecedent rainfall to assign **AMC-I / AMC-II / AMC-III** per day,
    5. Computes **daily runoff (mm)** maps and a **runoff time-series**,
    6. Finds the **day of maximum mean runoff** and shows its runoff map.
    
    All units are in **SI (mm)**. The analysis is for **one year only**.
    """
)

# -------------------------------------------------------
# 1. Upload inputs (one-year analysis only)
# -------------------------------------------------------
soil_file = st.file_uploader(
    "Upload SOIL raster (GeoTIFF, OpenLandMap USDA texture classes 1–12)",
    type=["tif", "tiff"],
)
lulc_file = st.file_uploader(
    "Upload LULC raster (GeoTIFF, ESA CCI Land Cover)",
    type=["tif", "tiff"],
)
rain_csv_file = st.file_uploader(
    "Upload daily Rainfall CSV (1 year, columns: date, rain_mm)",
    type=["csv"],
)

# Season type for AMC classification
season_type = st.selectbox(
    "Season type for AMC classification (SCS 5-day antecedent rainfall thresholds)",
    options=["Growing season", "Dormant season"],
    index=0
)
# ---- 1. Read rainfall CSV and enforce 1 year ----
df_rain = pd.read_csv(rain_csv_file)

# Normalize column names (lowercase, strip spaces)
df_rain.columns = [c.strip().lower() for c in df_rain.columns]

if "date" not in df_rain.columns or "rain_mm" not in df_rain.columns:
    st.error(
        "Rainfall CSV must have columns named 'date' and 'rain_mm' "
        "(case-insensitive). Current columns: "
        f"{list(df_rain.columns)}"
    )
    st.stop()

# Try to parse dates safely
df_rain["date"] = pd.to_datetime(
    df_rain["date"],
    errors="coerce",          # non-parsable → NaT
    infer_datetime_format=True,
    dayfirst=True,            # handles DD/MM/YYYY or DD-MM-YYYY
)

# Check if any dates failed to parse
if df_rain["date"].isna().any():
    bad_rows = df_rain[df_rain["date"].isna()]
    st.error(
        "Some rows in the 'date' column could not be parsed as dates. "
        "Please check the format (e.g., use YYYY-MM-DD or DD/MM/YYYY). "
        f"Example problematic entries:\n{bad_rows.head().to_string(index=False)}"
    )
    st.stop()

years = df_rain["date"].dt.year.unique()
if len(years) != 1:
    st.error(
        f"Rainfall CSV must contain data for exactly 1 year. Found years: {years}"
    )
    st.stop()

year_label = str(years[0])

# -------------------------------------------------------
# 2. Mappings (Soil, LULC, CN-II)
# -------------------------------------------------------

# OpenLandMap soil texture → Hydrologic Soil Group (HSG)
soil_to_hsg = {
    12: "A",  # Sand
    11: "A",  # Loamy sand
    9:  "A",  # Sandy loam

    7:  "B",  # Loam
    8:  "B",  # Silt loam
    10: "B",  # Silt

    6:  "C",  # Sandy clay loam
    4:  "C",  # Clay loam
    5:  "C",  # Silty clay loam

    1:  "D",  # Clay
    2:  "D",  # Silty clay
    3:  "D",  # Sandy clay
}
hsg_letter_to_id = {"A": 1, "B": 2, "C": 3, "D": 4}

# ESA CCI LULC → broader SCS land-use categories
cci_to_category = {
    0:  None,             # No data
    10: "cropland",
    11: "cropland",
    12: "cropland",
    20: "cropland",
    30: "mosaic_cropland",
    40: "mosaic_cropland",

    50: "forest",
    60: "forest",
    61: "forest",
    62: "forest",
    70: "forest",
    71: "forest",
    72: "forest",
    80: "forest",
    81: "forest",
    82: "forest",
    90: "forest",
    100:"forest",

    110:"grassland",
    120:"shrubland",
    121:"shrubland",
    122:"shrubland",
    130:"grassland",

    140:"barren",
    150:"barren",
    151:"barren",
    152:"barren",
    153:"barren",
    200:"barren",
    201:"barren",
    202:"barren",
    220:"barren",

    160:"water",
    170:"water",
    180:"water",
    210:"water",

    190:"urban",
}

# TR-55 CN table (AMC-II)
cn_table_ii = {
    "cropland":        {"A": 64, "B": 75, "C": 82, "D": 85},  # Row crops (good)
    "mosaic_cropland": {"A": 60, "B": 72, "C": 80, "D": 84},  # Small grain (good)
    "grassland":       {"A": 39, "B": 61, "C": 74, "D": 80},  # Pasture/grassland good
    "shrubland":       {"A": 35, "B": 56, "C": 70, "D": 77},  # Brush fair
    "forest":          {"A": 30, "B": 55, "C": 70, "D": 77},  # Woods good
    "barren":          {"A": 72, "B": 82, "C": 87, "D": 89},  # Dirt / bare
    "urban":           {"A": 98, "B": 98, "C": 98, "D": 98},  # Fully paved
    "water":           {"A":100, "B":100, "C":100, "D":100},  # Open water
}

# -------------------------------------------------------
# 3. Helper functions
# -------------------------------------------------------
def save_to_temp(uploaded_file, suffix=".tif"):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.read())
    tmp.flush()
    tmp.close()
    return tmp.name

def reproject_soil_to_lulc(soil_path, lulc_path):
    """Reproject soil raster to the grid of the LULC raster."""
    with rasterio.open(soil_path) as soil_src, rasterio.open(lulc_path) as lulc_src:
        soil = soil_src.read(1)
        soil_nodata = soil_src.nodata if soil_src.nodata is not None else 0

        soil_reproj = np.full((lulc_src.height, lulc_src.width),
                              soil_nodata, dtype=soil.dtype)

        reproject(
            source=soil,
            destination=soil_reproj,
            src_transform=soil_src.transform,
            src_crs=soil_src.crs,
            dst_transform=lulc_src.transform,
            dst_crs=lulc_src.crs,
            resampling=Resampling.nearest
        )

        return soil_reproj, soil_nodata, lulc_src.profile

def soil_to_hsg_id(soil_arr, soil_nodata):
    hsg_id = np.full_like(soil_arr, -1, dtype=np.int16)
    valid = soil_arr != soil_nodata
    for code, letter in soil_to_hsg.items():
        mask = (soil_arr == code) & valid
        hsg_id[mask] = hsg_letter_to_id[letter]
    return hsg_id

def compute_cn_ii(soil_arr, soil_nodata, lulc_arr, profile):
    """Compute CN-II array and HSG from soil+LULC."""
    hsg_id = soil_to_hsg_id(soil_arr, soil_nodata)
    cn_nodata = -9999.0
    cn_ii = np.full_like(lulc_arr, cn_nodata, dtype=np.float32)

    id_to_letter = {v: k for k, v in hsg_letter_to_id.items()}
    unl = np.unique(lulc_arr)
    unknown_codes = []

    for lc_val in unl:
        cat = cci_to_category.get(int(lc_val))
        if cat is None:
            unknown_codes.append(int(lc_val))
            continue
        lc_mask = (lulc_arr == lc_val)
        for hid, hletter in id_to_letter.items():
            sg_mask = (hsg_id == hid)
            mask = lc_mask & sg_mask
            if not np.any(mask):
                continue
            cn_val = cn_table_ii[cat][hletter]
            cn_ii[mask] = cn_val

    cn_profile = profile.copy()
    cn_profile.update(dtype="float32", nodata=cn_nodata)

    return hsg_id, cn_ii, cn_profile, sorted(set(unknown_codes))

def adjust_cn_for_amc_from_cn_ii(cn_ii):
    """Pre-compute CN-I and CN-III arrays from CN-II using SCS formulas."""
    cn_i = cn_ii.copy().astype(np.float32)
    cn_iii = cn_ii.copy().astype(np.float32)
    mask_valid = (cn_ii > 0) & (cn_ii < 100)

    # CN-I
    cn_i[mask_valid] = cn_ii[mask_valid] / (2.281 - 0.01281 * cn_ii[mask_valid])
    # CN-III
    cn_iii[mask_valid] = cn_ii[mask_valid] / (0.427 + 0.00573 * cn_ii[mask_valid])

    return cn_i, cn_iii

def compute_S_from_CN(cn):
    """Compute S (mm) from CN using SI conversion."""
    S_nodata = -9999.0
    S = np.full_like(cn, S_nodata, dtype=np.float32)
    valid = (cn > 0) & (cn < 100)
    S[valid] = (25400.0 / cn[valid]) - 254.0
    return S, S_nodata

def array_to_geotiff_bytes(arr, profile):
    with MemoryFile() as memfile:
        with memfile.open(**profile) as ds:
            ds.write(arr, 1)
        data = memfile.read()
    return BytesIO(data)

def plot_raster(arr, title, cbar_label=None, vmin=None, vmax=None,
                ticks=None, ticklabels=None):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(arr, interpolation="nearest", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.axis("off")
    cbar = fig.colorbar(im, ax=ax)
    if cbar_label:
        cbar.set_label(cbar_label)
    if ticks is not None and ticklabels is not None:
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(ticklabels)
    fig.tight_layout()
    return fig

def classify_lulc_categories(lulc_arr):
    """Map LULC codes to category IDs and return array + mapping."""
    cat_arr = np.full(lulc_arr.shape, -1, dtype=np.int16)
    codes = np.unique(lulc_arr)
    cats = []
    for code in codes:
        cat = cci_to_category.get(int(code))
        if cat is not None:
            cats.append(cat)
    cats = sorted(set(cats))
    cat_to_id = {c: i for i, c in enumerate(cats)}
    for code in codes:
        cat = cci_to_category.get(int(code))
        if cat is None:
            continue
        mask = (lulc_arr == code)
        cat_arr[mask] = cat_to_id[cat]
    return cat_arr, cat_to_id

def classify_amc_series(df: pd.DataFrame, season_type: str) -> pd.DataFrame:
    """
    Assign AMC-I/II/III based on 5-day antecedent rainfall (SCS).
    df has columns 'date' (datetime) and 'rain_mm'.
    """
    df = df.sort_values("date").reset_index(drop=True)
    p5 = df["rain_mm"].rolling(window=5, min_periods=5).sum()

    if "Growing" in season_type:
        thr1, thr2 = 35.0, 53.0  # mm (growing season)
    else:
        thr1, thr2 = 13.0, 28.0  # mm (dormant season)

    amc = np.array(["II"] * len(df), dtype=object)
    amc[p5 < thr1] = "I"
    amc[p5 > thr2] = "III"
    df["AMC"] = amc
    return df

def compute_runoff_q(P, S, S_nodata):
    """
    Compute runoff depth Q (mm) from rainfall P (mm) and S (mm).
    P is scalar (uniform rainfall), S is 2D array.
    """
    Q = np.zeros_like(S, dtype=np.float32)
    mask_valid = (S != S_nodata) & (S > 0) & (P > 0.0)

    # Q = ((P - 0.2S)^2) / (P + 0.8S) if P>0.2S
    cond = mask_valid & (P > 0.2 * S)
    num = (P - 0.2 * S)**2
    den = P + 0.8 * S
    Q[cond] = num[cond] / den[cond]
    # where P <= 0.2S, Q remains 0
    return Q

# -------------------------------------------------------
# 4. Main logic – single-year analysis
# -------------------------------------------------------
if soil_file and lulc_file and rain_csv_file:
    if st.button("Run Single-Year CN–Retention–Runoff Analysis"):
        with st.spinner("Processing..."):

            # ---- 1. Read rainfall CSV and enforce 1 year ----
            df_rain = pd.read_csv(rain_csv_file)
            if "date" not in df_rain.columns or "rain_mm" not in df_rain.columns:
                st.error("Rainfall CSV must have columns: 'date' and 'rain_mm'.")
                st.stop()

            df_rain["date"] = pd.to_datetime(df_rain["date"])
            years = df_rain["date"].dt.year.unique()
            if len(years) != 1:
                st.error(
                    f"Rainfall CSV must contain data for exactly 1 year. Found years: {years}"
                )
                st.stop()
            year_label = str(years[0])

            # AMC classification per day based on 5-day antecedent rainfall
            df_rain = classify_amc_series(df_rain, season_type)

            # ---- 2. Save rasters and compute CN-II ----
            soil_path_tmp = save_to_temp(soil_file)
            lulc_path_tmp = save_to_temp(lulc_file)

            soil_reproj, soil_nodata, _ = reproject_soil_to_lulc(
                soil_path_tmp, lulc_path_tmp
            )

            with rasterio.open(lulc_path_tmp) as lulc_src:
                lulc_arr = lulc_src.read(1)
                profile = lulc_src.profile

            hsg_id, cn_ii, cn_profile, unknown_codes = compute_cn_ii(
                soil_reproj, soil_nodata, lulc_arr, profile
            )

            # Pre-compute CN-I and CN-III
            cn_i, cn_iii = adjust_cn_for_amc_from_cn_ii(cn_ii)

            # Retention S-II (mm) from CN-II for base map
            S_ii, S_nodata = compute_S_from_CN(cn_ii)
            S_profile = cn_profile.copy()
            S_profile.update(dtype="float32", nodata=S_nodata)

        if unknown_codes:
            st.warning(
                f"These CCI LULC codes were not mapped and remain NoData in CN: {unknown_codes}"
            )

        # ------------------------------------------------
        # 5. CN-II & S-II maps (for this year)
        # ------------------------------------------------
        st.subheader(f"Curve Number & Retention Maps (Year: {year_label})")

        col1, col2 = st.columns(2)

        # LULC categories map
        lulc_cat_arr, cat_to_id = classify_lulc_categories(lulc_arr)
        cat_ids = list(cat_to_id.values())
        cat_names = list(cat_to_id.keys())

        with col1:
            st.markdown("**LULC (categories)**")
            fig_lulc = plot_raster(
                lulc_cat_arr,
                f"LULC Categories ({year_label})",
                cbar_label="Category",
                ticks=cat_ids,
                ticklabels=cat_names,
            )
            st.pyplot(fig_lulc)

        with col2:
            st.markdown("**Hydrologic Soil Group (HSG)**")
            hsg_ticks = [1, 2, 3, 4]
            hsg_labels = ["A", "B", "C", "D"]
            fig_hsg = plot_raster(
                hsg_id,
                "HSG",
                cbar_label="HSG",
                ticks=hsg_ticks,
                ticklabels=hsg_labels,
            )
            st.pyplot(fig_hsg)

        col3, col4 = st.columns(2)

        with col3:
            st.markdown("**Curve Number CN-II (Base)**")
            fig_cn = plot_raster(
                cn_ii,
                f"Curve Number CN-II ({year_label})",
                cbar_label="CN-II",
                vmin=60,
                vmax=100,
            )
            st.pyplot(fig_cn)

        with col4:
            st.markdown("**Retention S-II (mm)**")
            fig_S = plot_raster(
                S_ii,
                f"Retention S-II (mm) ({year_label})",
                cbar_label="S-II (mm)",
            )
            st.pyplot(fig_S)

        # ------------------------------------------------
        # 6. Daily runoff time-series for this year
        # ------------------------------------------------
        st.subheader("Runoff Time-Series (Daily, using AMC from rainfall)")

        runoff_means = []
        runoff_dates = []
        amc_list = []
        max_runoff_mean = -1.0
        max_runoff_date = None
        max_runoff_grid = None

        for _, row in df_rain.iterrows():
            date = row["date"]
            P = row["rain_mm"]
            AMC = row["AMC"]

            if AMC == "I":
                cn_day = cn_i
            elif AMC == "III":
                cn_day = cn_iii
            else:
                cn_day = cn_ii

            S_day, _ = compute_S_from_CN(cn_day)
            Q_day = compute_runoff_q(P, S_day, S_nodata)

            mask_valid = (cn_day > 0) & (cn_day < 100)
            if np.any(mask_valid):
                mean_Q = float(np.nanmean(Q_day[mask_valid]))
            else:
                mean_Q = 0.0

            runoff_means.append(mean_Q)
            runoff_dates.append(date)
            amc_list.append(AMC)

            if mean_Q > max_runoff_mean:
                max_runoff_mean = mean_Q
                max_runoff_date = date
                max_runoff_grid = Q_day.copy()

        df_runoff = pd.DataFrame(
            {"date": runoff_dates, "runoff_mm": runoff_means, "AMC": amc_list}
        ).set_index("date")

        st.line_chart(df_runoff[["runoff_mm"]])

        st.markdown(
            f"**Maximum mean runoff** occurs on **{max_runoff_date.date()}**, "
            f"with mean runoff ≈ **{max_runoff_mean:.1f} mm**."
        )

        # ------------------------------------------------
        # 7. Map of runoff on max-runoff day (for this year)
        # ------------------------------------------------
        st.subheader(f"Runoff Map on Maximum Runoff Day ({max_runoff_date.date()})")

        fig_Qmax = plot_raster(
            max_runoff_grid,
            f"Runoff (mm) on {max_runoff_date.date()}",
            cbar_label="Runoff (mm)",
        )
        st.pyplot(fig_Qmax)

        # ------------------------------------------------
        # 8. Downloads: CN-II and S-II GeoTIFFs
        # ------------------------------------------------
        st.subheader("Download CN-II & Retention S-II GeoTIFFs (Single Year)")

        cn_bytes = array_to_geotiff_bytes(cn_ii, cn_profile)
        st.download_button(
            label=f"Download CN-II GeoTIFF ({year_label})",
            data=cn_bytes,
            file_name=f"CurveNumber_CNII_{year_label}.tif",
            mime="image/tiff",
        )

        S_bytes = array_to_geotiff_bytes(S_ii, S_profile)
        st.download_button(
            label=f"Download Retention S-II (mm) GeoTIFF ({year_label})",
            data=S_bytes,
            file_name=f"Retention_SII_mm_{year_label}.tif",
            mime="image/tiff",
        )

        # Clean temp files
        try:
            os.remove(soil_path_tmp)
            os.remove(lulc_path_tmp)
        except Exception:
            pass

else:
    st.info(
        "Please upload SOIL raster, LULC raster, and daily Rainfall CSV (exactly 1 year) to run the analysis."
    )
