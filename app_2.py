"""app_2.py

Streamlit app for House Price Prediction.

The previously existing file was corrupted (embedded JSON / invalid Python).
This version is a valid Streamlit script that uses the trained model + scaler
from the House_price_prediction project directory.

Expected files (relative to current working directory when running `streamlit run app_2.py`):
- house_price_model.pkl
- scaler.pkl

If you run from inside the House_price_prediction folder, these files should be
present there.
"""

import os
import pickle

import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide",
)

st.title("🏠 House Price Prediction")

with st.container():
    st.markdown(
        """
### Enter key house details
Fill only the important fields below. The app will internally build the full
feature vector (one-hot encoding + scaling) to match the trained model.
"""
    )


# --- Resolve model/scaler paths ---
# Support running app_2.py either from the CAPSTONE root or from the
# House_price_prediction folder.
BASE_DIR = os.path.dirname(__file__)
CANDIDATE_DIRS = [
    BASE_DIR,  # CAPSTONE PROJECTS root (if pkl files copied here)
    os.path.join(BASE_DIR, "House_price_prediction"),
]

MODEL_PATH = None
SCALER_PATH = None
for d in CANDIDATE_DIRS:
    mp = os.path.join(d, "house_price_model.pkl")
    sp = os.path.join(d, "scaler.pkl")
    if os.path.exists(mp) and os.path.exists(sp):
        MODEL_PATH = mp
        SCALER_PATH = sp
        break

if not MODEL_PATH or not SCALER_PATH:
    st.error(
        "Could not find required model files. Looked for house_price_model.pkl and scaler.pkl in: "
        + ", ".join(CANDIDATE_DIRS)
        + "\n\nCopy the pkl files into the House_price_prediction folder (recommended), "
        + "or run from the folder where they exist."
    )
    st.stop()


# Load model and scaler
model = pickle.load(open(MODEL_PATH, "rb"))
scaler = pickle.load(open(SCALER_PATH, "rb"))

if not hasattr(scaler, "feature_names_in_"):
    st.error("Scaler is missing feature_names_in_. Cannot align features reliably.")
    st.stop()

required_features = list(scaler.feature_names_in_)


# ---------------------------
# UI: key/important features
# ---------------------------

st.subheader("🏡 Property basics")
col1, col2, col3 = st.columns(3)

with col1:
    overall_qual = st.number_input("OverallQual (1-10)", min_value=1, max_value=10, value=7, step=1)
    overall_cond = st.number_input("OverallCond (1-10)", min_value=1, max_value=10, value=5, step=1)

with col2:
    year_built = st.number_input("YearBuilt", min_value=1800, max_value=2100, value=2003, step=1)
    year_remod_add = st.number_input("YearRemodAdd", min_value=0, max_value=2100, value=2003, step=1)

with col3:
    gr_liv_area = st.number_input("GrLivArea (above-ground living area)", min_value=100, max_value=20000, value=1710, step=10)
    lot_area = st.number_input("LotArea", min_value=500, max_value=200000, value=8450, step=100)

st.divider()

st.subheader("📍 Location")
col4, col5 = st.columns(2)
with col4:
    neighborhood = st.selectbox(
        "Neighborhood",
        options=[
            "CollgCr",
            "Veenker",
            "Crawfor",
            "NoRidge",
            "Mitchel",
            "NAmes",
            "Edwards",
            "Somerst",
        ],
        index=0,
    )
with col5:
    ms_zoning = st.selectbox("MSZoning", options=["RL", "RM", "FV", "RH"], index=0)

st.divider()

st.subheader("🏗️ Structure")
col6, col7, col8 = st.columns(3)
with col6:
    house_style = st.selectbox(
        "HouseStyle",
        options=["2Story", "1Story", "1.5Fin", "1.5Unf", "SFoyer", "2.5Unf", "2.5Fin"],
        index=0,
    )
    bldg_type = st.selectbox("BldgType", options=["1Fam", "2fmCon", "Duplex", "TwnhsE", "Twnhs"], index=0)

with col7:
    foundation = st.selectbox("Foundation", options=["PConc", "CBlock", "BrkTil", "Wood", "Slab"], index=0)
    exter_qual = st.selectbox("ExterQual", options=["Gd", "TA", "Ex", "Fa"], index=0)

with col8:
    exter_cond = st.selectbox("ExterCond", options=["TA", "Gd", "Ex", "Fa"], index=0)
    lot_frontage = st.number_input("LotFrontage", min_value=0, max_value=200, value=65, step=1)

st.divider()

st.subheader("🚗 Garage & rooms")
col9, col10, col11 = st.columns(3)
with col9:
    garage_cars = st.number_input("GarageCars", min_value=0, max_value=10, value=2, step=1)
    garage_area = st.number_input("GarageArea (approx)", min_value=0, max_value=2000, value=540, step=10)

with col10:
    bedroom_abv_gr = st.number_input("BedroomAbvGr", min_value=0, max_value=10, value=3, step=1)
    full_bath = st.number_input("FullBath", min_value=0, max_value=5, value=2, step=1)
    half_bath = st.number_input("HalfBath", min_value=0, max_value=5, value=1, step=1)

with col11:
    fireplaces = st.number_input("Fireplaces", min_value=0, max_value=5, value=0, step=1)
    fireplace_qu = st.selectbox("FireplaceQu", options=["None", "Gd", "TA"], index=0)

st.divider()

st.subheader("✨ Optional extras (keep defaults if unsure)")
col12, col13 = st.columns(2)
with col12:
    open_porch_sf = st.number_input("OpenPorchSF", min_value=0, max_value=5000, value=61, step=10)
    wood_deck_sf = st.number_input("WoodDeckSF", min_value=0, max_value=5000, value=0, step=10)

with col13:
    pool_area = st.number_input("PoolArea", min_value=0, max_value=2000, value=0, step=10)
    screen_porch = st.number_input("ScreenPorch", min_value=0, max_value=2000, value=0, step=10)

# Main action
predict_btn = st.button("🔮 Predict price")


def build_input_row():
    """Build a raw input row for one-hot encoding.

    The model/scaler expects a specific expanded feature set. We create a
    reasonable baseline for fields not covered by the UI.
    """

    return {
        "Id": 1,
        "MSSubClass": 60,
        "MSZoning": ms_zoning,
        "LotFrontage": lot_frontage,
        "LotArea": lot_area,
        "Street": "Pave",
        "Alley": "NA",
        "LotShape": "Reg",
        "LandContour": "Lvl",
        "Utilities": "AllPub",
        "LotConfig": "Inside",
        "LandSlope": "Gtl",
        "Neighborhood": neighborhood,
        "Condition1": "Norm",
        "Condition2": "Norm",
        "BldgType": bldg_type,
        "HouseStyle": house_style,
        "OverallQual": overall_qual,
        "OverallCond": overall_cond,
        "YearBuilt": year_built,
        "YearRemodAdd": year_remod_add,
        "RoofStyle": "Gable",
        "RoofMatl": "CompShg",
        "Exterior1st": "VinylSd",
        "Exterior2nd": "VinylSd",
        "MasVnrType": "None",
        "MasVnrArea": 0,
        "ExterQual": exter_qual,
        "ExterCond": exter_cond,
        "Foundation": foundation,
        "BsmtQual": "None",
        "BsmtCond": "None",
        "BsmtExposure": "None",
        "BsmtFinType1": "None",
        "BsmtFinSF1": 0,
        "BsmtFinType2": "None",
        "BsmtFinSF2": 0,
        "BsmtUnfSF": 0,
        "TotalBsmtSF": 0,
        "Heating": "GasA",
        "HeatingQC": "TA",
        "CentralAir": "Y",
        "Electrical": "SBrkr",
        "1stFlrSF": gr_liv_area,
        "2ndFlrSF": 0,
        "LowQualFinSF": 0,
        "GrLivArea": gr_liv_area,
        "BsmtFullBath": 0,
        "BsmtHalfBath": 0,
        "FullBath": full_bath,
        "HalfBath": half_bath,
        "BedroomAbvGr": bedroom_abv_gr,
        "KitchenAbvGr": 1,
        "KitchenQual": "TA",
        "TotRmsAbvGrd": 0,
        "Functional": "Typ",
        "Fireplaces": fireplaces,
        "FireplaceQu": "None" if fireplaces == 0 or fireplace_qu == "None" else fireplace_qu,
        "GarageType": "Attchd",
        "GarageYrBlt": 2003,
        "GarageFinish": "RFn",
        "GarageCars": garage_cars,
        "GarageArea": garage_area,
        "GarageQual": "TA",
        "GarageCond": "TA",
        "PavedDrive": "Y",
        "WoodDeckSF": wood_deck_sf,
        "OpenPorchSF": open_porch_sf,
        "EnclosedPorch": 0,
        "3SsnPorch": 0,
        "ScreenPorch": screen_porch,
        "PoolArea": pool_area,
        "PoolQC": "None",
        "Fence": "None",
        "MiscFeature": "None",
        "MiscVal": 0,
        "MoSold": 2,
        "YrSold": 2008,
        "SaleType": "WD",
        "SaleCondition": "Normal",
    }


if predict_btn:
    try:
        input_row = build_input_row()
        df = pd.DataFrame([input_row])

        # One-hot encode categoricals
        df_encoded = pd.get_dummies(df, drop_first=True)

        # Align to training features exactly
        for col in required_features:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
        df_encoded = df_encoded[required_features]

        X_scaled = scaler.transform(df_encoded)
        pred = model.predict(X_scaled)[0]

        st.success(f"Predicted House Price: ${pred:,.0f}")

        with st.expander("Show the feature alignment (debug)"):
            st.write("Input (raw):")
            st.dataframe(df)
            st.write("Encoded row shape:", df_encoded.shape)
    except Exception as e:
        st.error(
            "Prediction failed. Feature names/scaling alignment may not match.\n\n"
            f"Error details: {e}"
        )

