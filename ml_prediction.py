
import sys
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from custom_transformers import (
    CorrelationReducer,
    AdaptiveMISelector
)

# The saved pipeline references __main__ classes.
# Register the original classes before loading.
sys.modules["__main__"].CorrelationReducer = CorrelationReducer
sys.modules["__main__"].AdaptiveMISelector = AdaptiveMISelector

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "models/final_fully_nested_rf_pipeline.joblib"
METADATA_PATH = BASE / "models/final_rf_metadata.json"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_metadata():
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def render_prediction_page():
    st.title("🧠 ML Prediction")
    st.caption("ThalScreen AI | Research Demonstration")

    st.warning(
        "For research demonstration only. "
        "Not intended for clinical diagnosis or medical decisions."
    )

    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        st.error("Model assets are unavailable.")
        return

    try:
        model = load_model()
        metadata = load_metadata()
    except Exception as e:
        st.error("Unable to load the trained model.")
        st.code(f"{type(e).__name__}: {e}")
        return

    feature_order = metadata["input_feature_order"]

    st.subheader("Upload 47-Feature CSV")
    st.write(
        "Upload a CSV containing the 47 formula features "
        "in the original research feature format."
    )

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        key="formula_csv_upload"
    )

    if uploaded_file is None:
        st.info("Upload a CSV to begin.")
        return

    try:
        df = pd.read_csv(uploaded_file)

        if df.empty:
            st.error("The uploaded CSV contains no rows.")
            return

        if len(df) > 100:
            st.error(
                "Maximum 100 rows per upload for this demo."
            )
            return

        missing = [
            col for col in feature_order
            if col not in df.columns
        ]

        if missing:
            st.error("Required feature columns are missing.")
            st.write(missing)
            return

        X = df.loc[:, feature_order].copy()

        for column in feature_order:
            X[column] = pd.to_numeric(
                X[column],
                errors="coerce"
            )

        if X.isna().any().any():
            st.error(
                "Input contains missing or non-numeric values."
            )
            return

        if not np.isfinite(X.to_numpy(dtype=float)).all():
            st.error(
                "Input contains infinite or invalid values."
            )
            return

        st.success(
            f"Validated {len(X)} row(s) "
            f"and {len(feature_order)} features."
        )

        st.subheader("Input Preview")
        st.dataframe(X.head(5), use_container_width=True)

        if st.button(
            "Run Research Prediction",
            type="primary",
            use_container_width=True
        ):
            predictions = model.predict(X)
            probabilities = model.predict_proba(X)

            classes = list(model.classes_)
            p0 = probabilities[:, classes.index(0)]
            p1 = probabilities[:, classes.index(1)]

            results = pd.DataFrame({
                "Predicted Class": predictions,
                "Research Screening Label": [
                    "β-Thalassemia Trait"
                    if value == 1 else "Anemia"
                    for value in predictions
                ],
                "Anemia Probability (%)": np.round(
                    p0 * 100, 2
                ),
                "βTT Probability (%)": np.round(
                    p1 * 100, 2
                )
            })

            st.subheader("Prediction Results")
            st.dataframe(
                results,
                use_container_width=True
            )

            st.download_button(
                "Download Prediction Results",
                data=results.to_csv(index=False),
                file_name="thalscreen_predictions.csv",
                mime="text/csv"
            )

            st.caption(
                "Model probabilities are not clinically "
                "validated disease probabilities."
            )

    except Exception as e:
        st.error("Prediction could not be completed.")
        st.code(f"{type(e).__name__}: {e}")
