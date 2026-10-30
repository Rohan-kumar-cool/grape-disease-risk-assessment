
from pathlib import Path

import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image


from src.risk_fis import calculate_risk

# ============================================================
# PROJECT PATHS AND MODEL CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

IMAGE_MODEL_PATH = (
    BASE_DIR
    / "model_outputs_grouped"
    / "mobilenetv2_ngld_best.keras"
)

IMAGE_SIZE = (224, 224)

CLASS_NAMES = [
    "Downy Mildew",
    "Healthy Leaves",
    "Powdery Mildew",
]


# ============================================================
# STREAMLIT PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Grape Disease Risk Assessment",
    page_icon="🍇",
    layout="wide",
)

st.title("🍇 Grape Disease Detection and Risk Assessment")

st.write(
    """
    Upload a grape leaf image to identify its likely condition.
    You can also enter five-day environmental observations to
    assess disease-favouring weather conditions and calculate a
    prototype Mamdani risk index.
    """
)

st.warning(
    "Research prototype: the image classifier identifies visual "
    "patterns, while the environmental rules and Mamdani risk index "
    "provide supporting assessments. The risk index is not a validated "
    "probability of disease worsening."
)


# ============================================================
# LOAD IMAGE CLASSIFICATION MODEL
# ============================================================

@st.cache_resource
def load_image_model():
    """Load and cache the existing trained image model."""

    if not IMAGE_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Image model not found: {IMAGE_MODEL_PATH}"
        )

    return tf.keras.models.load_model(
        str(IMAGE_MODEL_PATH),
        compile=False,
    )


# ============================================================
# EXISTING PROVIDER-DEFINED WEATHER RULES
# ============================================================

def assess_weather(temp, humidity, rainfall, wet_days):
    """
    Assess weather conditions using the existing provider rules.

    Rainfall units are not specified by the source data.
    Night dew-point depression is not collected by the form.
    Missing values are not silently replaced with zero.
    """

    results = {}

    # --------------------------------------------------------
    # Powdery mildew rule:
    # 16 <= temperature <= 31
    # 45 <= relative humidity <= 90
    # --------------------------------------------------------

    if temp is None or humidity is None:
        results["Powdery mildew"] = (
            "Insufficient data",
            "Temperature and humidity are required.",
        )

    elif (
        16 <= temp <= 31
        and 45 <= humidity <= 90
    ):
        results["Powdery mildew"] = (
            "Favourable",
            "Temperature and humidity satisfy the provider's "
            "powdery mildew weather rule.",
        )

    else:
        results["Powdery mildew"] = (
            "Not favourable",
            "Temperature or humidity is outside the "
            "provider's powdery mildew conditions.",
        )

    # --------------------------------------------------------
    # Downy mildew rule:
    # temperature <= 23
    # humidity >= 70
    # AND a moisture condition:
    # rainfall > 0.05 OR wet days >= 2 OR night DPD <= 1.5
    #
    # Night DPD is not collected by the form.
    # --------------------------------------------------------

    if temp is None or humidity is None:
        results["Downy mildew"] = (
            "Insufficient data",
            "Temperature and humidity are required.",
        )

    elif temp > 23 or humidity < 70:
        results["Downy mildew"] = (
            "Not favourable",
            "Temperature or humidity does not satisfy the "
            "provider's downy mildew weather rule.",
        )

    elif rainfall is None and wet_days is None:
        results["Downy mildew"] = (
            "Insufficient data",
            "Rainfall and wet-day observations are missing. "
            "Night dew-point depression is also not collected.",
        )

    elif (
        (rainfall is not None and rainfall > 0.05)
        or (wet_days is not None and wet_days >= 2)
    ):
        results["Downy mildew"] = (
            "Favourable",
            "Temperature, humidity and at least one recorded "
            "moisture condition satisfy the provider's rule.",
        )

    else:
        results["Downy mildew"] = (
            "Insufficient data",
            "The recorded moisture conditions do not establish "
            "favourability, and night dew-point depression is "
            "not available.",
        )

    return results


# ============================================================
# DISPLAY WEATHER RESULTS
# ============================================================

def display_weather_results(results):
    st.subheader("Environmental weather assessment")

    for disease, (status, explanation) in results.items():
        with st.container(border=True):
            st.markdown(f"**{disease}**")

            if status == "Favourable":
                st.success(status)
            elif status == "Not favourable":
                st.info(status)
            else:
                st.warning(status)

            st.write(explanation)


# ============================================================
# MAMDANI RISK FIS DISPLAY
# ============================================================

def display_risk_assessment(
    predicted_class,
    confidence,
    temperature,
    humidity,
    rainfall,
    wet_days,
):
    st.subheader("Mamdani disease risk index")

    if predicted_class == "Healthy Leaves":
        st.info(
            "The image classifier predicted Healthy Leaves. "
            "The disease-specific risk index is not calculated "
            "for this prediction. A healthy-looking image does "
            "not rule out early or visually undetectable disease."
        )
        return

    try:
        risk = calculate_risk(
            disease=predicted_class,
            confidence=confidence,
            temperature=temperature,
            humidity=humidity,
            rainfall=rainfall,
            wet_days=wet_days,
        )

    except (ValueError, TypeError) as exc:
        st.error(f"Risk assessment could not be calculated: {exc}")
        return

    if risk["risk_index"] is None:
        st.warning("Risk index unavailable")
        st.write(risk["message"])
        return

    index = risk["risk_index"]
    category = risk["risk_category"]

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Prototype risk index", f"{index:.2f}/100")

    with col2:
        st.metric("Risk category", category)

    st.progress(
        int(np.clip(round(index), 0, 100))
    )

    if "environmental_match" in risk:
        st.metric(
            "Environmental match",
            f"{risk['environmental_match']:.2f}%",
        )

    st.caption(risk["message"])

    st.info(
        "This index is an expert-designed prototype. It is not a "
        "validated probability, a confirmed disease-severity score, "
        "or a validated prediction of future disease worsening."
    )


# ============================================================
# USER INPUTS
# ============================================================

st.header("1. Environmental observations")

st.caption(
    "Enter observations for the five-day period. Leave unknown "
    "values blank. Rainfall uses the source dataset's units, "
    "which are not specified."
)

with st.form("assessment_form"):

    col1, col2 = st.columns(2)

    with col1:
        temp_5d = st.number_input(
            "Five-day average temperature (°C)",
            min_value=-10.0,
            max_value=50.0,
            value=None,
            step=0.5,
            format="%.2f",
        )

        rh_5d = st.number_input(
            "Five-day average relative humidity (%)",
            min_value=0.0,
            max_value=100.0,
            value=None,
            step=1.0,
            format="%.2f",
        )

    with col2:
        rain_5d = st.number_input(
            "Five-day rainfall (dataset units)",
            min_value=0.0,
            value=None,
            step=0.01,
            format="%.3f",
        )

        wet_days_5d = st.number_input(
            "Number of wet days (out of 5)",
            min_value=0,
            max_value=5,
            value=None,
            step=1,
            format="%d",
        )

    st.header("2. Grape leaf image")

    uploaded_file = st.file_uploader(
        "Upload a grape leaf image",
        type=["jpg", "jpeg", "png"],
    )

    submitted = st.form_submit_button(
        "Assess grape leaf",
        type="primary",
    )


# ============================================================
# RUN ASSESSMENT
# ============================================================

if submitted:

    if uploaded_file is None:
        st.error("Please upload a grape leaf image first.")

    else:
        try:
            image = Image.open(uploaded_file).convert("RGB")

        except Exception:
            st.error(
                "The uploaded file could not be read as an image."
            )
            st.stop()

        st.header("3. Image classification")

        image_col, result_col = st.columns([1, 1])

        with image_col:
            st.image(
                image,
                caption="Uploaded grape leaf",
                use_container_width=True,
            )

        try:
            image_model = load_image_model()

            processed_image = image.resize(IMAGE_SIZE)

            image_array = np.asarray(
                processed_image,
                dtype=np.float32,
            )

            image_array = (
                tf.keras.applications.mobilenet_v2.preprocess_input(
                    image_array
                )
            )

            image_array = np.expand_dims(
                image_array,
                axis=0,
            )

            predictions = image_model.predict(
                image_array,
                verbose=0,
            )[0]

            if len(predictions) != len(CLASS_NAMES):
                st.error(
                    "The model output does not match the configured "
                    "class list. Check the model's class mapping."
                )
                st.stop()

            predicted_index = int(np.argmax(predictions))
            predicted_class = CLASS_NAMES[predicted_index]

            confidence = (
                float(predictions[predicted_index]) * 100.0
            )

        except Exception as exc:
            st.error(
                f"Image classification failed: {exc}"
            )
            st.stop()

        with result_col:
            st.subheader("Prediction")

            st.write(f"**Predicted class:** {predicted_class}")

            st.metric(
                "Model confidence",
                f"{confidence:.2f}%",
            )

            st.progress(
                int(np.clip(round(confidence), 0, 100))
            )

            st.caption(
                "Model confidence is not the same as the "
                "probability that the disease will worsen."
            )

            with st.expander("View class scores"):
                for class_name, score in zip(
                    CLASS_NAMES,
                    predictions,
                ):
                    st.write(
                        f"**{class_name}:** "
                        f"{float(score) * 100:.2f}%"
                    )

        # ----------------------------------------------------
        # ENVIRONMENTAL ASSESSMENT
        # ----------------------------------------------------

        weather_results = assess_weather(
            temp_5d,
            rh_5d,
            rain_5d,
            wet_days_5d,
        )

        display_weather_results(weather_results)

        # ----------------------------------------------------
        # DISEASE-SPECIFIC MAMDANI RISK INDEX
        # ----------------------------------------------------

        display_risk_assessment(
            predicted_class=predicted_class,
            confidence=confidence,
            temperature=temp_5d,
            humidity=rh_5d,
            rainfall=rain_5d,
            wet_days=wet_days_5d,
        )

        # ----------------------------------------------------
        # SUMMARY
        # ----------------------------------------------------

        st.header("4. Assessment summary")

        st.write(f"**Image prediction:** {predicted_class}")
        st.write(f"**Image confidence:** {confidence:.2f}%")

        st.write(
            "Environmental conditions and the image prediction "
            "are displayed separately from the prototype risk index."
        )

        st.caption(
            "Use this output for research and demonstration only. "
            "Field observations and expert diagnosis are needed "
            "to validate actual disease progression."
        )

else:
    st.info(
        "Enter any available environmental observations, upload "
        "a grape leaf image, and select **Assess grape leaf**."
    )
