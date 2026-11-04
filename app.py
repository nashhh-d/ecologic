import io
from datetime import datetime

import streamlit as st
from PIL import Image, UnidentifiedImageError

# This app runs on the SAME laptop as the ECOLOGIC backend.
# It imports the real analysis function; it does not fabricate predictions.
try:
    from ecologic.service import analyze_image
    BACKEND_IMPORT_ERROR = None
except Exception as exc:
    analyze_image = None
    BACKEND_IMPORT_ERROR = exc


st.set_page_config(
    page_title="ECOLOGIC | Waste Intelligence",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

FOREST = "#214E34"
SAGE = "#A9C5A0"
OFFWHITE = "#F7F8F3"
CHARCOAL = "#26332B"
MUTED = "#6B756E"

st.markdown(
    f"""
    <style>
    .stApp {{ background: {OFFWHITE}; color: {CHARCOAL}; }}
    [data-testid="stSidebar"] {{
        background: #EEF2E9 !important;
        border-right: 1px solid #DCE4D8;
        color: #26332B !important;
    }}
    /* Force readable dark text throughout sidebar navigation. */
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] *,
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"],
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] *,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] label *,
    [data-testid="stSidebar"] [role="radiogroup"],
    [data-testid="stSidebar"] [role="radio"],
    [data-testid="stSidebar"] [role="radio"] *,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span {{
        color: #26332B !important;
    }}
    [data-testid="stSidebar"] [role="radio"][aria-checked="true"] {{
        background: #DCE8D8 !important;
        border-radius: 8px;
    }}
    [data-testid="stSidebar"] [role="radio"] svg {{
        color: #214E34 !important;
        fill: #214E34 !important;
    }}
    .brand {{
        font-size: 1.7rem;
        font-weight: 800;
        letter-spacing: .08em;
        color: {FOREST};
    }}
    .brand-sub {{ color: {MUTED}; font-size: .75rem; }}
    .hero {{
        background: {FOREST};
        color: white;
        padding: 2rem;
        border-radius: 18px;
    }}
    .hero h1 {{ color: white; font-size: 2.3rem; margin: 0 0 .5rem 0; }}
    .hero p {{ color: #E5EEE4; margin-bottom: 0; }}
    .eyebrow {{
        color: {SAGE};
        font-size: .75rem;
        letter-spacing: .15em;
        font-weight: 700;
        text-transform: uppercase;
    }}
    .panel {{
        background: white;
        border: 1px solid #E1E7DD;
        border-radius: 15px;
        padding: 1.1rem 1.25rem;
    }}
    .metric-label {{ color: {MUTED}; font-size: .82rem; }}
    .metric-value {{
        color: {FOREST};
        font-size: 1.2rem;
        font-weight: 750;
        margin-top: .25rem;
    }}
    .small-note {{ color: {MUTED}; font-size: .85rem; }}
    .result-title {{
        color: {FOREST};
        font-size: 1.55rem;
        font-weight: 750;
    }}
    div.stButton > button {{
        background: {FOREST};
        color: white;
        border: 1px solid {FOREST};
        border-radius: 9px;
        font-weight: 650;
    }}
    div.stButton > button:hover {{
        background: #173B27;
        color: white;
        border-color: #173B27;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

if "history" not in st.session_state:
    st.session_state.history = []
if "latest_result" not in st.session_state:
    st.session_state.latest_result = None
if "latest_image_bytes" not in st.session_state:
    st.session_state.latest_image_bytes = None


def pretty(value):
    """Make backend category labels easier to read."""
    return str(value or "Not available").replace("_", " ").title()


def show_result(result, image=None):
    """Display a real result returned by ecologic.service.analyze_image."""
    category = result.get("predicted_category", result.get("label", "Not available"))
    confidence = float(result.get("confidence_percent", 0))
    method = result.get("recommended_method", "Not available")
    reason = result.get("recommendation_reason", "No recommendation reason was returned.")
    review = result.get("requires_review", result.get("low_confidence", False))

    if review:
        st.warning(
            "Manual review recommended. Verify the predicted category before using this recommendation."
        )

    left, right = st.columns([1, 1.5], gap="large")
    with left:
        if image is not None:
            st.image(image, caption="Uploaded waste image", use_container_width=True)

    with right:
        st.markdown('<div class="eyebrow">MODEL CLASSIFICATION</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="result-title">{pretty(category)}</div>',
            unsafe_allow_html=True,
        )
        st.metric("Model confidence", f"{confidence:.1f}%")
        st.progress(max(0.0, min(1.0, confidence / 100.0)))
        st.markdown("**Recommended processing**")
        st.write(method)
        st.write(reason)
        st.caption(
            "Image classification does not measure pH, moisture, carbon-to-nitrogen ratio, "
            "volatile solids, or microbial activity. Confidence is not a guarantee of correctness."
        )

    predictions = result.get("top_predictions") or []
    if predictions:
        st.subheader("Top predictions")
        rows = []
        for item in predictions:
            value = float(item.get("confidence", 0))
            if value <= 1:
                value *= 100
            rows.append(
                {
                    "Category": pretty(item.get("category", item.get("label"))),
                    "Confidence": f"{value:.2f}%",
                }
            )
        st.dataframe(rows, hide_index=True, use_container_width=True)


# Sidebar navigation and status
with st.sidebar:
    st.markdown(
        '<div class="brand">ECOLOGIC</div>'
        '<div class="brand-sub">SMART WASTE-TO-MICROBE CONVERTER</div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")
    page = st.radio(
        "NAVIGATION",
        ["Dashboard", "Analyze Waste", "Treatment Pathways", "Analysis History", "About"],
    )
    st.markdown("---")

    # IMPORTANT: use a normal if/else. Do not put Streamlit return values
    # into a conditional expression, as Streamlit may display object details.
    if BACKEND_IMPORT_ERROR is None:
        st.success("Backend service imported")
    else:
        st.warning("Backend import needs attention")
        st.caption("Analysis will not work until this import issue is fixed.")
        with st.expander("Import error details"):
            st.code(str(BACKEND_IMPORT_ERROR))

    st.caption("Local prototype · Same laptop")


# Dashboard
if page == "Dashboard":
    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">AI · BIOLOGICAL RESOURCE RECOVERY</div>
          <h1>Transform Waste Into Resources</h1>
          <p>Classify waste from an image and explore a suitable processing recommendation,
          while keeping missing measurements transparent.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")

    latest = st.session_state.latest_result or {}
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f'<div class="panel"><div class="metric-label">LATEST CATEGORY</div>'
            f'<div class="metric-value">{pretty(latest.get("predicted_category")) if latest else "No analysis yet"}</div></div>',
            unsafe_allow_html=True,
        )
    with col2:
        recommendation = latest.get("recommended_method", "No analysis yet") if latest else "No analysis yet"
        st.markdown(
            f'<div class="panel"><div class="metric-label">LATEST RECOMMENDATION</div>'
            f'<div class="metric-value">{recommendation}</div></div>',
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f'<div class="panel"><div class="metric-label">SESSION ANALYSES</div>'
            f'<div class="metric-value">{len(st.session_state.history)}</div></div>',
            unsafe_allow_html=True,
        )

    st.write("")
    st.subheader("How ECOLOGIC works")
    columns = st.columns(5)
    steps = [
        ("01", "Upload", "Provide a waste image."),
        ("02", "Classify", "The model predicts a category."),
        ("03", "Review", "Check confidence and warnings."),
        ("04", "Recommend", "Review the suggested method."),
        ("05", "Recover", "Consider possible resource recovery."),
    ]
    for column, (number, title, description) in zip(columns, steps):
        with column:
            st.markdown(
                f'<div class="panel"><div class="eyebrow">{number}</div>'
                f'<h3>{title}</h3><p class="small-note">{description}</p></div>',
                unsafe_allow_html=True,
            )

    st.write("")
    st.info(
        "An image alone cannot determine chemical properties or guarantee a treatment outcome. "
        "Use recommendations as decision support."
    )


# Analyze Waste
elif page == "Analyze Waste":
    st.title("Analyze Waste")
    st.write("Upload a clear image of one waste item or a small waste sample.")
    uploaded = st.file_uploader(
        "Choose a waste image",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False,
    )

    if uploaded is not None:
        image_bytes = uploaded.getvalue()
        try:
            with Image.open(io.BytesIO(image_bytes)) as check_image:
                check_image.verify()
            with Image.open(io.BytesIO(image_bytes)) as opened_image:
                image = opened_image.convert("RGB")

            st.caption(f"{uploaded.name} · {image.width} × {image.height} pixels")
            st.image(image, caption="Image preview", use_container_width=True)

            if st.button("Analyze Waste", type="primary", key="analyze_waste_button"):
                if analyze_image is None:
                    st.error(
                        "The backend analysis function could not be imported. "
                        "See the sidebar's import error details."
                    )
                else:
                    try:
                        with st.spinner("Analyzing image with the ECOLOGIC model..."):
                            result = analyze_image(image)

                        if not isinstance(result, dict):
                            raise TypeError(
                                "The analysis function did not return a dictionary. "
                                f"Received: {type(result).__name__}"
                            )

                        st.session_state.latest_result = result
                        st.session_state.latest_image_bytes = image_bytes
                        confidence = float(result.get("confidence_percent", 0))
                        st.session_state.history.insert(
                            0,
                            {
                                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "Category": pretty(
                                    result.get("predicted_category", result.get("label"))
                                ),
                                "Confidence": f"{confidence:.1f}%",
                                "Recommended method": result.get(
                                    "recommended_method", "Not available"
                                ),
                                "Status": (
                                    "Manual review"
                                    if result.get("requires_review", False)
                                    else "Completed"
                                ),
                            },
                        )
                        st.success("Analysis completed.")
                    except Exception as exc:
                        st.error(f"Analysis failed: {exc}")

            if st.session_state.latest_result is not None:
                st.divider()
                st.subheader("Latest analysis result")
                latest_image = None
                if st.session_state.latest_image_bytes:
                    latest_image = Image.open(
                        io.BytesIO(st.session_state.latest_image_bytes)
                    ).convert("RGB")
                show_result(st.session_state.latest_result, latest_image)

        except (UnidentifiedImageError, OSError, ValueError):
            st.error("This file is not a valid readable image. Upload a JPG, JPEG, or PNG.")

    else:
        st.markdown(
            '<div class="panel"><h3>Image upload tips</h3>'
            '<p class="small-note">Use good lighting, keep the waste item visible, and avoid mixing '
            'several different materials in one image when possible.</p></div>',
            unsafe_allow_html=True,
        )


# Treatment pathways
elif page == "Treatment Pathways":
    st.title("Treatment Pathway")
    st.caption(
        "This page uses the category and recommendation from your latest real ECOLOGIC analysis."
    )

    latest = st.session_state.latest_result

    if not latest:
        st.info(
            "No waste has been analyzed in this session yet. Open **Analyze Waste**, "
            "upload an image, and click **Analyze Waste**. Your category-specific treatment "
            "recommendation will appear here."
        )
    else:
        category = latest.get("predicted_category", latest.get("label", "Not available"))
        confidence = float(latest.get("confidence_percent", 0))
        method = latest.get("recommended_method", "No treatment recommendation returned")
        reason = latest.get(
            "recommendation_reason",
            "The backend did not return a reason for this recommendation.",
        )
        requires_review = latest.get("requires_review", latest.get("low_confidence", False))
        category_key = str(category).lower().replace(" ", "_")

        # Supplementary explanations are chosen by predicted category.
        # The actual method and reason always come from the backend recommender.
        category_details = {
            "food": {
                "heading": "Food waste",
                "context": "Food waste may be biodegradable, but the appropriate route depends on its composition and condition.",
                "checks": "Check contamination, moisture, acidity, and feedstock consistency before processing.",
                "potential": "Potential outputs depend on the selected process and operating conditions.",
            },
            "leaf": {
                "heading": "Leaf waste",
                "context": "Leaves are organic material that may be suitable for biological processing when appropriately prepared.",
                "checks": "Consider particle size, moisture, aeration, and the mix with other feedstocks.",
                "potential": "Potential outputs depend on the selected process and feedstock mixture.",
            },
            "wood": {
                "heading": "Wood waste",
                "context": "Wood can contain lignocellulosic material that may be slow or difficult to break down without suitable pretreatment.",
                "checks": "Check whether the material is untreated wood and assess particle size and suitability for the selected process.",
                "potential": "Processing options depend strongly on wood type and any required pretreatment.",
            },
            "paper": {
                "heading": "Paper waste",
                "context": "Paper-based material may contain fibres that can biodegrade, but coatings, inks, and contamination affect its suitability.",
                "checks": "Separate coated, laminated, wet-strength, or contaminated paper before selecting a route.",
                "potential": "The best route depends on paper composition and local recovery options.",
            },
            "plastic_bottles": {
                "heading": "Plastic bottles",
                "context": "Conventional plastic bottles are not suitable for ordinary composting or anaerobic digestion.",
                "checks": "Identify the polymer and check local recycling acceptance; empty and sort bottles according to local guidance.",
                "potential": "Material recovery may be possible when the bottle type and local recycling stream accept it.",
            },
            "plastic_bags": {
                "heading": "Plastic bags",
                "context": "Conventional plastic bags are not suitable for ordinary composting or anaerobic digestion and can contaminate organic waste streams.",
                "checks": "Separate bags from organic feedstock and check local recycling or disposal options.",
                "potential": "Recovery options depend on polymer type and local collection infrastructure.",
            },
            "metal_cans": {
                "heading": "Metal cans",
                "context": "Metal cans are not a biological feedstock and should be kept out of composting and anaerobic digestion systems.",
                "checks": "Empty and sort the can by material type according to local recycling instructions.",
                "potential": "Metal recovery may be possible through an accepted recycling stream.",
            },
            "non_biodegradable": {
                "heading": "Non-biodegradable or mixed material",
                "context": "This category should not be sent directly to biological treatment without further identification and sorting.",
                "checks": "Inspect the material and separate recyclable, hazardous, and residual fractions as appropriate.",
                "potential": "The next step depends on the actual material identified during sorting.",
            },
        }

        detail = category_details.get(
            category_key,
            {
                "heading": str(category).replace("_", " ").title(),
                "context": "Use the backend recommendation below as the starting point for further assessment.",
                "checks": "Verify the material and its condition before processing; image classification does not measure chemical properties.",
                "potential": "Potential outputs cannot be determined from the image alone.",
            },
        )

        if requires_review:
            st.warning(
                "Manual review required by the model. Verify the predicted category before acting on this recommendation."
            )

        st.markdown(
            f'<div class="panel"><div class="eyebrow">LATEST DETECTION</div>'
            f'<h2 style="color:#214E34;margin-bottom:.2rem;">{detail["heading"]}</h2>'
            f'<p style="color:#6B756E;margin-top:0;">Predicted category: {pretty(category)} '
            f'· Model confidence: {confidence:.1f}%</p></div>',
            unsafe_allow_html=True,
        )
        st.write("")

        st.subheader("Recommended next step")
        st.markdown(
            f'<div class="panel"><div class="eyebrow">FROM YOUR ECOLOGIC BACKEND</div>'
            f'<h3 style="color:#214E34;">{method}</h3>'
            f'<p>{reason}</p></div>',
            unsafe_allow_html=True,
        )

        st.write("")
        st.subheader("Why this category matters")
        st.write(detail["context"])

        left, right = st.columns(2, gap="medium")
        with left:
            st.markdown("#### Checks before processing")
            st.write(detail["checks"])
        with right:
            st.markdown("#### Potential recovery")
            st.write(detail["potential"])

        st.info(
            "These are decision-support notes, not measurements. An image classifier does not "
            "measure pH, moisture, carbon-to-nitrogen ratio, volatile solids, or microbial activity. "
            "Do not treat potential outputs as guaranteed yields."
        )

        with st.expander("View full model result"):
            st.json(latest)

# Analysis history
elif page == "Analysis History":
    st.title("Analysis History")
    st.caption("This history is stored only in the current Streamlit session, not in a database.")
    if st.session_state.history:
        st.dataframe(
            st.session_state.history,
            hide_index=True,
            use_container_width=True,
        )
        if st.button("Clear session history"):
            st.session_state.history = []
            st.session_state.latest_result = None
            st.session_state.latest_image_bytes = None
            st.rerun()
    else:
        st.info("No analyses yet. Analyze an image to create a history entry.")


# About
else:
    st.title("About ECOLOGIC")
    st.markdown(
        """
**ECOLOGIC — Smart Waste-to-Microbe Converter** explores how image-based waste classification
can support decisions about biological processing and resource recovery.

**Problem:** Mixed waste streams can make it difficult to choose an appropriate treatment route.

**Proposed solution:** A computer-vision model predicts a waste category, and a rule-based
recommender suggests a next step.

**Current limitations**
- Images do not measure moisture, pH, carbon-to-nitrogen ratio, volatile solids, or microbial activity.
- Model confidence does not guarantee correctness.
- Recommendations should be validated against actual feedstock and process measurements.
- This application does not physically introduce microorganisms or control treatment equipment.

**Future scope:** Sensor integration, measured process parameters, uncertainty handling, and validated treatment guidance.
"""
    )
    st.caption("ECOLOGIC · Local prototype")
