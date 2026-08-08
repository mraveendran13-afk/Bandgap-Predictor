"""
Bandgap Predictor — Streamlit app
Portfolio project by Amba Prasad
"""
import json
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from pymatgen.core import Composition

st.set_page_config(page_title="Bandgap Predictor", page_icon="⚛️", layout="wide")

# ---- Model definition (must match training) ----
class BandgapNN(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.fc1 = nn.Linear(n_features, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.2)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x))); x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x))); x = self.dropout(x)
        x = F.relu(self.fc3(x))
        return self.fc4(x).squeeze(-1)


@st.cache_resource
def load_model_and_featurizer():
    with open("bandgap_featurizer.pkl", "rb") as f:
        blob = pickle.load(f)
    ep = blob["featurizer"]
    scaler = blob["scaler"]
    n_features = blob["n_features"]
    
    model = BandgapNN(n_features)
    model.load_state_dict(torch.load("bandgap_model.pth", map_location="cpu", weights_only=True))
    model.eval()
    return model, ep, scaler


@st.cache_data
def load_metrics():
    with open("bandgap_metrics.json") as f:
        return json.load(f)


@st.cache_resource
def load_delta_model():
    try:
        with open("delta_model.pkl", "rb") as f:
            blob = pickle.load(f)
        return blob
    except FileNotFoundError:
        return None


@st.cache_data
def load_delta_metrics():
    try:
        with open("delta_metrics.json") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


METAL_GATE_THRESHOLD = 0.15  # below this, treat as metal — skip correction


def predict_bandgap(formula, model, ep, scaler, delta_blob=None):
    """Predict bandgap for a chemical formula string.
    Returns (pbe_pred, corrected_pred_or_None, error_or_None)."""
    try:
        comp = Composition(formula)
    except Exception as e:
        return None, None, f"Invalid formula: {e}"
    
    try:
        feats = np.array(ep.featurize(comp), dtype=np.float32).reshape(1, -1)
    except Exception as e:
        return None, None, f"Featurization failed: {e}"
    
    if np.isnan(feats).any():
        return None, None, "Could not featurize (missing element data — check formula)"
    
    feats_s = scaler.transform(feats)
    with torch.no_grad():
        pred = model(torch.tensor(feats_s, dtype=torch.float32)).item()
    pred = max(0.0, pred)
    
    corrected = None
    if delta_blob is not None and pred >= METAL_GATE_THRESHOLD:
        # Metal gate: only apply HSE06 correction to non-metallic predictions.
        # The delta model was trained mostly on semiconductors/insulators and
        # over-corrects near-zero-gap metals if applied unconditionally.
        if delta_blob["type"] == "rf":
            delta = delta_blob["model"].predict(feats_s)[0]
            corrected = max(0.0, pred + delta)
    elif delta_blob is not None and pred < METAL_GATE_THRESHOLD:
        corrected = pred  # metal — no correction needed
    
    return pred, corrected, None


# ---- UI ----
st.title("⚛️ Bandgap Predictor")
st.markdown(
    "Predict the electronic bandgap of any inorganic compound from its chemical formula. "
    "Trained on ~97,000 Materials Project entries using 132 element-based (Magpie) descriptors."
)

model, ep, scaler = load_model_and_featurizer()
metrics = load_metrics()
delta_blob = load_delta_model()
delta_metrics = load_delta_metrics()
has_correction = delta_blob is not None

# --- Sidebar with metrics ---
with st.sidebar:
    st.subheader("Model performance")
    st.metric("Test MAE (eV)", f"{metrics['nn_test_mae']:.3f}")
    st.metric("Test R²", f"{metrics['nn_test_r2']:.3f}")
    st.caption(f"Test set: {metrics['n_test']:,} compounds unseen during training")
    st.caption(f"Training set: {metrics['n_train']:,} compounds")
    st.markdown("---")
    st.caption(
        f"**Random Forest baseline** (for reference):\n\n"
        f"Test MAE: {metrics['rf_test_mae']:.3f} eV | R²: {metrics['rf_test_r2']:.3f}"
    )
    if has_correction and delta_metrics:
        st.markdown("---")
        st.subheader("HSE06 correction")
        st.caption(
            "PBE (raw DFT) systematically underestimates bandgaps. A second "
            "Δ-learning model, trained on 10,481 compounds with hybrid-functional "
            "(HSE06) bandgaps, corrects for this."
        )
        st.metric("Corrected MAE vs HSE06", f"{delta_metrics['best_corrected_mae']:.3f} eV",
                   delta=f"-{delta_metrics['uncorrected_mae'] - delta_metrics['best_corrected_mae']:.3f} eV",
                   delta_color="normal")
        st.caption(f"(uncorrected: {delta_metrics['uncorrected_mae']:.3f} eV)")

tab1, tab2 = st.tabs(["🔍 Single prediction", "📦 Batch prediction (CSV)"])

with tab1:
    # --- Main input ---
    # Initialize state
    if "formula_input" not in st.session_state:
        st.session_state.formula_input = "SrTiO3"

    def set_formula(f):
        st.session_state.formula_input = f

    # Example buttons — on_click updates the input BEFORE it's rendered
    st.markdown("**Quick examples** — click any to try:")
    examples = ["Si", "GaAs", "SiO2", "TiO2", "MoS2", "Cu", "SrTiO3", "Fe2O3", "Al2O3", "ZnO"]
    ex_cols = st.columns(len(examples))
    for i, ex in enumerate(examples):
        ex_cols[i].button(ex, key=f"ex_{ex}", use_container_width=True,
                           on_click=set_formula, args=(ex,))

    formula = st.text_input(
        "Chemical formula",
        key="formula_input",
        help="Enter any chemical formula, e.g. Si, GaAs, Fe2O3, SrTiO3, MoS2"
    ).strip()

    if formula:
        pred, corrected, error = predict_bandgap(formula, model, ep, scaler, delta_blob)
        
        if error:
            st.error(error)
        else:
            display_val = corrected if corrected is not None else pred
            
            # Result card
            col1, col2, col3 = st.columns(3)
            if has_correction and corrected is not None:
                col1.metric("HSE06-corrected estimate", f"{corrected:.2f} eV",
                            help=f"Raw PBE prediction: {pred:.2f} eV")
            else:
                col1.metric("Predicted bandgap", f"{pred:.2f} eV")
            
            # Classification (based on corrected value if available)
            if display_val < 0.1:
                classification = "Metal"
                color = "🟡"
            elif display_val < 3.0:
                classification = "Semiconductor"
                color = "🔵"
            else:
                classification = "Insulator"
                color = "🟣"
            col2.metric("Material type", f"{color} {classification}")
            
            # Uncertainty estimate
            uncertainty = delta_metrics['best_corrected_mae'] if (has_correction and delta_metrics) else metrics['nn_test_mae']
            col3.metric("Typical error", f"± {uncertainty:.2f} eV")
            
            if has_correction and corrected is not None and abs(corrected - pred) > 0.05:
                st.caption(f"ℹ️ Raw PBE-level prediction was {pred:.2f} eV — corrected upward to account for "
                           f"DFT's known bandgap underestimation (see sidebar).")
            
            pred = display_val  # downstream interpretation text uses the best available estimate
            
            # Interpretation
            st.markdown("---")
            st.subheader("Interpretation")
            
            if pred < 0.1:
                st.info(f"**{formula}** is predicted to be metallic (bandgap ≈ 0). "
                        "Electrons can move freely — good conductor.")
            elif pred < 1.5:
                st.info(f"**{formula}** is predicted to be a narrow-gap semiconductor. "
                        f"Bandgap of {pred:.2f} eV is typical of infrared/thermal photovoltaic materials.")
            elif pred < 3.5:
                st.info(f"**{formula}** is predicted to be a semiconductor with bandgap {pred:.2f} eV. "
                        f"This range covers visible-light absorbers (solar cells, LEDs).")
            else:
                st.info(f"**{formula}** is predicted to be a wide-gap insulator ({pred:.2f} eV). "
                        "Typical of ceramics, transparent conductors, UV-transparent materials.")
            
            # Comparison bar chart
            with st.expander("Compare with well-known compounds", expanded=False):
                reference = {
                    'Copper (Cu)': 0.0,
                    'Silicon (Si)': 1.11,
                    'GaAs': 1.42,
                    'CdTe': 1.44,
                    'ZnO': 3.37,
                    'TiO2': 3.20,
                    'GaN': 3.40,
                    'SiO2': 9.00,
                    'Al2O3': 8.80,
                }
                reference[f'YOUR: {formula}'] = pred
                
                df_ref = pd.DataFrame({
                    'Material': list(reference.keys()),
                    'Bandgap (eV)': list(reference.values()),
                }).sort_values('Bandgap (eV)')
                
                fig, ax = plt.subplots(figsize=(10, 5))
                colors = ['red' if 'YOUR' in m else 'steelblue' for m in df_ref['Material']]
                ax.barh(df_ref['Material'], df_ref['Bandgap (eV)'], color=colors)
                ax.set_xlabel('Bandgap (eV)')
                ax.axvspan(0, 0.1, alpha=0.1, color='yellow', label='Metal')
                ax.axvspan(0.1, 3.0, alpha=0.1, color='blue', label='Semiconductor')
                ax.axvspan(3.0, 12, alpha=0.1, color='purple', label='Insulator')
                ax.legend(loc='lower right')
                ax.grid(alpha=0.3, axis='x')
                plt.tight_layout()
                st.pyplot(fig)

with tab2:
    st.markdown("**Upload a CSV** with a column named `formula` — get bandgap predictions for all compounds at once.")
    st.markdown("Useful for screening large candidate lists.")
    
    with st.expander("CSV format example", expanded=False):
        st.code("formula\nSi\nGaAs\nSrTiO3\nMoS2\nFe2O3", language="text")
        st.caption("One formula per row. Other columns are preserved in the output.")
    
    # Sample CSV download
    sample = pd.DataFrame({
        "formula": ["Si", "GaAs", "SrTiO3", "MoS2", "CdS", "ZnO", "TiO2", "Fe2O3",
                    "CsPbI3", "BaTiO3", "MgO", "Al2O3", "Cu", "Fe"]
    })
    st.download_button("📄 Download sample CSV template", sample.to_csv(index=False),
                       file_name="bandgap_input_template.csv", mime="text/csv")
    
    uploaded = st.file_uploader("Upload your CSV", type=["csv"])
    
    if uploaded is not None:
        try:
            input_df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            input_df = None
        
        if input_df is not None:
            if "formula" not in input_df.columns:
                st.error("CSV must contain a column named `formula`.")
            elif len(input_df) == 0:
                st.error("CSV is empty.")
            elif len(input_df) > 5000:
                st.error(f"CSV has {len(input_df)} rows — please cap at 5000 to keep the app responsive.")
            else:
                st.success(f"Loaded {len(input_df)} formulas. Predicting…")
                progress = st.progress(0.0, text="Processing…")
                
                predictions = []
                corrected_predictions = []
                errors = []
                types = []
                for i, formula in enumerate(input_df["formula"].astype(str)):
                    pred, corrected, err = predict_bandgap(formula.strip(), model, ep, scaler, delta_blob)
                    if err:
                        predictions.append(None)
                        corrected_predictions.append(None)
                        errors.append(err)
                        types.append(None)
                    else:
                        display_val = corrected if corrected is not None else pred
                        predictions.append(round(pred, 3))
                        corrected_predictions.append(round(corrected, 3) if corrected is not None else None)
                        errors.append("")
                        if display_val < 0.1:
                            types.append("Metal")
                        elif display_val < 3.0:
                            types.append("Semiconductor")
                        else:
                            types.append("Insulator")
                    if (i + 1) % 10 == 0 or i == len(input_df) - 1:
                        progress.progress((i + 1) / len(input_df),
                                          text=f"Processed {i+1}/{len(input_df)}")
                progress.empty()
                
                output_df = input_df.copy()
                output_df["predicted_bandgap_PBE_eV"] = predictions
                if has_correction:
                    output_df["predicted_bandgap_HSE06corrected_eV"] = corrected_predictions
                    output_df["typical_error_eV"] = delta_metrics["best_corrected_mae"]
                else:
                    output_df["typical_error_eV"] = metrics["nn_test_mae"]
                output_df["material_type"] = types
                output_df["error_note"] = errors
                
                # Show summary
                n_ok = sum(1 for e in errors if not e)
                n_err = len(errors) - n_ok
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total", len(input_df))
                col2.metric("Predicted", n_ok)
                col3.metric("Failed", n_err)
                if n_ok > 0:
                    n_metal = sum(1 for t in types if t == "Metal")
                    n_semi = sum(1 for t in types if t == "Semiconductor")
                    n_ins = sum(1 for t in types if t == "Insulator")
                    col4.metric("Metal/Semi/Ins", f"{n_metal}/{n_semi}/{n_ins}")
                
                st.markdown("### Results")
                st.dataframe(output_df, use_container_width=True, height=400)
                
                st.download_button(
                    "⬇️ Download results as CSV",
                    output_df.to_csv(index=False),
                    file_name="bandgap_predictions.csv",
                    mime="text/csv",
                )

st.markdown("---")

with st.expander("How this works", expanded=False):
    st.markdown(f"""
    **Data**: Materials Project database, {metrics['n_train'] + metrics['n_val'] + metrics['n_test']:,} inorganic compounds with computed bandgaps.
    
    **Features**: 132 Magpie descriptors — statistical summaries (mean, std, min, max, mode, range) of atomic properties (atomic number, electronegativity, ionization energy, atomic radius, valence electrons, etc.) across the elements in each compound.
    
    **Model**: 3-layer feed-forward neural network with batch norm + dropout (85K parameters). Trained with Smooth L1 loss + AdamW optimizer.
    
    **Performance**:
    - Test MAE: {metrics['nn_test_mae']:.3f} eV (competitive with published benchmarks)
    - Test R²: {metrics['nn_test_r2']:.3f}
    - Random Forest baseline gave slightly better MAE ({metrics['rf_test_mae']:.3f}) — as often happens on tabular data with limited features
    
    **Limitations**:
    - Trained on DFT-calculated bandgaps (typically underestimate experimental values by 30-50%)
    - Only inorganic compounds well-represented; organics/polymers may be unreliable
    - Composition-only — doesn't account for polymorphs (e.g., different TiO2 phases share the same predicted value)
    
    **What it's good for**: Screening large numbers of compositions before running expensive DFT calculations or experiments.
    """)

st.markdown("---")
st.caption(
    "Built by Amba Prasad — nanophysics Ph.D. transitioning to ML/AI. "
    "Portfolio project bridging materials science with modern machine learning."
)
