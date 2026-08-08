# Bandgap Predictor

Predict the electronic bandgap of any inorganic compound from its chemical formula.

**Live**: [amba-bandgap-predictor.streamlit.app](https://amba-bandgap-predictor.streamlit.app)

## About

Portfolio project by Amba Prasad (nanophysics Ph.D., transitioning to ML/AI) with support from Manu Raveendran.

## What it does

Enter any chemical formula (Si, GaAs, SrTiO3, MoS2, etc.) — get a predicted bandgap in electron volts, plus material classification (metal / semiconductor / insulator) and comparison to well-known compounds.

## How it works

- **Data**: 97,394 inorganic compounds from Materials Project with computed bandgaps
- **Features**: 132 Magpie descriptors (element property statistics) via matminer
- **Model**: 3-layer feed-forward neural network with batch normalization and dropout
- **Performance**: Test MAE 0.40 eV, R² 0.80 (competitive with published benchmarks)

## Model performance

| Metric | Neural Network | Random Forest (baseline) |
|---|---|---|
| Test MAE (eV) | 0.397 | 0.351 |
| Test R² | 0.805 | 0.853 |
| Test samples | 9,740 | 9,740 |

Interesting observation: on tabular material-composition data with only 132 features, RandomForest slightly outperforms the neural network. This is documented in the materials informatics literature — with more features (e.g., adding structure descriptors), NNs typically pull ahead.

## Files

- `streamlit_app.py` — main Streamlit application
- `bandgap_model.pth` — trained NN weights (85K params)
- `bandgap_featurizer.pkl` — matminer featurizer + StandardScaler
- `bandgap_metrics.json` — training/test metrics
- `requirements.txt` — Python dependencies

## Portfolio narrative

This project demonstrates:
- **End-to-end ML pipeline**: data acquisition → featurization → training → deployment
- **Domain-informed feature engineering**: Magpie descriptors chosen for materials informatics
- **Honest benchmarking**: NN vs RF comparison shows real-world tradeoffs
- **Practical deployment**: web app that runs pre-trained models client-side, no GPU needed at inference
- **Materials science expertise**: interpretation module classifies materials and contextualizes predictions

## Limitations

- Trained on DFT-calculated bandgaps (typically underestimate experimental values by 30-50%)
- Only inorganic compounds well-represented — organics/polymers may be unreliable
- Composition-only model — doesn't distinguish polymorphs (e.g., anatase vs rutile TiO2)
