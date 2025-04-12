# Visualizing Bias Travel in ML Lifecycle
This repository has the case study for "Visualizing bias travel in ML lifecycle: A metric driven framework for fairness evolutional analysis" research paper.

---

# Bias Travel Visualizer

This project presents a code-based implementation to visualize how bias propagates across different stages of the machine learning lifecycle using the COMPAS dataset. The goal is to observe changes in fairness-related metrics at each step, from raw data to deployment, using interpretable visualizations.

## Overview

This is a research-focused codebase developed to:
- Demonstrate how bias can evolve through the ML pipeline.
- Track and visualize fairness metrics stage by stage.
- Enable inspection of metric shifts across the lifecycle (e.g., preprocessing, modeling, prediction).

## Stages Covered

- Data Collection
- Preprocessing
- Modeling
- Prediction
- Deployment

Each stage is visualized using expanders in the Streamlit interface to compare relevant metrics between demographic groups.

## Metrics Tracked

| Stage          | Metric                         | Description                                       |
|----------------|--------------------------------|---------------------------------------------------|
| Data Collection | Demographic Distribution       | Group-wise representation ratio                  |
| Preprocessing   | KL Divergence                  | Distribution shift after scaling/cleaning        |
| Modeling        | Statistical Parity Difference  | Label outcome parity between groups              |
| Prediction      | Prediction Gap                 | Difference in predicted probabilities or scores  |
| Deployment      | Population Stability Index     | Drift in feature distributions post-deployment   |

## Dataset

- **COMPAS Dataset** from ProPublica  
- Filtered for binary classification on recidivism (`two_year_recid`)
- Focused comparison between `African-American` and `Caucasian` groups  
- Source: [https://github.com/propublica/compas-analysis](https://github.com/propublica/compas-analysis)

## Running the Code

```bash
pip install -r requirements.txt
streamlit run casestudy.py
```

## File Structure

```
├── casestudy.py          # Streamlit code for bias visualization
├── requirements.txt      # Dependencies
├── README.md             # Documentation
```

## Notes

This is not intended as a production-ready application but rather a research visualization script to support studies on bias across the ML lifecycle.

---
