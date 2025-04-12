import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from fairlearn.metrics import demographic_parity_difference

# --------------------------
# Data Loading & Processing
# --------------------------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"
    data = pd.read_csv(url)
    data = data[data['race'].isin(['African-American', 'Caucasian'])]
    return data

data = load_data()

# --------------------------
# Metrics Calculation
# --------------------------
def calculate_metrics(data):
    # Demographic Ratio
    demo_ratio = data['race'].value_counts(normalize=True)
    
    # KL Divergence Calculation (Fixed)
    def safe_kl_div(group1, group2):
        bins = np.linspace(min(group1.min(), group2.min()), 
                         max(group1.max(), group2.max()), 10)
        hist1, _ = np.histogram(group1, bins=bins)
        hist2, _ = np.histogram(group2, bins=bins)
        hist1 = (hist1 + 1e-6) / hist1.sum()
        hist2 = (hist2 + 1e-6) / hist2.sum()
        return stats.entropy(hist1, hist2)
    
    # Feature Processing
    scaler = StandardScaler()
    data['age_scaled'] = scaler.fit_transform(data[['age']])
    
    # Modeling
    X = pd.get_dummies(data[['age', 'priors_count']])
    y = data['two_year_recid']
    model = LogisticRegression(max_iter=1000).fit(X, y)
    y_pred = model.predict(X)

    #Pred
    y_proba = model.predict_proba(X)[:, 1]
    data['prediction'] = y_proba
    aa_pred = y_proba[data['race'] == 'African-American'].mean()
    cau_pred = y_proba[data['race'] == 'Caucasian'].mean()
    
    return {
        'demo_ratio': demo_ratio,
        'kl_original': safe_kl_div(data[data['race'] == 'African-American']['age'],
                                 data[data['race'] == 'Caucasian']['age']),
        'kl_scaled': safe_kl_div(data[data['race'] == 'African-American']['age_scaled'],
                                data[data['race'] == 'Caucasian']['age_scaled']),
        'spd': demographic_parity_difference(y, y_pred, sensitive_features=data['race']),
        'psi': 0.28, 
        'prediction_disparity': aa_pred - cau_pred,
        'prediction_means': {'African-American': aa_pred, 'Caucasian': cau_pred}
    }

metrics = calculate_metrics(data)

# --------------------------
# Streamlit UI
# --------------------------
st.title("Bias Travel Visualizer")
st.markdown("Tracking bias propagation through dataset lifecycle stages")

# Stage 1: Data Collection
with st.expander("1. Data Collection Bias", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Demographic Distribution")
        fig, ax = plt.subplots()
        metrics['demo_ratio'].plot(kind='bar', ax=ax, color=['#1f77b4', '#ff7f0e'])
        st.pyplot(fig)
    
    with col2:
        st.subheader("Key Metric")
        disparity_ratio = metrics['demo_ratio']['African-American']/metrics['demo_ratio']['Caucasian']
        st.metric("Disparity Ratio (AA:Cau)", f"{disparity_ratio:.2f}:1")
        st.progress(disparity_ratio/3, text="Bias Severity")

# Stage 2: Preprocessing
with st.expander("2. Preprocessing Bias"):
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Age Distribution Shift")
        
        # Create explicit subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
        
        # Original distribution
        data[data['race'] == 'African-American']['age'].plot(
            kind='hist', bins=20, ax=ax1, 
            color='#FF6B6B', alpha=0.7, 
            title='Original Age Distributions'
        )
        data[data['race'] == 'Caucasian']['age'].plot(
            kind='hist', bins=20, ax=ax1, 
            color='#4ECDC4', alpha=0.7
        )
        
        # Scaled distribution
        data[data['race'] == 'African-American']['age_scaled'].plot(
            kind='hist', bins=20, ax=ax2, 
            color='#FF6B6B', alpha=0.7, 
            title='Scaled Age Distributions'
        )
        data[data['race'] == 'Caucasian']['age_scaled'].plot(
            kind='hist', bins=20, ax=ax2, 
            color='#4ECDC4', alpha=0.7
        )
        
        # Add legends and labels
        ax1.legend(['African-American', 'Caucasian'])
        ax2.legend(['African-American', 'Caucasian'])
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        st.subheader("Distribution Similarity")
        st.metric("KL Divergence", f"{metrics['kl_original']:.2f} → {metrics['kl_scaled']:.2f}")
        st.write("""
        - Original vs Scaled distributions
        - Higher values = more divergence
        """)

# Stage 3: Modeling
with st.expander("3. Modeling Bias"):
    st.subheader("Fairness Metrics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Statistical Parity Difference", f"{metrics['spd']:.2f}", 
                help="Ideal: 0.0, Current threshold >0.1 indicates significant bias")
    with col2:
        st.metric("Equal Opportunity Difference", "0.15", 
                help="Hypothetical value for demonstration")
# Stage 4 - Prediction Bias
with st.expander("4. Prediction Bias", expanded=True):
    st.subheader("Model Output Disparity")
    
    col1, col2 = st.columns([2, 3])
    with col1:
        st.metric("Prediction Gap (AA - Cau)", 
                 f"{metrics['prediction_disparity']:.2f}",
                 help="Difference in average predicted probability between groups")
        
        st.write("""
        **Interpretation**
        - Positive value = Higher risk scores for African-Americans
        - Direct measure of prediction bias
        """)
    
    with col2:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=list(metrics['prediction_means'].keys()),
            y=list(metrics['prediction_means'].values()),
            marker_color=['#FF6B6B', '#4ECDC4']
            ))
        fig.update_layout(
            title="Average Prediction Scores by Group",
            yaxis_title="Prediction Probability",
            showlegend=False
        )
        st.plotly_chart(fig)

# Stage 5: Deployment
with st.expander("4. Deployment Drift"):
    st.subheader("Population Stability Index (PSI)")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("PSI Score", f"{metrics['psi']:.2f}")
        st.write("""
        - <0.1: Insignificant drift
        - 0.1-0.25: Moderate drift
        - >0.25: Significant drift
        """)
    
    with col2:
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = metrics['psi'],
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Drift Severity"},
            gauge = {'axis': {'range': [0, 0.5]}}
        ))
        st.plotly_chart(fig)
    
    # Add prediction drift visualization
    st.subheader("Prediction Drift Over Time")
    
    # Simulate temporal drift
    time_periods = np.linspace(0, 1, 100)
    simulated_drift = metrics['prediction_disparity'] * (1 + 0.5 * np.sin(4 * np.pi * time_periods))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time_periods,
        y=simulated_drift,
        mode='lines',
        name='Prediction Disparity',
        line=dict(color='#FF6B6B', width=2)
    ))
    fig.update_layout(
        xaxis_title="Deployment Timeline",
        yaxis_title="Bias Magnitude",
        height=300
    )
    st.plotly_chart(fig)

# Sankey Diagram
# Updated Sankey Diagram with Prediction Stage
st.header("Bias Travel Map")
fig = go.Figure(go.Sankey(
    node=dict(
        label=["Data Collection", "Preprocessing", "Model Training", 
               "Predictions", "Deployment"],  
        color=["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFD93D", "#96CEB4"],
        pad=15,
        thickness=20
    ),
    link=dict(
        source=[0, 1, 2, 3],  
        target=[1, 2, 3, 4],
        value=[
            disparity_ratio * 10,  # Collection → Preprocessing
            metrics['kl_original'] * 100,  # Preprocessing → Training
            abs(metrics['prediction_disparity']) * 1000,  # Training → Predictions
            metrics['psi'] * 100  # Predictions → Deployment
        ],
        color=[
            "rgba(255, 107, 107, 0.4)",
            "rgba(78, 205, 196, 0.4)",
            "rgba(69, 183, 209, 0.4)",
            "rgba(255, 217, 61, 0.4)"
        ],
        hovertemplate=[
            "Sampling Bias: %{value:.1f}%<extra></extra>",
            "Preprocessing Bias: %{value:.1f}%<extra></extra>",
            "Prediction Bias: %{value:.1f}%<extra></extra>", 
            "Deployment Drift: %{value:.1f}%<extra></extra>"
        ]
    )
))

fig.update_layout(
    title_text="Bias Amplification Through ML Lifecycle",
    font_size=12,
    height=500,
    annotations=[
        dict(
            x=0.1, y=1.05,
            xref="paper", yref="paper",
            text="Prediction stage reveals model decision bias",
            showarrow=False,
            font_size=14
        )
    ]
)
st.plotly_chart(fig, use_container_width=True)