"""Streamlit app for Credit Card Fraud Detection - Simple and Focused"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_manager import ModelManager

# ════════════════════════════════════════════════════════════════════════════
# PAGE CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .fraud-card {
        background-color: #ffe6e6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #ff0000;
    }
    .normal-card {
        background-color: #e6ffe6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #00cc00;
    }
    </style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# SESSION STATE & INITIALIZATION
# ════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_model_and_scaler():
    """Load model and scaler once using ModelManager."""
    try:
        manager = ModelManager()
        artifacts = manager.load_all("Random_Forest_tuned", "scaler")
        return artifacts['model'], artifacts['scaler'], None
    except Exception as e:
        return None, None, str(e)

# ════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ════════════════════════════════════════════════════════════════════════════

st.title("Credit Card Fraud Detection System")
st.markdown("Real-time fraud detection using Machine Learning")
st.markdown("---")

# Load model and scaler
model, scaler, error = load_model_and_scaler()

if error:
    st.error(f"Error loading model: {error}")
    st.stop()

if model is None or scaler is None:
    st.error("Failed to load model or scaler")
    st.stop()

# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR - INFO & SETTINGS
# ════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.header("ℹ️ Model Information")
    st.info("""
    **Model:** Random Forest (Optimized)
    
    **Features:** 6 numerical
    
    **Optimal Threshold:** 0.21
    
    **Key Features:**
    - newbalanceOrig
    - oldbalanceOrg
    - amount
    - step
    - oldbalanceDest
    - newbalanceDest
    """)
    
    st.markdown("---")
    st.markdown("**Built with:** scikit-learn, Streamlit, joblib")

# ════════════════════════════════════════════════════════════════════════════
# MAIN INTERFACE - TWO TABS
# ════════════════════════════════════════════════════════════════════════════

tab1, tab2 = st.tabs(["🔍 Single Prediction", "📊 Batch Analysis"])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1: SINGLE PREDICTION
# ════════════════════════════════════════════════════════════════════════════

with tab1:
    st.header("Single Transaction Prediction")
    st.markdown("Enter transaction details below to predict if it's fraudulent")
    
    # Create columns for input
    col1, col2, col3 = st.columns(3)
    
    with col1:
        step = st.number_input(
            "Step (Transaction #)",
            min_value=1,
            max_value=743,
            value=100,
            help="Transaction step in the sequence"
        )
        amount = st.number_input(
            "Amount ($)",
            min_value=0.0,
            value=5000.0,
            step=100.0,
            help="Transaction amount"
        )
    
    with col2:
        oldbalanceOrg = st.number_input(
            "Orig. Balance Before ($)",
            min_value=0.0,
            value=50000.0,
            step=1000.0,
            help="Originator's balance before transaction"
        )
        newbalanceOrig = st.number_input(
            "Orig. Balance After ($)",
            min_value=0.0,
            value=45000.0,
            step=1000.0,
            help="Originator's balance after transaction"
        )
    
    with col3:
        oldbalanceDest = st.number_input(
            "Dest. Balance Before ($)",
            min_value=0.0,
            value=0.0,
            step=1000.0,
            help="Destination's balance before transaction"
        )
        newbalanceDest = st.number_input(
            "Dest. Balance After ($)",
            min_value=0.0,
            value=5000.0,
            step=1000.0,
            help="Destination's balance after transaction"
        )
    
    st.markdown("---")
    
    # Prediction button
    if st.button("🔮 Predict Fraud", type="primary", use_container_width=True):
        try:
            # Prepare input data
            features = np.array([[
                step,
                amount,
                oldbalanceOrg,
                newbalanceOrig,
                oldbalanceDest,
                newbalanceDest
            ]])
            
            # Scale features
            features_scaled = scaler.transform(features)
            
            # Get prediction probability
            fraud_probability = model.predict_proba(features_scaled)[0][1]
            prediction = model.predict(features_scaled)[0]
            
            # Determine if fraud using optimal threshold
            optimal_threshold = 0.21
            is_fraud = fraud_probability > optimal_threshold
            
            # Display results
            st.markdown("---")
            st.subheader("Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if is_fraud:
                    st.error("FRAUD DETECTED")
                else:
                    st.success("LEGITIMATE")
            
            with col2:
                st.metric(
                    "Fraud Probability",
                    f"{fraud_probability:.1%}",
                    help="Confidence score for fraud"
                )
            
            with col3:
                st.metric(
                    "Risk Level",
                    "HIGH" if is_fraud else "LOW",
                    delta="Alert" if is_fraud else "Safe"
                )
            
            # Detailed probability display
            st.markdown("---")
            st.subheader("Detailed Analysis")
            
            # Probability visualization
            col1, col2 = st.columns(2)
            
            with col1:
                prob_data = {
                    'Prediction': ['Legitimate', 'Fraud'],
                    'Probability': [(1 - fraud_probability), fraud_probability]
                }
                prob_df = pd.DataFrame(prob_data)
                
                import plotly.express as px
                fig = px.bar(
                    prob_df,
                    x='Prediction',
                    y='Probability',
                    color='Prediction',
                    color_discrete_map={'Legitimate': '#00cc00', 'Fraud': '#ff0000'},
                    title="Prediction Confidence",
                    range_y=[0, 1]
                )
                fig.update_layout(height=300, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Threshold visualization
                st.write("**Classification Threshold**")
                st.write(f"Your probability: **{fraud_probability:.4f}**")
                st.write(f"Threshold: **{optimal_threshold:.2f}**")
                
                if fraud_probability > optimal_threshold:
                    st.progress(1.0, text=f"FRAUD: {fraud_probability:.4f} > {optimal_threshold:.2f}")
                else:
                    progress_pct = fraud_probability / optimal_threshold
                    st.progress(progress_pct, text=f"SAFE: {fraud_probability:.4f} < {optimal_threshold:.2f}")
            
            # Feature explanation
            st.markdown("---")
            st.subheader("📋 Transaction Summary")
            
            summary_data = {
                'Feature': ['Step', 'Amount', 'Orig. Balance Before', 'Orig. Balance After',
                           'Dest. Balance Before', 'Dest. Balance After'],
                'Value': [f"${step}", f"${amount:,.2f}", f"${oldbalanceOrg:,.2f}",
                         f"${newbalanceOrig:,.2f}", f"${oldbalanceDest:,.2f}",
                         f"${newbalanceDest:,.2f}"]
            }
            summary_df = pd.DataFrame(summary_data)
            st.table(summary_df)
            
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")


# ════════════════════════════════════════════════════════════════════════════
# TAB 2: BATCH ANALYSIS
# ════════════════════════════════════════════════════════════════════════════

with tab2:
    st.header("Batch Transaction Analysis")
    st.markdown("Upload a CSV file with transaction data for batch predictions")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type="csv",
        help="CSV should have columns: step, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest"
    )
    
    if uploaded_file is not None:
        try:
            # Read CSV
            df = pd.read_csv(uploaded_file)
            
            # Check required columns
            required_cols = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 
                           'oldbalanceDest', 'newbalanceDest']
            
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                st.error(f"Missing columns: {', '.join(missing_cols)}")
            else:
                st.write(f"📊 Analyzing {len(df)} transactions...")
                
                # Extract features
                X = df[required_cols].values
                
                # Scale features
                X_scaled = scaler.transform(X)
                
                # Get predictions
                predictions = model.predict(X_scaled)
                probabilities = model.predict_proba(X_scaled)[:, 1]
                
                # Apply optimal threshold
                optimal_threshold = 0.21
                fraud_flags = probabilities > optimal_threshold
                
                # Add results to dataframe
                results_df = df.copy()
                results_df['Fraud_Probability'] = probabilities
                results_df['Is_Fraud'] = fraud_flags
                results_df['Risk_Level'] = results_df['Is_Fraud'].apply(
                    lambda x: "🔴 HIGH (Fraud)" if x else "🟢 LOW (Legitimate)"
                )
                
                # Display statistics
                st.markdown("---")
                st.subheader("📈 Batch Statistics")
                
                col1, col2, col3, col4 = st.columns(4)
                
                fraud_count = fraud_flags.sum()
                fraud_rate = (fraud_count / len(df)) * 100 if len(df) > 0 else 0
                
                with col1:
                    st.metric("Total Transactions", len(df))
                
                with col2:
                    st.metric("Fraudulent", fraud_count)
                
                with col3:
                    st.metric("Legitimate", len(df) - fraud_count)
                
                with col4:
                    st.metric("Fraud Rate", f"{fraud_rate:.1f}%")
                
                # Display results table
                st.markdown("---")
                st.subheader("📋 Detailed Results")
                
                # Format probability for display
                display_df = results_df.copy()
                display_df['Fraud_Probability'] = display_df['Fraud_Probability'].apply(
                    lambda x: f"{x:.2%}"
                )
                
                st.dataframe(
                    display_df[['amount', 'oldbalanceOrg', 'newbalanceOrig', 
                               'Fraud_Probability', 'Risk_Level']],
                    use_container_width=True,
                    height=400
                )
                
                # Download results
                st.markdown("---")
                csv = display_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results (CSV)",
                    data=csv,
                    file_name="fraud_predictions.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

# ════════════════════════════════════════════════════════════════════════════
# FOOTER
# ════════════════════════════════════════════════════════════════════════════

st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.caption("🔒 **Privacy:** Data is not stored or logged")

with col2:
    st.caption("⚙️ **Model:** Random Forest (tuned)")

with col3:
    st.caption("📅 **Last Updated:** April 2026")
