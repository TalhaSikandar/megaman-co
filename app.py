import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from datetime import datetime
from train import load_model, evaluate, train, save_model
from data_utils import load_data
from metadata import MODEL_PATH, PREPROCESSED_DATA_PATH
from infer import predict

# App configuration
st.set_page_config(
    page_title="AI Stock Predictor",
    page_icon="📈",
    layout="wide"
)

# Load data and model (cache for performance)
@st.cache_data
def get_data():
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(PREPROCESSED_DATA_PATH)
    return X_train, y_train, X_val, y_val, X_test, y_test

@st.cache_resource
def get_model():
    return load_model(MODEL_PATH)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .progress-container {
        height: 20px;
        background-color: #e0e0e0;
        border-radius: 10px;
        margin: 10px 0;
    }
    .progress-bar {
        height: 100%;
        border-radius: 10px;
        background-color: #4CAF50;
        text-align: center;
        line-height: 20px;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar for navigation
with st.sidebar:
    st.image("https://via.placeholder.com/150x50?text=StockAI", width=150)
    page = st.selectbox("Navigation", ["Dashboard", "Live Prediction", "Model Training", "Performance", "About"])

# Main content
if page == "Dashboard":
    st.title("📊 Stock Prediction Dashboard")
    X_train, y_train, X_val, y_val, X_test, y_test = get_data()
    model = get_model()
    # Evaluate model
    test_metrics = evaluate(model, X_test, y_test)
    last_pred = predict(model, X_test[-1:])[0]
    last_actual = y_test[-1]
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"<div class='metric-card'><h3>Test MAE</h3><h2>{test_metrics['mae']:.4f}</h2></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='metric-card'><h3>Last Prediction</h3><h2>${last_pred:.2f}</h2></div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div class='metric-card'><h3>Training Samples</h3><h2>{len(X_train)}</h2></div>", unsafe_allow_html=True)

    # Recent predictions chart
    st.subheader("Recent Predictions vs Actual")
    preds = predict(model, X_test[-20:])
    chart_data = pd.DataFrame({
        'Index': np.arange(len(preds)),
        'Predicted': preds.flatten(),
        'Actual': y_test[-20:].flatten()
    })
    st.line_chart(chart_data.set_index('Index'))

    st.subheader("Model Architecture")
    st.markdown("""
    - **Input Layer**: 6 features (Open, High, Low, Close, Volume, Change %)
    - **LSTM Layers**: 2 layers with 50 units each
    - **Dense Layer**: 1 unit with linear activation
    - **Optimizer**: Adam
    - **Loss Function**: Mean Squared Error
    """)

elif page == "Live Prediction":
    st.title("🔮 Live Stock Prediction")
    X_train, y_train, X_val, y_val, X_test, y_test = get_data()
    model = get_model()
    lookback = X_test.shape[1]

    st.subheader("Latest Market Data (from test set)")
    last_features = X_test[-1].reshape(lookback, 6)
    last_df = pd.DataFrame(last_features, columns=["Open", "High", "Low", "Close", "Volume", "Change %"])
    st.dataframe(last_df)

    st.subheader("Prediction Input")
    with st.form("prediction_form"):
        use_last = st.checkbox("Use latest data from test set", value=True)
        if use_last:
            features = last_features
        else:
            features = []
            for i in range(lookback):
                st.markdown(f"**Step {i+1}:**")
                row = []
                row.append(st.number_input(f"Open [{i+1}]", value=float(last_features[i,0])))
                row.append(st.number_input(f"High [{i+1}]", value=float(last_features[i,1])))
                row.append(st.number_input(f"Low [{i+1}]", value=float(last_features[i,2])))
                row.append(st.number_input(f"Close [{i+1}]", value=float(last_features[i,3])))
                row.append(st.number_input(f"Volume [{i+1}]", value=float(last_features[i,4])))
                row.append(st.number_input(f"Change % [{i+1}]", value=float(last_features[i,5])))
                features.append(row)
            features = np.array(features)
        submitted = st.form_submit_button("Predict Next Day")
        if submitted:
            with st.spinner("Predicting..."):
                # Use your custom predict function
                pred = predict(model, features.reshape(1, lookback, 6))[0]
                st.success(f"Predicted Next Closing Price: **${pred:.2f}**")

elif page == "Model Training":
    st.title("🤖 Model Training Center")
    X_train, y_train, X_val, y_val, X_test, y_test = get_data()
    model = get_model()
    st.subheader("Start New Training")
    epochs = st.slider("Epochs", 1, 100, 10)
    patience = st.slider("Early Stopping Patience", 1, 20, 5)
    if st.button("Train Model"):
        with st.spinner("Training in progress..."):
            history = train(
                model, X_train, y_train, X_val, y_val,
                epochs=epochs, patience=patience, verbose=1, checkpoint_path=MODEL_PATH
            )
            save_model(model, MODEL_PATH)
            st.success("Training completed!")
            st.write("Training Loss History:", history.get('loss', []))
            st.write("Validation Loss History:", history.get('val_loss', []))
            fig, ax = plt.subplots()
            ax.plot(history.get('loss', []), label='Train Loss')
            ax.plot(history.get('val_loss', []), label='Val Loss')
            ax.legend()
            st.pyplot(fig)

elif page == "Performance":
    st.title("📊 Model Performance Analysis")
    X_train, y_train, X_val, y_val, X_test, y_test = get_data()
    model = get_model()
    st.subheader("Evaluation Metrics")
    metrics = evaluate(model, X_test, y_test)
    for k, v in metrics.items():
        st.metric(k, f"{v:.4f}")

    st.subheader("Actual vs Predicted")
    preds = predict(X_test)
    fig, ax = plt.subplots()
    ax.plot(y_test.flatten(), label='Actual')
    ax.plot(preds.flatten(), label='Predicted')
    ax.legend()
    st.pyplot(fig)

elif page == "About":
    st.title("ℹ️ About StockAI Predictor")
    st.image("https://via.placeholder.com/800x200?text=StockAI+Predictor", use_column_width=True)
    st.markdown("""
    ## Advanced LSTM Stock Price Prediction System

    This application uses deep learning to predict future stock prices based on historical market data.

    ### Key Features:
    - **Real-time predictions** using the latest market data
    - **LSTM neural network** architecture for time series forecasting
    - **Training dashboard** to monitor model learning
    - **Performance metrics** to evaluate prediction accuracy
    - **Backtesting** capabilities to validate model performance

    ### Technical Details:
    - Built with Python and numPy
    - Uses Streamlit for the interactive web interface
    - Processes 6 key market indicators for predictions
    - Model retraining available through the interface

    ### Data Sources:
    - Yahoo Finance API
    - Alpha Vantage
    - Custom data pipelines
    - Investing.com

    """)
    st.markdown("---")
    st.markdown("© 2023 StockAI Predictor | Version 1.2.0")