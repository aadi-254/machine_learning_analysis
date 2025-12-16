import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import os

BASE_DIR = os.path.dirname(__file__)
df = pd.read_csv(os.path.join(BASE_DIR, "data.csv"))


# Page configuration
st.set_page_config(page_title="Regression Hyperparameter Tuning", layout="wide")

# Title
st.title("🎯 Regression Algorithms - Hyperparameter Tuning")
st.markdown("---")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('data.csv')
    return df

df = load_data()

# Sidebar for hyperparameters
st.sidebar.header("🔧 Hyperparameter Settings")

# Algorithm selection
algorithm = st.sidebar.selectbox(
    "Select Algorithm",
    ["Linear Regression", "Ridge", "Lasso", "ElasticNet", "Compare All"]
)

st.sidebar.markdown("---")

# Hyperparameters based on selected algorithm
if algorithm == "Ridge":
    alpha_ridge = st.sidebar.slider("Ridge Alpha (Regularization Strength)", 
                                     min_value=0.01, max_value=10.0, value=1.0, step=0.01)
    
elif algorithm == "Lasso":
    alpha_lasso = st.sidebar.slider("Lasso Alpha (Regularization Strength)", 
                                     min_value=0.01, max_value=10.0, value=1.0, step=0.01)
    
elif algorithm == "ElasticNet":
    alpha_elastic = st.sidebar.slider("ElasticNet Alpha (Regularization Strength)", 
                                       min_value=0.01, max_value=10.0, value=1.0, step=0.01)
    l1_ratio = st.sidebar.slider("L1 Ratio (Lasso vs Ridge balance)", 
                                  min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    st.sidebar.info("L1 Ratio = 1: Pure Lasso\nL1 Ratio = 0: Pure Ridge")

elif algorithm == "Compare All":
    st.sidebar.subheader("Ridge Settings")
    alpha_ridge = st.sidebar.slider("Ridge Alpha", 
                                     min_value=0.01, max_value=10.0, value=1.0, step=0.01, key="ridge")
    
    st.sidebar.subheader("Lasso Settings")
    alpha_lasso = st.sidebar.slider("Lasso Alpha", 
                                     min_value=0.01, max_value=10.0, value=1.0, step=0.01, key="lasso")
    
    st.sidebar.subheader("ElasticNet Settings")
    alpha_elastic = st.sidebar.slider("ElasticNet Alpha", 
                                       min_value=0.01, max_value=10.0, value=1.0, step=0.01, key="elastic")
    l1_ratio = st.sidebar.slider("L1 Ratio", 
                                  min_value=0.0, max_value=1.0, value=0.5, step=0.01)

# Train-test split settings
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Data Split")
test_size = st.sidebar.slider("Test Size (%)", min_value=10, max_value=40, value=20, step=5) / 100
random_state = st.sidebar.number_input("Random State", min_value=0, max_value=100, value=42, step=1)

# Prepare data
X = df[['CGPA']]
y = df['IQ']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=int(random_state))

# Function to train and evaluate model
def train_and_evaluate(model, name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    return model, y_pred, {
        'R² Score': r2,
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse
    }

# Main content area
if algorithm == "Compare All":
    # Create all models with hyperparameters
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge': Ridge(alpha=alpha_ridge),
        'Lasso': Lasso(alpha=alpha_lasso),
        'ElasticNet': ElasticNet(alpha=alpha_elastic, l1_ratio=l1_ratio)
    }
    
    colors = {
        'Linear Regression': 'red',
        'Ridge': 'blue',
        'Lasso': 'green',
        'ElasticNet': 'purple'
    }
    
    # Train all models
    results = {}
    trained_models = {}
    for name, model in models.items():
        trained_model, y_pred, metrics = train_and_evaluate(model, name)
        results[name] = metrics
        trained_models[name] = trained_model
    
    # Create two columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📈 Regression Lines Comparison")
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Scatter plot
        ax.scatter(df['CGPA'], df['IQ'], alpha=0.5, c='gray', edgecolors='k', s=50, label='Data points')
        
        # Generate prediction line for each model
        x_range = pd.DataFrame(
            np.linspace(df['CGPA'].min(), df['CGPA'].max(), 100),
            columns=['CGPA']
        )
        
        for name, model in trained_models.items():
            y_pred_line = model.predict(x_range)
            ax.plot(x_range, y_pred_line, color=colors[name], linewidth=2.5, label=name, alpha=0.8)
        
        ax.set_xlabel('CGPA', fontsize=13, fontweight='bold')
        ax.set_ylabel('IQ', fontsize=13, fontweight='bold')
        ax.set_title('CGPA vs IQ - All Regression Models', fontsize=15, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
    
    with col2:
        st.subheader("📊 Performance Metrics")
        
        # Create metrics dataframe
        metrics_df = pd.DataFrame(results).T
        metrics_df = metrics_df.round(4)
        
        # Highlight best values
        st.dataframe(
            metrics_df.style.highlight_max(subset=['R² Score'], color='lightgreen')
                           .highlight_min(subset=['MAE', 'MSE', 'RMSE'], color='lightgreen'),
            width='stretch'
        )
        
        # Best model
        best_model = metrics_df['R² Score'].idxmax()
        st.success(f"🏆 Best Model: **{best_model}**")
        st.metric("Best R² Score", f"{metrics_df.loc[best_model, 'R² Score']:.4f}")
    
    # Detailed metrics in expandable section
    with st.expander("📋 Detailed Metrics Breakdown"):
        for name, metrics in results.items():
            st.write(f"**{name}**")
            cols = st.columns(4)
            cols[0].metric("R² Score", f"{metrics['R² Score']:.4f}")
            cols[1].metric("MAE", f"{metrics['MAE']:.4f}")
            cols[2].metric("MSE", f"{metrics['MSE']:.4f}")
            cols[3].metric("RMSE", f"{metrics['RMSE']:.4f}")
            st.markdown("---")

else:
    # Single algorithm mode
    if algorithm == "Linear Regression":
        model = LinearRegression()
    elif algorithm == "Ridge":
        model = Ridge(alpha=alpha_ridge)
    elif algorithm == "Lasso":
        model = Lasso(alpha=alpha_lasso)
    else:  # ElasticNet
        model = ElasticNet(alpha=alpha_elastic, l1_ratio=l1_ratio)
    
    # Train and evaluate
    trained_model, y_pred, metrics = train_and_evaluate(model, algorithm)
    
    # Create two columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(f"📈 {algorithm} - Regression Line")
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Scatter plot
        ax.scatter(df['CGPA'], df['IQ'], alpha=0.6, c='blue', edgecolors='k', s=60, label='Data points')
        
        # Regression line
        x_range = pd.DataFrame(
            np.linspace(df['CGPA'].min(), df['CGPA'].max(), 100),
            columns=['CGPA']
        )
        y_pred_line = trained_model.predict(x_range)
        ax.plot(x_range, y_pred_line, color='red', linewidth=3, label=f'{algorithm} fit', alpha=0.9)
        
        # Test predictions
        ax.scatter(X_test, y_test, alpha=0.8, c='orange', edgecolors='k', s=80, 
                   label='Test data', marker='s')
        ax.scatter(X_test, y_pred, alpha=0.8, c='green', edgecolors='k', s=80, 
                   label='Predictions', marker='^')
        
        ax.set_xlabel('CGPA', fontsize=13, fontweight='bold')
        ax.set_ylabel('IQ', fontsize=13, fontweight='bold')
        ax.set_title(f'CGPA vs IQ - {algorithm}', fontsize=15, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
    
    with col2:
        st.subheader("📊 Performance Metrics")
        
        st.metric("R² Score (Accuracy)", f"{metrics['R² Score']:.4f}")
        st.metric("Mean Absolute Error", f"{metrics['MAE']:.4f}")
        st.metric("Mean Squared Error", f"{metrics['MSE']:.4f}")
        st.metric("Root Mean Squared Error", f"{metrics['RMSE']:.4f}")
        
        # Model coefficients
        st.markdown("---")
        st.subheader("🔢 Model Parameters")
        if hasattr(trained_model, 'coef_'):
            st.write(f"**Coefficient:** {trained_model.coef_[0]:.4f}")
        if hasattr(trained_model, 'intercept_'):
            st.write(f"**Intercept:** {trained_model.intercept_:.4f}")
        
        # Equation
        if hasattr(trained_model, 'coef_') and hasattr(trained_model, 'intercept_'):
            st.info(f"**Equation:**\n\nIQ = {trained_model.coef_[0]:.4f} × CGPA + {trained_model.intercept_:.4f}")

# Data info section
with st.expander("📁 Dataset Information"):
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Samples", len(df))
    col2.metric("Training Samples", len(X_train))
    col3.metric("Test Samples", len(X_test))
    
    st.write("**Dataset Preview:**")
    st.dataframe(df.head(10), width='stretch')
    
    st.write("**Statistical Summary:**")
    st.dataframe(df.describe(), width='stretch')

# Footer
st.markdown("---")
st.markdown("💡 **Tip:** Adjust the hyperparameters in the sidebar to see how they affect the model's performance!")

