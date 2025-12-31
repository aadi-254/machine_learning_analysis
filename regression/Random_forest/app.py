import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.datasets import make_classification, make_regression
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="Random Forest Algorithm Explorer",
    page_icon="🌲",
    layout="wide"
)

# Title and description
st.title("🌲 Random Forest Algorithm Explorer")
st.markdown("### Interactive ML Application for Classification & Regression")
st.markdown("---")

# Sidebar for parameters
st.sidebar.header("⚙️ Random Forest Parameters")

# Problem type selection
problem_type = st.sidebar.selectbox(
    "Select Problem Type",
    ["Classification", "Regression"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("Model Parameters")

# Number of estimators (decision trees)
n_estimators = st.sidebar.slider(
    "Number of Estimators (Decision Trees)",
    min_value=10,
    max_value=500,
    value=100,
    step=10,
    help="Number of decision trees in the forest"
)

# Max features
max_features_option = st.sidebar.selectbox(
    "Max Features",
    ["sqrt", "log2", "None (all features)"],
    help="Number of features to consider when looking for the best split"
)
max_features = None if max_features_option == "None (all features)" else max_features_option.replace(" (all features)", "")

# Max samples
max_samples = st.sidebar.slider(
    "Max Samples (fraction of rows)",
    min_value=0.1,
    max_value=1.0,
    value=0.8,
    step=0.1,
    help="Fraction of samples to use for training each tree"
)

# Bootstrap
bootstrap = st.sidebar.checkbox(
    "Bootstrap",
    value=True,
    help="Whether bootstrap samples are used when building trees"
)

# Additional parameters
st.sidebar.markdown("---")
st.sidebar.subheader("Additional Parameters")

max_depth = st.sidebar.slider(
    "Max Depth",
    min_value=1,
    max_value=50,
    value=None,
    step=1,
    help="Maximum depth of each tree (None = unlimited)"
)
if max_depth == 1:
    max_depth = None

min_samples_split = st.sidebar.slider(
    "Min Samples Split",
    min_value=2,
    max_value=20,
    value=2,
    step=1,
    help="Minimum number of samples required to split an internal node"
)

min_samples_leaf = st.sidebar.slider(
    "Min Samples Leaf",
    min_value=1,
    max_value=20,
    value=1,
    step=1,
    help="Minimum number of samples required to be at a leaf node"
)

# Test size
test_size = st.sidebar.slider(
    "Test Set Size",
    min_value=0.1,
    max_value=0.5,
    value=0.2,
    step=0.05,
    help="Fraction of data to use for testing"
)

random_state = st.sidebar.number_input(
    "Random State",
    min_value=0,
    max_value=1000,
    value=42,
    help="Random state for reproducibility"
)

# Generate dataset button
if st.sidebar.button("🎲 Generate New Dataset", type="primary"):
    st.rerun()

# Main content area
tab1, tab2, tab3 = st.tabs(["📊 Dataset & Results", "📈 Visualizations", "ℹ️ About"])

with tab1:
    # Generate dataset based on problem type
    if problem_type == "Classification":
        st.subheader("🎯 Classification Problem")
        
        # Generate classification dataset
        X, y = make_classification(
            n_samples=700,
            n_features=10,
            n_informative=8,
            n_redundant=2,
            n_classes=3,
            random_state=random_state,
            class_sep=1.0
        )
        
        # Create DataFrame
        feature_names = [f"Feature_{i+1}" for i in range(X.shape[1])]
        df = pd.DataFrame(X, columns=feature_names)
        df['Target'] = y
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.write("**Dataset Info:**")
            st.write(f"- Total Samples: {len(df)}")
            st.write(f"- Number of Features: {X.shape[1]}")
            st.write(f"- Number of Classes: {len(np.unique(y))}")
            st.write(f"- Class Distribution:")
            for cls in np.unique(y):
                count = np.sum(y == cls)
                st.write(f"  - Class {cls}: {count} samples ({count/len(y)*100:.1f}%)")
        
        with col2:
            st.write("**Dataset Preview:**")
            st.dataframe(df.head(10), use_container_width=True)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Train Random Forest Classifier
        with st.spinner("🌲 Training Random Forest Classifier..."):
            rf_classifier = RandomForestClassifier(
                n_estimators=n_estimators,
                max_features=max_features if max_features != "None" else None,
                max_samples=max_samples if bootstrap else None,
                bootstrap=bootstrap,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=random_state,
                n_jobs=-1
            )
            rf_classifier.fit(X_train, y_train)
            
            # Predictions
            y_pred_train = rf_classifier.predict(X_train)
            y_pred_test = rf_classifier.predict(X_test)
            
            # Metrics
            train_accuracy = accuracy_score(y_train, y_pred_train)
            test_accuracy = accuracy_score(y_test, y_pred_test)
        
        st.success("✅ Model trained successfully!")
        
        # Display metrics
        st.subheader("📊 Model Performance")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Train Accuracy", f"{train_accuracy:.4f}")
        col2.metric("Test Accuracy", f"{test_accuracy:.4f}")
        col3.metric("Training Samples", len(X_train))
        col4.metric("Test Samples", len(X_test))
        
        # Classification report
        st.subheader("📋 Classification Report")
        report = classification_report(y_test, y_pred_test, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.background_gradient(cmap='Blues'), use_container_width=True)
        
        # Feature importance
        st.subheader("🎯 Feature Importance")
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': rf_classifier.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        fig_importance = px.bar(
            feature_importance,
            x='Importance',
            y='Feature',
            orientation='h',
            title='Feature Importance',
            color='Importance',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig_importance, use_container_width=True)
        
    else:  # Regression
        st.subheader("📈 Regression Problem")
        
        # Generate regression dataset
        X, y = make_regression(
            n_samples=700,
            n_features=10,
            n_informative=8,
            noise=10.0,
            random_state=random_state
        )
        
        # Create DataFrame
        feature_names = [f"Feature_{i+1}" for i in range(X.shape[1])]
        df = pd.DataFrame(X, columns=feature_names)
        df['Target'] = y
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.write("**Dataset Info:**")
            st.write(f"- Total Samples: {len(df)}")
            st.write(f"- Number of Features: {X.shape[1]}")
            st.write(f"- Target Range: [{y.min():.2f}, {y.max():.2f}]")
            st.write(f"- Target Mean: {y.mean():.2f}")
            st.write(f"- Target Std: {y.std():.2f}")
        
        with col2:
            st.write("**Dataset Preview:**")
            st.dataframe(df.head(10), use_container_width=True)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Train Random Forest Regressor
        with st.spinner("🌲 Training Random Forest Regressor..."):
            rf_regressor = RandomForestRegressor(
                n_estimators=n_estimators,
                max_features=max_features if max_features != "None" else None,
                max_samples=max_samples if bootstrap else None,
                bootstrap=bootstrap,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=random_state,
                n_jobs=-1
            )
            rf_regressor.fit(X_train, y_train)
            
            # Predictions
            y_pred_train = rf_regressor.predict(X_train)
            y_pred_test = rf_regressor.predict(X_test)
            
            # Metrics
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            train_mae = mean_absolute_error(y_train, y_pred_train)
            test_mae = mean_absolute_error(y_test, y_pred_test)
        
        st.success("✅ Model trained successfully!")
        
        # Display metrics
        st.subheader("📊 Model Performance")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Train R² Score", f"{train_r2:.4f}")
        col2.metric("Test R² Score", f"{test_r2:.4f}")
        col3.metric("Overfitting", f"{(train_r2 - test_r2):.4f}")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Train RMSE", f"{train_rmse:.4f}")
        col2.metric("Test RMSE", f"{test_rmse:.4f}")
        col3.metric("Test MAE", f"{test_mae:.4f}")
        
        # Feature importance
        st.subheader("🎯 Feature Importance")
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': rf_regressor.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        fig_importance = px.bar(
            feature_importance,
            x='Importance',
            y='Feature',
            orientation='h',
            title='Feature Importance',
            color='Importance',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig_importance, use_container_width=True)

with tab2:
    st.subheader("📈 Visualizations")
    
    if problem_type == "Classification":
        # Confusion Matrix
        st.write("**Confusion Matrix**")
        cm = confusion_matrix(y_test, y_pred_test)
        
        fig_cm = px.imshow(
            cm,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=[f"Class {i}" for i in range(len(cm))],
            y=[f"Class {i}" for i in range(len(cm))],
            color_continuous_scale='Blues',
            text_auto=True,
            title="Confusion Matrix"
        )
        st.plotly_chart(fig_cm, use_container_width=True)
        
        # Prediction distribution
        col1, col2 = st.columns(2)
        
        with col1:
            fig_actual = px.histogram(
                x=y_test,
                nbins=len(np.unique(y)),
                title="Actual Class Distribution",
                labels={'x': 'Class', 'y': 'Count'},
                color_discrete_sequence=['#636EFA']
            )
            st.plotly_chart(fig_actual, use_container_width=True)
        
        with col2:
            fig_pred = px.histogram(
                x=y_pred_test,
                nbins=len(np.unique(y)),
                title="Predicted Class Distribution",
                labels={'x': 'Class', 'y': 'Count'},
                color_discrete_sequence=['#EF553B']
            )
            st.plotly_chart(fig_pred, use_container_width=True)
            
    else:  # Regression
        # Actual vs Predicted
        fig_scatter = go.Figure()
        
        fig_scatter.add_trace(go.Scatter(
            x=y_test,
            y=y_pred_test,
            mode='markers',
            name='Predictions',
            marker=dict(size=8, color='#636EFA', opacity=0.6)
        ))
        
        # Perfect prediction line
        min_val = min(y_test.min(), y_pred_test.min())
        max_val = max(y_test.max(), y_pred_test.max())
        fig_scatter.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', dash='dash')
        ))
        
        fig_scatter.update_layout(
            title='Actual vs Predicted Values',
            xaxis_title='Actual Values',
            yaxis_title='Predicted Values',
            height=500
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Residual plot
        residuals = y_test - y_pred_test
        
        fig_residual = go.Figure()
        fig_residual.add_trace(go.Scatter(
            x=y_pred_test,
            y=residuals,
            mode='markers',
            marker=dict(size=8, color='#EF553B', opacity=0.6)
        ))
        
        fig_residual.add_hline(y=0, line_dash="dash", line_color="black")
        fig_residual.update_layout(
            title='Residual Plot',
            xaxis_title='Predicted Values',
            yaxis_title='Residuals',
            height=500
        )
        st.plotly_chart(fig_residual, use_container_width=True)
        
        # Distribution of predictions
        col1, col2 = st.columns(2)
        
        with col1:
            fig_actual_dist = px.histogram(
                x=y_test,
                nbins=30,
                title="Actual Values Distribution",
                labels={'x': 'Value', 'y': 'Count'},
                color_discrete_sequence=['#636EFA']
            )
            st.plotly_chart(fig_actual_dist, use_container_width=True)
        
        with col2:
            fig_pred_dist = px.histogram(
                x=y_pred_test,
                nbins=30,
                title="Predicted Values Distribution",
                labels={'x': 'Value', 'y': 'Count'},
                color_discrete_sequence=['#EF553B']
            )
            st.plotly_chart(fig_pred_dist, use_container_width=True)

with tab3:
    st.subheader("ℹ️ About Random Forest Algorithm")
    
    st.markdown("""
    ### What is Random Forest?
    
    Random Forest is an ensemble learning method that operates by constructing multiple decision trees 
    during training and outputting the class that is the mode of the classes (classification) or mean 
    prediction (regression) of the individual trees.
    
    ### Key Parameters Explained:
    
    **1. Number of Estimators (n_estimators)**
    - Number of decision trees in the forest
    - More trees = better performance but slower training
    - Typical range: 100-500
    
    **2. Max Features**
    - Number of features to consider when looking for the best split
    - `sqrt`: Square root of total features (good for classification)
    - `log2`: Log base 2 of total features
    - `None`: Use all features
    
    **3. Max Samples**
    - Fraction of samples to use for training each tree
    - Controls the size of bootstrap samples
    - Range: 0.1 to 1.0
    
    **4. Bootstrap**
    - Whether to use bootstrap sampling
    - `True`: Sample with replacement (more diversity)
    - `False`: Use entire dataset for each tree
    
    **5. Max Depth**
    - Maximum depth of each tree
    - Controls model complexity
    - `None` = unlimited depth
    
    **6. Min Samples Split**
    - Minimum samples required to split an internal node
    - Higher values prevent overfitting
    
    **7. Min Samples Leaf**
    - Minimum samples required at leaf node
    - Higher values create simpler models
    
    ### Advantages:
    - ✅ Handles both classification and regression
    - ✅ Robust to outliers and noise
    - ✅ Feature importance ranking
    - ✅ Handles missing values well
    - ✅ Reduces overfitting compared to single decision trees
    
    ### Use Cases:
    - 🎯 Classification: spam detection, image classification, customer churn
    - 📈 Regression: price prediction, demand forecasting, risk assessment
    
    ### For More Information:
    Visit: **https://github.com/aadi-254/machine_learning**
    """)
    
    st.markdown("---")
    st.info("💡 **Tip**: Adjust the parameters in the sidebar to see how they affect model performance!")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 Resources")
st.sidebar.markdown("[GitHub Repository](https://github.com/aadi-254/machine_learning)")
st.sidebar.markdown("Made with ❤️ using Streamlit")
