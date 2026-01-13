import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression, load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                            confusion_matrix, classification_report, roc_auc_score, roc_curve,
                            mean_squared_error, mean_absolute_error, r2_score)
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
import xgboost as xgb
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time

# Page configuration
st.set_page_config(
    page_title="XGBoost & GradientBoosting Hyperparameter Tuning",
    page_icon="🚀",
    layout="wide"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stAlert {
        margin-top: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🚀 XGBoost & GradientBoosting Explorer</h1>', unsafe_allow_html=True)
st.markdown("### Interactive Hyperparameter Tuning for Gradient Boosting Algorithms")
st.markdown("---")

# Sidebar configuration
st.sidebar.header("🎯 Model Configuration")

# Algorithm selection
algorithm = st.sidebar.selectbox(
    "Select Algorithm",
    ["XGBoost", "GradientBoosting (Sklearn)"],
    help="Choose between XGBoost and Scikit-learn's GradientBoosting"
)

# Problem type selection
problem_type = st.sidebar.selectbox(
    "Select Problem Type",
    ["Classification", "Regression"]
)

# Dataset selection
st.sidebar.markdown("---")
st.sidebar.header("📊 Dataset Selection")

if problem_type == "Classification":
    dataset_option = st.sidebar.selectbox(
        "Choose Dataset",
        ["Breast Cancer (Built-in)", "Synthetic Dataset"],
        help="Breast Cancer dataset shows clear changes with hyperparameter tuning"
    )
else:
    dataset_option = st.sidebar.selectbox(
        "Choose Dataset",
        ["Diabetes (Built-in)", "Synthetic Dataset"],
        help="Diabetes dataset shows clear changes with hyperparameter tuning"
    )

# Load dataset
@st.cache_data
def load_dataset(dataset_name, problem_type):
    if problem_type == "Classification":
        if dataset_name == "Breast Cancer (Built-in)":
            data = load_breast_cancer()
            X = pd.DataFrame(data.data, columns=data.feature_names)
            y = pd.Series(data.target, name='target')
            return X, y, data.feature_names, ['Malignant', 'Benign']
        else:
            X, y = make_classification(
                n_samples=1000,
                n_features=20,
                n_informative=15,
                n_redundant=3,
                n_classes=2,
                random_state=42,
                flip_y=0.1
            )
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
            X = pd.DataFrame(X, columns=feature_names)
            y = pd.Series(y, name='target')
            return X, y, feature_names, ['Class 0', 'Class 1']
    else:
        if dataset_name == "Diabetes (Built-in)":
            data = load_diabetes()
            X = pd.DataFrame(data.data, columns=data.feature_names)
            y = pd.Series(data.target, name='target')
            return X, y, data.feature_names
        else:
            X, y = make_regression(
                n_samples=1000,
                n_features=20,
                n_informative=15,
                noise=10,
                random_state=42
            )
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
            X = pd.DataFrame(X, columns=feature_names)
            y = pd.Series(y, name='target')
            return X, y, feature_names

# Load data
if problem_type == "Classification":
    X, y, feature_names, class_names = load_dataset(dataset_option, problem_type)
else:
    X, y, feature_names = load_dataset(dataset_option, problem_type)

# Train-test split configuration
st.sidebar.markdown("---")
st.sidebar.header("🔀 Train-Test Split")
test_size = st.sidebar.slider(
    "Test Size (%)",
    min_value=10,
    max_value=40,
    value=20,
    step=5
) / 100

random_state = st.sidebar.number_input(
    "Random State",
    min_value=0,
    max_value=100,
    value=42
)

# Hyperparameters
st.sidebar.markdown("---")
st.sidebar.header("🎛️ Hyperparameters")

# Common hyperparameters
n_estimators = st.sidebar.slider(
    "Number of Estimators (Trees)",
    min_value=10,
    max_value=500,
    value=100,
    step=10,
    help="Number of boosting stages. More trees = better fit but slower"
)

learning_rate = st.sidebar.slider(
    "Learning Rate",
    min_value=0.01,
    max_value=1.0,
    value=0.1,
    step=0.01,
    help="Step size shrinkage. Lower values prevent overfitting but require more trees"
)

max_depth = st.sidebar.slider(
    "Max Depth",
    min_value=1,
    max_value=15,
    value=3,
    step=1,
    help="Maximum depth of trees. Deeper trees = more complex patterns but risk overfitting"
)

min_child_weight = st.sidebar.slider(
    "Min Child Weight (XGBoost) / Min Samples Split",
    min_value=1,
    max_value=10,
    value=1,
    step=1,
    help="Minimum sum of instance weight needed in a child"
)

subsample = st.sidebar.slider(
    "Subsample Ratio",
    min_value=0.1,
    max_value=1.0,
    value=1.0,
    step=0.1,
    help="Fraction of samples used for each tree. Lower values prevent overfitting"
)

if algorithm == "XGBoost":
    colsample_bytree = st.sidebar.slider(
        "Column Sample by Tree",
        min_value=0.1,
        max_value=1.0,
        value=1.0,
        step=0.1,
        help="Fraction of features used per tree"
    )
    
    gamma = st.sidebar.slider(
        "Gamma (Min Split Loss)",
        min_value=0.0,
        max_value=10.0,
        value=0.0,
        step=0.1,
        help="Minimum loss reduction required to make a split"
    )

# Train button
st.sidebar.markdown("---")
train_button = st.sidebar.button("🚀 Train Model", type="primary", use_container_width=True)

# Show dataset info
st.header("📊 Dataset Overview")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Samples", X.shape[0])
with col2:
    st.metric("Features", X.shape[1])
with col3:
    st.metric("Training Samples", int(X.shape[0] * (1 - test_size)))
with col4:
    st.metric("Test Samples", int(X.shape[0] * test_size))

# Dataset preview
with st.expander("📋 View Dataset Preview"):
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Features (X)")
        st.dataframe(X.head(10), use_container_width=True)
    with col2:
        st.subheader("Target (y)")
        st.dataframe(pd.DataFrame(y.head(10)), use_container_width=True)
    
    # Target distribution
    if problem_type == "Classification":
        fig = px.histogram(y, title="Target Class Distribution", 
                          labels={'value': 'Class', 'count': 'Count'},
                          color=y)
    else:
        fig = px.histogram(y, title="Target Value Distribution",
                          labels={'value': 'Target Value', 'count': 'Count'})
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Training and evaluation
if train_button:
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Train model
    status_text.text("🔄 Training model...")
    progress_bar.progress(20)
    
    start_time = time.time()
    
    if algorithm == "XGBoost":
        if problem_type == "Classification":
            model = xgb.XGBClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                min_child_weight=min_child_weight,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                gamma=gamma,
                random_state=random_state,
                eval_metric='logloss',
                use_label_encoder=False
            )
        else:
            model = xgb.XGBRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                min_child_weight=min_child_weight,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                gamma=gamma,
                random_state=random_state
            )
    else:  # GradientBoosting
        if problem_type == "Classification":
            model = GradientBoostingClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                min_samples_split=min_child_weight,
                subsample=subsample,
                random_state=random_state
            )
        else:
            model = GradientBoostingRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                min_samples_split=min_child_weight,
                subsample=subsample,
                random_state=random_state
            )
    
    progress_bar.progress(40)
    model.fit(X_train, y_train)
    progress_bar.progress(70)
    
    # Predictions
    status_text.text("📊 Making predictions...")
    y_pred = model.predict(X_test)
    
    if problem_type == "Classification":
        y_pred_proba = model.predict_proba(X_test)
    
    training_time = time.time() - start_time
    progress_bar.progress(100)
    status_text.text("✅ Training complete!")
    
    time.sleep(0.5)
    progress_bar.empty()
    status_text.empty()
    
    # Display results
    st.header("📈 Model Performance")
    
    if problem_type == "Classification":
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Metrics display
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Accuracy", f"{accuracy:.4f}")
        with col2:
            st.metric("Precision", f"{precision:.4f}")
        with col3:
            st.metric("Recall", f"{recall:.4f}")
        with col4:
            st.metric("F1-Score", f"{f1:.4f}")
        with col5:
            st.metric("Training Time", f"{training_time:.2f}s")
        
        # Visualizations
        st.markdown("---")
        st.subheader("📊 Detailed Analysis")
        
        tab1, tab2, tab3, tab4 = st.tabs(["Confusion Matrix", "ROC Curve", "Feature Importance", "Classification Report"])
        
        with tab1:
            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            
            fig = go.Figure(data=go.Heatmap(
                z=cm,
                x=class_names if 'class_names' in locals() else ['Class 0', 'Class 1'],
                y=class_names if 'class_names' in locals() else ['Class 0', 'Class 1'],
                text=cm,
                texttemplate='%{text}',
                colorscale='Blues',
                showscale=True
            ))
            
            fig.update_layout(
                title='Confusion Matrix',
                xaxis_title='Predicted',
                yaxis_title='Actual',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            # ROC Curve
            if len(np.unique(y_test)) == 2:
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
                auc_score = roc_auc_score(y_test, y_pred_proba[:, 1])
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', 
                                        name=f'ROC Curve (AUC = {auc_score:.4f})',
                                        line=dict(color='blue', width=2)))
                fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                                        name='Random Classifier',
                                        line=dict(color='red', dash='dash')))
                
                fig.update_layout(
                    title='ROC Curve',
                    xaxis_title='False Positive Rate',
                    yaxis_title='True Positive Rate',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                st.metric("ROC AUC Score", f"{auc_score:.4f}")
            else:
                st.info("ROC Curve is only available for binary classification")
        
        with tab3:
            # Feature Importance
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                feature_importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importances
                }).sort_values('Importance', ascending=True)
                
                fig = px.bar(feature_importance_df.tail(20), 
                            x='Importance', 
                            y='Feature',
                            orientation='h',
                            title='Top 20 Feature Importances')
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
                
                with st.expander("📋 View All Feature Importances"):
                    st.dataframe(
                        feature_importance_df.sort_values('Importance', ascending=False),
                        use_container_width=True
                    )
        
        with tab4:
            # Classification Report
            report = classification_report(y_test, y_pred, 
                                          target_names=class_names if 'class_names' in locals() else None,
                                          output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df, use_container_width=True)
    
    else:  # Regression
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Metrics display
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("R² Score", f"{r2:.4f}")
        with col2:
            st.metric("RMSE", f"{rmse:.4f}")
        with col3:
            st.metric("MAE", f"{mae:.4f}")
        with col4:
            st.metric("MSE", f"{mse:.4f}")
        with col5:
            st.metric("Training Time", f"{training_time:.2f}s")
        
        # Visualizations
        st.markdown("---")
        st.subheader("📊 Detailed Analysis")
        
        tab1, tab2, tab3 = st.tabs(["Prediction vs Actual", "Residual Analysis", "Feature Importance"])
        
        with tab1:
            # Scatter plot: Predicted vs Actual
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=y_test,
                y=y_pred,
                mode='markers',
                name='Predictions',
                marker=dict(size=8, color='blue', opacity=0.6)
            ))
            
            # Perfect prediction line
            min_val = min(y_test.min(), y_pred.min())
            max_val = max(y_test.max(), y_pred.max())
            fig.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='red', dash='dash')
            ))
            
            fig.update_layout(
                title='Predicted vs Actual Values',
                xaxis_title='Actual Values',
                yaxis_title='Predicted Values',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            # Residual plot
            residuals = y_test - y_pred
            
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Residual Plot', 'Residual Distribution')
            )
            
            # Residual scatter
            fig.add_trace(
                go.Scatter(x=y_pred, y=residuals, mode='markers',
                          marker=dict(size=8, color='blue', opacity=0.6),
                          name='Residuals'),
                row=1, col=1
            )
            fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)
            
            # Residual histogram
            fig.add_trace(
                go.Histogram(x=residuals, name='Distribution',
                           marker=dict(color='lightblue')),
                row=1, col=2
            )
            
            fig.update_xaxes(title_text="Predicted Values", row=1, col=1)
            fig.update_yaxes(title_text="Residuals", row=1, col=1)
            fig.update_xaxes(title_text="Residuals", row=1, col=2)
            fig.update_yaxes(title_text="Count", row=1, col=2)
            
            fig.update_layout(height=500, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # Residual statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Mean Residual", f"{residuals.mean():.4f}")
            with col2:
                st.metric("Std Residual", f"{residuals.std():.4f}")
            with col3:
                st.metric("Min Residual", f"{residuals.min():.4f}")
            with col4:
                st.metric("Max Residual", f"{residuals.max():.4f}")
        
        with tab3:
            # Feature Importance
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                feature_importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importances
                }).sort_values('Importance', ascending=True)
                
                fig = px.bar(feature_importance_df.tail(20), 
                            x='Importance', 
                            y='Feature',
                            orientation='h',
                            title='Top 20 Feature Importances')
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
                
                with st.expander("📋 View All Feature Importances"):
                    st.dataframe(
                        feature_importance_df.sort_values('Importance', ascending=False),
                        use_container_width=True
                    )
    
    # Learning curves (if using GradientBoosting from sklearn)
    if algorithm == "GradientBoosting (Sklearn)":
        st.markdown("---")
        st.subheader("📉 Learning Curves")
        
        train_scores = []
        test_scores = []
        
        for i, (train_pred, test_pred) in enumerate(zip(
            model.staged_predict(X_train),
            model.staged_predict(X_test)
        )):
            if problem_type == "Classification":
                train_scores.append(accuracy_score(y_train, train_pred))
                test_scores.append(accuracy_score(y_test, test_pred))
            else:
                train_scores.append(r2_score(y_train, train_pred))
                test_scores.append(r2_score(y_test, test_pred))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(1, len(train_scores) + 1)),
            y=train_scores,
            mode='lines',
            name='Training Score',
            line=dict(color='blue', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=list(range(1, len(test_scores) + 1)),
            y=test_scores,
            mode='lines',
            name='Test Score',
            line=dict(color='red', width=2)
        ))
        
        fig.update_layout(
            title='Model Performance vs Number of Trees',
            xaxis_title='Number of Trees',
            yaxis_title='Accuracy' if problem_type == "Classification" else 'R² Score',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Find optimal number of trees
        optimal_n_trees = np.argmax(test_scores) + 1
        st.info(f"💡 **Insight**: Optimal number of trees is around **{optimal_n_trees}** for this configuration. "
               f"The test score is **{test_scores[optimal_n_trees-1]:.4f}**. "
               "Try adjusting other hyperparameters to improve performance!")

# Information section
st.markdown("---")
st.header("📚 Hyperparameter Guide")

with st.expander("🔍 Understanding Hyperparameters"):
    st.markdown("""
    ### Key Hyperparameters and Their Effects:
    
    **1. Number of Estimators (Trees)**
    - More trees generally improve performance but increase training time
    - Too many trees can lead to overfitting
    - **Tip**: Start with 100 and adjust based on learning curves
    
    **2. Learning Rate**
    - Controls how much each tree contributes to the final prediction
    - Lower values (0.01-0.1) are more conservative and often better
    - Lower learning rate requires more trees
    - **Tip**: Use lower learning rate with more trees for better generalization
    
    **3. Max Depth**
    - Controls the maximum depth of each tree
    - Deeper trees can capture more complex patterns
    - Too deep can lead to overfitting
    - **Tip**: Start with 3-5 for most problems
    
    **4. Subsample**
    - Fraction of samples used to train each tree
    - Values < 1.0 introduce randomness and prevent overfitting
    - **Tip**: Try 0.8 to reduce overfitting
    
    **5. Min Child Weight (XGBoost)**
    - Minimum sum of instance weight needed in a child
    - Higher values prevent overfitting
    - **Tip**: Increase if you see overfitting
    
    **6. Gamma (XGBoost)**
    - Minimum loss reduction required to make a split
    - Higher values make the algorithm more conservative
    - **Tip**: Start with 0 and increase if overfitting
    
    ### Common Patterns:
    - **High training score, low test score** → Overfitting
      - *Solution*: Reduce max_depth, increase min_child_weight, reduce learning_rate
    
    - **Low training and test scores** → Underfitting
      - *Solution*: Increase max_depth, increase n_estimators, increase learning_rate
    
    - **Gap between train and test scores increasing** → Overfitting
      - *Solution*: Add regularization (gamma, subsample < 1.0)
    """)

with st.expander("🎯 Quick Tuning Tips"):
    st.markdown("""
    ### Step-by-step Tuning Strategy:
    
    1. **Start with defaults**: Use default parameters to get a baseline
    
    2. **Tune n_estimators and learning_rate together**:
       - Try learning_rate = 0.1 with n_estimators = 100-300
       - Try learning_rate = 0.01 with n_estimators = 1000-3000
    
    3. **Tune tree-specific parameters**:
       - Adjust max_depth (3-10)
       - Adjust min_child_weight (1-10)
    
    4. **Add randomness to prevent overfitting**:
       - Set subsample to 0.8
       - Set colsample_bytree to 0.8 (XGBoost)
    
    5. **Fine-tune regularization**:
       - Increase gamma if still overfitting (XGBoost)
    
    ### Example Configurations:
    
    **Conservative (Less Overfitting)**:
    - Learning Rate: 0.05
    - N Estimators: 200
    - Max Depth: 3
    - Subsample: 0.8
    
    **Balanced**:
    - Learning Rate: 0.1
    - N Estimators: 100
    - Max Depth: 5
    - Subsample: 1.0
    
    **Aggressive (More Complex Patterns)**:
    - Learning Rate: 0.3
    - N Estimators: 50
    - Max Depth: 8
    - Subsample: 1.0
    """)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>Built with ❤️ using Streamlit | XGBoost & GradientBoosting Explorer</p>
        <p>💡 Experiment with different hyperparameters to see their impact on model performance!</p>
    </div>
    """, unsafe_allow_html=True)
