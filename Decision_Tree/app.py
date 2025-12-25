import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score, mean_squared_error, mean_absolute_error
import plotly.graph_objects as go
import time

# Page configuration
st.set_page_config(page_title="Decision Tree Hyperparameter Tuning", layout="wide")

# Title
st.title("🌳 Decision Tree Hyperparameter Tuning")
st.markdown("---")

# Sidebar for model selection and hyperparameters
st.sidebar.header("Model Configuration")

# Select problem type
problem_type = st.sidebar.selectbox(
    "Select Problem Type",
    ["Classification", "Regression"]
)

# Select mode
mode = st.sidebar.radio(
    "Select Mode",
    ["Manual Tuning", "Grid Search CV"],
    help="Manual: Adjust hyperparameters manually. Grid Search: Find best parameters automatically."
)

st.sidebar.markdown("---")

if mode == "Grid Search CV":
    st.sidebar.header("Grid Search Configuration")
    cv_folds = st.sidebar.slider(
        "Cross-Validation Folds",
        min_value=2,
        max_value=10,
        value=5,
        help="Number of folds for cross-validation"
    )
    st.sidebar.markdown("---")
    st.sidebar.info("⚡ Grid Search will test multiple parameter combinations to find the best model.")

st.sidebar.markdown("---")
st.sidebar.header("Hyperparameters")

if mode == "Manual Tuning":
    # Hyperparameter controls for manual tuning
    max_depth = st.sidebar.slider(
        "Max Depth of Tree",
        min_value=1,
        max_value=30,
        value=5,
        help="Maximum depth of the tree. Deeper trees can model more complex patterns."
    )

    if problem_type == "Classification":
        criterion = st.sidebar.selectbox(
            "Criterion",
            ["gini", "entropy"],
            help="Function to measure the quality of a split"
        )
    else:
        criterion = st.sidebar.selectbox(
            "Criterion",
            ["squared_error", "friedman_mse", "absolute_error", "poisson"],
            help="Function to measure the quality of a split"
        )

    splitter = st.sidebar.selectbox(
        "Splitter",
        ["best", "random"],
        help="Strategy to choose the split at each node"
    )

    min_samples_split = st.sidebar.slider(
        "Min Samples Split",
        min_value=2,
        max_value=20,
        value=2,
        help="Minimum number of samples required to split an internal node"
    )

    min_samples_leaf = st.sidebar.slider(
        "Min Samples Leaf",
        min_value=1,
        max_value=20,
        value=1,
        help="Minimum number of samples required to be at a leaf node"
    )

    max_leaf_nodes = st.sidebar.slider(
        "Max Leaf Nodes",
        min_value=2,
        max_value=100,
        value=None,
        help="Maximum number of leaf nodes"
    )

    if max_leaf_nodes == 2:
        max_leaf_nodes = None

    min_impurity_decrease = st.sidebar.slider(
        "Min Impurity Decrease",
        min_value=0.0,
        max_value=0.5,
        value=0.0,
        step=0.01,
        help="A node will be split if this split induces a decrease of the impurity greater than or equal to this value"
    )

    ccp_alpha = st.sidebar.slider(
        "CCP Alpha (Cost Complexity Pruning)",
        min_value=0.0,
        max_value=0.1,
        value=0.0,
        step=0.001,
        help="Complexity parameter for post-pruning. Higher values prune more nodes."
    )
else:
    st.sidebar.info("📊 Grid Search will automatically test multiple parameter combinations.")
    run_grid_search = st.sidebar.button("🚀 Run Grid Search", type="primary")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header(f"{problem_type} Model - {mode}")
    
    # Load appropriate dataset
    try:
        if problem_type == "Classification":
            data = pd.read_csv('classification_data.csv')
            st.success("✅ Classification dataset loaded (500 rows)")
        else:
            data = pd.read_csv('regression_data.csv')
            st.success("✅ Regression dataset loaded (500 rows)")
        
        # Display dataset info
        with st.expander("📊 View Dataset Information"):
            st.write(f"**Dataset Shape:** {data.shape}")
            st.write("**First 5 rows:**")
            st.dataframe(data.head())
            st.write("**Statistical Summary:**")
            st.dataframe(data.describe())
        
        # Prepare data
        X = data.drop('target', axis=1)
        y = data['target']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Mode-specific training
        if mode == "Manual Tuning":
            # Train model with manual parameters
            if problem_type == "Classification":
                model = DecisionTreeClassifier(
                    max_depth=max_depth,
                    criterion=criterion,
                    splitter=splitter,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    max_leaf_nodes=max_leaf_nodes,
                    min_impurity_decrease=min_impurity_decrease,
                    ccp_alpha=ccp_alpha,
                    random_state=42
                )
            else:
                model = DecisionTreeRegressor(
                    max_depth=max_depth,
                    criterion=criterion,
                    splitter=splitter,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    max_leaf_nodes=max_leaf_nodes,
                    min_impurity_decrease=min_impurity_decrease,
                    ccp_alpha=ccp_alpha,
                    random_state=42
                )
            
            # Fit model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
        else:  # Grid Search CV mode
            if 'run_grid_search' in locals() and run_grid_search:
                st.info("🔍 Running Grid Search... This may take a few moments.")
                
                # Define parameter grid
                if problem_type == "Classification":
                    param_grid = {
                        'max_depth': [3, 5, 7, 10, 15, None],
                        'criterion': ['gini', 'entropy'],
                        'splitter': ['best', 'random'],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 4],
                        'max_leaf_nodes': [None, 10, 20, 50],
                        'min_impurity_decrease': [0.0, 0.01, 0.05],
                        'ccp_alpha': [0.0, 0.01, 0.05]
                    }
                    base_model = DecisionTreeClassifier(random_state=42)
                    scoring = 'accuracy'
                else:
                    param_grid = {
                        'max_depth': [3, 5, 7, 10, 15, None],
                        'criterion': ['squared_error', 'friedman_mse', 'absolute_error'],
                        'splitter': ['best', 'random'],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 4],
                        'max_leaf_nodes': [None, 10, 20, 50],
                        'min_impurity_decrease': [0.0, 0.01, 0.05],
                        'ccp_alpha': [0.0, 0.01, 0.05]
                    }
                    base_model = DecisionTreeRegressor(random_state=42)
                    scoring = 'r2'
                
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                start_time = time.time()
                
                # Run Grid Search
                grid_search = GridSearchCV(
                    base_model,
                    param_grid,
                    cv=cv_folds,
                    scoring=scoring,
                    n_jobs=-1,
                    verbose=0
                )
                
                grid_search.fit(X_train, y_train)
                
                elapsed_time = time.time() - start_time
                progress_bar.progress(100)
                status_text.success(f"✅ Grid Search completed in {elapsed_time:.2f} seconds!")
                
                # Get best model
                model = grid_search.best_estimator_
                
                # Make predictions
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)
                
                # Display best parameters
                st.markdown("---")
                st.subheader("🏆 Best Parameters Found")
                
                best_params_col1, best_params_col2 = st.columns(2)
                
                params_list = list(grid_search.best_params_.items())
                mid_point = len(params_list) // 2
                
                with best_params_col1:
                    for param, value in params_list[:mid_point]:
                        st.metric(param, str(value))
                
                with best_params_col2:
                    for param, value in params_list[mid_point:]:
                        st.metric(param, str(value))
                
                st.info(f"**Best CV Score ({scoring}):** {grid_search.best_score_:.4f}")
                
                # Show top 5 parameter combinations
                with st.expander("🔍 View Top 5 Parameter Combinations"):
                    results_df = pd.DataFrame(grid_search.cv_results_)
                    results_df = results_df.sort_values('rank_test_score')
                    top_results = results_df[['params', 'mean_test_score', 'std_test_score', 'rank_test_score']].head(5)
                    st.dataframe(top_results, use_container_width=True)
                
            else:
                st.warning("⚠️ Click the 'Run Grid Search' button in the sidebar to start the search.")
                st.stop()
        
        # Calculate metrics
        st.markdown("---")
        st.subheader("📈 Model Performance")
        
        metric_col1, metric_col2 = st.columns(2)
        
        if problem_type == "Classification":
            train_accuracy = accuracy_score(y_train, y_pred_train)
            test_accuracy = accuracy_score(y_test, y_pred_test)
            
            with metric_col1:
                st.metric("Training Accuracy", f"{train_accuracy:.4f}")
                st.metric("Precision (Test)", f"{precision_score(y_test, y_pred_test, average='weighted'):.4f}")
            
            with metric_col2:
                st.metric("Test Accuracy", f"{test_accuracy:.4f}")
                st.metric("Recall (Test)", f"{recall_score(y_test, y_pred_test, average='weighted'):.4f}")
            
            st.metric("F1 Score (Test)", f"{f1_score(y_test, y_pred_test, average='weighted'):.4f}")
            
            # Feature importance
            st.markdown("---")
            st.subheader("🔍 Feature Importance")
            feature_importance = pd.DataFrame({
                'Feature': X.columns,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            fig = go.Figure(go.Bar(
                x=feature_importance['Importance'],
                y=feature_importance['Feature'],
                orientation='h',
                marker=dict(color='lightblue')
            ))
            fig.update_layout(
                title="Feature Importance",
                xaxis_title="Importance",
                yaxis_title="Feature",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            test_mse = mean_squared_error(y_test, y_pred_test)
            test_rmse = np.sqrt(test_mse)
            test_mae = mean_absolute_error(y_test, y_pred_test)
            
            with metric_col1:
                st.metric("Training R² Score", f"{train_r2:.4f}")
                st.metric("Test MSE", f"{test_mse:.4f}")
                st.metric("Test RMSE", f"{test_rmse:.4f}")
            
            with metric_col2:
                st.metric("Test R² Score", f"{test_r2:.4f}")
                st.metric("Test MAE", f"{test_mae:.4f}")
            
            # Feature importance
            st.markdown("---")
            st.subheader("🔍 Feature Importance")
            feature_importance = pd.DataFrame({
                'Feature': X.columns,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            fig = go.Figure(go.Bar(
                x=feature_importance['Importance'],
                y=feature_importance['Feature'],
                orientation='h',
                marker=dict(color='lightgreen')
            ))
            fig.update_layout(
                title="Feature Importance",
                xaxis_title="Importance",
                yaxis_title="Feature",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Prediction vs Actual plot
            st.markdown("---")
            st.subheader("📊 Predictions vs Actual Values")
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=y_test,
                y=y_pred_test,
                mode='markers',
                name='Predictions',
                marker=dict(color='blue', size=8, opacity=0.6)
            ))
            fig2.add_trace(go.Scatter(
                x=[y_test.min(), y_test.max()],
                y=[y_test.min(), y_test.max()],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='red', dash='dash')
            ))
            fig2.update_layout(
                xaxis_title="Actual Values",
                yaxis_title="Predicted Values",
                height=400
            )
            st.plotly_chart(fig2, use_container_width=True)
        
    except FileNotFoundError:
        st.error("⚠️ Dataset files not found. Please run the notebook to generate datasets first.")
    except Exception as e:
        st.error(f"⚠️ An error occurred: {str(e)}")

with col2:
    st.header("ℹ️ Model Info")
    
    if mode == "Manual Tuning":
        st.markdown(f"""
        **Current Configuration:**
        - **Problem Type:** {problem_type}
        - **Max Depth:** {max_depth}
        - **Criterion:** {criterion}
        - **Splitter:** {splitter}
        - **Min Samples Split:** {min_samples_split}
        - **Min Samples Leaf:** {min_samples_leaf}
        - **Max Leaf Nodes:** {max_leaf_nodes if max_leaf_nodes else 'None'}
        - **Min Impurity Decrease:** {min_impurity_decrease}
        - **CCP Alpha:** {ccp_alpha}
        """)
    else:
        st.markdown(f"""
        **Current Mode:**
        - **Problem Type:** {problem_type}
        - **Mode:** Grid Search CV
        - **CV Folds:** {cv_folds}
        - **Status:** {'✅ Completed' if 'model' in locals() else '⏳ Waiting'}
        """)
    
    if 'model' in locals():
        st.markdown("---")
        st.markdown(f"""
        **Tree Info:**
        - **Number of Leaves:** {model.get_n_leaves()}
        - **Tree Depth:** {model.get_depth()}
        """)

st.markdown("---")
st.markdown("""
### 📚 Hyperparameter Guide:
- **Max Depth**: Controls how deep the tree can grow. Higher values may lead to overfitting.
- **Criterion**: Measures split quality (gini/entropy for classification, mse/mae for regression).
- **Splitter**: 'best' chooses best split, 'random' adds randomness.
- **Min Samples Split**: Minimum samples needed to split a node.
- **Min Samples Leaf**: Minimum samples required at leaf nodes.
- **Max Leaf Nodes**: Limits the number of leaf nodes.
- **Min Impurity Decrease**: Minimum decrease in impurity required for a split.
- **CCP Alpha**: Cost complexity pruning parameter for post-pruning. Higher values prune more nodes.

### 🔍 Grid Search CV:
Grid Search performs an exhaustive search over specified parameter values to find the best combination.
It uses cross-validation to evaluate each parameter combination and prevents overfitting.
""")