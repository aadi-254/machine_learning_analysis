import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="KNN Parameter Impact Visualizer",
    page_icon="🎯",
    layout="wide"
)

# Title and description
st.title("🎯 KNN Hyperparameter Impact Visualizer")
st.markdown("### See How Parameters Affect Decision Boundaries & Accuracy")
st.markdown("---")

# Generate dataset
@st.cache_data
def generate_dataset(dataset_type, n_samples, noise):
    if dataset_type == "Moons":
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)
    elif dataset_type == "Circles":
        X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=42)
    else:  # Blobs
        X, y = make_classification(n_samples=n_samples, n_features=2, n_redundant=0, 
                                   n_informative=2, n_clusters_per_class=1, 
                                   random_state=42, class_sep=1.5)
        X += noise * np.random.randn(*X.shape)
    return X, y

# Sidebar for dataset and parameter selection
st.sidebar.header("📊 Dataset Configuration")

dataset_type = st.sidebar.selectbox(
    "Dataset Type",
    ["Moons", "Circles", "Blobs"],
    help="Choose dataset pattern"
)

n_samples = st.sidebar.slider(
    "Number of Samples",
    min_value=100,
    max_value=500,
    value=300,
    step=50
)

noise = st.sidebar.slider(
    "Noise Level",
    min_value=0.0,
    max_value=0.3,
    value=0.1,
    step=0.05
)

# Generate data
X, y = generate_dataset(dataset_type, n_samples, noise)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

st.sidebar.markdown("---")
st.sidebar.header("🔧 Current Hyperparameters")

# Main hyperparameters
n_neighbors = st.sidebar.slider(
    "n_neighbors",
    min_value=1,
    max_value=50,
    value=5,
    help="Number of neighbors - Lower = more complex boundaries"
)

weights = st.sidebar.radio(
    "weights",
    ["uniform", "distance"],
    help="uniform: all neighbors equal | distance: closer neighbors matter more"
)

metric = st.sidebar.selectbox(
    "metric",
    ["euclidean", "manhattan", "chebyshev"],
    help="Distance calculation method"
)

p = st.sidebar.slider(
    "p (Minkowski power)",
    min_value=1,
    max_value=5,
    value=2,
    help="1=Manhattan, 2=Euclidean"
)

algorithm = st.sidebar.selectbox(
    "algorithm",
    ["auto", "ball_tree", "kd_tree", "brute"],
    help="Neighbor search method"
)

leaf_size = st.sidebar.slider(
    "leaf_size",
    min_value=10,
    max_value=100,
    value=30,
    step=10,
    help="Affects tree algorithm speed"
)

n_jobs = st.sidebar.selectbox(
    "n_jobs",
    [-1, 1],
    format_func=lambda x: "All CPUs" if x == -1 else "1 CPU",
    help="Parallel processing"
)

# Function to create decision boundary
def plot_decision_boundary(X, y, model, title):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    fig = go.Figure()
    
    # Decision boundary
    fig.add_trace(go.Contour(
        x=np.arange(x_min, x_max, h),
        y=np.arange(y_min, y_max, h),
        z=Z,
        colorscale=[[0, 'lightblue'], [1, 'lightcoral']],
        showscale=False,
        opacity=0.5,
        hoverinfo='skip'
    ))
    
    # Data points
    for class_val in np.unique(y):
        mask = y == class_val
        fig.add_trace(go.Scatter(
            x=X[mask, 0],
            y=X[mask, 1],
            mode='markers',
            name=f'Class {class_val}',
            marker=dict(size=8, line=dict(width=1, color='white'))
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Feature 1",
        yaxis_title="Feature 2",
        showlegend=True,
        height=500
    )
    return fig

# Train current model
knn = KNeighborsClassifier(
    n_neighbors=n_neighbors,
    weights=weights,
    metric=metric,
    p=p,
    algorithm=algorithm,
    leaf_size=leaf_size,
    n_jobs=n_jobs
)
knn.fit(X_train_scaled, y_train)

# Predictions
y_train_pred = knn.predict(X_train_scaled)
y_test_pred = knn.predict(X_test_scaled)

train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)

# Main content area
tab1, tab2, tab3 = st.tabs(["🎨 Decision Boundaries", "📊 Parameter Impact Analysis", "🔍 Hyperparameter Comparison"])

with tab1:
    st.header("Decision Boundary Visualization")
    st.markdown("**See how your current parameters shape the decision regions**")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Training Accuracy", f"{train_acc:.1%}")
    col2.metric("Test Accuracy", f"{test_acc:.1%}")
    col3.metric("Overfitting", f"{(train_acc - test_acc):.1%}", 
                delta_color="inverse")
    
    st.markdown("---")
    
    # Show decision boundary
    fig = plot_decision_boundary(X_train_scaled, y_train, knn, 
                                 f"Decision Boundary (k={n_neighbors}, {weights})")
    st.plotly_chart(fig, use_container_width=True)
    
    st.info(f"💡 **Current Settings**: {n_neighbors} neighbors, {weights} weighting, {metric} distance")

with tab2:
    st.header("How Hyperparameters Affect Performance")
    st.markdown("**Watch accuracy change as you vary each parameter**")
    
    # Analysis: n_neighbors impact
    st.subheader("1️⃣ Impact of n_neighbors")
    k_values = range(1, 51, 2)
    train_scores = []
    test_scores = []
    
    progress_bar = st.progress(0)
    for i, k in enumerate(k_values):
        knn_temp = KNeighborsClassifier(n_neighbors=k, weights=weights, metric=metric, p=p)
        knn_temp.fit(X_train_scaled, y_train)
        train_scores.append(knn_temp.score(X_train_scaled, y_train))
        test_scores.append(knn_temp.score(X_test_scaled, y_test))
        progress_bar.progress((i + 1) / len(k_values))
    
    progress_bar.empty()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(k_values), y=train_scores, mode='lines+markers',
                            name='Training Accuracy', line=dict(color='blue', width=2)))
    fig.add_trace(go.Scatter(x=list(k_values), y=test_scores, mode='lines+markers',
                            name='Test Accuracy', line=dict(color='red', width=2)))
    fig.add_vline(x=n_neighbors, line_dash="dash", line_color="green",
                 annotation_text=f"Current k={n_neighbors}")
    fig.update_layout(
        title="Accuracy vs Number of Neighbors",
        xaxis_title="Number of Neighbors (k)",
        yaxis_title="Accuracy",
        height=400,
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Find optimal k
    optimal_k = list(k_values)[np.argmax(test_scores)]
    st.success(f"📈 **Optimal k for test accuracy**: {optimal_k} (Accuracy: {max(test_scores):.1%})")
    
    st.markdown("---")
    
    # Analysis: weights comparison
    st.subheader("2️⃣ Impact of weights (uniform vs distance)")
    
    weights_comparison = []
    for w in ['uniform', 'distance']:
        knn_temp = KNeighborsClassifier(n_neighbors=n_neighbors, weights=w, metric=metric, p=p)
        knn_temp.fit(X_train_scaled, y_train)
        weights_comparison.append({
            'weights': w,
            'train_acc': knn_temp.score(X_train_scaled, y_train),
            'test_acc': knn_temp.score(X_test_scaled, y_test)
        })
    
    df_weights = pd.DataFrame(weights_comparison)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(name='Training', x=df_weights['weights'], y=df_weights['train_acc'],
                        marker_color='lightblue'))
    fig.add_trace(go.Bar(name='Test', x=df_weights['weights'], y=df_weights['test_acc'],
                        marker_color='salmon'))
    fig.update_layout(
        title="Accuracy Comparison: Uniform vs Distance Weights",
        yaxis_title="Accuracy",
        barmode='group',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Analysis: metric comparison
    st.subheader("3️⃣ Impact of Distance Metrics")
    
    metrics_to_test = ['euclidean', 'manhattan', 'chebyshev']
    metrics_comparison = []
    
    for m in metrics_to_test:
        knn_temp = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, metric=m)
        knn_temp.fit(X_train_scaled, y_train)
        metrics_comparison.append({
            'metric': m,
            'train_acc': knn_temp.score(X_train_scaled, y_train),
            'test_acc': knn_temp.score(X_test_scaled, y_test)
        })
    
    df_metrics = pd.DataFrame(metrics_comparison)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(name='Training', x=df_metrics['metric'], y=df_metrics['train_acc'],
                        marker_color='lightgreen'))
    fig.add_trace(go.Bar(name='Test', x=df_metrics['metric'], y=df_metrics['test_acc'],
                        marker_color='lightcoral'))
    fig.update_layout(
        title="Accuracy Comparison: Different Distance Metrics",
        yaxis_title="Accuracy",
        barmode='group',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    best_metric = df_metrics.loc[df_metrics['test_acc'].idxmax(), 'metric']
    st.success(f"📊 **Best metric for this data**: {best_metric}")

with tab3:
    st.header("Side-by-Side Hyperparameter Comparison")
    st.markdown("**Compare different parameter settings visually**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Small k (k=3)")
        knn_small = KNeighborsClassifier(n_neighbors=3, weights=weights, metric=metric)
        knn_small.fit(X_train_scaled, y_train)
        acc_small = knn_small.score(X_test_scaled, y_test)
        st.metric("Test Accuracy", f"{acc_small:.1%}")
        fig_small = plot_decision_boundary(X_train_scaled, y_train, knn_small, 
                                          f"k=3 (More Complex)")
        st.plotly_chart(fig_small, use_container_width=True)
    
    with col2:
        st.subheader("Large k (k=30)")
        knn_large = KNeighborsClassifier(n_neighbors=30, weights=weights, metric=metric)
        knn_large.fit(X_train_scaled, y_train)
        acc_large = knn_large.score(X_test_scaled, y_test)
        st.metric("Test Accuracy", f"{acc_large:.1%}")
        fig_large = plot_decision_boundary(X_train_scaled, y_train, knn_large, 
                                           f"k=30 (Smoother)")
        st.plotly_chart(fig_large, use_container_width=True)
    
    st.markdown("---")
    
    # Weights comparison
    st.subheader("Uniform vs Distance Weights")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Uniform Weights")
        knn_uniform = KNeighborsClassifier(n_neighbors=n_neighbors, weights='uniform', metric=metric)
        knn_uniform.fit(X_train_scaled, y_train)
        acc_uniform = knn_uniform.score(X_test_scaled, y_test)
        st.metric("Test Accuracy", f"{acc_uniform:.1%}")
        fig_uniform = plot_decision_boundary(X_train_scaled, y_train, knn_uniform, 
                                            "Uniform (All Equal)")
        st.plotly_chart(fig_uniform, use_container_width=True)
    
    with col2:
        st.subheader("Distance Weights")
        knn_dist = KNeighborsClassifier(n_neighbors=n_neighbors, weights='distance', metric=metric)
        knn_dist.fit(X_train_scaled, y_train)
        acc_dist = knn_dist.score(X_test_scaled, y_test)
        st.metric("Test Accuracy", f"{acc_dist:.1%}")
        fig_dist = plot_decision_boundary(X_train_scaled, y_train, knn_dist, 
                                         "Distance (Closer = More Weight)")
        st.plotly_chart(fig_dist, use_container_width=True)
    
    st.markdown("---")
    
    # Summary table
    st.subheader("📋 Hyperparameter Summary Table")
    summary_df = pd.DataFrame({
        'Hyperparameter': ['n_neighbors', 'weights', 'metric', 'p', 'algorithm', 'leaf_size', 'n_jobs'],
        'Current Value': [n_neighbors, weights, metric, p, algorithm, leaf_size, n_jobs],
        'Purpose': [
            'Number of neighbors to consider',
            'How to weight neighbor votes',
            'Method to calculate distances',
            'Minkowski power (1=Manhattan, 2=Euclidean)',
            'Algorithm for neighbor search',
            'Leaf size for tree algorithms',
            'Number of parallel jobs'
        ],
        'Effect': [
            'Lower → complex boundaries | Higher → smoother',
            'uniform → equal vote | distance → closer matters more',
            'Different distance calculations affect boundaries',
            'Controls distance metric behavior',
            'Affects computation speed, not predictions',
            'Affects speed of tree algorithms',
            'Affects training speed only'
        ]
    })
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        <p>🎯 KNN Parameter Impact Visualizer | Built with Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
)
