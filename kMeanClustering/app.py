import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs, make_moons, make_circles, load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time

# Page configuration
st.set_page_config(
    page_title="K-Means Clustering Explorer",
    page_icon="🎯",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🎯 K-Means Clustering Explorer</h1>', unsafe_allow_html=True)
st.markdown("### Interactive Visualization and Understanding of K-Means Algorithm")
st.markdown("---")

# Sidebar configuration
st.sidebar.header("🎛️ K-Means Configuration")

# Dataset selection
st.sidebar.header("📊 Dataset Selection")
dataset_option = st.sidebar.selectbox(
    "Choose Dataset",
    ["Blobs (Easy Clusters)", "Iris Dataset", "Moons (Complex)", "Circles (Complex)", "Random Data"],
    help="Different datasets show how K-Means performs on various data distributions"
)

# Dataset parameters
if dataset_option == "Blobs (Easy Clusters)":
    st.sidebar.subheader("Dataset Parameters")
    n_samples = st.sidebar.slider("Number of Samples", 100, 1000, 300, 50)
    n_features = st.sidebar.slider("Number of Features", 2, 10, 2, 1)
    true_clusters = st.sidebar.slider("True Number of Clusters", 2, 8, 3, 1)
    cluster_std = st.sidebar.slider("Cluster Std Deviation", 0.1, 3.0, 1.0, 0.1,
                                    help="Higher values make clusters more spread out")
elif dataset_option in ["Moons (Complex)", "Circles (Complex)"]:
    st.sidebar.subheader("Dataset Parameters")
    n_samples = st.sidebar.slider("Number of Samples", 100, 1000, 300, 50)
    noise = st.sidebar.slider("Noise Level", 0.0, 0.3, 0.05, 0.01)
elif dataset_option == "Random Data":
    st.sidebar.subheader("Dataset Parameters")
    n_samples = st.sidebar.slider("Number of Samples", 100, 1000, 300, 50)
    n_features = st.sidebar.slider("Number of Features", 2, 10, 2, 1)

# K-Means hyperparameters
st.sidebar.markdown("---")
st.sidebar.header("🎯 K-Means Hyperparameters")

n_clusters = st.sidebar.slider(
    "Number of Clusters (K)",
    min_value=2,
    max_value=10,
    value=3,
    step=1,
    help="Number of clusters to form"
)

init_method = st.sidebar.selectbox(
    "Initialization Method",
    ["k-means++", "random"],
    help="k-means++: Smart initialization (recommended). random: Random initialization"
)

max_iter = st.sidebar.slider(
    "Maximum Iterations",
    min_value=10,
    max_value=500,
    value=300,
    step=10,
    help="Maximum number of iterations for convergence"
)

n_init = st.sidebar.slider(
    "Number of Initializations",
    min_value=1,
    max_value=20,
    value=10,
    step=1,
    help="Number of times to run with different centroid seeds"
)

random_state = st.sidebar.number_input(
    "Random State",
    min_value=0,
    max_value=100,
    value=42
)

# Visualization options
st.sidebar.markdown("---")
st.sidebar.header("📊 Visualization Options")

show_animation = st.sidebar.checkbox(
    "Show Clustering Animation",
    value=False,
    help="Visualize how clusters form step by step (only for 2D data)"
)

show_elbow = st.sidebar.checkbox(
    "Show Elbow Method",
    value=True,
    help="Find optimal number of clusters"
)

show_silhouette = st.sidebar.checkbox(
    "Show Silhouette Analysis",
    value=True,
    help="Evaluate cluster quality"
)

# Train button
st.sidebar.markdown("---")
train_button = st.sidebar.button("🚀 Run K-Means Clustering", type="primary", use_container_width=True)

# Generate dataset
@st.cache_data
def generate_dataset(dataset_name, random_state, **kwargs):
    if dataset_name == "Blobs (Easy Clusters)":
        X, y_true = make_blobs(
            n_samples=kwargs.get('n_samples', 300),
            n_features=kwargs.get('n_features', 2),
            centers=kwargs.get('true_clusters', 3),
            cluster_std=kwargs.get('cluster_std', 1.0),
            random_state=random_state
        )
        feature_names = [f'Feature {i+1}' for i in range(X.shape[1])]
        return X, y_true, feature_names
    
    elif dataset_name == "Iris Dataset":
        data = load_iris()
        X = data.data
        y_true = data.target
        feature_names = data.feature_names
        return X, y_true, feature_names
    
    elif dataset_name == "Moons (Complex)":
        X, y_true = make_moons(
            n_samples=kwargs.get('n_samples', 300),
            noise=kwargs.get('noise', 0.05),
            random_state=random_state
        )
        feature_names = ['Feature 1', 'Feature 2']
        return X, y_true, feature_names
    
    elif dataset_name == "Circles (Complex)":
        X, y_true = make_circles(
            n_samples=kwargs.get('n_samples', 300),
            noise=kwargs.get('noise', 0.05),
            factor=0.5,
            random_state=random_state
        )
        feature_names = ['Feature 1', 'Feature 2']
        return X, y_true, feature_names
    
    else:  # Random Data
        X = np.random.randn(
            kwargs.get('n_samples', 300),
            kwargs.get('n_features', 2)
        )
        y_true = None
        feature_names = [f'Feature {i+1}' for i in range(X.shape[1])]
        return X, y_true, feature_names

# Generate dataset based on selection
dataset_params = {}
if dataset_option == "Blobs (Easy Clusters)":
    dataset_params = {
        'n_samples': n_samples,
        'n_features': n_features,
        'true_clusters': true_clusters,
        'cluster_std': cluster_std
    }
elif dataset_option in ["Moons (Complex)", "Circles (Complex)"]:
    dataset_params = {
        'n_samples': n_samples,
        'noise': noise
    }
elif dataset_option == "Random Data":
    dataset_params = {
        'n_samples': n_samples,
        'n_features': n_features
    }

X, y_true, feature_names = generate_dataset(dataset_option, random_state, **dataset_params)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dataset info
st.header("📊 Dataset Overview")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Samples", X.shape[0])
with col2:
    st.metric("Features", X.shape[1])
with col3:
    st.metric("Selected K", n_clusters)
with col4:
    if y_true is not None:
        st.metric("True Clusters", len(np.unique(y_true)))
    else:
        st.metric("True Clusters", "Unknown")

# Show dataset preview
with st.expander("📋 View Dataset Preview"):
    df = pd.DataFrame(X_scaled, columns=feature_names)
    if y_true is not None:
        df['True Label'] = y_true
    st.dataframe(df.head(20), use_container_width=True)
    
    # If 2D, show scatter plot
    if X.shape[1] == 2:
        fig = px.scatter(
            x=X_scaled[:, 0], 
            y=X_scaled[:, 1],
            color=y_true if y_true is not None else None,
            title="Original Data Distribution",
            labels={'x': feature_names[0], 'y': feature_names[1]},
            color_continuous_scale='viridis'
        )
        fig.update_traces(marker=dict(size=8, opacity=0.7))
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Main clustering execution
if train_button:
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Train K-Means
    status_text.text("🔄 Running K-Means clustering...")
    progress_bar.progress(20)
    
    start_time = time.time()
    
    kmeans = KMeans(
        n_clusters=n_clusters,
        init=init_method,
        max_iter=max_iter,
        n_init=n_init,
        random_state=random_state
    )
    
    progress_bar.progress(40)
    labels = kmeans.fit_predict(X_scaled)
    
    training_time = time.time() - start_time
    
    progress_bar.progress(70)
    status_text.text("📊 Calculating metrics...")
    
    # Calculate metrics
    inertia = kmeans.inertia_
    silhouette_avg = silhouette_score(X_scaled, labels)
    calinski = calinski_harabasz_score(X_scaled, labels)
    davies_bouldin = davies_bouldin_score(X_scaled, labels)
    n_iter = kmeans.n_iter_
    
    progress_bar.progress(100)
    status_text.text("✅ Clustering complete!")
    
    time.sleep(0.5)
    progress_bar.empty()
    status_text.empty()
    
    # Display results
    st.header("📈 Clustering Results")
    
    # Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Inertia (WCSS)", f"{inertia:.2f}", help="Within-Cluster Sum of Squares (lower is better)")
    with col2:
        st.metric("Silhouette Score", f"{silhouette_avg:.3f}", help="Range [-1, 1], higher is better")
    with col3:
        st.metric("Calinski-Harabasz", f"{calinski:.2f}", help="Higher is better")
    with col4:
        st.metric("Davies-Bouldin", f"{davies_bouldin:.3f}", help="Lower is better")
    with col5:
        st.metric("Iterations", f"{n_iter}", help="Number of iterations to converge")
    
    st.info(f"⏱️ **Training Time**: {training_time:.3f} seconds")
    
    # Main visualization
    st.markdown("---")
    st.subheader("🎨 Cluster Visualization")
    
    # Reduce to 2D if needed for visualization
    if X.shape[1] > 2:
        pca = PCA(n_components=2, random_state=random_state)
        X_viz = pca.fit_transform(X_scaled)
        explained_var = pca.explained_variance_ratio_
        st.info(f"📊 Using PCA for visualization. Explained variance: {explained_var[0]:.1%} + {explained_var[1]:.1%} = {sum(explained_var):.1%}")
    else:
        X_viz = X_scaled
    
    # Create main cluster plot
    cluster_centers = kmeans.cluster_centers_
    if X.shape[1] > 2:
        cluster_centers_viz = pca.transform(cluster_centers)
    else:
        cluster_centers_viz = cluster_centers
    
    # Create figure with subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('K-Means Clustering Results', 'True Labels (if available)'),
        specs=[[{"type": "scatter"}, {"type": "scatter"}]]
    )
    
    # Plot K-Means clusters
    colors = px.colors.qualitative.Set1
    for i in range(n_clusters):
        mask = labels == i
        fig.add_trace(
            go.Scatter(
                x=X_viz[mask, 0],
                y=X_viz[mask, 1],
                mode='markers',
                name=f'Cluster {i}',
                marker=dict(size=8, opacity=0.6, color=colors[i % len(colors)]),
                showlegend=True
            ),
            row=1, col=1
        )
    
    # Plot cluster centers
    fig.add_trace(
        go.Scatter(
            x=cluster_centers_viz[:, 0],
            y=cluster_centers_viz[:, 1],
            mode='markers',
            name='Centroids',
            marker=dict(
                size=20,
                color='black',
                symbol='x',
                line=dict(width=2, color='white')
            ),
            showlegend=True
        ),
        row=1, col=1
    )
    
    # Plot true labels if available
    if y_true is not None:
        for i in np.unique(y_true):
            mask = y_true == i
            fig.add_trace(
                go.Scatter(
                    x=X_viz[mask, 0],
                    y=X_viz[mask, 1],
                    mode='markers',
                    name=f'True Class {i}',
                    marker=dict(size=8, opacity=0.6, color=colors[i % len(colors)]),
                    showlegend=True
                ),
                row=1, col=2
            )
    else:
        fig.add_trace(
            go.Scatter(
                x=X_viz[:, 0],
                y=X_viz[:, 1],
                mode='markers',
                marker=dict(size=8, opacity=0.3, color='gray'),
                showlegend=False,
                text=['No true labels available']
            ),
            row=1, col=2
        )
    
    fig.update_xaxes(title_text="Component 1" if X.shape[1] > 2 else feature_names[0], row=1, col=1)
    fig.update_yaxes(title_text="Component 2" if X.shape[1] > 2 else feature_names[1], row=1, col=1)
    fig.update_xaxes(title_text="Component 1" if X.shape[1] > 2 else feature_names[0], row=1, col=2)
    fig.update_yaxes(title_text="Component 2" if X.shape[1] > 2 else feature_names[1], row=1, col=2)
    
    fig.update_layout(height=600, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)
    
    # Show animation if requested (only for 2D data)
    if show_animation and X.shape[1] == 2:
        st.markdown("---")
        st.subheader("🎬 Clustering Animation")
        st.info("💡 This animation shows how K-Means iteratively updates cluster centers and reassigns points")
        
        # Run K-Means with max_iter=1 multiple times to get iterations
        animation_kmeans = KMeans(
            n_clusters=n_clusters,
            init=init_method,
            max_iter=1,
            n_init=1,
            random_state=random_state
        )
        
        frames = []
        centers_history = []
        labels_history = []
        
        # Initialize
        animation_kmeans.fit(X_scaled)
        centers_history.append(animation_kmeans.cluster_centers_.copy())
        labels_history.append(animation_kmeans.labels_.copy())
        
        # Iterate
        for iteration in range(min(10, max_iter)):
            animation_kmeans.max_iter = iteration + 2
            animation_kmeans.fit(X_scaled)
            centers_history.append(animation_kmeans.cluster_centers_.copy())
            labels_history.append(animation_kmeans.labels_.copy())
            
            if animation_kmeans.n_iter_ < animation_kmeans.max_iter:
                break
        
        # Create animation frames
        for idx, (centers, lbls) in enumerate(zip(centers_history, labels_history)):
            frame_data = []
            
            for i in range(n_clusters):
                mask = lbls == i
                frame_data.append(
                    go.Scatter(
                        x=X_viz[mask, 0],
                        y=X_viz[mask, 1],
                        mode='markers',
                        name=f'Cluster {i}',
                        marker=dict(size=8, opacity=0.6, color=colors[i % len(colors)]),
                        showlegend=(idx == 0)
                    )
                )
            
            # Add centroids
            frame_data.append(
                go.Scatter(
                    x=centers[:, 0],
                    y=centers[:, 1],
                    mode='markers',
                    name='Centroids',
                    marker=dict(
                        size=20,
                        color='black',
                        symbol='x',
                        line=dict(width=2, color='white')
                    ),
                    showlegend=(idx == 0)
                )
            )
            
            frames.append(go.Frame(data=frame_data, name=str(idx)))
        
        # Create figure with frames
        fig_anim = go.Figure(
            data=frames[0].data,
            frames=frames,
            layout=go.Layout(
                title=f"K-Means Clustering Animation (Iteration 0)",
                xaxis=dict(title=feature_names[0]),
                yaxis=dict(title=feature_names[1]),
                updatemenus=[{
                    "type": "buttons",
                    "showactive": False,
                    "buttons": [
                        {
                            "label": "▶ Play",
                            "method": "animate",
                            "args": [None, {
                                "frame": {"duration": 800, "redraw": True},
                                "fromcurrent": True,
                                "mode": "immediate"
                            }]
                        },
                        {
                            "label": "⏸ Pause",
                            "method": "animate",
                            "args": [[None], {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate"
                            }]
                        }
                    ],
                    "x": 0.1,
                    "y": 1.15
                }],
                sliders=[{
                    "active": 0,
                    "steps": [
                        {
                            "args": [[f.name], {
                                "frame": {"duration": 0, "redraw": True},
                                "mode": "immediate"
                            }],
                            "label": f"Iter {k}",
                            "method": "animate"
                        }
                        for k, f in enumerate(frames)
                    ],
                    "x": 0.1,
                    "len": 0.9,
                    "xanchor": "left",
                    "y": 0,
                    "yanchor": "top"
                }]
            )
        )
        
        fig_anim.update_layout(height=600)
        st.plotly_chart(fig_anim, use_container_width=True)
    
    # Additional analysis tabs
    st.markdown("---")
    st.subheader("📊 Detailed Analysis")
    
    tabs = ["Cluster Statistics", "Feature Analysis"]
    if show_elbow:
        tabs.append("Elbow Method")
    if show_silhouette:
        tabs.append("Silhouette Analysis")
    
    tab_objects = st.tabs(tabs)
    tab_idx = 0
    
    # Tab 1: Cluster Statistics
    with tab_objects[tab_idx]:
        st.markdown("### Cluster Statistics")
        
        cluster_stats = []
        for i in range(n_clusters):
            mask = labels == i
            cluster_data = X_scaled[mask]
            
            stats = {
                'Cluster': i,
                'Size': mask.sum(),
                'Percentage': f"{(mask.sum() / len(labels)) * 100:.1f}%",
                'Mean Distance to Center': f"{np.mean(np.linalg.norm(cluster_data - cluster_centers[i], axis=1)):.3f}",
                'Std Distance to Center': f"{np.std(np.linalg.norm(cluster_data - cluster_centers[i], axis=1)):.3f}"
            }
            
            if y_true is not None:
                # Most common true label in this cluster
                true_labels_in_cluster = y_true[mask]
                most_common = np.bincount(true_labels_in_cluster).argmax()
                purity = (true_labels_in_cluster == most_common).sum() / len(true_labels_in_cluster) * 100
                stats['Most Common True Label'] = most_common
                stats['Purity'] = f"{purity:.1f}%"
            
            cluster_stats.append(stats)
        
        df_stats = pd.DataFrame(cluster_stats)
        st.dataframe(df_stats, use_container_width=True)
        
        # Cluster sizes visualization
        fig = px.bar(
            df_stats,
            x='Cluster',
            y='Size',
            title='Cluster Sizes',
            color='Cluster',
            color_continuous_scale='viridis'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    tab_idx += 1
    
    # Tab 2: Feature Analysis
    with tab_objects[tab_idx]:
        st.markdown("### Feature Distribution by Cluster")
        
        # Select features to visualize
        if len(feature_names) > 2:
            selected_features = st.multiselect(
                "Select features to analyze",
                feature_names,
                default=feature_names[:2]
            )
        else:
            selected_features = feature_names
        
        if selected_features:
            # Create violin plots for each feature
            for feature in selected_features:
                feature_idx = feature_names.index(feature)
                
                df_feature = pd.DataFrame({
                    'Value': X_scaled[:, feature_idx],
                    'Cluster': labels
                })
                
                fig = px.violin(
                    df_feature,
                    x='Cluster',
                    y='Value',
                    title=f'Distribution of {feature} by Cluster',
                    box=True,
                    points='all',
                    color='Cluster',
                    color_discrete_sequence=colors
                )
                st.plotly_chart(fig, use_container_width=True)
    
    tab_idx += 1
    
    # Tab 3: Elbow Method
    if show_elbow:
        with tab_objects[tab_idx]:
            st.markdown("### Elbow Method - Finding Optimal K")
            st.info("💡 The elbow point suggests the optimal number of clusters where adding more clusters doesn't significantly reduce inertia")
            
            # Calculate inertia for different K values
            with st.spinner("Calculating elbow curve..."):
                K_range = range(2, min(11, X.shape[0] // 10))
                inertias = []
                silhouettes = []
                
                for k in K_range:
                    km = KMeans(n_clusters=k, init=init_method, random_state=random_state, n_init=5)
                    km.fit(X_scaled)
                    inertias.append(km.inertia_)
                    silhouettes.append(silhouette_score(X_scaled, km.labels_))
            
            # Create subplot with both metrics
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Elbow Curve (Inertia)', 'Silhouette Score')
            )
            
            # Elbow curve
            fig.add_trace(
                go.Scatter(
                    x=list(K_range),
                    y=inertias,
                    mode='lines+markers',
                    name='Inertia',
                    line=dict(color='blue', width=3),
                    marker=dict(size=10)
                ),
                row=1, col=1
            )
            
            # Highlight current K
            current_k_idx = list(K_range).index(n_clusters) if n_clusters in K_range else None
            if current_k_idx is not None:
                fig.add_trace(
                    go.Scatter(
                        x=[n_clusters],
                        y=[inertias[current_k_idx]],
                        mode='markers',
                        name='Current K',
                        marker=dict(size=20, color='red', symbol='star')
                    ),
                    row=1, col=1
                )
            
            # Silhouette scores
            fig.add_trace(
                go.Scatter(
                    x=list(K_range),
                    y=silhouettes,
                    mode='lines+markers',
                    name='Silhouette',
                    line=dict(color='green', width=3),
                    marker=dict(size=10)
                ),
                row=1, col=2
            )
            
            # Highlight current K
            if current_k_idx is not None:
                fig.add_trace(
                    go.Scatter(
                        x=[n_clusters],
                        y=[silhouettes[current_k_idx]],
                        mode='markers',
                        name='Current K',
                        marker=dict(size=20, color='red', symbol='star'),
                        showlegend=False
                    ),
                    row=1, col=2
                )
            
            fig.update_xaxes(title_text="Number of Clusters (K)", row=1, col=1)
            fig.update_yaxes(title_text="Inertia (WCSS)", row=1, col=1)
            fig.update_xaxes(title_text="Number of Clusters (K)", row=1, col=2)
            fig.update_yaxes(title_text="Silhouette Score", row=1, col=2)
            
            fig.update_layout(height=500, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
            
            # Recommendations
            optimal_k_silhouette = list(K_range)[np.argmax(silhouettes)]
            st.success(f"🎯 **Recommendation**: Based on Silhouette Score, optimal K = **{optimal_k_silhouette}**")
            
            if optimal_k_silhouette != n_clusters:
                st.warning(f"💡 Try setting K = {optimal_k_silhouette} for potentially better clustering")
        
        tab_idx += 1
    
    # Tab 4: Silhouette Analysis
    if show_silhouette:
        with tab_objects[tab_idx]:
            st.markdown("### Silhouette Analysis")
            st.info("💡 Silhouette coefficient measures how similar a point is to its own cluster compared to other clusters. Range: [-1, 1], higher is better.")
            
            from sklearn.metrics import silhouette_samples
            
            silhouette_vals = silhouette_samples(X_scaled, labels)
            
            # Create silhouette plot
            fig = go.Figure()
            
            y_lower = 10
            for i in range(n_clusters):
                cluster_silhouette_vals = silhouette_vals[labels == i]
                cluster_silhouette_vals.sort()
                
                size_cluster_i = cluster_silhouette_vals.shape[0]
                y_upper = y_lower + size_cluster_i
                
                fig.add_trace(go.Bar(
                    x=cluster_silhouette_vals,
                    y=np.arange(y_lower, y_upper),
                    orientation='h',
                    name=f'Cluster {i}',
                    marker=dict(color=colors[i % len(colors)]),
                    showlegend=True
                ))
                
                y_lower = y_upper + 10
            
            # Add average silhouette line
            fig.add_vline(
                x=silhouette_avg,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Avg: {silhouette_avg:.3f}",
                annotation_position="top"
            )
            
            fig.update_layout(
                title='Silhouette Plot for Each Cluster',
                xaxis_title='Silhouette Coefficient',
                yaxis_title='Cluster',
                height=600,
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Per-cluster silhouette scores
            col1, col2 = st.columns([2, 1])
            
            with col1:
                cluster_silhouettes = []
                for i in range(n_clusters):
                    cluster_silhouettes.append({
                        'Cluster': i,
                        'Avg Silhouette': f"{silhouette_vals[labels == i].mean():.3f}",
                        'Min': f"{silhouette_vals[labels == i].min():.3f}",
                        'Max': f"{silhouette_vals[labels == i].max():.3f}"
                    })
                
                df_silhouettes = pd.DataFrame(cluster_silhouettes)
                st.dataframe(df_silhouettes, use_container_width=True)
            
            with col2:
                st.markdown("**Interpretation:**")
                if silhouette_avg > 0.7:
                    st.success("Excellent clustering! 🎉")
                elif silhouette_avg > 0.5:
                    st.success("Good clustering ✅")
                elif silhouette_avg > 0.3:
                    st.warning("Moderate clustering ⚠️")
                else:
                    st.error("Poor clustering ❌")

# Information section
st.markdown("---")
st.header("📚 K-Means Guide")

with st.expander("🔍 How K-Means Works"):
    st.markdown("""
    ### K-Means Algorithm Steps:
    
    1. **Initialize**: Randomly select K points as initial cluster centers (centroids)
    2. **Assign**: Assign each data point to the nearest centroid
    3. **Update**: Recalculate centroids as the mean of all points in each cluster
    4. **Repeat**: Steps 2-3 until convergence (centroids don't change significantly)
    
    ### Key Concepts:
    
    **Inertia (WCSS - Within-Cluster Sum of Squares)**
    - Sum of squared distances from each point to its cluster center
    - Lower is better
    - Always decreases as K increases
    
    **Silhouette Score**
    - Measures how well-separated clusters are
    - Range: [-1, 1]
    - > 0.7: Excellent, 0.5-0.7: Good, 0.3-0.5: Moderate, < 0.3: Poor
    
    **Calinski-Harabasz Index**
    - Ratio of between-cluster dispersion to within-cluster dispersion
    - Higher is better
    
    **Davies-Bouldin Index**
    - Average similarity between each cluster and its most similar cluster
    - Lower is better
    """)

with st.expander("🎯 Choosing the Right K"):
    st.markdown("""
    ### Methods to Find Optimal K:
    
    **1. Elbow Method**
    - Plot inertia vs number of clusters
    - Look for the "elbow" point where inertia stops decreasing rapidly
    - This is often the optimal K
    
    **2. Silhouette Analysis**
    - Calculate silhouette score for different K values
    - Choose K with the highest silhouette score
    
    **3. Domain Knowledge**
    - Sometimes you know the expected number of clusters
    - Use business context to guide your choice
    
    ### Tips:
    - Start with Elbow Method for initial estimate
    - Validate with Silhouette Analysis
    - Try K values around the elbow point
    - Consider computational cost vs performance trade-off
    """)

with st.expander("⚙️ Hyperparameters Explained"):
    st.markdown("""
    ### K-Means Hyperparameters:
    
    **Number of Clusters (K)**
    - Most important parameter
    - Too low: Underfitting (merges distinct groups)
    - Too high: Overfitting (splits natural groups)
    
    **Initialization Method**
    - **k-means++**: Smart initialization, faster convergence (recommended)
    - **random**: Random initialization, may need more iterations
    
    **Maximum Iterations**
    - Maximum number of iterations before stopping
    - Default (300) is usually sufficient
    - Increase if algorithm doesn't converge
    
    **Number of Initializations (n_init)**
    - Number of times to run K-Means with different seeds
    - Best result is kept
    - Higher values = more robust but slower
    - Default (10) is good balance
    
    ### When K-Means Works Well:
    - ✅ Spherical/round shaped clusters
    - ✅ Similar sized clusters
    - ✅ Well-separated clusters
    - ✅ Clearly defined cluster centers
    
    ### When K-Means Struggles:
    - ❌ Non-spherical shapes (moons, circles)
    - ❌ Very different cluster sizes
    - ❌ Overlapping clusters
    - ❌ Outliers (heavily affect centroids)
    """)

with st.expander("💡 Quick Tips"):
    st.markdown("""
    ### Best Practices:
    
    1. **Always standardize/scale your features** before K-Means
    2. **Try different K values** - use Elbow and Silhouette methods
    3. **Run multiple times** with different random states
    4. **Check cluster sizes** - very imbalanced sizes may indicate issues
    5. **Visualize results** - especially important for validation
    
    ### Troubleshooting:
    
    **Problem: Poor silhouette scores**
    - Solution: Try different K values, check if data is suitable for K-Means
    
    **Problem: Empty clusters**
    - Solution: Reduce K or increase n_init
    
    **Problem: Clusters merge distinct groups**
    - Solution: Increase K
    
    **Problem: One large cluster, others small**
    - Solution: Check for outliers, try different K
    """)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>Built with ❤️ using Streamlit | K-Means Clustering Explorer</p>
        <p>🎯 Experiment with different datasets and hyperparameters to understand clustering!</p>
    </div>
    """, unsafe_allow_html=True)
