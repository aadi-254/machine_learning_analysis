# KNN Parameter Tuning App 🎯

An interactive Streamlit application for K-Nearest Neighbors hyperparameter tuning and visualization.

## Features

- **Multiple Datasets**: Choose from Iris, Wine, or Breast Cancer datasets
- **Comprehensive Hyperparameter Tuning**: Tune all major KNN parameters
- **Real-time Visualization**: See confusion matrix, classification reports, and performance metrics
- **Cross-Validation**: Optional 5-fold cross-validation for robust evaluation
- **Feature Scaling**: Standardize features before training
- **Detailed Analysis**: View misclassified samples and confidence metrics

## Hyperparameters

| Hyperparameter | Purpose                    | Options/Range           |
| -------------- | -------------------------- | ----------------------- |
| n_neighbors    | Number of neighbors        | 1-50                    |
| weights        | Influence of neighbors     | uniform, distance       |
| metric         | Distance calculation       | euclidean, manhattan, etc. |
| p              | Type of Minkowski distance | 1-5                     |
| algorithm      | Search method              | auto, ball_tree, kd_tree, brute |
| leaf_size      | Tree performance           | 10-100                  |
| n_jobs         | Parallel processing        | -1, 1, 2, 4             |

## Installation

1. Install the required packages:
```bash
pip install -r requirements.txt
```

## Running the App

```bash
streamlit run app.py
```

The app will open in your default web browser at `http://localhost:8501`

## How to Use

1. **Select Dataset**: Choose a dataset from the sidebar (Iris, Wine, or Breast Cancer)
2. **Configure Test Split**: Adjust the test size percentage and random state
3. **Tune Hyperparameters**: Modify KNN parameters in the sidebar
4. **Additional Options**: 
   - Enable feature scaling for better results
   - Use cross-validation for more robust evaluation
5. **Train Model**: Click the "Train Model" button
6. **Explore Results**: Navigate through tabs to view:
   - Dataset Overview
   - Model Performance (accuracy, confusion matrix, classification report)
   - Visualizations (feature variance, prediction confidence)
   - Detailed Analysis (misclassified samples, hyperparameter summary)

## Tips for Better Results

- **Feature Scaling**: Always recommended for KNN (enabled by default)
- **n_neighbors**: Start with 5, increase for smoother boundaries
- **weights='distance'**: Better for unevenly distributed data
- **Cross-Validation**: Use for reliable performance estimation
- **metric**: Euclidean works well for most cases, try Manhattan for high-dimensional data

## Author

Built with ❤️ using Streamlit and Scikit-learn
