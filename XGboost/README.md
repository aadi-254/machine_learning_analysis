# 🚀 XGBoost & GradientBoosting Hyperparameter Tuning App

An interactive Streamlit application for exploring and understanding hyperparameter tuning in XGBoost and GradientBoosting algorithms.

## 🎯 Features

- **Two Powerful Algorithms**: Compare XGBoost and Scikit-learn's GradientBoosting
- **Classification & Regression**: Support for both problem types
- **Real Datasets**: Uses Breast Cancer (classification) and Diabetes (regression) datasets that show clear changes with hyperparameter adjustments
- **Interactive Hyperparameter Tuning**: Real-time visualization of how parameters affect model performance
- **Comprehensive Metrics**: Detailed performance metrics and visualizations
- **Feature Importance**: Understand which features matter most
- **Learning Curves**: Visualize model performance vs number of trees (GradientBoosting)
- **Educational Guides**: Built-in explanations for all hyperparameters

## 📊 Available Datasets

### Classification
- **Breast Cancer Dataset**: 569 samples, 30 features - Perfect for seeing hyperparameter effects
- **Synthetic Dataset**: 1000 samples, 20 features - Customizable complexity

### Regression
- **Diabetes Dataset**: 442 samples, 10 features - Shows clear performance changes
- **Synthetic Dataset**: 1000 samples, 20 features - Adjustable noise levels

## 🛠️ Installation

1. Install the required packages:
```bash
pip install -r requirements.txt
```

## 🚀 Usage

1. Run the Streamlit app:
```bash
streamlit run app.py
```

2. The app will open in your browser at `http://localhost:8501`

## 🎛️ Key Hyperparameters to Experiment With

### Number of Estimators (Trees)
- Range: 10-500
- Effect: More trees = better fit but slower training
- **Try this**: Start with 100, then try 50 and 300 to see the difference

### Learning Rate
- Range: 0.01-1.0
- Effect: Lower values = more conservative, better generalization
- **Try this**: Compare 0.01, 0.1, and 0.5 to see the impact on overfitting

### Max Depth
- Range: 1-15
- Effect: Deeper trees capture complex patterns but risk overfitting
- **Try this**: Start with 3, then try 1 and 10 to see underfitting vs overfitting

### Subsample Ratio
- Range: 0.1-1.0
- Effect: Using less data per tree adds randomness and prevents overfitting
- **Try this**: Compare 1.0 (no subsampling) with 0.8 to see regularization effect

## 🎓 Learning Path

### For Beginners:
1. Start with **Classification** using **Breast Cancer** dataset
2. Use default parameters and train the model
3. Observe the baseline metrics
4. Adjust **one parameter at a time** and see how metrics change

### Recommended Experiments:

#### Experiment 1: Effect of Learning Rate
- Set n_estimators = 100
- Try learning_rate = 0.01, 0.1, 0.5, 1.0
- Observe: How does accuracy and training time change?

#### Experiment 2: Effect of Max Depth
- Set learning_rate = 0.1, n_estimators = 100
- Try max_depth = 1, 3, 5, 10
- Observe: When does overfitting start?

#### Experiment 3: Preventing Overfitting
- Set max_depth = 10 (likely to overfit)
- Try subsample = 1.0, then 0.8, then 0.5
- Observe: How does the gap between training and test scores change?

#### Experiment 4: Optimal Number of Trees
- Use **GradientBoosting** algorithm
- Train with n_estimators = 300
- Check the **Learning Curves** tab
- Observe: At what point does adding more trees stop helping?

## 📈 Interpreting Results

### Classification Metrics:
- **Accuracy**: Overall correctness (aim for > 0.90 on Breast Cancer)
- **Precision**: When model predicts positive, how often is it correct?
- **Recall**: Out of all actual positives, how many did we catch?
- **F1-Score**: Harmonic mean of precision and recall

### Regression Metrics:
- **R² Score**: Proportion of variance explained (closer to 1.0 is better)
- **RMSE**: Average prediction error in original units
- **MAE**: Average absolute error (more robust to outliers than RMSE)

### Signs of Overfitting:
- ✅ High training accuracy but lower test accuracy
- ✅ Large gap between train and test scores
- ✅ Learning curve shows divergence
- **Solution**: Reduce max_depth, increase regularization (subsample, gamma)

### Signs of Underfitting:
- ❌ Low training and test accuracy
- ❌ Similar low scores for both
- **Solution**: Increase max_depth, increase n_estimators

## 🔍 Advanced Features

### XGBoost Specific Parameters:
- **Column Sample by Tree**: Randomly use subset of features per tree
- **Gamma**: Minimum loss reduction needed to split (higher = more conservative)

### Visualization Tabs:
1. **Confusion Matrix**: See which classes are confused (classification)
2. **ROC Curve**: Trade-off between true positive and false positive rates
3. **Feature Importance**: Which features drive predictions?
4. **Learning Curves**: How performance changes with number of trees

## 💡 Tips for Best Results

1. **Start Simple**: Use default parameters first
2. **One at a Time**: Change one parameter at a time to understand its effect
3. **Monitor Both**: Always compare training and test scores
4. **Use Learning Curves**: They reveal optimal number of trees
5. **Balance Performance**: Sometimes slightly lower accuracy with faster training is better

## 🎯 Common Use Cases

### When to use XGBoost:
- Need maximum performance
- Have sufficient computational resources
- Want more hyperparameter control

### When to use GradientBoosting:
- Prefer simpler, more interpretable models
- Want to see learning curves
- Need reliable, well-tested algorithm

## 🤝 Contributing

Feel free to extend this app with:
- More datasets
- Additional algorithms (LightGBM, CatBoost)
- Grid search / Random search functionality
- Model export/import features

## 📚 Resources

- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Scikit-learn GradientBoosting](https://scikit-learn.org/stable/modules/ensemble.html#gradient-boosting)
- [Streamlit Documentation](https://docs.streamlit.io/)

## 🐛 Troubleshooting

### If you see an error about missing xgboost:
```bash
pip install xgboost
```

### If plots don't show:
```bash
pip install plotly --upgrade
```

### If app is slow:
- Reduce n_estimators
- Use smaller test_size
- Use synthetic dataset with fewer samples

---

**Happy Learning! 🎉**

Experiment with different combinations and observe how hyperparameters affect model performance. The best way to learn is by doing!
