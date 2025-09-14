# A2 Demo Notebook Usage Guide

This directory contains `A2_Colab_Demo.ipynb` - a self-contained cloud-executable demonstration of the PantryPal Recipe Recommendation System for university assignment A2.

## 🚀 Quick Start

### Option 1: Google Colab (Recommended)
1. Click this badge: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/marcel-qayoom-taylor/PantryPalML/blob/main/notebooks/A2_Colab_Demo.ipynb)
2. Wait for Colab to load the notebook
3. Click **Runtime → Run all** (or press `Ctrl+F9`)
4. Wait ~2-3 minutes for complete execution
5. Scroll down to see results and smoke test output

### Option 2: Binder (Jupyter Lab)
1. Click this badge: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/marcel-qayoom-taylor/PantryPalML/HEAD?labpath=notebooks%2FA2_Colab_Demo.ipynb)
2. Wait for Binder to build the environment (~1-2 minutes)
3. Once Jupyter Lab opens, the notebook should load automatically
4. Click **Kernel → Restart & Run All**
5. Wait for execution to complete

### Option 3: Local Jupyter
```bash
# Clone the repository
git clone https://github.com/marcel-qayoom-taylor/PantryPalML.git
cd PantryPalML

# Install dependencies
pip install lightgbm pandas numpy scikit-learn matplotlib seaborn jupyter

# Start Jupyter
jupyter notebook notebooks/A2_Colab_Demo.ipynb

# In the notebook: Kernel → Restart & Run All
```

## 📋 What the Notebook Does

The notebook demonstrates a complete machine learning pipeline in 6 cells:

### Cell 1: Environment Setup
- **Purpose**: Installs dependencies and clones repo (Colab only)
- **Output**: "Environment ready. Project root: /path/to/project"
- **Time**: ~30-60 seconds on first run

### Cell 2: Data Loading
- **Purpose**: Loads training/validation/test datasets
- **Behavior**: 
  - First tries to load existing CSV files from the repo
  - Falls back to generating synthetic data if CSVs are missing
- **Output**: Dataset shapes, e.g., `(100, 6) (20, 6) (20, 6)`

### Cell 3: Model Training
- **Purpose**: Trains a LightGBM binary classifier
- **Features**: Uses numeric columns (rating, ingredient count, complexity)
- **Output**: Performance metrics (AUC, Precision, Recall, F1)
- **Time**: ~10-20 seconds

### Cell 4: Recommendation Demo
- **Purpose**: Shows inference - scoring and ranking recipes
- **Output**: DataFrame with top-ranked recipes for a demo user
- **Columns**: `recipe_id`, `score`, `user_id`

### Cell 5: Smoke Test
- **Purpose**: Validates the entire pipeline works correctly
- **Output**: Either "SMOKE TEST: PASS" or detailed error messages
- **Checks**: Data loading, feature columns, model predictions, recommendation output

## 🎯 Expected Outputs

### Successful Run Example:
```
Environment ready. Project root: /content/PantryPalML
(100, 6) (20, 6) (20, 6)
{'AUC': 0.95, 'Precision': 0.89, 'Recall': 0.85, 'F1': 0.87}
[DataFrame showing top 10 recipe recommendations]
SMOKE TEST: PASS
Train/Val/Test sizes: 100, 20, 20
Features used: 3
```

### Performance Expectations:
- **AUC**: Should be > 0.7 (typically 0.8-0.95)
- **F1-Score**: Should be > 0.6 (typically 0.7-0.9)
- **Recommendations**: Should return 10 ranked recipes with scores

## 🔧 A2 Assignment Criteria Compliance

### Task Definition (Input/Output)
- **Training Input**: CSV with user-recipe pairs and features (`user_id`, `recipe_id`, `label`, `avg_rating`, `ingredient_count`, `complexity_score`)
- **Training Output**: Trained LightGBM model with binary classification capability
- **Inference Input**: User ID and candidate recipes DataFrame
- **Inference Output**: Ranked list of recipes with relevance scores (0-1 range)

### Model Implementation
- **Algorithm**: Gradient Boosting Machine (LightGBM) for binary classification
- **Features**: Numeric features extracted from user-recipe interactions
- **Loss Function**: Binary cross-entropy (logistic loss)
- **Evaluation**: AUC, Precision, Recall, F1-score on held-out validation set

### System Workflow
1. Data loading and feature preparation
2. Train/validation/test split
3. LightGBM model training with early stopping
4. Performance evaluation on validation set
5. Inference demonstration with recommendation ranking

## 🐛 Troubleshooting

### Common Issues & Solutions:

#### "No module named 'lightgbm'"
- **Solution**: Re-run Cell 1 or manually install: `!pip install lightgbm`

#### "SMOKE TEST: FAIL - Dataframes not defined"
- **Cause**: Cells were run out of order
- **Solution**: Run **Runtime → Restart & Run All** to execute in sequence

#### "SMOKE TEST: FAIL - Non-numeric features found"
- **Cause**: Data loading issue with feature types
- **Solution**: This indicates the fallback synthetic data generator ran - this is normal if CSV files are missing

#### Low performance metrics (AUC < 0.6)
- **Cause**: Using synthetic fallback data instead of real training data
- **Status**: Expected behavior - the synthetic data is for demonstration only

#### Colab "Runtime disconnected" 
- **Solution**: Click **Runtime → Reconnect** and run cells again

#### Binder fails to load
- **Alternative**: Use Colab link or run locally

## 📊 Understanding the Results

### Model Performance Metrics:
- **AUC (0.7-0.95)**: How well the model distinguishes positive/negative recipe preferences
- **Precision (0.6-0.9)**: Of recipes predicted as relevant, how many actually are
- **Recall (0.6-0.9)**: Of truly relevant recipes, how many were identified
- **F1-Score (0.6-0.9)**: Balanced measure combining precision and recall

### Recommendation Scores:
- **Range**: 0.0 (least relevant) to 1.0 (most relevant)
- **Interpretation**: Higher scores indicate higher predicted user preference
- **Usage**: Rank recipes by score to create personalized recommendation lists

## 🔄 Customization Options

### Modify User Interactions:
Edit Cell 4 to test different recommendation scenarios:
```python
# Add more diverse test data
sample_candidates = pd.DataFrame({
    'user_id': ['test_user'] * 5,
    'recipe_id': ['recipe_1', 'recipe_2', 'recipe_3', 'recipe_4', 'recipe_5'],
    'avg_rating': [4.5, 3.2, 4.8, 2.1, 4.0],
    'ingredient_count': [8, 12, 6, 15, 10],
    'complexity_score': [5.2, 8.1, 3.5, 9.2, 6.0]
})
recs = recommend_for_user("test_user", sample_candidates)
```

### Adjust Model Parameters:
Modify Cell 3 training parameters:
```python
params = {
    "learning_rate": 0.05,  # Slower learning
    "num_leaves": 50,       # More complex trees
    "min_data_in_leaf": 5,  # Less regularization
}
```

## 📈 Next Steps After A2

This demo notebook provides a foundation. For production deployment:

1. **Scale up**: Use the full training pipeline in `recipe_recommender/models/`
2. **Real data**: Connect to Supabase database for complete recipe catalog
3. **Rich features**: Add user behavior patterns and recipe content features
4. **API deployment**: Integrate with FastAPI for real-time recommendations

## 💡 Tips for Presentation (A3)

- **Focus on workflow**: Explain the 5-step ML pipeline clearly
- **Highlight trade-offs**: Discuss why LightGBM vs other algorithms
- **Show live demo**: Run a cell or two during presentation
- **Explain metrics**: What do AUC=0.85 and F1=0.80 mean for recommendations?
- **Discuss limitations**: Synthetic data vs real-world performance

## 📞 Support

If you encounter issues:
1. Check this README troubleshooting section
2. Try the alternative platforms (Colab → Binder → Local)
3. Verify the smoke test output for specific error details
4. Ensure you run cells in order (use "Restart & Run All")

---

**Ready to run? Click the Colab badge at the top and execute Runtime → Run all!** 🚀
