# A2 Production Notebooks Usage Guide

This directory contains production-ready notebooks for the PantryPal Recipe Recommendation System:
- **`A2_Production_Training.ipynb`** - Complete training pipeline from data to model
- **`A2_Production_Inference.ipynb`** - Real-time inference with pre-trained models

## 🚀 Quick Start

### Option 1: Google Colab (Recommended)

**Training Notebook:**
1. Click: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/marcel-qayoom-taylor/PantryPalML/blob/main/notebooks/A2_Production_Training.ipynb)
2. Runtime → Run all (or press `Ctrl+F9`)
3. Wait ~10 seconds for training to complete
4. View model performance metrics and evaluation results

**Inference Notebook:**
1. Click: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/marcel-qayoom-taylor/PantryPalML/blob/main/notebooks/A2_Production_Inference.ipynb)
2. Runtime → Run all
3. See personalized recommendations generated in real-time

### Option 2: Binder (Jupyter Lab)

**Training:**
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/marcel-qayoom-taylor/PantryPalML/HEAD?labpath=notebooks%2FA2_Production_Training.ipynb)

**Inference:**
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/marcel-qayoom-taylor/PantryPalML/HEAD?labpath=notebooks%2FA2_Production_Inference.ipynb)

1. Click badge and wait for environment to build (~1-2 minutes)
2. Once Jupyter Lab opens, the notebook loads automatically
3. Click **Kernel → Restart & Run All**

### Option 3: Local Jupyter
```bash
# Clone the repository
git clone https://github.com/marcel-qayoom-taylor/PantryPalML.git
cd PantryPalML

# Install dependencies
pip install lightgbm pandas numpy scikit-learn matplotlib seaborn jupyter supabase python-dotenv

# Start Jupyter (choose one notebook)
jupyter notebook notebooks/A2_Production_Training.ipynb
jupyter notebook notebooks/A2_Production_Inference.ipynb

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
- **Output**: Ranking metrics (NDCG@k, Recall@k, Spearman correlation)
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

### Training Notebook Output:
```
Environment ready. Project root: /content/PantryPalML
Loading datasets...
Train: (16,584, 22) | Val: (2,073, 22) | Test: (2,073, 22)

Training LightGBM model...
[LightGBM] [Info] Training until validation scores don't improve for 50 rounds
[LightGBM] [Info] Early stopping at iteration 145

Ranking Performance:
  NDCG@5: 0.6545
  NDCG@10: 0.6545
  Recall@5: 0.9555
  Recall@10: 0.9894
  Spearman: 0.9958

Model saved to: recipe_recommender/output/hybrid_models/
```

### Inference Notebook Output:
```
Loading pre-trained model...
Model loaded successfully: hybrid_lightgbm_model.txt

Loading recipe catalog...
1,967 recipes loaded with complete features

Generating recommendations for user: demo_user_001
User interaction history: 3 events

Top 10 Recommendations:
┌──────────┬───────────────────────────┬───────┬────────────────┬─────────┐
│ recipe_id │ recipe_name              │ score │ author_name    │ time    │
├──────────┼───────────────────────────┼───────┼────────────────┼─────────┤
│ 1234     │ Spicy Thai Basil Chicken │ 0.945 │ Chef Williams  │ 30 min  │
│ 5678     │ Mediterranean Pasta      │ 0.923 │ Chef Anderson  │ 25 min  │
│ ...      │ ...                      │ ...   │ ...            │ ...     │
└──────────┴───────────────────────────┴───────┴────────────────┴─────────┘
```

### Performance Expectations:
- **NDCG@5**: Around 0.65 on the provided dataset
- **Recall@10**: Around 0.98 on the provided dataset
- **Spearman**: Around 0.99 (rank correlation)
- **Recommendations**: Returns top-N ranked recipes with scores 0.0-1.0
- **Inference Speed**: ~0.08 seconds to score 1,967 recipes

## 🔧 A2 Assignment Criteria Compliance

### Task Definition (Input/Output)
- **Training Input**: CSV files with user-recipe pairs and 22 engineered features
  - Features include: user behavior patterns, recipe content, compatibility scores
  - Files: `hybrid_train_data.csv`, `hybrid_val_data.csv`, `hybrid_test_data.csv`
- **Training Output**: Trained LightGBM model with metadata
  - Model file: `hybrid_lightgbm_model.txt`
  - Metadata: `hybrid_lightgbm_metadata.json`, `best_params.json`
- **Inference Input**: User ID and interaction history (list of recipe events)
  - Format: `[{"recipe_id": "100", "event_type": "Recipe Cooked", "timestamp": 1755666000}]`
- **Inference Output**: Ranked list of recipes with relevance scores (0-1 range)
  - Includes recipe metadata: name, author, time, servings, ingredients

### Model Implementation
- **Algorithm**: Gradient Boosting Machine (LightGBM) for binary classification
- **Features**: Numeric features extracted from user-recipe interactions
- **Loss Function**: Binary cross-entropy (logistic loss)
- **Evaluation**: NDCG@k, Recall@k, Spearman correlation on held-out sets

### System Workflow
1. Data loading and feature preparation
2. Train/validation/test split
3. LightGBM model training with early stopping
4. Performance evaluation on validation set
5. Inference demonstration with recommendation ranking

## 🐛 Troubleshooting

### Common Issues & Solutions:

#### "No module named 'lightgbm'" or other import errors
- **Solution**: Re-run the first cell (Environment Setup) or manually install:
  ```python
  !pip install lightgbm pandas numpy scikit-learn matplotlib seaborn supabase python-dotenv
  ```

#### "File not found" errors for CSV or model files
- **Cause**: Repository files not properly cloned or path issues
- **Solution**: 
  1. Verify repo was cloned: Check if `/content/PantryPalML` exists (Colab)
  2. Re-run the environment setup cell
  3. Check that `recipe_recommender/output/` directory exists with data files

#### Cells run out of order
- **Cause**: Running cells individually without dependencies
- **Solution**: Run **Runtime → Restart & Run All** to execute in sequence

#### Training notebook shows low performance (NDCG@5 < 0.60)
- **Cause**: Possible data loading issue or incomplete training
- **Solution**: 
  1. Check that training data was loaded correctly
  2. Verify all 22 features are present
  3. Ensure training completed (look for "Model saved" message)

#### Inference notebook: "No model found"
- **Cause**: Pre-trained model not loaded or path issue
- **Solution**: 
  1. Verify model file exists: `recipe_recommender/output/hybrid_models/hybrid_lightgbm_model.txt`
  2. If running locally, train the model first using training notebook
  3. Check paths in notebook configuration cells

#### Colab "Runtime disconnected"
- **Solution**: Click **Runtime → Reconnect** and **Runtime → Run all** again
- **Prevention**: Keep browser tab active during execution

#### Binder fails to load or times out
- **Alternative**: Use Colab link (more reliable) or run locally
- **Note**: Binder can be slow on first build (~2-5 minutes)

## 📊 Understanding the Results

### Model Performance Metrics:
- **NDCG@k**: Ranking quality emphasizing top-k positions (higher is better)
- **Recall@k**: Fraction of relevant items appearing in the top-k list
- **Spearman correlation**: Agreement between predicted ranking and ground truth order

### Recommendation Scores:
- **Range**: 0.0 (least relevant) to 1.0 (most relevant)
- **Interpretation**: Higher scores indicate higher predicted user preference
- **Usage**: Rank recipes by score to create personalized recommendation lists

## 🔄 Customization Options

### Training Notebook Customization:

**Adjust Model Hyperparameters:**
Modify the training parameters to experiment with different model configurations:
```python
# In the training cell, modify these parameters
params = {
    "learning_rate": 0.05,      # Slower learning (default: 0.1)
    "num_leaves": 50,           # More complex trees (default: 31)
    "min_data_in_leaf": 10,     # More regularization (default: 20)
    "max_depth": 8,             # Limit tree depth (default: -1)
    "feature_fraction": 0.8,    # Use 80% of features per tree
}
```

**Change Train/Val/Test Split:**
```python
# Modify the split ratios
train_size = 0.70  # 70% training (default: 80%)
val_size = 0.15    # 15% validation (default: 10%)
test_size = 0.15   # 15% test (default: 10%)
```

### Inference Notebook Customization:

**Test Different User Interactions:**
```python
# Modify user interaction history
user_interactions = [
    {"recipe_id": "100", "event_type": "Recipe Viewed", "timestamp": 1755665991},
    {"recipe_id": "100", "event_type": "Recipe Cooked", "timestamp": 1755666000},
    {"recipe_id": "1291", "event_type": "Recipe Favourited", "timestamp": 1755666100},
    {"recipe_id": "500", "event_type": "Recipe Shared", "timestamp": 1755666200},
]
```

**Adjust Recommendation Count:**
```python
# Get more/fewer recommendations
recommendations = scorer.get_user_recommendations(
    user_id="test_user",
    interaction_history=user_interactions,
    n_recommendations=20  # Change from default 10 to 20
)
```

**Filter Recommendations:**
```python
# Filter by cooking time (implement custom filtering)
quick_recipes = [rec for rec in recommendations['recommendations'] 
                 if rec['total_time'] <= 30]  # 30 minutes or less
```

## 📈 Next Steps After A2

These notebooks provide a production-ready foundation. For further enhancements:

1. **Enhanced Features**: Add text embeddings for recipe descriptions and instructions
2. **Real-time Updates**: Implement online learning to adapt to new user interactions
3. **A/B Testing**: Test different model configurations and feature sets
4. **API Integration**: Deploy with FastAPI for real-time recommendation serving
5. **Monitoring**: Track recommendation CTR and user engagement metrics
6. **Explainability**: Add SHAP values to explain why recipes were recommended

## 💡 Tips for Presentation (A3)

### Demonstrate Understanding
- **Training Notebook**: Show the complete ML workflow from data to model
- **Inference Notebook**: Demonstrate real-time recommendations with different user profiles
- **Compare Users**: Show how recommendations differ based on interaction history

### Key Discussion Points
- **Algorithm Choice**: Why LightGBM? (Fast, accurate, handles mixed features well)
- **Hybrid Approach**: Benefits of combining collaborative + content-based filtering
- **Feature Engineering**: How user profiles and recipe content create powerful features
- **Metrics Interpretation**: What do NDCG@5≈0.65, Recall@10≈0.99, Spearman≈0.996 mean for recommendations?
- **Cold Start**: How the system handles new users with minimal interaction history

### Live Demo Strategy
1. Start with **Inference Notebook** for immediate impact
2. Show recommendations for different user types (e.g., beginner cook vs. experienced)
3. Modify interaction history live to show personalization
4. Then show **Training Notebook** to explain how the model was built
5. Highlight model performance metrics and feature importance

### Address These Questions
- How does the system balance popularity with personalization?
- What happens when a new recipe is added?
- How often should the model be retrained?
- What are the computational requirements for real-time serving?

## 📞 Support

If you encounter issues:
1. Check this README troubleshooting section above
2. Try alternative platforms: Colab → Binder → Local Jupyter
3. Verify all dependencies are installed (check first cell output)
4. Ensure cells run in order (use "Restart & Run All")
5. For file not found errors, verify repo cloning was successful
6. Check the main project README for additional troubleshooting

### Useful Commands for Local Development
```bash
# Verify environment
python --version  # Should be Python 3.8+
pip list | grep lightgbm  # Check LightGBM installed

# Check data files exist
ls -lh recipe_recommender/output/hybrid_*.csv
ls -lh recipe_recommender/output/hybrid_models/

# Run notebooks from command line
jupyter nbconvert --to notebook --execute notebooks/A2_Production_Training.ipynb
jupyter nbconvert --to notebook --execute notebooks/A2_Production_Inference.ipynb
```

---

**Ready to run? Click the Colab badges at the top and execute Runtime → Run all!** 🚀

📚 **For complete documentation, see the [main project README](../README.md)**

