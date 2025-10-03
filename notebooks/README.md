# A2 Production Notebooks Usage Guide

This directory contains the end-to-end production notebook for the PantryPal Recipe Recommendation System:
- **`A2_Production_EndToEnd.ipynb`** - Complete pipeline from data to model training to real-time inference

## 🚀 Quick Start

### Option 1: Google Colab (Recommended)

**End-to-End Notebook:**
1. Click: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/marcel-qayoom-taylor/PantryPalML/blob/main/notebooks/A2_Production_EndToEnd.ipynb)
2. Runtime → Run all (or press `Ctrl+F9`)
3. View training metrics and generated recommendations in one run

### Option 2: Binder (Jupyter Lab)

**End-to-End:**
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/marcel-qayoom-taylor/PantryPalML/HEAD?labpath=notebooks%2FA2_Production_EndToEnd.ipynb)

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

# Start Jupyter (end-to-end)
jupyter notebook notebooks/A2_Production_EndToEnd.ipynb

# In the notebook: Kernel → Restart & Run All
```

## 📋 What the Notebook Does

The notebook demonstrates a complete machine learning pipeline in 6 cells:

### Cell 1: Environment Setup
- **Purpose**: Installs dependencies and clones repo (Colab only)
- **Output**: "Environment ready. Project root: /path/to/project"

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

---

**Ready to run? Click the Colab badges at the top and execute Runtime → Run all!** 🚀

📚 **For complete documentation, see the [main project README](../README.md)**

