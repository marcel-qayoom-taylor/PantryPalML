# Recipe Recommendation ML Pipeline

This module contains the complete machine learning pipeline for PantryPal's hybrid recipe recommendation system.

## 🎯 System Overview

A production-ready recommendation system that combines:
- **User interaction history** from analytics events
- **Rich recipe metadata** from Supabase database  
- **Hybrid GBM model** (LightGBM) for scoring and ranking

## 📁 Module Structure

```
ml_etl/
├── config.py          # 🆕 Centralized configuration management
├── utils/             # 🆕 Common utilities and logging
├── database/          # Supabase integration
├── models/            # ML model training (improved architecture)
├── inference/         # Production API (enhanced with config)
├── transformations/   # Data preprocessing
├── analysis/          # Exploratory analysis (reference)
├── input/             # Raw event data
└── output/            # Processed data and trained models
```

## 🚀 Usage Workflows

### Getting Recommendations (Production)
```python
from recipe_recommender.inference.production_recipe_scorer import ProductionRecipeScorer

scorer = ProductionRecipeScorer()
recommendations = scorer.get_user_recipe_recommendations(
    user_id="user123",
    interaction_history=[
        {"recipe_id": "100", "event_type": "Recipe Cooked", "timestamp": 1755666000}
    ],
    n_recommendations=10
)
```

### Configuration Management
```python
from recipe_recommender.config import get_ml_config

# Get default configuration
config = get_ml_config()

# Customize settings
config.learning_rate = 0.05
config.negative_sampling_ratio = 3

# Use with models
from recipe_recommender.models.hybrid_gbm_recommender import HybridGBMRecommender
recommender = HybridGBMRecommender(config=config)
```

### Retraining Pipeline
```bash
# 1. Extract latest recipe data
python recipe_recommender/etl/database/fetch_real_recipes.py

# 2. Build training dataset
python recipe_recommender/models/hybrid_recommendation_data_builder.py

# 3. Train model
python recipe_recommender/models/hybrid_gbm_recommender.py

# 4. Test updated model
python recipe_recommender/inference/production_recipe_scorer.py
```

## 📊 Key Components

### Configuration & Utilities
- **`config.py`**: Centralized settings and path management 
- **`utils/common.py`**: Shared functions, logging, file operations

### Database Layer (`database/`)
- **`supabase_config.py`**: Database connection management
- **`fetch_real_recipes.py`**: Extract complete recipe catalog

### Model Training (`models/`)  
- **`hybrid_recommendation_data_builder.py`**: Build training dataset with config support
- **`hybrid_gbm_recommender.py`**: Train LightGBM model with improved architecture

### Production API (`inference/`)
- **`production_recipe_scorer.py`**: Real-time recommendation scoring with config

### Data Processing (`transformations/`)
- **`event_transformation.py`**: Process user interaction events
- **`helpers.py`**: Utility functions for data cleaning

## 🎯 Model Details

### Features Used (22 total)
- **User Behavior**: Total interactions, engagement score, activity patterns
- **Recipe Content**: Ingredient count, complexity, cooking time, tags
- **Compatibility**: User-recipe matching scores

### Training Dataset
- **20,730 samples** (20% positive, 80% negative)
- **831 users** with interaction history
- **1,967 recipes** with complete metadata
- **Temporal train/val/test split** (60/20/20)

### Performance Metrics
- **Classification**: 99.92% AUC, 97.55% F1-score
- **Ranking**: 62.18% NDCG@5
- **Speed**: ~0.08 seconds to score 1,967 recipes

## ⚙️ Configuration

### Required Environment Variables
```bash
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_ANON_KEY=your-anon-key  
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key
```

### Database Schema Requirements
- `recipe` table: Recipe metadata
- `ingredient` table: Ingredient catalog  
- `ingredients_of_recipe` table: Recipe-ingredient relationships

## 🔧 Maintenance

### Regular Tasks
- **Monitor model performance** via evaluation metrics
- **Retrain monthly** with new interaction data
- **Update recipe database** when new recipes are added
- **Review feature importance** to understand model behavior

### Model Files Location
- **Trained Model**: `output/hybrid_models/hybrid_lightgbm_model.txt`
- **Model Metadata**: `output/hybrid_models/hybrid_lightgbm_metadata.json`  
- **Training Data**: `output/hybrid_train_data.csv` (and val/test)
- **Recipe Features**: `output/enhanced_recipe_features_from_db.csv`