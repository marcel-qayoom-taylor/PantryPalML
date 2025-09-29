# PantryPal Utils Makefile
# Run 'make help' to see available commands

.PHONY: help check format fix install clean train-data train-model score-recipes etl-transform setup

# Default target
help: ## Show this help message
	@echo "PantryPal Utils - Available Commands:"
	@echo "======================================"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Code Quality
check: ## Run Ruff linting checks
	@echo "🔍 Running Ruff static analysis..."
	uv run ruff check .
	@echo "📊 Issue summary:"
	@uv run ruff check --statistics . | head -10

format: ## Format code with Ruff
	@echo "✨ Formatting code with Ruff..."
	uv run ruff format .

fix: ## Auto-fix Ruff issues where possible
	@echo "🔧 Auto-fixing Ruff issues..."
	uv run ruff check --fix .

# Environment Setup
install: ## Install all dependencies
	@echo "📦 Installing dependencies..."
	uv sync --dev

setup: install ## Full project setup (install + verify)
	@echo "✅ Verifying installation..."
	uv run python -c "import pandas, lightgbm, sklearn; print('All dependencies working!')"

# ML Pipeline Commands
etl-transform: ## Transform raw event data
	@echo "🔄 Running ETL transformation..."
	uv run python -m recipe_recommender.etl.events.event_transformation

train-data: ## Build training data
	@echo "📊 Building training dataset..."
	uv run python -m recipe_recommender.models.training_data_builder

train-model: ## Train the recipe ranker model
	@echo "🚀 Training recipe ranker..."
	uv run python -m recipe_recommender.models.recipe_ranker

score-recipes: ## Run recipe scoring inference
	@echo "🍳 Running recipe scoring..."
	uv run python -m recipe_recommender.inference.recipe_scorer

fetch-data: ## Fetch fresh data from Supabase database
	@echo "📥 Fetching data from database..."
	uv run python -m recipe_recommender.etl.database.fetch_real_recipes

# Analysis
analyze: ## Run exploratory data analysis
	@echo "📈 Running exploratory data analysis..."
	uv run python -m recipe_recommender.analysis.exploratory_data_analysis

feature-eng: ## Run feature engineering
	@echo "🔧 Running feature engineering..."
	uv run python -m recipe_recommender.analysis.feature_engineering

# Full ML Pipeline
pipeline: etl-transform train-data train-model ## Run complete ML pipeline
	@echo "🎉 Complete ML pipeline finished!"

# Cleanup
clean: ## Clean generated files and caches
	@echo "🧹 Cleaning up..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .ruff_cache 2>/dev/null || true
	@echo "✅ Cleanup complete!"

# Development helpers
dev-check: check format ## Quick dev check (lint + format)
	@echo "✅ Development checks complete!"

# Hyperparameter tuning
tune-model: ## Tune hyperparameters (optional - params are hardcoded)
	@echo "🎯 Tuning hyperparameters..."
	uv run python -m recipe_recommender.tuning.tune_ranker --trials 50 --early-stop 15

all: clean install check train-data train-model ## Run everything from scratch
	@echo "🎊 Full build complete!"
