"""
Main Pipeline for GNSS Signal Classification 
Complete workflow from data to deployment
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Import custom modules
from constants import LABEL_MAP
from feature_engineering import GNSSFeatureEngineer
from labeling import GNSSLabeler
from models import GNSSModelFactory
from evaluation import GNSSEvaluator

class GNSSClassificationPipeline:
    """End-to-end pipeline for GNSS signal classification"""
    
    def __init__(self, data_path: str):
        """
        Args:
            data_path: Path to CSV data file
        """
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.feature_engineer = GNSSFeatureEngineer()
        self.labeler = GNSSLabeler()
        self.evaluator = GNSSEvaluator()
        
    def load_and_preprocess(self):
        """Load and preprocess data"""
        print("Loading data...")
        self.df = pd.read_csv(self.data_path)
        print(f"Loaded {len(self.df)} samples with {len(self.df.columns)} columns")
        
        return self
    
    def engineer_features(self):
        """Perform feature engineering"""
        print("\nEngineering features...")
        
        # Basic feature engineering
        self.df = self.feature_engineer.extract_basic_features(
            self.df, 
            'Carrier Delay-4', 
            'Clock Bias', 
            'PR-4', 
            'Elevation-4'
        )
        
        # Advanced features
        self.df = self.feature_engineer.extract_advanced_features(self.df)
        
        print(f"Created {len(self.feature_engineer.get_feature_list())} features")
        
        return self
    
    def create_labels(self):
        """Create pseudo-labels"""
        print("\nCreating pseudo-labels...")
        self.df = self.labeler.label_with_uncertainty(self.df)
        
        return self
    
    def prepare_data(self, test_size: float = 0.3):
        """Prepare train/test split"""
        print("\nPreparing train/test split...")
        
        # Select features
        feature_cols = [col for col in self.feature_engineer.get_feature_list() 
                       if col not in ['label', 'label_confidence', 'label_str', 'low_confidence']]
        
        X = self.df[feature_cols].fillna(0)
        y = self.df['label']
        
        # Train/test split with stratification
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale features
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        print(f"Training set: {len(self.X_train)} samples")
        print(f"Test set: {len(self.X_test)} samples")
        
        return self
    
    def run_baseline_models(self):
        """Run baseline models"""
        print("\n" + "=" * 70)
        print("RUNNING BASELINE MODELS")
        print("=" * 70)
        
        baseline_models = GNSSModelFactory.get_basic_models()
        baseline_results = {}
        
        for name, model in baseline_models.items():
            print(f"\nTraining {name}...")
            
            # Train model
            model.fit(self.X_train, self.y_train)
            
            # Predictions
            y_pred = model.predict(self.X_test)
            y_proba = model.predict_proba(self.X_test) if hasattr(model, 'predict_proba') else None
            
            # Evaluation
            metrics = self.evaluator.print_detailed_report(self.y_test, y_pred, name)
            baseline_results[name] = {
                'model': model,
                'metrics': metrics,
                'predictions': y_pred
            }
        
        return baseline_results
    
    def run_advanced_models(self):
        """Run advanced models"""
        print("\n" + "=" * 70)
        print("RUNNING ADVANCED MODELS")
        print("=" * 70)
        
        advanced_models = GNSSModelFactory.get_advanced_models()
        advanced_results = {}
        
        for name, model in advanced_models.items():
            print(f"\nTraining {name}...")
            
            # Train model
            model.fit(self.X_train, self.y_train)
            
            # Predictions
            y_pred = model.predict(self.X_test)
            y_proba = model.predict_proba(self.X_test)
            
            # Evaluation
            metrics = self.evaluator.comprehensive_evaluation(self.y_test, y_pred, y_proba, name)
            advanced_results[name] = {
                'model': model,
                'metrics': metrics,
                'predictions': y_pred,
                'probabilities': y_proba
            }
            
            # Print summary
            print(f"  Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_macro']:.4f}")
        
        return advanced_results
    
    def cross_validation(self, model, cv_folds: int = 5):
        """Perform cross-validation"""
        print(f"\nPerforming {cv_folds}-fold cross-validation...")
        
        cv_results = cross_validate(
            model, self.X_train, self.y_train,
            cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
            scoring=['accuracy', 'f1_macro', 'precision_macro', 'recall_macro'],
            return_train_score=True,
            n_jobs=-1
        )
        
        # Print CV results
        print(f"CV Accuracy: {cv_results['test_accuracy'].mean():.4f} (±{cv_results['test_accuracy'].std():.4f})")
        print(f"CV F1-Score: {cv_results['test_f1_macro'].mean():.4f} (±{cv_results['test_f1_macro'].std():.4f})")
        
        return cv_results
    
    def feature_importance_analysis(self, model):
        """Analyze feature importance"""
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            
            # Get feature names
            feature_cols = [col for col in self.feature_engineer.get_feature_list() 
                           if col not in ['label', 'label_confidence', 'label_str', 'low_confidence']]
            
            # Create importance DataFrame
            importance_df = pd.DataFrame({
                'feature': feature_cols,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            print("\nTop 10 Most Important Features:")
            print(importance_df.head(10).to_string(index=False))
            
            return importance_df
        
        return None
    
    def run_full_pipeline(self):
        """Run complete pipeline"""
        print("=" * 70)
        print("GNSS SIGNAL CLASSIFICATION PIPELINE")
        print("=" * 70)
        
        # Execute pipeline steps
        (self.load_and_preprocess()
         .engineer_features()
         .create_labels()
         .prepare_data())
        
        # Run models
        baseline_results = self.run_baseline_models()
        advanced_results = self.run_advanced_models()
        
        # Statistical comparison
        all_predictions = []
        all_model_names = []
        
        for name, result in {**baseline_results, **advanced_results}.items():
            all_predictions.append(result['predictions'])
            all_model_names.append(name)
        
        # Save results
        self.evaluator.save_results()
        
        return {
            'baseline': baseline_results,
            'advanced': advanced_results,
            'data_info': {
                'train_samples': len(self.X_train),
                'test_samples': len(self.X_test),
                'features': self.X_train.shape[1],
                'class_distribution': dict(self.df['label_str'].value_counts())
            }
        }


# Main execution
if __name__ == "__main__":
    # Initialize pipeline
    pipeline = GNSSClassificationPipeline('../data/sample.csv')
    
    # Run complete pipeline
    results = pipeline.run_full_pipeline()
    
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)

