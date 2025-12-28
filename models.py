"""
Advanced ML Models for GNSS Classification
Includes ensemble methods and neural networks 
"""

import numpy as np
from typing import Dict, Any
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
import lightgbm as lgb

class GNSSModelFactory:
    """Factory for creating and configuring ML models"""
    
    @staticmethod
    def get_basic_models() -> Dict[str, Any]:
        """Return dictionary of basic models"""
        return {
            'LogisticRegression': LogisticRegression(
                max_iter=1000,
                class_weight='balanced',
                random_state=42,
                solver='lbfgs',
                multi_class='multinomial'
            ),
            'RandomForest': RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
        }
    
    @staticmethod
    def get_advanced_models() -> Dict[str, Any]:
        """Return dictionary of advanced models"""
        return {
            'XGBoost': xgb.XGBClassifier(
                n_estimators=300,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                use_label_encoder=False,
                eval_metric='mlogloss'
            ),
            'LightGBM': lgb.LGBMClassifier(
                n_estimators=300,
                max_depth=7,
                learning_rate=0.05,
                num_leaves=31,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=5,
                random_state=42
            ),
            'SVM_RBF': SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                class_weight='balanced',
                probability=True,
                random_state=42
            )
        }
    
    @staticmethod
    def get_neural_network(input_dim: int) -> Dict[str, Any]:
        """Create neural network model"""
        return {
            'MLP': MLPClassifier(
                hidden_layer_sizes=(64, 32, 16),
                activation='relu',
                solver='adam',
                alpha=0.0001,
                batch_size='auto',
                learning_rate='adaptive',
                max_iter=500,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1
            )
        }
    
    @staticmethod
    def create_ensemble() -> VotingClassifier:
        """Create ensemble model"""
        estimators = [
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
            ('xgb', xgb.XGBClassifier(random_state=42, use_label_encoder=False)),
            ('lr', LogisticRegression(max_iter=1000, random_state=42))
        ]
        
        return VotingClassifier(
            estimators=estimators,
            voting='soft',
            weights=[2, 3, 1]
        )
    
    @staticmethod
    def get_calibrated_model(base_model, method='sigmoid'):
        """Return calibrated version of model"""
        return CalibratedClassifierCV(
            base_model,
            method=method,
            cv=3
        )

