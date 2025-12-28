"""
Intelligent Pseudo-Labeling Module
Implements rule-based labeling with uncertainty estimation
"""

import numpy as np
import pandas as pd
from typing import Tuple
from constants import THRESHOLDS, LABEL_MAP

class GNSSLabeler:
    """Advanced pseudo-labeling with uncertainty modeling"""
    
    def __init__(self, method='hybrid'):
        """
        Args:
            method: 'threshold', 'clustering', or 'hybrid'
        """
        self.method = method
        self.label_distribution = None
        
    def threshold_based_labeling(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Rule-based labeling using domain knowledge
        
        Args:
            df: DataFrame with features
            
        Returns:
            Tuple of (labels, confidence_scores)
        """
        labels = []
        confidences = []
        
        for _, row in df.iterrows():
            code_carrier_div = abs(row['code_carrier_div'])
            elevation = row['elevation_norm'] * 90.0  # Convert back to degrees
            
            # LOS: High elevation, small code-carrier divergence
            if (elevation > THRESHOLDS['LOS']['elevation_min'] and 
                code_carrier_div < THRESHOLDS['LOS']['code_carrier_max']):
                labels.append(0)  # LOS
                confidence = 1.0 - (code_carrier_div / THRESHOLDS['LOS']['code_carrier_max'])
                
            # NLOS: Low elevation, large divergence
            elif (elevation < THRESHOLDS['NLOS']['elevation_max'] and 
                  code_carrier_div > THRESHOLDS['NLOS']['code_carrier_min']):
                labels.append(2)  # NLOS
                confidence = min(1.0, code_carrier_div / 10.0)
                
            # MULTIPATH: Medium conditions
            else:
                labels.append(1)  # MULTIPATH
                # Confidence based on distance from thresholds
                elev_conf = 1 - abs(elevation - 30) / 45  # Peak at 30 degrees
                div_conf = 1 - abs(code_carrier_div - 3.5) / 3.5  # Peak at 3.5
                confidence = (elev_conf + div_conf) / 2
            
            confidences.append(min(max(confidence, 0.1), 1.0))
        
        self.label_distribution = pd.Series(labels).value_counts()
        return pd.Series(labels, index=df.index), pd.Series(confidences, index=df.index)
    
    def label_with_uncertainty(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Comprehensive labeling with uncertainty estimation
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with labels and confidence scores
        """
        df = df.copy()
        
        # Get labels and confidence
        labels, confidence = self.threshold_based_labeling(df)
        
        # Add to dataframe
        df['label'] = labels
        df['label_confidence'] = confidence
        df['label_str'] = df['label'].map(LABEL_MAP)
        
        # Add uncertainty flags
        df['low_confidence'] = df['label_confidence'] < 0.7
        
        # Statistics
        self.print_label_statistics(df)
        
        return df
    
    def print_label_statistics(self, df: pd.DataFrame):
        """Print detailed label statistics"""
        print("=" * 50)
        print("LABEL DISTRIBUTION")
        print("=" * 50)
        
        total = len(df)
        for label_id, label_name in LABEL_MAP.items():
            count = (df['label'] == label_id).sum()
            percentage = (count / total) * 100
            avg_conf = df[df['label'] == label_id]['label_confidence'].mean()
            
            print(f"{label_name:12s}: {count:5d} samples ({percentage:5.1f}%) | "
                  f"Avg Confidence: {avg_conf:.3f}")
        
        low_conf_count = df['low_confidence'].sum()
        print(f"\nLow confidence samples: {low_conf_count} ({low_conf_count/total*100:.1f}%)")
        print("=" * 50)
