"""
Feature Engineering Module - Domain Knowledge Based Feature Categories
"""

import pandas as pd
from typing import Dict, List, Tuple
import logging

FEATURE_CATEGORIES = {
    'hematologi': {
        'name': 'Hematologi / Darah Lengkap',
        'features': ['HB', 'LEUKOSIT', 'LED', 'EOSINOPIL', 'BASOPIL', 'SEGMENT', 'LYMPOSIT', 'MONOSIT', 'TROMBOSIT'],
        'description': '9 fitur untuk analisis darah lengkap'
    },
    'liver_function': {
        'name': 'Fungsi Hati / Liver Function',
        'features': ['BILIRUBIN_TOTAL', 'BILIRUBIN_DIRECT', 'BILIRUBIN_INDIRECT', 'ALKALINE_PHOSPAT', 'SGPT', 'SGOT', 'GAMMA_GT'],
        'description': '7 fitur untuk fungsi hati'
    },
    'kidney_function': {
        'name': 'Fungsi Ginjal / Kidney Function',
        'features': ['UREUM', 'KREATININ', 'ASAM_URAT_GINJAL'],
        'description': '3 fitur untuk fungsi ginjal'
    },
    'lipid_profile': {
        'name': 'Profil Lipid / Lipid Profile',
        'features': ['KOLEST_TOTAL', 'TRIGLISERIDA', 'HDL_KOLEST', 'LDL_KOLEST'],
        'description': '4 fitur untuk profil lipid'
    },
    'glucose_metabolism': {
        'name': 'Metabolisme Glukosa / Glucose Metabolism',
        'features': ['GULA_DARAH_PUASA', 'GULA_DARAH_2JAMPP'],
        'description': '2 fitur untuk metabolisme glukosa'
    },
    'vital_signs': {
        'name': 'Vital Signs',
        'features': ['TINGGI', 'BERAT', 'NADI', 'PERNAPASAN', 'SUHU'],
        'description': '5 fitur untuk vital signs'
    },
    'urine_test': {
        'name': 'Tes Urin / Urine Test',
        'features': ['UROBILINOGEN', 'BILIRUBIN_1', 'ASAM_URAT_URIN_1', 'TRIPLE_PHOSP_1'],
        'description': '4 fitur untuk tes urin'
    }
}


def setup_logger(name: str, verbose: bool = True) -> logging.Logger:
    """Setup logger utility - prevent duplicate handlers"""
    logger = logging.getLogger(name)
    
    # Clear existing handlers untuk avoid duplikat
    logger.handlers.clear()
    logger.propagate = False
    
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO if verbose else logging.WARNING)
    return logger


class FeatureSelector:
    """Memilih feature berdasarkan kategori atau custom"""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.logger = setup_logger(__name__, verbose)
    
    def select_by_category(self, data: pd.DataFrame, category: str) -> Tuple[pd.DataFrame, List[str]]:
        """Select features berdasarkan kategori"""
        if category not in FEATURE_CATEGORIES:
            raise ValueError(f"Category '{category}' tidak ditemukan. Available: {list(FEATURE_CATEGORIES.keys())}")
        
        features = FEATURE_CATEGORIES[category]['features']
        available = [f for f in features if f in data.columns]
        
        self.logger.info(f"Category: {FEATURE_CATEGORIES[category]['name']} ({len(available)} features)")
        return data[available], available
    
    def select_multiple_categories(self, data: pd.DataFrame, categories: List[str]) -> Tuple[pd.DataFrame, List[str]]:
        """Select features dari multiple kategori"""
        all_features = []
        for cat in categories:
            if cat in FEATURE_CATEGORIES:
                all_features.extend(FEATURE_CATEGORIES[cat]['features'])
        
        all_features = list(dict.fromkeys(all_features))  # Remove duplicates
        available = [f for f in all_features if f in data.columns]
        
        self.logger.info(f"Selected {len(categories)} categories ({len(available)} features)")
        return data[available], available
    
    def select_custom(self, data: pd.DataFrame, features: List[str]) -> Tuple[pd.DataFrame, List[str]]:
        """Select custom features"""
        available = [f for f in features if f in data.columns]
        self.logger.info(f"Selected {len(available)} custom features")
        return data[available], available
    
    def get_categories(self) -> Dict:
        """Get semua kategori feature"""
        return FEATURE_CATEGORIES
    
    def print_categories(self):
        """Print semua kategori yang tersedia"""
        print("\n" + "="*70)
        print("AVAILABLE FEATURE CATEGORIES")
        print("="*70)
        for key, info in FEATURE_CATEGORIES.items():
            print(f"\n{key.upper()}: {info['name']}")
            print(f"  Description: {info['description']}")
            print(f"  Features: {', '.join(info['features'])}")
