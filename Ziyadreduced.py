import streamlit as st
import pandas as pd
import numpy as np
import tempfile
import os
import platform
from io import BytesIO
import base64
import re
import math
import random

# Check for optional dependencies
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    from rdkit.Chem.Draw import rdMolDraw2D
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

class SimpleMolecularDescriptorExtractor:
    """Simple molecular descriptor extraction using basic SMILES parsing."""
    
    def __init__(self):
        self.descriptor_names = [
            'MolWt', 'NumAtoms', 'NumBonds', 
            'NumRings', 'NumHeteroAtoms', 'SMILESLength'
        ]
    
    def parse_smiles_basic(self, smiles):
        """Basic SMILES parsing for simple descriptors."""
        try:
            # Clean SMILES
            smiles = smiles.strip()
            if not smiles:
                return None
            
            # Count atoms (simple approximation)
            # Remove brackets and count uppercase letters (atoms)
            atom_pattern = r'[A-Z][a-z]?'
            atoms = re.findall(atom_pattern, smiles)
            num_atoms = len(atoms)
            
            # Estimate molecular weight (very basic)
            atomic_weights = {
                'C': 12.01, 'N': 14.01, 'O': 16.00, 'S': 32.06,
                'P': 30.97, 'F': 19.00, 'Cl': 35.45, 'Br': 79.90,
                'I': 126.90, 'H': 1.008
            }
            
            mol_weight = 0
            for atom in atoms:
                mol_weight += atomic_weights.get(atom, 12.01)  # Default to carbon
            
            # Count bonds (approximate)
            bond_chars = ['-', '=', '#', ':']
            num_bonds = sum(smiles.count(char) for char in bond_chars)
            
            # Count rings (approximate by counting ring closure numbers)
            ring_numbers = re.findall(r'\d', smiles)
            num_rings = len(set(ring_numbers)) // 2  # Each ring has 2 closure points
            
            # Count heteroatoms (non-carbon atoms)
            heteroatoms = [atom for atom in atoms if atom != 'C']
            num_heteroatoms = len(heteroatoms)
            
            descriptors = {
                'MolWt': mol_weight,
                'NumAtoms': num_atoms,
                'NumBonds': max(num_bonds, num_atoms - 1),  # At least n-1 bonds for n atoms
                'NumRings': num_rings,
                'NumHeteroAtoms': num_heteroatoms,
                'SMILESLength': len(smiles)
            }
            
            return descriptors
            
        except Exception as e:
            st.error(f"Error parsing SMILES: {e}")
            return None
    
    def smiles_to_descriptors(self, smiles):
        """Convert SMILES to descriptors using available method."""
        if RDKIT_AVAILABLE:
            return self.rdkit_descriptors(smiles)
        else:
            return self.parse_smiles_basic(smiles)
    
    def rdkit_descriptors(self, smiles):
        """Use RDKit for accurate descriptors."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            descriptors = {
                'XLogP': Descriptors.MolLogP(mol),
                'TPSA': Descriptors.TPSA(mol),
                'MolWt': Descriptors.MolWt(mol),
                'HBD': Descriptors.NumHDonors(mol),
                'HBA': Descriptors.NumHAcceptors(mol),
                'RotatableBonds': Descriptors.NumRotatableBonds(mol)
            }
            self.descriptor_names = list(descriptors.keys())
            return descriptors
        except Exception as e:
            return self.parse_smiles_basic(smiles)
    
    def batch_extract_descriptors(self, smiles_list):
        """Extract descriptors for multiple SMILES."""
        descriptors_list = []
        for smiles in smiles_list:
            desc = self.smiles_to_descriptors(smiles)
            if desc is not None:
                descriptors_list.append(desc)
            else:
                # Fill with default values for invalid SMILES
                descriptors_list.append({name: 0 for name in self.descriptor_names})
        
        return pd.DataFrame(descriptors_list)

class SimpleRandomForest:
    """Simple Random Forest implementation when sklearn is not available."""
    
    def __init__(self, n_estimators=10, max_depth=5, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.trees = []
        self.feature_importances_ = None
        self.is_trained = False
        
        random.seed(random_state)
        np.random.seed(random_state)
    
    class DecisionTree:
        def __init__(self, max_depth=5):
            self.max_depth = max_depth
            self.tree = None
        
        def fit(self, X, y):
            self.tree = self._build_tree(X, y, 0)
        
        def _build_tree(self, X, y, depth):
            if depth >= self.max_depth or len(set(y)) == 1 or len(X) < 2:
                return {'prediction': np.mean(y) > 0.5}
            
            best_feature, best_threshold = self._find_best_split(X, y)
            if best_feature is None:
                return {'prediction': np.mean(y) > 0.5}
            
            left_mask = X[:, best_feature] <= best_threshold
            right_mask = ~left_mask
            
            return {
                'feature': best_feature,
                'threshold': best_threshold,
                'left': self._build_tree(X[left_mask], y[left_mask], depth + 1),
                'right': self._build_tree(X[right_mask], y[right_mask], depth + 1)
            }
        
        def _find_best_split(self, X, y):
            best_gini = float('inf')
            best_feature, best_threshold = None, None
            
            for feature in range(X.shape[1]):
                thresholds = np.unique(X[:, feature])
                for threshold in thresholds:
                    left_mask = X[:, feature] <= threshold
                    if np.sum(left_mask) == 0 or np.sum(left_mask) == len(y):
                        continue
                    
                    gini = self._calculate_gini(y[left_mask], y[~left_mask])
                    if gini < best_gini:
                        best_gini = gini
                        best_feature, best_threshold = feature, threshold
            
            return best_feature, best_threshold
        
        def _calculate_gini(self, left_y, right_y):
            def gini_impurity(y):
                if len(y) == 0:
                    return 0
                p = np.mean(y)
                return 2 * p * (1 - p)
            
            total = len(left_y) + len(right_y)
            return (len(left_y) / total) * gini_impurity(left_y) + \
                   (len(right_y) / total) * gini_impurity(right_y)
        
        def predict_single(self, x):
            node = self.tree
            while 'feature' in node:
                if x[node['feature']] <= node['threshold']:
                    node = node['left']
                else:
                    node = node['right']
            return 1 if node['prediction'] else 0
        
        def predict(self, X):
            return np.array([self.predict_single(x) for x in X])
    
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        
        self.trees = []
        feature_counts = np.zeros(X.shape[1])
        
        for _ in range(self.n_estimators):
            # Bootstrap sampling
            indices = np.random.choice(len(X), len(X), replace=True)
            X_boot, y_boot = X[indices], y[indices]
            
            # Train tree
            tree = self.DecisionTree(self.max_depth)
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)
            
            # Simple feature importance (count usage)
            for i in range(X.shape[1]):
                feature_counts[i] += 1
        
        self.feature_importances_ = feature_counts / np.sum(feature_counts)
        self.is_trained = True
    
    def predict(self, X):
        X = np.array(X)
        predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.round(np.mean(predictions, axis=0)).astype(int)
    
    def predict_proba(self, X):
        X = np.array(X)
        predictions = np.array([tree.predict(X) for tree in self.trees])
        prob_positive = np.mean(predictions, axis=0)
        return np.column_stack([1 - prob_positive, prob_positive])

class ToxicityPredictor:
    """Toxicity prediction with fallback implementations."""
    
    def __init__(self):
        if SKLEARN_AVAILABLE:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.preprocessing import StandardScaler
            self.model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
            self.scaler = StandardScaler()
        else:
            self.model = SimpleRandomForest(n_estimators=20, max_depth=5, random_state=42)
            self.scaler = None
        
        self.is_trained = False
        self.feature_names = None
    
    def standardize_features(self, X):
        """Simple standardization when sklearn is not available."""
        if self.scaler is not None:
            return self.scaler.fit_transform(X) if not self.is_trained else self.scaler.transform(X)
        else:
            # Simple standardization
            if not hasattr(self, 'feature_means'):
                self.feature_means = np.mean(X, axis=0)
                self.feature_stds = np.std(X, axis=0) + 1e-8  # Avoid division by zero
            
            return (X - self.feature_means) / self.feature_stds
    
    def train(self, X, y):
        """Train the model."""
        X = X.fillna(X.mean())
        X_array = np.array(X)
        y_array = np.array(y)
        
        # Simple train-test split
        n_train = int(0.8 * len(X_array))
        indices = np.random.permutation(len(X_array))
        train_idx, test_idx = indices[:n_train], indices[n_train:]
        
        X_train, X_test = X_array[train_idx], X_array[test_idx]
        y_train, y_test = y_array[train_idx], y_array[test_idx]
        
        # Standardize features
        X_train_scaled = self.standardize_features(X_train)
        X_test_scaled = self.standardize_features(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        self.feature_names = X.columns.tolist()
        self.is_trained = True
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        
        # Simple metrics calculation
        correct = np.sum(y_pred == y_test)
        total = len(y_test)
        accuracy = correct / total if total > 0 else 0
        
        # Basic precision/recall for binary classification
        tp = np.sum((y_pred == 1) & (y_test == 1))
        fp = np.sum((y_pred == 1) & (y_test == 0))
        fn = np.sum((y_pred == 0) & (y_test == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        return metrics, X_test_scaled, y_test
    
    def predict(self, X):
        """Predict toxicity."""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        X = X.fillna(X.mean()) if hasattr(X, 'fillna') else X
        X_scaled = self.standardize_features(np.array(X))
        
        prediction = self.model.predict(X_scaled)
        probability = self.model.predict_proba(X_scaled)
        
        return prediction, probability, X_scaled

class MolecularVisualizer:
    """Molecular visualization with fallbacks."""
    
    def draw_molecule_simple(self, smiles):
        """Create simple text representation when RDKit is not available."""
        try:
            # Create a simple molecular representation
            info = f"""
            SMILES: {smiles}
            
            Basic Structure Analysis:
            - Length: {len(smiles)} characters
            - Contains rings: {'Yes' if any(c.isdigit() for c in smiles) else 'No'}
            - Aromatic: {'Yes' if any(c.islower() for c in smiles) else 'No'}
            - Branched: {'Yes' if '(' in smiles else 'No'}
            """
            return info
        except:
            return f"SMILES: {smiles}"
    
    def draw_molecule(self, smiles):
        """Generate molecular visualization."""
        if RDKIT_AVAILABLE:
            return self.draw_molecule_rdkit(smiles)
        else:
            return self.draw_molecule_simple(smiles)
    
    def draw_molecule_rdkit(self, smiles):
        """Generate 2D molecular structure using RDKit."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            from rdkit.Chem import rdDepictor
            rdDepictor.Compute2DCoords(mol)
            
            drawer = rdMolDraw2D.MolDraw2DCairo(400, 400)
            drawer.DrawMolecule(mol)
            drawer.FinishDrawing()
            
            return drawer.GetDrawingText()
        except Exception as e:
            return self.draw_molecule_simple(smiles)
    
    def create_importance_chart(self, model, feature_names):
        """Create feature importance visualization."""
        try:
            importances = model.feature_importances_
            top_indices = np.argsort(importances)[-3:][::-1]
            top_features = [feature_names[i] for i in top_indices]
            top_values = importances[top_indices]
            
            return top_features, top_values
        except Exception as e:
            return None, None

def create_sample_data():
    """Create sample dataset."""
    sample_data = {
        'SMILES': [
            'CCO',  # Ethanol
            'CC(=O)OC1=CC=CC=C1C(=O)O',  # Aspirin
            'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O',  # Ibuprofen
            'C1=CC=C(C=C1)O',  # Phenol
            'CC1=CC=CC=C1',  # Toluene
            'C(C(=O)O)N',  # Glycine
            'CCCCCCCC(=O)O',  # Octanoic acid
            'C1=CC=CC=C1',  # Benzene
        ],
        'Toxicity': [1, 0, 0, 1, 1, 0, 0, 1]
    }
    return pd.DataFrame(sample_data)

def main():
    st.set_page_config(
        page_title="ToxAgents: Simplified Toxicity Prediction",
        page_icon="🧪",
        layout="wide"
    )
    
    st.title("🧪 ToxAgents: Simplified Toxicity Prediction")
    st.markdown("""
    This application predicts the toxicity of chemical compounds using molecular descriptors and machine learning.
    **This version works with minimal dependencies and provides fallback implementations.**
    """)
    
    # Dependency status
    st.sidebar.header("System Status")
    dependencies = {
        "🧬 RDKit": RDKIT_AVAILABLE,
        "🤖 Scikit-learn": SKLEARN_AVAILABLE,
        "📊 Matplotlib": MATPLOTLIB_AVAILABLE,
        "🔍 SHAP": SHAP_AVAILABLE
    }
    
    for lib, available in dependencies.items():
        if available:
            st.sidebar.success(f"{lib} ✅")
        else:
            st.sidebar.warning(f"{lib} ❌ (Using fallback)")
    
    # Show what's being used
    if not RDKIT_AVAILABLE:
        st.info("💡 Using basic SMILES parsing (install RDKit for accurate molecular descriptors)")
    if not SKLEARN_AVAILABLE:
        st.info("💡 Using simple Random Forest implementation (install scikit-learn for full features)")
    
    # Initialize components
    if 'extractor' not in st.session_state:
        st.session_state.extractor = SimpleMolecularDescriptorExtractor()
        st.session_state.predictor = ToxicityPredictor()
        st.session_state.visualizer = MolecularVisualizer()
    
    # Sidebar controls
    st.sidebar.header("Controls")
    
    # Model training
    st.sidebar.subheader("1. Train Model")
    if st.sidebar.button("🚀 Train with Sample Data"):
        with st.spinner("Training model..."):
            sample_df = create_sample_data()
            X = st.session_state.extractor.batch_extract_descriptors(sample_df['SMILES'])
            y = sample_df['Toxicity']
            
            try:
                metrics, X_test, y_test = st.session_state.predictor.train(X, y)
                st.sidebar.success("✅ Model trained!")
                st.sidebar.metric("Accuracy", f"{metrics['accuracy']:.3f}")
                st.sidebar.metric("F1-Score", f"{metrics['f1']:.3f}")
            except Exception as e:
                st.sidebar.error(f"Training failed: {e}")
    
    # Input section
    st.sidebar.subheader("2. Predict Toxicity")
    smiles_examples = [
        "CCO",  # Ethanol
        "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
        "C1=CC=C(C=C1)O",  # Phenol
        "CC1=CC=CC=C1"  # Toluene
    ]
    
    selected_example = st.sidebar.selectbox("Choose example:", 
                                          ["Custom"] + smiles_examples)
    
    if selected_example != "Custom":
        smiles_input = selected_example
    else:
        smiles_input = st.sidebar.text_input("Enter SMILES:", value="CCO")
    
    predict_button = st.sidebar.button("🔍 Predict Toxicity")
    
    # Main content
    if predict_button and smiles_input:
        if not st.session_state.predictor.is_trained:
            st.error("❌ Please train the model first!")
        else:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("🎯 Prediction Results")
                
                descriptors = st.session_state.extractor.smiles_to_descriptors(smiles_input)
                
                if descriptors is None:
                    st.error("❌ Could not process SMILES string")
                else:
                    X_new = pd.DataFrame([descriptors])
                    
                    try:
                        prediction, probability, X_scaled = st.session_state.predictor.predict(X_new)
                        
                        toxicity_label = "Toxic" if prediction[0] == 1 else "Non-Toxic"
                        confidence = max(probability[0]) * 100
                        
                        if prediction[0] == 1:
                            st.error(f"⚠️ **{toxicity_label}** ({confidence:.1f}% confidence)")
                        else:
                            st.success(f"✅ **{toxicity_label}** ({confidence:.1f}% confidence)")
                        
                        # Show descriptors
                        st.subheader("📊 Molecular Descriptors")
                        desc_df = pd.DataFrame(list(descriptors.items()), 
                                             columns=['Descriptor', 'Value'])
                        st.dataframe(desc_df, use_container_width=True)
                        
                        # Feature importance
                        st.subheader("📈 Feature Importance")
                        top_features, top_values = st.session_state.visualizer.create_importance_chart(
                            st.session_state.predictor.model,
                            st.session_state.predictor.feature_names
                        )
                        
                        if top_features is not None:
                            importance_df = pd.DataFrame({
                                'Feature': top_features,
                                'Importance': top_values
                            })
                            st.bar_chart(importance_df.set_index('Feature'))
                            st.dataframe(importance_df, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Prediction failed: {e}")
            
            with col2:
                st.subheader("🔬 Molecular Structure")
                mol_viz = st.session_state.visualizer.draw_molecule(smiles_input)
                
                if RDKIT_AVAILABLE and isinstance(mol_viz, bytes):
                    st.image(mol_viz, caption=f"SMILES: {smiles_input}")
                else:
                    st.text_area("Structure Info:", mol_viz, height=300)
    
    # Help section
    st.markdown("---")
    st.subheader("📋 Quick Start Guide")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **Step 1: Train Model**
        - Click "🚀 Train with Sample Data"
        - Wait for training completion
        - Check accuracy metrics
        """)
    
    with col2:
        st.markdown("""
        **Step 2: Enter SMILES**
        - Choose from examples or enter custom
        - Examples: CCO (ethanol), C1=CC=CC=C1 (benzene)
        - Click "🔍 Predict Toxicity"
        """)
    
    with col3:
        st.markdown("""
        **Step 3: Interpret Results**
        - Green = Non-toxic prediction
        - Red = Toxic prediction
        - Check confidence and descriptors
        """)

if __name__ == "__main__":
    main()
