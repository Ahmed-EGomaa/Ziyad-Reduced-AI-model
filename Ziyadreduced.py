import streamlit as st
import pandas as pd
import numpy as np
import tempfile
import os
import platform
from io import BytesIO
import base64

# Try to import matplotlib
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# Core ML libraries
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    st.error("Scikit-learn is not available. Please install scikit-learn to use this application.")

# RDKit for molecular descriptors and visualization
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    from rdkit.Chem.Draw import rdMolDraw2D
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')  # Disable RDKit warnings
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

# SHAP for feature importance
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

class MolecularDescriptorExtractor:
    """Extract molecular descriptors from SMILES strings using RDKit."""
    
    def __init__(self):
        self.descriptor_names = [
            'XLogP', 'TPSA', 'MolWt', 
            'HBD', 'HBA', 'RotatableBonds'
        ]
    
    def smiles_to_descriptors(self, smiles):
        """Convert SMILES to molecular descriptors."""
        if not RDKIT_AVAILABLE:
            return None
            
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
            return descriptors
        except Exception as e:
            st.error(f"Error processing SMILES: {e}")
            return None
    
    def batch_extract_descriptors(self, smiles_list):
        """Extract descriptors for a list of SMILES."""
        descriptors_list = []
        for smiles in smiles_list:
            desc = self.smiles_to_descriptors(smiles)
            if desc is not None:
                descriptors_list.append(desc)
            else:
                # Fill with NaN for invalid SMILES
                descriptors_list.append({name: np.nan for name in self.descriptor_names})
        
        return pd.DataFrame(descriptors_list)

class ToxicityPredictor:
    """Machine learning pipeline for toxicity prediction."""
    
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = None
        
    def train(self, X, y):
        """Train the Random Forest model."""
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        self.feature_names = X.columns.tolist()
        self.is_trained = True
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred)
        }
        
        return metrics, X_test_scaled, y_test
    
    def predict(self, X):
        """Predict toxicity for new samples."""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Predict
        prediction = self.model.predict(X_scaled)
        probability = self.model.predict_proba(X_scaled)
        
        return prediction, probability, X_scaled

class MolecularVisualizer:
    """Create molecular visualizations and SHAP plots."""
    
    def __init__(self):
        pass
    
    def draw_molecule(self, smiles):
        """Generate 2D molecular structure from SMILES."""
        if not RDKIT_AVAILABLE:
            return None
            
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            # Generate 2D coordinates
            from rdkit.Chem import rdDepictor
            rdDepictor.Compute2DCoords(mol)
            
            # Create drawer
            drawer = rdMolDraw2D.MolDraw2DCairo(400, 400)
            drawer.DrawMolecule(mol)
            drawer.FinishDrawing()
            
            # Get image data
            img_data = drawer.GetDrawingText()
            
            return img_data
        except Exception as e:
            st.error(f"Error drawing molecule: {e}")
            return None
    
    def create_shap_plot(self, model, X_sample, feature_names):
        """Create SHAP summary plot for top 3 features."""
        if not SHAP_AVAILABLE or not MATPLOTLIB_AVAILABLE:
            return None
            
        try:
            # Create explainer
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
            
            # Get feature importance (mean absolute SHAP values)
            if len(shap_values) == 2:  # Binary classification
                shap_vals = shap_values[1]  # Use positive class
            else:
                shap_vals = shap_values
            
            mean_shap = np.mean(np.abs(shap_vals), axis=0)
            
            # Get top 3 features
            top_indices = np.argsort(mean_shap)[-3:][::-1]
            top_features = [feature_names[i] for i in top_indices]
            top_values = mean_shap[top_indices]
            
            # Create plot
            fig, ax = plt.subplots(figsize=(8, 4))
            bars = ax.barh(top_features, top_values, color='skyblue')
            ax.set_xlabel('Mean |SHAP Value|')
            ax.set_title('Top 3 Most Important Molecular Descriptors')
            
            # Add value labels on bars
            for bar, value in zip(bars, top_values):
                width = bar.get_width()
                ax.text(width + 0.01 * max(top_values), bar.get_y() + bar.get_height()/2,
                       f'{value:.3f}', ha='left', va='center')
            
            plt.tight_layout()
            
            # Save to bytes
            img_buffer = BytesIO()
            plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
            img_buffer.seek(0)
            plt.close()
            
            return img_buffer
            
        except Exception as e:
            st.error(f"Error creating SHAP plot: {e}")
            return None
    
    def create_simple_importance_chart(self, model, feature_names):
        """Create simple feature importance chart without matplotlib."""
        try:
            # Get feature importance from Random Forest
            importances = model.feature_importances_
            
            # Get top 3 features
            top_indices = np.argsort(importances)[-3:][::-1]
            top_features = [feature_names[i] for i in top_indices]
            top_values = importances[top_indices]
            
            return top_features, top_values
            
        except Exception as e:
            st.error(f"Error getting feature importance: {e}")
            return None, None

def create_sample_data():
    """Create sample dataset for demonstration."""
    sample_data = {
        'SMILES': [
            'CC(=O)OC1=CC=CC=C1C(=O)O',  # Aspirin (non-toxic)
            'CCO',  # Ethanol (toxic)
            'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O',  # Ibuprofen (non-toxic)
            'CCCCCCCCCCCCCCCCCCCCCC(=O)O',  # Behenic acid (non-toxic)
            'C1=CC=C(C=C1)O',  # Phenol (toxic)
            'CC(=O)N(C1=CC=CC=C1)C2=CC=CC=C2',  # Acetanilide (toxic)
            'CC1=CC=CC=C1',  # Toluene (toxic)
            'C(C(=O)O)N',  # Glycine (non-toxic)
        ],
        'Toxicity': [0, 1, 0, 0, 1, 1, 1, 0]  # 0: non-toxic, 1: toxic
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
    Enter a SMILES string to get toxicity prediction, feature importance analysis, and molecular visualization.
    """)
    
    # Check dependencies and show status
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
            st.sidebar.error(f"{lib} ❌")
    
    # Check critical dependencies
    if not RDKIT_AVAILABLE:
        st.error("⚠️ RDKit is required for molecular descriptor computation. Please install RDKit.")
        st.code("pip install rdkit", language="bash")
        st.stop()
    
    if not SKLEARN_AVAILABLE:
        st.error("⚠️ Scikit-learn is required for machine learning. Please install scikit-learn.")
        st.code("pip install scikit-learn", language="bash")
        st.stop()
    
    # Show warnings for optional dependencies
    if not MATPLOTLIB_AVAILABLE and not SHAP_AVAILABLE:
        st.warning("⚠️ Matplotlib and SHAP are not available. Feature importance will be shown as text only.")
    elif not MATPLOTLIB_AVAILABLE:
        st.warning("⚠️ Matplotlib is not available. Charts will be shown as text only.")
    elif not SHAP_AVAILABLE:
        st.warning("⚠️ SHAP is not available. Basic feature importance will be used instead.")
    
    # Initialize components
    if 'extractor' not in st.session_state:
        st.session_state.extractor = MolecularDescriptorExtractor()
        st.session_state.predictor = ToxicityPredictor()
        st.session_state.visualizer = MolecularVisualizer()
    
    # Sidebar for input and model training
    st.sidebar.header("Input & Configuration")
    
    # Model training section
    st.sidebar.subheader("1. Train Model")
    if st.sidebar.button("Train with Sample Data"):
        with st.spinner("Training model..."):
            # Create sample data
            sample_df = create_sample_data()
            
            # Extract descriptors
            X = st.session_state.extractor.batch_extract_descriptors(sample_df['SMILES'])
            y = sample_df['Toxicity']
            
            # Train model
            try:
                metrics, X_test, y_test = st.session_state.predictor.train(X, y)
                st.sidebar.success("Model trained successfully!")
                st.sidebar.write(f"Accuracy: {metrics['accuracy']:.3f}")
                st.sidebar.write(f"F1-Score: {metrics['f1']:.3f}")
                
                # Store test data for SHAP
                st.session_state.X_test = X_test
                
            except Exception as e:
                st.sidebar.error(f"Training failed: {e}")
    
    # SMILES input section
    st.sidebar.subheader("2. Predict Toxicity")
    smiles_input = st.sidebar.text_input(
        "Enter SMILES string:",
        value="CC(=O)OC1=CC=CC=C1C(=O)O",
        help="Example: CC(=O)OC1=CC=CC=C1C(=O)O (Aspirin)"
    )
    
    predict_button = st.sidebar.button("🔍 Predict Toxicity")
    
    # Main panel
    if predict_button and smiles_input:
        if not st.session_state.predictor.is_trained:
            st.error("Please train the model first using the 'Train with Sample Data' button.")
        else:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("Prediction Results")
                
                # Extract descriptors
                descriptors = st.session_state.extractor.smiles_to_descriptors(smiles_input)
                
                if descriptors is None:
                    st.error("Invalid SMILES string. Please check your input.")
                else:
                    # Convert to DataFrame
                    X_new = pd.DataFrame([descriptors])
                    
                    # Make prediction
                    try:
                        prediction, probability, X_scaled = st.session_state.predictor.predict(X_new)
                        
                        # Display results
                        toxicity_label = "Toxic" if prediction[0] == 1 else "Non-Toxic"
                        confidence = max(probability[0]) * 100
                        
                        if prediction[0] == 1:
                            st.error(f"⚠️ Predicted: **{toxicity_label}** (Confidence: {confidence:.1f}%)")
                        else:
                            st.success(f"✅ Predicted: **{toxicity_label}** (Confidence: {confidence:.1f}%)")
                        
                        # Display descriptors
                        st.subheader("Molecular Descriptors")
                        desc_df = pd.DataFrame([descriptors]).T
                        desc_df.columns = ['Value']
                        st.dataframe(desc_df)
                        
                        # SHAP plot or feature importance
                        if SHAP_AVAILABLE and MATPLOTLIB_AVAILABLE and hasattr(st.session_state, 'X_test'):
                            st.subheader("Feature Importance (SHAP)")
                            shap_plot = st.session_state.visualizer.create_shap_plot(
                                st.session_state.predictor.model,
                                X_scaled,
                                st.session_state.predictor.feature_names
                            )
                            if shap_plot:
                                st.image(shap_plot, use_column_width=True)
                        else:
                            # Fallback: Simple feature importance
                            st.subheader("Feature Importance")
                            if st.session_state.predictor.is_trained:
                                top_features, top_values = st.session_state.visualizer.create_simple_importance_chart(
                                    st.session_state.predictor.model,
                                    st.session_state.predictor.feature_names
                                )
                                
                                if top_features is not None:
                                    importance_df = pd.DataFrame({
                                        'Feature': top_features,
                                        'Importance': top_values
                                    })
                                    
                                    # Display as bar chart using Streamlit
                                    st.bar_chart(importance_df.set_index('Feature'))
                                    
                                    # Also display as table
                                    st.dataframe(importance_df)
                        
                    except Exception as e:
                        st.error(f"Prediction failed: {e}")
            
            with col2:
                st.subheader("Molecular Structure")
                
                # Draw molecule
                mol_image = st.session_state.visualizer.draw_molecule(smiles_input)
                if mol_image:
                    # Convert bytes to image
                    st.image(mol_image, caption=f"SMILES: {smiles_input}", use_column_width=True)
                else:
                    st.error("Could not generate molecular structure.")
    
    # Information section
    st.markdown("---")
    st.subheader("About This System")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **Molecular Descriptors:**
        - XLogP: Lipophilicity
        - TPSA: Polar Surface Area
        - MolWt: Molecular Weight
        - HBD: H-bond Donors
        - HBA: H-bond Acceptors
        - Rotatable Bonds
        """)
    
    with col2:
        st.markdown("""
        **Machine Learning:**
        - Algorithm: Random Forest
        - Features: 6 molecular descriptors
        - Preprocessing: StandardScaler
        - Evaluation: 80-20 split
        """)
    
    with col3:
        st.markdown("""
        **Sample SMILES:**
        - Aspirin: CC(=O)OC1=CC=CC=C1C(=O)O
        - Ethanol: CCO
        - Caffeine: CN1C=NC2=C1C(=O)N(C(=O)N2C)C
        - Benzene: C1=CC=CC=C1
        """)

if __name__ == "__main__":
    main()
