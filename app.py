import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Import utility modules
from utils.data_preprocessing import load_and_preprocess_data, scale_features
from utils.model_training import train_models
from utils.model_evaluation import evaluate_models, plot_roc_curves, plot_confusion_matrices
from utils.visualization import create_eda_plots, plot_feature_importance, plot_shap_values

# Configure page
st.set_page_config(
    page_title="Breast Cancer Prediction System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    button[kind="header"] {display: none !important;}
    </style>
    """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Main title
st.title("🩺 Breast Cancer Prediction System")
st.markdown("### Machine Learning-Based Diagnostic Tool with Explainable AI")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Choose a page:",
    ["Overview", "Dataset Analysis", "Model Training", "Model Comparison", "Prediction Tool", "Model Interpretability"]
)

# Load data (cached for performance)
@st.cache_data
def load_data():
    return load_and_preprocess_data()

# Load models (cached for performance)
@st.cache_resource
def get_trained_models():
    X, y, feature_names, target_names = load_data()
    X_scaled = scale_features(X)
    models, X_train, X_test, y_train, y_test, scaler = train_models(X_scaled, y)
    return models, X_train, X_test, y_train, y_test, scaler, feature_names, target_names

try:
    X, y, feature_names, target_names = load_data()
    models, X_train, X_test, y_train, y_test, scaler, _, _ = get_trained_models()
    
    if page == "Overview":
        st.header("Project Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Objective")
            st.write("""
            This application demonstrates a comprehensive machine learning approach to breast cancer diagnosis 
            using the Wisconsin Breast Cancer Dataset. The system employs multiple ML algorithms to predict 
            whether a tumor is malignant or benign based on various cellular features.
            """)
            
            st.subheader("🔬 Key Features")
            st.write("""
            - **Multiple ML Models**: Random Forest, SVM, XGBoost, Logistic Regression
            - **Model Comparison**: Performance metrics and visualizations
            - **Explainable AI**: SHAP values for model interpretability
            - **Real-time Predictions**: Interactive tool for new diagnoses
            - **Comprehensive EDA**: Statistical analysis and visualizations
            """)
        
        with col2:
            st.subheader("📊 Dataset Statistics")
            st.metric("Total Samples", X.shape[0])
            st.metric("Features", X.shape[1])
            st.metric("Malignant Cases", sum(y))
            st.metric("Benign Cases", len(y) - sum(y))
            
            # Class distribution pie chart
            fig = px.pie(
                values=[len(y) - sum(y), sum(y)],
                names=['Benign', 'Malignant'],
                title="Class Distribution",
                color_discrete_map={'Benign': '#2E8B57', 'Malignant': '#DC143C'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    elif page == "Dataset Analysis":
        st.header("📈 Exploratory Data Analysis")
        
        # Dataset info
        st.subheader("Dataset Information")
        df = pd.DataFrame(X, columns=feature_names)
        df['target'] = y
        df['diagnosis'] = df['target'].map({0: 'Benign', 1: 'Malignant'})
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Dataset Shape:**", df.shape)
            st.write("**Missing Values:**", df.isnull().sum().sum())
        
        with col2:
            st.write("**Feature Types:**")
            st.write(f"- Numerical: {len(feature_names)}")
            st.write(f"- Categorical: 1 (target)")
        
        # Feature statistics
        st.subheader("Feature Statistics")
        st.dataframe(df.describe())
        
        # Correlation heatmap
        st.subheader("Feature Correlation Matrix")
        correlation_matrix = df[feature_names].corr()
        fig, ax = plt.subplots(figsize=(20, 16))
        sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0, ax=ax)
        st.pyplot(fig)
        plt.close()
        
        # Feature distributions
        st.subheader("Feature Distributions by Diagnosis")
        
        # Select features to display
        selected_features = st.multiselect(
            "Select features to visualize:",
            feature_names,
            default=feature_names[:4]
        )
        
        if selected_features:
            for feature in selected_features:
                fig = px.histogram(
                    df, x=feature, color='diagnosis',
                    marginal='box',
                    title=f'Distribution of {feature}',
                    color_discrete_map={'Benign': '#2E8B57', 'Malignant': '#DC143C'}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Pairwise relationships
        st.subheader("Pairwise Feature Relationships")
        if len(selected_features) >= 2:
            fig = px.scatter_matrix(
                df, dimensions=selected_features[:4], color='diagnosis',
                title="Pairwise Feature Relationships",
                color_discrete_map={'Benign': '#2E8B57', 'Malignant': '#DC143C'}
            )
            fig.update_layout(height=800)
            st.plotly_chart(fig, use_container_width=True)
    
    elif page == "Model Training":
        st.header("🤖 Model Training Results")
        
        # Training information
        st.subheader("Training Configuration")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Training Samples", X_train.shape[0])
        with col2:
            st.metric("Testing Samples", X_test.shape[0])
        with col3:
            st.metric("Features Used", X_train.shape[1])
        
        # Model architectures
        st.subheader("Model Architectures")
        
        model_info = {
            "Random Forest": {
                "Type": "Ensemble Method",
                "Key Parameters": "n_estimators=100, max_depth=10",
                "Strengths": "Handles overfitting well, feature importance"
            },
            "SVM": {
                "Type": "Support Vector Machine",
                "Key Parameters": "kernel=rbf, C=1.0, gamma=scale",
                "Strengths": "Effective in high dimensions, memory efficient"
            },
            "XGBoost": {
                "Type": "Gradient Boosting",
                "Key Parameters": "n_estimators=100, max_depth=6, learning_rate=0.1",
                "Strengths": "High performance, handles missing values"
            },
            "Logistic Regression": {
                "Type": "Linear Model",
                "Key Parameters": "C=1.0, max_iter=1000",
                "Strengths": "Interpretable, probabilistic output"
            }
        }
        
        for model_name, info in model_info.items():
            with st.expander(f"{model_name} Details"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Type:** {info['Type']}")
                    st.write(f"**Parameters:** {info['Key Parameters']}")
                with col2:
                    st.write(f"**Strengths:** {info['Strengths']}")
        
        # Feature importance for tree-based models
        st.subheader("Feature Importance Analysis")
        
        # Random Forest feature importance
        if 'Random Forest' in models:
            st.write("**Random Forest Feature Importance**")
            rf_model = models['Random Forest']
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            fig = px.bar(
                feature_importance.head(10),
                x='importance',
                y='feature',
                orientation='h',
                title='Top 10 Most Important Features (Random Forest)'
            )
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
        
        # XGBoost feature importance
        if 'XGBoost' in models:
            st.write("**XGBoost Feature Importance**")
            xgb_model = models['XGBoost']
            feature_importance_xgb = pd.DataFrame({
                'feature': feature_names,
                'importance': xgb_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            fig = px.bar(
                feature_importance_xgb.head(10),
                x='importance',
                y='feature',
                orientation='h',
                title='Top 10 Most Important Features (XGBoost)'
            )
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
    
    elif page == "Model Comparison":
        st.header("📊 Model Performance Comparison")
        
        # Evaluate models
        results = evaluate_models(models, X_test, y_test)
        
        # Performance metrics table
        st.subheader("Performance Metrics")
        metrics_df = pd.DataFrame(results).T
        st.dataframe(metrics_df.style.highlight_max(axis=0))
        
        # Best model identification
        best_model_name = metrics_df['Accuracy'].idxmax()
        st.success(f"🏆 Best Performing Model: **{best_model_name}** (Accuracy: {metrics_df.loc[best_model_name, 'Accuracy']:.4f})")
        
        # Performance comparison charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Accuracy comparison
            fig = px.bar(
                x=list(results.keys()),
                y=[results[model]['Accuracy'] for model in results.keys()],
                title='Model Accuracy Comparison',
                labels={'x': 'Models', 'y': 'Accuracy'},
                color=[results[model]['Accuracy'] for model in results.keys()],
                color_continuous_scale='viridis'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # F1-Score comparison
            fig = px.bar(
                x=list(results.keys()),
                y=[results[model]['F1-Score'] for model in results.keys()],
                title='Model F1-Score Comparison',
                labels={'x': 'Models', 'y': 'F1-Score'},
                color=[results[model]['F1-Score'] for model in results.keys()],
                color_continuous_scale='plasma'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # ROC Curves
        st.subheader("ROC Curves")
        roc_fig = plot_roc_curves(models, X_test, y_test)
        st.pyplot(roc_fig)
        plt.close()
        
        # Confusion Matrices
        st.subheader("Confusion Matrices")
        conf_fig = plot_confusion_matrices(models, X_test, y_test, target_names)
        st.pyplot(conf_fig)
        plt.close()
        
        # Detailed metrics
        st.subheader("Detailed Performance Analysis")
        
        for model_name in results.keys():
            with st.expander(f"{model_name} Detailed Metrics"):
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Accuracy", f"{results[model_name]['Accuracy']:.4f}")
                with col2:
                    st.metric("Precision", f"{results[model_name]['Precision']:.4f}")
                with col3:
                    st.metric("Recall", f"{results[model_name]['Recall']:.4f}")
                with col4:
                    st.metric("ROC-AUC", f"{results[model_name]['ROC-AUC']:.4f}")
    
    elif page == "Prediction Tool":
        st.header("🔮 Real-time Breast Cancer Prediction")
        
        st.write("Enter patient data to get a prediction:")
        
        # Create input form
        with st.form("prediction_form"):
            st.subheader("Patient Information")
            
            # Create input fields for all features
            input_data = {}
            
            # Group features by category for better organization
            feature_groups = {
                'Mean Values': [f for f in feature_names if 'mean' in f.lower()],
                'Standard Error Values': [f for f in feature_names if 'error' in f.lower()],
                'Worst Values': [f for f in feature_names if 'worst' in f.lower()]
            }
            
            # If no clear grouping, create columns
            if not any(feature_groups.values()):
                cols = st.columns(3)
                for i, feature in enumerate(feature_names):
                    with cols[i % 3]:
                        # Get reasonable default and range based on training data
                        feature_idx = feature_names.index(feature)
                        min_val = float(X[:, feature_idx].min())
                        max_val = float(X[:, feature_idx].max())
                        mean_val = float(X[:, feature_idx].mean())
                        
                        input_data[feature] = st.number_input(
                            feature.replace('_', ' ').title(),
                            min_value=min_val,
                            max_value=max_val,
                            value=mean_val,
                            help=f"Range: {min_val:.2f} - {max_val:.2f}"
                        )
            else:
                # Use feature groups
                for group_name, group_features in feature_groups.items():
                    if group_features:
                        st.write(f"**{group_name}**")
                        cols = st.columns(min(3, len(group_features)))
                        for i, feature in enumerate(group_features):
                            with cols[i % len(cols)]:
                                feature_idx = feature_names.index(feature)
                                min_val = float(X[:, feature_idx].min())
                                max_val = float(X[:, feature_idx].max())
                                mean_val = float(X[:, feature_idx].mean())
                                
                                input_data[feature] = st.number_input(
                                    feature.replace('_', ' ').title(),
                                    min_value=min_val,
                                    max_value=max_val,
                                    value=mean_val,
                                    help=f"Range: {min_val:.2f} - {max_val:.2f}",
                                    key=feature
                                )
            
            # Model selection
            selected_model = st.selectbox("Choose Model:", list(models.keys()))
            
            submitted = st.form_submit_button("Predict")
            
            if submitted:
                # Prepare input array
                input_array = np.array([[input_data[feature] for feature in feature_names]])
                input_scaled = scaler.transform(input_array)
                
                # Make prediction
                model = models[selected_model]
                prediction = model.predict(input_scaled)[0]
                prediction_proba = model.predict_proba(input_scaled)[0]
                
                # Display results
                st.subheader("Prediction Results")
                
                col1, col2 = st.columns(2)
                with col1:
                    if prediction == 0:
                        st.success("🟢 **Prediction: BENIGN**")
                        st.write(f"Confidence: {prediction_proba[0]:.2%}")
                    else:
                        st.error("🔴 **Prediction: MALIGNANT**")
                        st.write(f"Confidence: {prediction_proba[1]:.2%}")
                
                with col2:
                    # Confidence chart
                    fig = px.bar(
                        x=['Benign', 'Malignant'],
                        y=prediction_proba,
                        title='Prediction Confidence',
                        color=['Benign', 'Malignant'],
                        color_discrete_map={'Benign': '#2E8B57', 'Malignant': '#DC143C'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Model explanation (if available)
                st.subheader("Prediction Explanation")
                st.write(f"This prediction was made using the **{selected_model}** model.")
                
                if selected_model in ['Random Forest', 'XGBoost']:
                    st.write("The model considered the following features as most important for this prediction:")
                    
                    # Get feature importance
                    importance = model.feature_importances_
                    feature_importance_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': importance,
                        'Input_Value': [input_data[f] for f in feature_names]
                    }).sort_values('Importance', ascending=False).head(5)
                    
                    st.dataframe(feature_importance_df)
    
    elif page == "Model Interpretability":
        st.header("🔍 Model Interpretability & Explainability")
        
        st.write("Understanding how models make predictions is crucial for medical applications. This section provides insights into model decision-making processes.")
        
        # Select model for interpretation
        selected_model_name = st.selectbox("Select Model for Interpretation:", list(models.keys()))
        
        if selected_model_name:
            model = models[selected_model_name]
            
            # Try SHAP analysis first
            shap_available = False
            try:
                import shap
                shap_available = True
                
                # Create SHAP explainer
                with st.spinner("Calculating SHAP values..."):
                    if selected_model_name in ['XGBoost', 'Random Forest']:
                        # Tree-based models
                        explainer = shap.TreeExplainer(model)
                        shap_values = explainer.shap_values(X_test[:50])
                        # For binary classification, take the positive class values
                        if isinstance(shap_values, list) and len(shap_values) == 2:
                            shap_values = shap_values[1]
                        elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
                            # Handle case where shape is (n_samples, n_features, n_classes)
                            shap_values = shap_values[:, :, 1]  # Take positive class
                        # Ensure 2D array structure
                        if len(shap_values.shape) == 1:
                            shap_values = shap_values.reshape(1, -1)
                    elif selected_model_name == 'Logistic Regression':
                        # Linear model
                        explainer = shap.LinearExplainer(model, X_train[:50])
                        shap_values = explainer.shap_values(X_test[:50])
                        # Ensure 2D array structure
                        if len(shap_values.shape) == 1:
                            shap_values = shap_values.reshape(1, -1)
                    elif selected_model_name == 'SVM':
                        # Use Permutation explainer for SVM (faster than Kernel)
                        from shap import Explainer
                        def predict_fn(X):
                            return model.predict_proba(X)[:, 1]  # Return positive class probabilities
                        
                        explainer = shap.PermutationExplainer(predict_fn, X_train[:20])
                        shap_values = explainer(X_test[:20])
                        # Convert to numpy array if needed
                        if hasattr(shap_values, 'values'):
                            shap_values = shap_values.values
                        # Ensure 2D array structure
                        if len(shap_values.shape) == 1:
                            shap_values = shap_values.reshape(1, -1)
                    else:
                        # Fallback to Explainer (SHAP v0.40+)
                        explainer = shap.Explainer(model, X_train[:50])
                        shap_values = explainer(X_test[:50])
                        if hasattr(shap_values, 'values'):
                            shap_values = shap_values.values
                
                st.success("SHAP analysis completed!")
                
                # Debug info for troubleshooting
                with st.expander("Debug Information"):
                    st.write(f"Model: {selected_model_name}")
                    st.write(f"SHAP values shape: {shap_values.shape}")
                    st.write(f"SHAP values type: {type(shap_values)}")
                    st.write(f"X_sample shape: {X_sample.shape if 'X_sample' in locals() else 'Not defined yet'}")
                
                # Determine sample size and data for plotting
                if selected_model_name == 'SVM':
                    X_sample = X_test[:20]
                    sample_size = 20
                    max_sample_idx = min(19, len(X_test)-1)
                else:
                    X_sample = X_test[:50]
                    sample_size = 50
                    max_sample_idx = min(49, len(X_test)-1)
                
                # Summary plot
                st.subheader("🎯 Feature Impact Summary")
                fig_summary = plt.figure(figsize=(12, 8))
                shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
                st.pyplot(fig_summary)
                plt.close()
                
                # Feature importance plot
                st.subheader("📊 Mean Feature Importance")
                fig_importance = plt.figure(figsize=(10, 8))
                shap.summary_plot(shap_values, X_sample, feature_names=feature_names, plot_type="bar", show=False)
                st.pyplot(fig_importance)
                plt.close()
                
                # Individual prediction explanation
                st.subheader("🔬 Individual Prediction Explanation")
                sample_idx = st.slider("Select sample to explain:", 0, max_sample_idx, 0)
                
                # Show actual vs predicted
                actual = y_test[sample_idx]
                predicted = model.predict(X_test[sample_idx:sample_idx+1])[0]
                prediction_proba = model.predict_proba(X_test[sample_idx:sample_idx+1])[0]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Actual Diagnosis", target_names[actual])
                with col2:
                    st.metric("Predicted Diagnosis", target_names[predicted])
                with col3:
                    st.metric("Confidence", f"{max(prediction_proba):.1%}")
                
                # Force plot for individual explanation
                st.subheader("📈 Individual Feature Contributions")
                try:
                    # Show top contributing features for this prediction as alternative to force plot
                    st.write("**Top Contributing Features for this Prediction:**")
                    
                    # Ensure we have the right sample index
                    if sample_idx < len(shap_values):
                        feature_contributions = pd.DataFrame({
                            'Feature': feature_names,
                            'SHAP_Value': shap_values[sample_idx],
                            'Feature_Value': X_sample[sample_idx]
                        })
                        feature_contributions['Abs_SHAP'] = np.abs(feature_contributions['SHAP_Value'])
                        feature_contributions['Impact'] = feature_contributions['SHAP_Value'].apply(
                            lambda x: 'Increases Risk' if x > 0 else 'Decreases Risk'
                        )
                        
                        # Sort by absolute SHAP value
                        top_contributors = feature_contributions.nlargest(10, 'Abs_SHAP')
                        
                        # Display in a nice format
                        st.dataframe(
                            top_contributors[['Feature', 'SHAP_Value', 'Feature_Value', 'Impact']],
                            use_container_width=True
                        )
                        
                        # Create a simple bar chart for visualization
                        import plotly.express as px
                        fig_contrib = px.bar(
                            top_contributors.head(8), 
                            x='SHAP_Value', 
                            y='Feature',
                            orientation='h',
                            title='Top 8 Feature Contributions (SHAP Values)',
                            color='SHAP_Value',
                            color_continuous_scale='RdYlBu_r'
                        )
                        fig_contrib.update_layout(height=400)
                        st.plotly_chart(fig_contrib, use_container_width=True)
                        
                    else:
                        st.error("Sample index out of range for SHAP values")
                        
                except Exception as e:
                    st.error(f"Error creating individual explanation: {str(e)}")
                    st.write("Debug info:")
                    st.write(f"SHAP values shape: {shap_values.shape}")
                    st.write(f"Sample index: {sample_idx}")
                    st.write(f"X_sample shape: {X_sample.shape}")
                
            except ImportError:
                st.warning("SHAP library not installed. Showing alternative interpretability features.")
                shap_available = False
            except Exception as e:
                st.warning(f"SHAP analysis failed: {str(e)}. Showing alternative interpretability features.")
                shap_available = False
            
            # Alternative interpretability features (always available)
            if not shap_available:
                st.subheader("📊 Feature Importance Analysis")
                
                if hasattr(model, 'feature_importances_'):
                    # Tree-based model feature importance
                    importance_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': model.feature_importances_
                    }).sort_values('Importance', ascending=False)
                    
                    fig = px.bar(
                        importance_df.head(15),
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        title=f'Feature Importance - {selected_model_name}',
                        color='Importance',
                        color_continuous_scale='viridis'
                    )
                    fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=600)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show top contributing features
                    st.subheader("🏆 Top Contributing Features")
                    top_features_df = importance_df.head(10)
                    st.dataframe(top_features_df, use_container_width=True)
                    
                elif hasattr(model, 'coef_'):
                    # Linear model coefficients
                    coef_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Coefficient': model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_,
                        'Abs_Coefficient': np.abs(model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_)
                    }).sort_values('Abs_Coefficient', ascending=False)
                    
                    fig = px.bar(
                        coef_df.head(15),
                        x='Coefficient',
                        y='Feature',
                        orientation='h',
                        title=f'Feature Coefficients - {selected_model_name}',
                        color='Coefficient',
                        color_continuous_scale='RdBu'
                    )
                    fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=600)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show feature coefficients
                    st.subheader("📈 Feature Coefficients")
                    st.dataframe(coef_df.head(10), use_container_width=True)
            
            # Model-specific insights
            st.subheader("🧠 Model-Specific Insights")
            
            if selected_model_name == 'Random Forest':
                st.write("""
                **Random Forest Interpretability:**
                - Feature importance is calculated based on how much each feature decreases impurity when used for splits
                - Higher values indicate more important features for making predictions
                - The model uses ensemble voting from multiple decision trees
                """)
                
            elif selected_model_name == 'SVM':
                st.write("""
                **Support Vector Machine Interpretability:**
                - SVM finds the optimal decision boundary between classes
                - Features with higher coefficient magnitudes have more influence on the decision
                - The model works by finding support vectors that define the decision boundary
                """)
                
            elif selected_model_name == 'XGBoost':
                st.write("""
                **XGBoost Interpretability:**
                - Feature importance shows how useful each feature is for making predictions
                - Based on gradient boosting, where each tree corrects errors from previous trees
                - Higher importance values indicate features that contribute more to reducing prediction error
                """)
                
            elif selected_model_name == 'Logistic Regression':
                st.write("""
                **Logistic Regression Interpretability:**
                - Coefficients represent the change in log-odds for a unit change in the feature
                - Positive coefficients increase the probability of malignant diagnosis
                - Negative coefficients decrease the probability of malignant diagnosis
                """)
            
            # Performance insights
            st.subheader("🎯 Model Performance Context")
            results = evaluate_models({selected_model_name: model}, X_test, y_test)
            model_metrics = results[selected_model_name]
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Accuracy", f"{model_metrics['Accuracy']:.3f}")
            with col2:
                st.metric("Precision", f"{model_metrics['Precision']:.3f}")
            with col3:
                st.metric("Recall", f"{model_metrics['Recall']:.3f}")
            with col4:
                st.metric("F1-Score", f"{model_metrics['F1-Score']:.3f}")
            
            # Clinical interpretation
            st.subheader("🏥 Clinical Interpretation Guidelines")
            st.info("""
            **For Medical Use:**
            - High Precision: Fewer false positives (fewer healthy patients incorrectly diagnosed as having cancer)
            - High Recall: Fewer false negatives (fewer cancer cases missed)
            - Feature importance helps identify which cellular characteristics are most indicative of malignancy
            - Always use AI as a diagnostic aid, not a replacement for professional medical judgment
            """)

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.write("Please check that all required libraries are installed and try again.")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.markdown("This application was developed as a comprehensive machine learning tool for breast cancer diagnosis using the Wisconsin Breast Cancer Dataset.")
st.sidebar.markdown("**Technologies Used:**")
st.sidebar.markdown("- Streamlit")
st.sidebar.markdown("- Scikit-learn")
st.sidebar.markdown("- XGBoost")
st.sidebar.markdown("- SHAP")
st.sidebar.markdown("- Plotly")
