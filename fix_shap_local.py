#!/usr/bin/env python3
"""
Quick fix script for SHAP implementation in app.py
Run this script to update your local app.py with the corrected SHAP code
"""

import os
import shutil

def update_shap_implementation():
    """Update the SHAP implementation in app.py"""
    
    # Read the current app.py
    with open('app.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Define the old SHAP explainer code
    old_shap_code = '''                # Create SHAP explainer
                with st.spinner("Calculating SHAP values..."):
                    if selected_model_name == 'XGBoost':
                        explainer = shap.TreeExplainer(model)
                        shap_values = explainer.shap_values(X_test[:100])
                    elif selected_model_name == 'Random Forest':
                        explainer = shap.TreeExplainer(model)
                        shap_values = explainer.shap_values(X_test[:100])
                    else:
                        # For SVM and Logistic Regression
                        explainer = shap.LinearExplainer(model, X_train[:100])
                        shap_values = explainer.shap_values(X_test[:100])'''
    
    # Define the new SHAP explainer code
    new_shap_code = '''                # Create SHAP explainer
                with st.spinner("Calculating SHAP values..."):
                    if selected_model_name in ['XGBoost', 'Random Forest']:
                        # Tree-based models
                        explainer = shap.TreeExplainer(model)
                        shap_values = explainer.shap_values(X_test[:100])
                        # For binary classification, take the positive class values
                        if isinstance(shap_values, list) and len(shap_values) == 2:
                            shap_values = shap_values[1]
                    elif selected_model_name == 'Logistic Regression':
                        # Linear model
                        explainer = shap.LinearExplainer(model, X_train[:100])
                        shap_values = explainer.shap_values(X_test[:100])
                    elif selected_model_name == 'SVM':
                        # Use KernelExplainer for SVM (more general but slower)
                        explainer = shap.KernelExplainer(model.predict_proba, X_train[:50])
                        shap_values = explainer.shap_values(X_test[:50])
                        # Take positive class for binary classification
                        if isinstance(shap_values, list) and len(shap_values) == 2:
                            shap_values = shap_values[1]
                    else:
                        # Fallback to Explainer (SHAP v0.40+)
                        explainer = shap.Explainer(model, X_train[:100])
                        shap_values = explainer(X_test[:100])
                        shap_values = shap_values.values'''
    
    # Update the content
    if old_shap_code in content:
        content = content.replace(old_shap_code, new_shap_code)
        print("✓ Updated SHAP explainer implementation")
    else:
        print("! Could not find old SHAP code to replace")
    
    # Also update the plotting section
    old_plot_code = '''                # Summary plot
                st.subheader("🎯 Feature Impact Summary")
                fig_summary = plt.figure(figsize=(12, 8))
                shap.summary_plot(shap_values, X_test[:100], feature_names=feature_names, show=False)
                st.pyplot(fig_summary)
                plt.close()'''
    
    new_plot_code = '''                # Determine sample size and data for plotting
                if selected_model_name == 'SVM':
                    X_sample = X_test[:50]
                    sample_size = 50
                    max_sample_idx = min(49, len(X_test)-1)
                else:
                    X_sample = X_test[:100]
                    sample_size = 100
                    max_sample_idx = min(99, len(X_test)-1)
                
                # Summary plot
                st.subheader("🎯 Feature Impact Summary")
                fig_summary = plt.figure(figsize=(12, 8))
                shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
                st.pyplot(fig_summary)
                plt.close()'''
    
    if old_plot_code in content:
        content = content.replace(old_plot_code, new_plot_code)
        print("✓ Updated SHAP plotting section")
    
    # Backup the original file
    shutil.copy('app.py', 'app_backup.py')
    print("✓ Created backup: app_backup.py")
    
    # Write the updated content
    with open('app.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✓ Updated app.py with corrected SHAP implementation")
    print("\nThe fix addresses the SVM SHAP error by:")
    print("- Using KernelExplainer for SVM models")
    print("- Handling binary classification outputs correctly")
    print("- Using appropriate sample sizes for each model type")
    print("\nRestart your Streamlit app to see the changes:")
    print("streamlit run app.py --server.address localhost --server.port 8501")

if __name__ == "__main__":
    if os.path.exists('app.py'):
        update_shap_implementation()
    else:
        print("Error: app.py not found in current directory")
        print("Make sure you're in the correct project folder")