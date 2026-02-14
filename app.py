import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef, confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import LabelEncoder

st.title("ML Model Classifier - BITS Assignment 2")

# a. Dataset upload option [cite: 91]
st.subheader("📥 Upload Dataset")

# Sample dataset download
try:
    sample_data = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')
    csv_bytes = sample_data.to_csv(index=False).encode()
    st.download_button(
        label="📊 Download Sample Dataset (HR Employee Attrition)",
        data=csv_bytes,
        file_name="HR_Employee_Attrition.csv",
        mime="text/csv",
        help="Download pre-loaded sample dataset for testing"
    )
except FileNotFoundError:
    st.info("Sample dataset not found in repo")

uploaded_file = st.file_uploader("Upload your Test CSV data", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Data Preview and Info
    st.subheader("📊 Dataset Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", len(df))
    col2.metric("Total Features", len(df.columns))
    col3.metric("Missing Values", df.isnull().sum().sum())
    
    st.write("**Data Preview:**", df.head())
    
    with st.expander("📈 Data Statistics"):
        st.write(df.describe())
    
    # Data Processing
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    # Encode categorical columns to numeric
    label_encoders = {}
    for column in X.columns:
        if X[column].dtype == 'object':
            le = LabelEncoder()
            X[column] = le.fit_transform(X[column].astype(str))
            label_encoders[column] = le
    
    # Encode target variable if categorical
    if y.dtype == 'object':
        le_target = LabelEncoder()
        y = le_target.fit_transform(y.astype(str))
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    st.divider()

    # Model dictionary
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "kNN": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
        "XGBoost": XGBClassifier(random_state=42, n_estimators=100, verbosity=0)
    }

   # Compare All Models Section
    st.subheader("🏆 Global Model Comparison (Mandatory Metrics)")
    
    if st.checkbox("Show Comparison of All 6 Models", value=False):
        comparison_data = []
        
        # We wrap this in a spinner because training 6 models takes a few seconds
        with st.spinner('Training all 6 models... please wait.'):
            for model_name, model in models.items():
                # 1. Train
                model.fit(X_train, y_train)
                y_pred_temp = model.predict(X_test)
                
                # 2. Probabilities for AUC
                y_prob_temp = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred_temp
                
                # 3. Calculate metrics (Crucial for Documentation)
                acc = accuracy_score(y_test, y_pred_temp)
                prec = precision_score(y_test, y_pred_temp, average='weighted', zero_division=0)
                rec = recall_score(y_test, y_pred_temp, average='weighted', zero_division=0)
                f1_temp = f1_score(y_test, y_pred_temp, average='weighted', zero_division=0)
                mcc_temp = matthews_corrcoef(y_test, y_pred_temp)
                
                try:
                    auc_val = roc_auc_score(y_test, y_prob_temp)
                except:
                    auc_val = 0.5 
                
                # 4. Append ALL 6 required parameters
                comparison_data.append({
                    'Model': model_name,
                    'Accuracy': round(acc, 4),
                    'MCC': round(mcc_temp, 4),      
                    'AUC': round(auc_val, 4),
                    'Precision': round(prec, 4),
                    'Recall': round(rec, 4),
                    'F1 Score': round(f1_temp, 4)
                })
        
        # Display the Table
        comparison_df = pd.DataFrame(comparison_data)
        st.table(comparison_df) 
        
        # Visual Output (Comparison Chart)
        st.write("**Visual Comparison: All Performance Metrics**")
        
        # Define the list of all metrics we want to see
        all_metrics = ['Accuracy', 'MCC', 'AUC', 'Precision', 'Recall', 'F1 Score']
        
        fig, ax = plt.subplots(figsize=(14, 7)) # Increased width for 6 metrics
        
        # Plotting all 6 metrics
        comparison_df.set_index('Model')[all_metrics].plot(kind='bar', ax=ax)
        
        ax.set_title("Comprehensive Model Comparison (All 6 Metrics)", fontsize=16)
        ax.set_ylabel("Score", fontsize=12)
        ax.set_ylim(0, 1.1) 
        
        
        ax.legend(title="Metrics", bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        
        st.pyplot(fig)
        # Markdown based code to see it in terminal
        print(comparison_df.to_markdown(index=False))
        st.success(f"Best Model by Accuracy: {comparison_df.loc[comparison_df['Accuracy'].idxmax(), 'Model']}")
    
    st.divider()

    # b. Model selection dropdown [cite: 92]
    st.subheader("🤖 Model Selection and Training")
    model_option = st.selectbox("Select Model for Detailed Analysis", 
        ["Logistic Regression", "Decision Tree", "kNN", "Naive Bayes", "Random Forest", "XGBoost"])

    model = models[model_option]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

    # c. Display of evaluation metrics [cite: 40, 93]
    st.subheader(f"📊 Evaluation Metrics for {model_option}")
    
    # Calculate all metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)
    
    # Display metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{accuracy:.4f}", delta=None)
    col2.metric("Precision", f"{precision:.4f}", delta=None)
    col3.metric("Recall", f"{recall:.4f}", delta=None)
    col4.metric("F1 Score", f"{f1:.4f}", delta=None)
    
    col5, col6 = st.columns(2)
    col5.metric("MCC Score", f"{mcc:.4f}", delta=None)
    
    # ROC-AUC Score (for binary classification)
    try:
        if len(np.unique(y_test)) == 2:
            roc_auc = roc_auc_score(y_test, y_prob)
            col6.metric("ROC-AUC Score", f"{roc_auc:.4f}", delta=None)
    except:
        st.write("*ROC-AUC not applicable for multi-class classification*")
    
    st.divider()

    # d. Confusion matrix with visualization [cite: 94]
    st.subheader("📈 Confusion Matrix and Classification Report")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Confusion Matrix:**")
        cm = confusion_matrix(y_test, y_pred)
        
        # Plot confusion matrix as heatmap
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar_kws={'label': 'Count'})
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title(f'Confusion Matrix - {model_option}')
        st.pyplot(fig)
    
    with col2:
        st.write("**Classification Report:**")
        st.text(classification_report(y_test, y_pred))
    
    st.divider()
    
    # ROC Curve (for binary classification)
    if len(np.unique(y_test)) == 2:
        st.subheader("📉 ROC Curve Analysis")
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curve - {model_option}')
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        
        # Precision-Recall Curve
        st.subheader("📊 Precision-Recall Curve")
        precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_prob)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(recall_vals, precision_vals, color='green', lw=2, label='Precision-Recall curve')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(f'Precision-Recall Curve - {model_option}')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.legend(loc="upper right")
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        
        st.divider()
    
    # Feature Importance (for tree-based models)
    if model_option in ["Decision Tree", "Random Forest", "XGBoost"]:
        st.subheader("🎯 Feature Importance Analysis")
        
        if hasattr(model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'Feature': X.columns,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(feature_importance['Feature'], feature_importance['Importance'], color='skyblue')
            ax.set_xlabel('Importance Score')
            ax.set_title(f'Feature Importance - {model_option}')
            ax.invert_yaxis()
            st.pyplot(fig)
            
            st.write("**Feature Importance Values:**")
            st.dataframe(feature_importance, use_container_width=True)
        
        st.divider()
    
    # Cross-Validation Scores
    st.subheader("✅ Cross-Validation Analysis")
    
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    
    col1, col2, col3 = st.columns(3)
    col1.metric("CV Mean Accuracy", f"{cv_scores.mean():.4f}")
    col2.metric("CV Std Dev", f"{cv_scores.std():.4f}")
    col3.metric("CV Min/Max", f"{cv_scores.min():.4f} / {cv_scores.max():.4f}")
    
    st.write("**Cross-Validation Scores (5-Fold):**")
    cv_df = pd.DataFrame({
        'Fold': [f'Fold {i+1}' for i in range(len(cv_scores))],
        'Accuracy': [f"{score:.4f}" for score in cv_scores]
    })
    st.dataframe(cv_df, use_container_width=True)
    
    st.divider()
    
    # Model Performance Summary and Recommendation
    st.subheader("📋 Model Summary & Recommendation")
    
    summary_text = f"""
    **Selected Model:** {model_option}
    
    **Performance Metrics:**
    - Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)
    - Precision: {precision:.4f}
    - Recall: {recall:.4f}
    - F1 Score: {f1:.4f}
    - MCC Score: {mcc:.4f}
    - Cross-Validation Mean: {cv_scores.mean():.4f}
    
    **Model Reliability:**
    """
    
    if cv_scores.std() < 0.05:
        summary_text += "✓ Very stable across different data splits (low variance)\n"
    elif cv_scores.std() < 0.10:
        summary_text += "✓ Reasonably stable across different data splits\n"
    else:
        summary_text += "⚠ Shows high variance - might be sensitive to data samples\n"
    
    if accuracy > 0.85:
        summary_text += "✓ Strong performance (Accuracy > 85%)\n"
    elif accuracy > 0.70:
        summary_text += "✓ Good performance (Accuracy > 70%)\n"
    else:
        summary_text += "⚠ Consider trying other models\n"
    
    st.info(summary_text)