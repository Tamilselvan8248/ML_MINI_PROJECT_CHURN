import pandas as pd
import numpy as np
import os
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import plot_tree
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, roc_curve, auc)


# ─────────────────────────────────────────────
# 1. PREPROCESSING
# ─────────────────────────────────────────────
def preprocess(df: pd.DataFrame):
    df = df.copy()

    # Drop duplicates
    df.drop_duplicates(inplace=True)

    # Drop rows where target is missing
    if 'Churn' in df.columns:
        df.dropna(subset=['Churn'], inplace=True)

    # Fill numeric NaN with median
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in num_cols:
        df[col].fillna(df[col].median(), inplace=True)

    # Fill categorical NaN with mode
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    for col in cat_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)

    # Encode target
    target_col = None
    for col in df.columns:
        if col.lower() == 'churn':
            target_col = col
            break
    if target_col is None:
        raise ValueError("Dataset must have a 'Churn' column.")

    le_target = LabelEncoder()
    df[target_col] = le_target.fit_transform(df[target_col].astype(str).str.strip())

    # Encode remaining categoricals
    le_dict = {}
    cat_cols_remaining = df.select_dtypes(include=['object']).columns.tolist()
    for col in cat_cols_remaining:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le

    # Separate customer IDs if present
    id_col = None
    for col in df.columns:
        if col.lower() in ('customerid', 'customer_id', 'id'):
            id_col = col
            break

    customer_ids = df[id_col].values.tolist() if id_col else list(range(len(df)))

    # Feature / target split
    drop_cols = [target_col]
    if id_col:
        drop_cols.append(id_col)
    X = df.drop(columns=drop_cols)
    y = df[target_col]

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y.values, scaler, le_dict, customer_ids, X.columns.tolist()


# ─────────────────────────────────────────────
# 2. TRAIN ALL MODELS
# ─────────────────────────────────────────────
def train_all_models(X_train, y_train):
    knn = KNeighborsClassifier(n_neighbors=5)
    lr = LogisticRegression(max_iter=1000, random_state=42)
    dt = DecisionTreeClassifier(random_state=42)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    
    ensemble = VotingClassifier(estimators=[
        ('knn', knn), ('lr', lr), ('dt', dt), ('rf', rf)
    ], voting='soft')

    models = {
        'K-Nearest Neighbors': knn,
        'Logistic Regression': lr,
        'Decision Tree':       dt,
        'Random Forest':       rf,
        'Voting Ensemble':     ensemble
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
    return models


# ─────────────────────────────────────────────
# 3. EVALUATE MODELS
# ─────────────────────────────────────────────
def evaluate_models(models, X_test, y_test):
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        cm  = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        try:
            y_prob = model.predict_proba(X_test)[:, 1]
        except AttributeError:
            y_prob = y_pred.astype(float)
            
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        results[name] = {
            'accuracy':  round(acc * 100, 2),
            'cm':        cm,
            'report':    report,
            'model':     model,
            'fpr':       fpr,
            'tpr':       tpr,
            'roc_auc':   roc_auc
        }
    return results


# ─────────────────────────────────────────────
# 4. PREDICT CHURN PER CUSTOMER
# ─────────────────────────────────────────────
def predict_churn(model, X_scaled, customer_ids, feature_names):
    preds = model.predict(X_scaled)
    # Not all models support predict_proba
    try:
        probs = model.predict_proba(X_scaled)[:, 1]
    except AttributeError:
        probs = preds.astype(float)

    rows = []
    for i, cid in enumerate(customer_ids):
        prob = round(float(probs[i]) * 100, 1)
        churn_label = 'Yes' if preds[i] == 1 else 'No'
        if prob >= 70:
            risk = 'High'
        elif prob >= 40:
            risk = 'Medium'
        else:
            risk = 'Low'
        rows.append({
            'CustomerID':  cid,
            'Churn':       churn_label,
            'Probability': prob,
            'RiskLevel':   risk,
        })
    return rows


# ─────────────────────────────────────────────
# 5. RETENTION SUGGESTIONS
# ─────────────────────────────────────────────
def get_suggestions(churn_rate: float):
    suggestions = []
    if churn_rate > 40:
        suggestions.append("Offer personalised loyalty discounts to high-risk customers.")
        suggestions.append("Introduce a VIP rewards programme to increase engagement.")
    if churn_rate > 20:
        suggestions.append("Conduct regular customer satisfaction surveys.")
        suggestions.append("Provide dedicated customer support for long-tenure customers.")
    suggestions.append("Send proactive renewal reminders before contract expiry.")
    suggestions.append("Bundle additional services (Tech Support, Streaming) at reduced rates.")
    suggestions.append("Improve onboarding experience for new customers to boost retention.")
    suggestions.append("Offer flexible payment options to reduce payment-related churn.")
    return suggestions


# ─────────────────────────────────────────────
# 6. TREE VISUALIZATION
# ─────────────────────────────────────────────
def get_tree_base64(model, feature_names):
    # Dynamically adjust figsize based on depth to make it readable
    depth = model.max_depth if model.max_depth else 5
    width = max(10, depth * 3)
    fig, ax = plt.subplots(figsize=(width, 8))
    
    # Let plot_tree render the actual depth of the model (up to a reasonable visual limit like 5 if None)
    plot_depth = model.max_depth if model.max_depth else 5
    
    plot_tree(model, feature_names=feature_names, class_names=['No', 'Yes'],
              filled=True, rounded=True, ax=ax, max_depth=plot_depth, fontsize=9)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')


