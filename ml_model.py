import pandas as pd
import numpy as np
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_score, recall_score, f1_score,
                             roc_curve, auc)
from sklearn.tree import plot_tree
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

def preprocess(df: pd.DataFrame):
    df = df.copy()
    df.drop_duplicates(inplace=True)

    if 'Churn' in df.columns:
        df.dropna(subset=['Churn'], inplace=True)

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    for col in cat_cols:
        if not df[col].mode().empty:
            df[col] = df[col].fillna(df[col].mode()[0])

    target_col = None
    for col in df.columns:
        if col.lower() == 'churn':
            target_col = col
            break
    if target_col is None:
        raise ValueError("Dataset must have a 'Churn' column.")

    le_target = LabelEncoder()
    df[target_col] = le_target.fit_transform(df[target_col].astype(str).str.strip())

    cat_cols_remaining = df.select_dtypes(include=['object']).columns.tolist()
    for col in cat_cols_remaining:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    id_col = None
    for col in df.columns:
        if col.lower() in ('customerid', 'customer_id', 'id'):
            id_col = col
            break

    customer_ids = df[id_col].values.tolist() if id_col else list(range(len(df)))

    drop_cols = [target_col]
    if id_col:
        drop_cols.append(id_col)
    X = df.drop(columns=drop_cols)
    y = df[target_col]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y.values, customer_ids, X.columns.tolist()

def train_all_models(X_train, y_train):
    knn = KNeighborsClassifier(n_neighbors=7)
    lr  = LogisticRegression(max_iter=1000, C=0.5, random_state=42)
    dt  = DecisionTreeClassifier(max_depth=8, min_samples_leaf=5, random_state=42)
    rf  = RandomForestClassifier(n_estimators=100, max_depth=12,
                                 min_samples_leaf=3, random_state=42)

    models = {
        'K-Nearest Neighbors': knn,
        'Logistic Regression': lr,
        'Decision Tree':       dt,
        'Random Forest':       rf,
    }
    for model in models.values():
        model.fit(X_train, y_train)
    return models

def evaluate_models(models, X_test, y_test):
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        acc       = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall    = recall_score(y_test, y_pred, zero_division=0)
        f1        = f1_score(y_test, y_pred, zero_division=0)
        cm        = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, int(sum(y_test)))
        results[name] = {
            'accuracy':  round(acc * 100, 2),
            'precision': round(precision * 100, 2),
            'recall':    round(recall * 100, 2),
            'f1':        round(f1 * 100, 2),
            'tp': int(tp), 'tn': int(tn),
            'fp': int(fp), 'fn': int(fn),
            'model': model
        }
    return results


# ─────────────────────────────────────────────
# DATA-DRIVEN INSIGHTS
# ─────────────────────────────────────────────
def generate_insights(df_orig: pd.DataFrame):
    """Analyse the raw (un-encoded) DataFrame and return list of insight dicts."""
    df = df_orig.copy()

    # Locate churn column
    churn_col = next((c for c in df.columns if c.lower() == 'churn'), None)
    if churn_col is None:
        return []

    churn_str = df[churn_col].astype(str).str.strip().str.lower()
    df['_c'] = churn_str.map({'yes': 1, '1': 1, 'true': 1}).fillna(0).astype(int)
    total_rate = df['_c'].mean() * 100
    insights = []

    def _rate_dict(series):
        return {str(k): round(float(v), 1) for k, v in series.items()}

    # 1. Contract type
    if 'Contract' in df.columns:
        grp = df.groupby('Contract')['_c'].mean() * 100
        worst = grp.idxmax(); worst_r = grp.max(); best = grp.idxmin(); best_r = grp.min()
        insights.append({
            'title':  f'{worst} contract has the highest churn ({worst_r:.1f}%)',
            'detail': (f'Customers on <b>{worst}</b> contracts churn at <b>{worst_r:.1f}%</b>, '
                       f'compared to just <b>{best_r:.1f}%</b> for <b>{best}</b> contracts. '
                       f'Encouraging long-term commitments with discounts can dramatically improve retention.'),
            'icon': 'fa-file-signature', 'color': '#f43f5e',
            'value': f'{worst_r:.1f}%', 'label': 'Highest Contract Churn',
            'stats': _rate_dict(grp)
        })

    # 2. Monthly charges tiers
    if 'MonthlyCharges' in df.columns:
        df['_bin'] = pd.cut(df['MonthlyCharges'], bins=[0, 40, 70, 9999],
                            labels=['Low (<$40)', 'Mid ($40-$70)', 'High (>$70)'])
        grp = df.groupby('_bin', observed=True)['_c'].mean() * 100
        high_r = float(grp.get('High (>$70)', 0))
        low_r  = float(grp.get('Low (<$40)', 0))
        insights.append({
            'title':  f'High monthly charges → {high_r:.1f}% churn vs {low_r:.1f}% for low charges',
            'detail': (f'Customers paying over $70/month churn at <b>{high_r:.1f}%</b>, '
                       f'nearly {high_r/max(low_r,1):.1f}× the rate of low-charge customers ({low_r:.1f}%). '
                       f'Targeted discount campaigns for high-spend accounts can cut churn significantly.'),
            'icon': 'fa-dollar-sign', 'color': '#f59e0b',
            'value': f'{high_r:.1f}%', 'label': 'High-Charge Churn',
            'stats': _rate_dict(grp)
        })

    # 3. Tech support
    if 'TechSupport' in df.columns:
        grp = df.groupby('TechSupport')['_c'].mean() * 100
        no_r  = float(grp.get('No',  grp.mean()))
        yes_r = float(grp.get('Yes', grp.mean()))
        insights.append({
            'title':  f'No tech support → {no_r:.1f}% churn vs {yes_r:.1f}% with support',
            'detail': (f'Customers without tech support churn at <b>{no_r:.1f}%</b>, compared to <b>{yes_r:.1f}%</b> '
                       f'for those subscribed. Promoting the value of tech support — especially to new customers — can reduce churn by up to {no_r-yes_r:.1f} percentage points.'),
            'icon': 'fa-headset', 'color': '#818cf8',
            'value': f'{no_r:.1f}%', 'label': 'No-Support Churn',
            'stats': _rate_dict(grp)
        })

    # 4. Tenure buckets
    if 'Tenure' in df.columns:
        df['_ten'] = pd.cut(df['Tenure'], bins=[0, 12, 36, 9999],
                            labels=['New (0-12 mo)', 'Mid (13-36 mo)', 'Loyal (36+ mo)'])
        grp = df.groupby('_ten', observed=True)['_c'].mean() * 100
        new_r   = float(grp.get('New (0-12 mo)', 0))
        loyal_r = float(grp.get('Loyal (36+ mo)', 0))
        insights.append({
            'title':  f'New customers churn at {new_r:.1f}% vs {loyal_r:.1f}% for loyal customers',
            'detail': (f'Customers in their first year have a <b>{new_r:.1f}%</b> churn rate, '
                       f'compared to only <b>{loyal_r:.1f}%</b> for 3+ year customers. '
                       f'A strong onboarding experience and first-year loyalty programme can cut early churn significantly.'),
            'icon': 'fa-clock', 'color': '#10b981',
            'value': f'{new_r:.1f}%', 'label': 'New-Customer Churn',
            'stats': _rate_dict(grp)
        })

    # 5. Internet service
    if 'InternetService' in df.columns:
        grp = df.groupby('InternetService')['_c'].mean() * 100
        worst_i = grp.idxmax(); worst_ir = grp.max()
        insights.append({
            'title':  f'{worst_i} internet service has {worst_ir:.1f}% churn',
            'detail': (f'Among internet service types, <b>{worst_i}</b> customers have the highest '
                       f'churn rate at <b>{worst_ir:.1f}%</b>. Investigating service quality or pricing '
                       f'issues in this segment can yield quick retention wins.'),
            'icon': 'fa-wifi', 'color': '#c084fc',
            'value': f'{worst_ir:.1f}%', 'label': f'{worst_i} Churn',
            'stats': _rate_dict(grp)
        })

    # 6. Payment method
    if 'PaymentMethod' in df.columns:
        grp = df.groupby('PaymentMethod')['_c'].mean() * 100
        worst_p = grp.idxmax(); worst_pr = grp.max()
        insights.append({
            'title':  f'"{worst_p}" payment has {worst_pr:.1f}% churn',
            'detail': (f'Customers who pay via <b>{worst_p}</b> churn at <b>{worst_pr:.1f}%</b>. '
                       f'Encouraging automatic payment methods (bank transfer / credit card) '
                       f'reduces friction and correlates with lower churn.'),
            'icon': 'fa-credit-card', 'color': '#f472b6',
            'value': f'{worst_pr:.1f}%', 'label': 'Payment Churn',
            'stats': _rate_dict(grp)
        })

    return insights

# ── Light Theme Style Helper ──────────────────────────────────
BG      = '#f8fafc'
BG2     = '#ffffff'
BORDER  = '#e2e8f0'
TEXT    = '#0f172a'
MUTED   = '#475569'
PRIMARY = '#4f46e5'
GREEN   = '#059669'
RED     = '#e11d48'
GOLD    = '#d97706'
PINK    = '#db2777'
PURPLE  = '#9333ea'

def _dark_fig(w=7, h=4.5):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG2)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.spines['bottom'].set_color(BORDER)
    ax.spines['left'].set_color(BORDER)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.label.set_color(MUTED)
    ax.xaxis.label.set_color(MUTED)
    ax.title.set_color(TEXT)
    ax.grid(axis='y', color=BORDER, linewidth=0.6, linestyle='--', alpha=0.6)
    return fig, ax

def get_base64_image():
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=130)
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def generate_graphs(models, results, best_name, X_test, y_test, X, feature_names):
    graphs = {'all_models': {}}
    MODEL_COLORS = {
        'K-Nearest Neighbors': PRIMARY,
        'Logistic Regression':  GREEN,
        'Decision Tree':        GOLD,
        'Random Forest':        PINK,
    }

    # 1. Accuracy Bar Chart ─────────────────────────────────
    fig, ax = _dark_fig(7, 4.5)
    names = list(results.keys())
    accs  = [results[n]['accuracy'] for n in names]
    bar_colors = [MODEL_COLORS.get(n, PRIMARY) for n in names]
    bars = ax.bar(names, accs, color=bar_colors, width=0.55, zorder=3,
                  edgecolor='none')
    # Highlight best
    for i, n in enumerate(names):
        if n == best_name:
            bars[i].set_linewidth(2)
            bars[i].set_edgecolor(TEXT)
    # Value labels
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.7,
                f'{acc}%', ha='center', va='bottom', fontsize=9,
                fontweight='bold', color=TEXT)
    ax.set_ylim(0, 110)
    ax.set_title('Model Accuracy Comparison', fontsize=12, fontweight='bold', pad=14)
    ax.set_ylabel('Accuracy (%)')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace(' ', '\n') for n in names], fontsize=8.5)
    plt.tight_layout()
    graphs['accuracy_chart'] = get_base64_image()

    for name, model in models.items():
        model_graphs = {}
        
        # 2. Confusion Matrix ───────────────────────────────────
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        fig.patch.set_facecolor(BG)
        ax.set_facecolor(BG)
        cmap = sns.diverging_palette(220, 20, as_cmap=True)
        sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm',
                    xticklabels=['No Churn', 'Churn'],
                    yticklabels=['No Churn', 'Churn'],
                    ax=ax, linewidths=2, linecolor=BG,
                    annot_kws={'size': 16, 'weight': 'bold', 'color': TEXT})
        ax.set_title(f'Confusion Matrix ({name})', fontsize=12, fontweight='bold', color=TEXT, pad=14)
        ax.set_xlabel('Predicted', color=MUTED, fontsize=9)
        ax.set_ylabel('Actual', color=MUTED, fontsize=9)
        ax.tick_params(colors=MUTED, labelsize=9)
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(colors=MUTED, labelsize=8)
        plt.tight_layout()
        model_graphs['confusion_matrix'] = get_base64_image()

        # 3. Pie Chart ──────────────────────────────────────────
        preds = model.predict(X)
        churn_count     = int(sum(preds))
        non_churn_count = int(len(preds) - churn_count)
        fig, ax = plt.subplots(figsize=(5.5, 4.8))
        fig.patch.set_facecolor(BG)
        ax.set_facecolor(BG)
        wedge_props = dict(width=0.55, edgecolor=BG, linewidth=3)
        wedges, texts, autotexts = ax.pie(
            [non_churn_count, churn_count],
            labels=['No Churn', 'Churn'],
            autopct='%1.1f%%',
            colors=[GREEN, RED],
            startangle=90,
            wedgeprops=wedge_props,
            textprops={'color': TEXT, 'fontsize': 10},
            pctdistance=0.75,
            labeldistance=1.12
        )
        for at in autotexts:
            at.set_fontweight('bold')
            at.set_fontsize(10)
        ax.set_title(f'Churn Distribution ({name})', fontsize=12, fontweight='bold', color=TEXT, pad=14)
        ax.text(0, 0, f'{churn_count}\nChurn', ha='center', va='center',
                fontsize=11, fontweight='bold', color=RED)
        plt.tight_layout()
        model_graphs['pie_chart'] = get_base64_image()

        # 4. Feature Importance / Coefficients ──────────────────
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feat_imp = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:10]
            title = f'Top Feature Importances ({name})'
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
            feat_imp = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:10]
            title = f'Top Absolute Coefficients ({name})'
        else:
            feat_imp = None
            
        if feat_imp:
            feats, imps = zip(*feat_imp)
            fig, ax = _dark_fig(7, 4.8)
            palette = [PRIMARY, PURPLE, PINK, GOLD, GREEN, '#60a5fa', '#a78bfa',
                       '#fb923c', '#34d399', '#f87171']
            bars = ax.barh(list(feats[::-1]), list(imps[::-1]),
                           color=palette[:len(feats)], zorder=3, height=0.6)
            for bar, imp in zip(bars, list(imps[::-1])):
                ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                        f'{imp:.3f}', va='center', fontsize=8, color=MUTED)
            ax.set_title(title, fontsize=11, fontweight='bold', pad=14)
            ax.set_xlabel('Importance / Coef Magnitude')
            ax.grid(axis='x', color=BORDER, linewidth=0.6, linestyle='--', alpha=0.6)
            ax.grid(axis='y', visible=False)
            plt.tight_layout()
            model_graphs['feature_importance'] = get_base64_image()
        else:
            # Empty plot with message for KNN
            fig, ax = _dark_fig(7, 4.8)
            ax.text(0.5, 0.5, 'Feature Importance not available\nfor K-Nearest Neighbors', 
                    ha='center', va='center', fontsize=11, color=MUTED)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f'Feature Importance ({name})', fontsize=11, fontweight='bold', pad=14)
            plt.tight_layout()
            model_graphs['feature_importance'] = get_base64_image()

        # 5. Algorithm Specific Visualization ──────────────────
        if name == 'Decision Tree':
            fig, ax = plt.subplots(figsize=(10, 6))
            fig.patch.set_facecolor(BG)
            plot_tree(model, max_depth=3, feature_names=feature_names, 
                      class_names=['Stay', 'Leave'], filled=True, 
                      rounded=True, fontsize=7, ax=ax, precision=2)
            ax.set_title('Decision Tree Structure (Max Depth: 3)', fontsize=12, fontweight='bold', pad=14)
            plt.tight_layout()
            model_graphs['algo_viz'] = get_base64_image()
        elif name == 'Random Forest':
            # Feature Importance is already good, but maybe show a single tree?
            # Or just use the same key for consistency
            model_graphs['algo_viz'] = model_graphs.get('feature_importance')
        else:
            model_graphs['algo_viz'] = None

        graphs['all_models'][name] = model_graphs

    return graphs

def predict_churn_model(model, X_scaled, customer_ids):
    """Return per-customer predictions with risk tier for any model."""
    preds = model.predict(X_scaled)
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

def generate_roc_curves(models, X_test, y_test):
    """Generate ROC curves for all models that support predict_proba."""
    fig, ax = _dark_fig(8, 5.5)
    MODEL_COLORS = {
        'K-Nearest Neighbors': PRIMARY,
        'Logistic Regression': GREEN,
        'Decision Tree':       GOLD,
        'Random Forest':       PINK,
    }
    auc_scores = {}
    for name, model in models.items():
        try:
            probs = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, probs)
            roc_auc = auc(fpr, tpr)
            auc_scores[name] = round(roc_auc, 4)
            ax.plot(fpr, tpr, color=MODEL_COLORS.get(name, PRIMARY),
                    linewidth=2.5, label=f'{name} (AUC={roc_auc:.3f})')
        except Exception:
            auc_scores[name] = None
    ax.plot([0, 1], [0, 1], color=BORDER, linewidth=1.5, linestyle='--', label='Random Classifier')
    ax.fill_between([0, 1], [0, 1], alpha=0.04, color=MUTED)
    ax.set_xlabel('False Positive Rate', fontsize=10)
    ax.set_ylabel('True Positive Rate', fontsize=10)
    ax.set_title('ROC Curves — All Models', fontsize=13, fontweight='bold', pad=16)
    ax.legend(loc='lower right', fontsize=8.5, framealpha=0.9,
              facecolor=BG2, edgecolor=BORDER)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    plt.tight_layout()
    return get_base64_image(), auc_scores


def generate_shap_plot(model, X_scaled, feature_names, model_name, max_samples=200):
    """Generate SHAP summary bar plot. Returns base64 string or None."""
    if not SHAP_AVAILABLE:
        return None
    try:
        X_sample = X_scaled[:max_samples]
        if hasattr(model, 'feature_importances_'):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
            if isinstance(shap_values, list):
                sv = shap_values[1]
            else:
                sv = shap_values
        elif hasattr(model, 'coef_'):
            explainer = shap.LinearExplainer(model, X_sample)
            sv = explainer.shap_values(X_sample)
        else:
            explainer = shap.KernelExplainer(model.predict_proba, X_sample[:50])
            sv = explainer.shap_values(X_sample[:50])[1]

        mean_abs = np.abs(sv).mean(axis=0)
        top_idx = np.argsort(mean_abs)[::-1][:12]
        top_feats = [feature_names[i] for i in top_idx]
        top_vals  = mean_abs[top_idx]

        fig, ax = _dark_fig(7, 5)
        palette = [PRIMARY, PURPLE, PINK, GOLD, GREEN, '#60a5fa', '#a78bfa',
                   '#fb923c', '#34d399', '#f87171', '#c084fc', '#38bdf8']
        bars = ax.barh(top_feats[::-1], top_vals[::-1],
                       color=palette[:len(top_feats)], height=0.6, zorder=3)
        for bar, val in zip(bars, top_vals[::-1]):
            ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                    f'{val:.4f}', va='center', fontsize=8, color=MUTED)
        ax.set_title(f'SHAP Feature Impact ({model_name})', fontsize=11, fontweight='bold', pad=14)
        ax.set_xlabel('Mean |SHAP Value| — Average Impact on Prediction')
        ax.grid(axis='x', color=BORDER, linewidth=0.6, linestyle='--', alpha=0.6)
        ax.grid(axis='y', visible=False)
        plt.tight_layout()
        return get_base64_image()
    except Exception:
        return None


def generate_customer_segments(df_orig, predictions):
    """Cluster customers into segments based on churn risk + value."""
    df = df_orig.copy()
    pred_map = {str(p['CustomerID']): p for p in predictions}

    id_col = next((c for c in df.columns if c.lower() in ('customerid', 'customer_id', 'id')), None)
    if id_col:
        df['_cid'] = df[id_col].astype(str)
    else:
        df['_cid'] = df.index.astype(str)

    df['_prob'] = df['_cid'].map(lambda x: pred_map.get(x, {}).get('Probability', 0))
    df['_churn'] = df['_cid'].map(lambda x: pred_map.get(x, {}).get('Churn', 'No'))

    charge_col  = next((c for c in df.columns if 'monthly' in c.lower() and 'charge' in c.lower()), None)
    tenure_col  = next((c for c in df.columns if 'tenure' in c.lower()), None)

    segments = []
    for _, row in df.iterrows():
        prob   = float(row['_prob'])
        churn  = row['_churn']
        charge = float(row[charge_col]) if charge_col and pd.notna(row[charge_col]) else 50
        tenure = float(row[tenure_col]) if tenure_col and pd.notna(row[tenure_col]) else 12

        if prob >= 70 and charge >= 70:
            seg = 'High-Value at Risk'
            color = '#f43f5e'; icon = 'fa-fire'
            action = 'Immediate VIP outreach — offer premium discount or dedicated account manager'
        elif prob >= 70 and charge < 70:
            seg = 'Low-Value at Risk'
            color = '#f59e0b'; icon = 'fa-triangle-exclamation'
            action = 'Cost-effective retention — offer contract upgrade or bundled services'
        elif prob < 40 and tenure >= 24:
            seg = 'Loyal Champions'
            color = '#10b981'; icon = 'fa-crown'
            action = 'Reward loyalty — referral program or exclusive perks'
        elif prob < 40 and tenure < 24:
            seg = 'Promising'
            color = '#818cf8'; icon = 'fa-seedling'
            action = 'Nurture engagement — onboarding support and feature education'
        else:
            seg = 'Needs Attention'
            color = '#c084fc'; icon = 'fa-eye'
            action = 'Monitor closely — proactive check-in and satisfaction survey'

        segments.append({
            'CustomerID': row['_cid'],
            'Segment':    seg,
            'Probability': prob,
            'Churn':      churn,
            'Color':      color,
            'Icon':       icon,
            'Action':     action,
            'Charge':     round(charge, 1),
            'Tenure':     int(tenure),
        })

    # Segment summary
    seg_counts = {}
    for s in segments:
        seg_counts[s['Segment']] = seg_counts.get(s['Segment'], 0) + 1

    return segments, seg_counts


def get_suggestions():
    return [
        "Customers with high monthly charges are more likely to churn. Consider offering customized discount plans.",
        "Customers on month-to-month contracts have a higher churn risk. Encourage them to switch to 1-year or 2-year contracts.",
        "Lack of tech support is a strong indicator of churn. Highlight the benefits of subscribing to premium support.",
        "Ensure new customers have a smooth onboarding process to increase their lifetime value."
    ]
