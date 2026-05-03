import pandas as pd
import numpy as np

def generate_base_features(n=1000):
    np.random.seed(42)
    tenure = np.random.randint(1, 73, n)
    monthly_charges = np.random.uniform(20, 120, n).round(2)
    total_charges = (tenure * monthly_charges).round(2)
    
    return pd.DataFrame({
        'CustomerID':      ['CUST-' + str(1000 + i) for i in range(n)],
        'Gender':          np.random.choice(['Male', 'Female'], n),
        'SeniorCitizen':   np.random.choice([0, 1], n, p=[0.8, 0.2]),
        'Partner':         np.random.choice(['Yes', 'No'], n),
        'Dependents':      np.random.choice(['Yes', 'No'], n),
        'Tenure':          tenure,
        'PhoneService':    np.random.choice(['Yes', 'No'], n),
        'MultipleLines':   np.random.choice(['Yes', 'No', 'No phone service'], n),
        'InternetService': np.random.choice(['Fiber optic', 'DSL', 'No'], n),
        'OnlineSecurity':  np.random.choice(['Yes', 'No'], n),
        'TechSupport':     np.random.choice(['Yes', 'No'], n),
        'StreamingTV':     np.random.choice(['Yes', 'No'], n),
        'StreamingMovies': np.random.choice(['Yes', 'No'], n),
        'Contract':        np.random.choice(['Month-to-month', 'One year', 'Two year'], n),
        'PaperlessBilling':np.random.choice(['Yes', 'No'], n),
        'PaymentMethod':   np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], n),
        'MonthlyCharges':  monthly_charges,
        'TotalCharges':    total_charges
    })

# 1. LOGISTIC REGRESSION BEST (Pure Linear Relationship)
def make_lr_best():
    df = generate_base_features()
    # Log-odds perfectly linear with Tenure and MonthlyCharges
    z = -0.15 * df['Tenure'] + 0.08 * df['MonthlyCharges'] - 1.0
    p = 1 / (1 + np.exp(-z))
    df['Churn'] = np.where(np.random.binomial(1, p) == 1, 'Yes', 'No')
    df.to_csv('data/lr_best_dataset.csv', index=False)
    print("LR dataset saved: data/lr_best_dataset.csv")

# 2. DECISION TREE BEST (Exact Axis-Aligned Splits, No Noise)
def make_dt_best():
    df = generate_base_features()
    # Pure Boolean logic rules
    c1 = (df['Contract'] == 'Month-to-month') & (df['InternetService'] == 'Fiber optic') & (df['Tenure'] < 15)
    c2 = (df['MonthlyCharges'] > 90) & (df['TechSupport'] == 'No')
    c3 = (df['Contract'] == 'One year') & (df['SeniorCitizen'] == 1) & (df['PaperlessBilling'] == 'Yes')
    
    df['Churn'] = np.where(c1 | c2 | c3, 'Yes', 'No')
    df.to_csv('data/dt_best_dataset.csv', index=False)
    print("DT dataset saved: data/dt_best_dataset.csv")

# 3. K-NEAREST NEIGHBORS BEST (Checkerboard / Non-linear Local Patterns)
def make_knn_best():
    df = generate_base_features()
    # Normalize Tenure and Charges to 0-1
    t_norm = df['Tenure'] / 72.0
    m_norm = df['MonthlyCharges'] / 120.0
    
    # Complex concentric rings / checkerboard pattern
    pattern = np.sin(t_norm * 15) * np.cos(m_norm * 15)
    
    # KNN captures local neighborhoods well; linear fails entirely
    df['Churn'] = np.where(pattern > 0, 'Yes', 'No')
    
    # Add a little noise
    flip = np.random.rand(len(df)) < 0.1
    df.loc[flip, 'Churn'] = np.where(df.loc[flip, 'Churn'] == 'Yes', 'No', 'Yes')
    
    df.to_csv('data/knn_best_dataset.csv', index=False)
    print("KNN dataset saved: data/knn_best_dataset.csv")

# 4. RANDOM FOREST BEST (Complex Interactions with Noise)
def make_rf_best():
    df = generate_base_features()
    # Complex interacting features
    score = np.zeros(len(df))
    
    score += np.where(df['Contract'] == 'Month-to-month', 2.0, -1.0)
    score += np.where((df['InternetService'] == 'Fiber optic') & (df['TechSupport'] == 'No'), 2.5, 0)
    score += (df['MonthlyCharges'] / 50.0) ** 2 # non-linear charge impact
    score -= (df['Tenure'] / 10.0) ** 1.5       # non-linear tenure impact
    score += np.where(df['PaymentMethod'] == 'Electronic check', 1.0, -0.5)
    
    # Convert to probability using sigmoid
    p = 1 / (1 + np.exp(-(score - 1.5)))
    
    # Random Forest excels at handling many interacting features and noise better than a single Tree
    df['Churn'] = np.where(np.random.binomial(1, p) == 1, 'Yes', 'No')
    df.to_csv('data/rf_best_dataset.csv', index=False)
    print("RF dataset saved: data/rf_best_dataset.csv")

if __name__ == '__main__':
    make_lr_best()
    make_dt_best()
    make_knn_best()
    make_rf_best()
