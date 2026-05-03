"""
Generate a balanced synthetic customer churn dataset.
- 500 Churn + 500 No Churn (50/50 split)
- Mixed monthly charges (low / medium / high) for both classes
- Realistic correlations: long-tenure + 2yr contracts = less churn, etc.
"""
import pandas as pd
import numpy as np

np.random.seed(42)

def gen_customers(n, churn_label):
    is_churn = (churn_label == 1)

    # Tenure: churners leave earlier, but not exclusively
    if is_churn:
        tenure = np.clip(np.random.exponential(scale=14, size=n).astype(int), 1, 72)
    else:
        tenure = np.clip(np.random.normal(loc=40, scale=18, size=n).astype(int), 1, 72)

    # Monthly charges: both classes have low/mid/high — but churners skew higher
    if is_churn:
        monthly_charges = np.concatenate([
            np.random.uniform(20, 45, int(n * 0.20)),   # 20% low
            np.random.uniform(45, 75, int(n * 0.35)),   # 35% medium
            np.random.uniform(75, 110, n - int(n*0.20) - int(n*0.35)),  # 45% high
        ])
    else:
        monthly_charges = np.concatenate([
            np.random.uniform(20, 45, int(n * 0.40)),   # 40% low
            np.random.uniform(45, 75, int(n * 0.40)),   # 40% medium
            np.random.uniform(75, 110, n - int(n*0.40) - int(n*0.40)),  # 20% high
        ])
    np.random.shuffle(monthly_charges)
    monthly_charges = monthly_charges.round(2)
    total_charges = (monthly_charges * tenure).round(2)

    # Contract: churners lean month-to-month, stayers lean long-term
    if is_churn:
        contract = np.random.choice(
            ['Month-to-month', 'One year', 'Two year'], n, p=[0.65, 0.22, 0.13])
    else:
        contract = np.random.choice(
            ['Month-to-month', 'One year', 'Two year'], n, p=[0.22, 0.33, 0.45])

    # Internet: fibre is faster but churners blame it more
    if is_churn:
        internet = np.random.choice(['Fiber optic', 'DSL', 'No'], n, p=[0.55, 0.33, 0.12])
    else:
        internet = np.random.choice(['Fiber optic', 'DSL', 'No'], n, p=[0.30, 0.42, 0.28])

    # Tech support & online security: non-churners have it more often
    if is_churn:
        tech_support    = np.random.choice(['Yes', 'No'], n, p=[0.25, 0.75])
        online_security = np.random.choice(['Yes', 'No'], n, p=[0.22, 0.78])
    else:
        tech_support    = np.random.choice(['Yes', 'No'], n, p=[0.62, 0.38])
        online_security = np.random.choice(['Yes', 'No'], n, p=[0.60, 0.40])

    # Payment: electronic check correlates with churn
    if is_churn:
        payment = np.random.choice(
            ['Electronic check', 'Mailed check',
             'Bank transfer (automatic)', 'Credit card (automatic)'],
            n, p=[0.46, 0.22, 0.18, 0.14])
    else:
        payment = np.random.choice(
            ['Electronic check', 'Mailed check',
             'Bank transfer (automatic)', 'Credit card (automatic)'],
            n, p=[0.14, 0.20, 0.36, 0.30])

    ids = ['CUST-' + str(1000 + i) for i in range(n)]

    return pd.DataFrame({
        'CustomerID':       ids,
        'Gender':           np.random.choice(['Male', 'Female'], n),
        'SeniorCitizen':    np.random.choice([0, 1], n, p=[0.84, 0.16]),
        'Partner':          np.random.choice(['Yes', 'No'], n, p=[0.48, 0.52]),
        'Dependents':       np.random.choice(['Yes', 'No'], n, p=[0.30, 0.70]),
        'Tenure':           tenure,
        'PhoneService':     np.random.choice(['Yes', 'No'], n, p=[0.90, 0.10]),
        'MultipleLines':    np.random.choice(['Yes', 'No', 'No phone service'], n,
                                              p=[0.42, 0.48, 0.10]),
        'InternetService':  internet,
        'OnlineSecurity':   online_security,
        'TechSupport':      tech_support,
        'StreamingTV':      np.random.choice(['Yes', 'No'], n, p=[0.44, 0.56]),
        'StreamingMovies':  np.random.choice(['Yes', 'No'], n, p=[0.44, 0.56]),
        'Contract':         contract,
        'PaperlessBilling': np.random.choice(['Yes', 'No'], n, p=[0.59, 0.41]),
        'PaymentMethod':    payment,
        'MonthlyCharges':   monthly_charges,
        'TotalCharges':     total_charges,
        'Churn':            ['Yes' if is_churn else 'No'] * n,
    })


churn_df    = gen_customers(500, 1)
no_churn_df = gen_customers(500, 0)

# Fix duplicate CustomerIDs
no_churn_df['CustomerID'] = ['CUST-' + str(2000 + i) for i in range(500)]

df = pd.concat([churn_df, no_churn_df]).sample(frac=1, random_state=42).reset_index(drop=True)

df.to_csv('data/balanced_customers.csv', index=False)

print(f"Dataset generated: {len(df)} rows")
print(df['Churn'].value_counts())
print("\nMonthlyCharges distribution:")
print(df.groupby('Churn')['MonthlyCharges'].describe().round(2))
print("\nContract distribution:")
print(df.groupby('Churn')['Contract'].value_counts())
print("\nSaved -> data/balanced_customers.csv")
