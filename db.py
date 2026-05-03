import os
from supabase import create_client, Client

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not required in production (Render sets env vars directly)

SUPABASE_URL = os.environ.get('SUPABASE_URL', 'https://yulygbawzddqamwmjnjd.supabase.co')
SUPABASE_KEY = os.environ.get('SUPABASE_KEY', '')

_client: Client = None


def get_client() -> Client:
    global _client
    if _client is None:
        if not SUPABASE_KEY:
            raise RuntimeError('SUPABASE_KEY environment variable not set.')
        _client = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _client


def save_analysis(session_id: str, data: dict):
    """Save a full analysis session to Supabase."""
    try:
        db = get_client()
        row = {
            'session_id':       session_id,
            'best_model':       data['best_name'],
            'total_customers':  data['total_customers'],
            'churn_yes':        data['churn_yes'],
            'churn_no':         data['churn_no'],
            'high_risk_count':  len(data['high_risk']),
            'churn_rate_pct':   round(data['churn_yes'] / data['total_customers'] * 100, 1),
            'potential_loss':   data['business_impact']['potential_loss'],
            'net_benefit':      data['business_impact']['net_benefit'],
            'roi':              data['business_impact']['roi'],
            'seg_counts':       data['seg_counts'],
            'auc_scores':       data['auc_scores'],
            'model_metrics':    [
                {
                    'name':      m['name'],
                    'accuracy':  m['accuracy'],
                    'precision': m['precision'],
                    'recall':    m['recall'],
                    'f1':        m['f1'],
                    'auc':       m.get('auc'),
                }
                for m in data['accuracies']
            ],
        }
        db.table('analyses').insert(row).execute()
        return True
    except Exception as e:
        print(f'[Supabase] save_analysis error: {e}')
        return False


def save_predictions(session_id: str, predictions: list):
    """Save per-customer predictions to Supabase."""
    try:
        db = get_client()
        rows = [
            {
                'session_id':  session_id,
                'customer_id': str(p['CustomerID']),
                'churn':       p['Churn'],
                'probability': p['Probability'],
                'risk_level':  p['RiskLevel'],
            }
            for p in predictions
        ]
        # Insert in batches of 500
        for i in range(0, len(rows), 500):
            db.table('predictions').insert(rows[i:i+500]).execute()
        return True
    except Exception as e:
        print(f'[Supabase] save_predictions error: {e}')
        return False


def get_recent_analyses(limit: int = 10):
    """Fetch recent analyses from Supabase."""
    try:
        db = get_client()
        res = db.table('analyses') \
                .select('*') \
                .order('created_at', desc=True) \
                .limit(limit) \
                .execute()
        return res.data or []
    except Exception as e:
        print(f'[Supabase] get_recent_analyses error: {e}')
        return []


def get_predictions(session_id: str):
    """Fetch predictions for a session from Supabase."""
    try:
        db = get_client()
        res = db.table('predictions') \
                .select('*') \
                .eq('session_id', session_id) \
                .execute()
        return res.data or []
    except Exception as e:
        print(f'[Supabase] get_predictions error: {e}')
        return []
