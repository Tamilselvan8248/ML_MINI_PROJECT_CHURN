import os, io
import pandas as pd
import numpy as np
from flask import (Flask, render_template, request,
                   redirect, url_for, session, send_file)
from ml_model import (preprocess, train_test_split, train_all_models, evaluate_models,
                      predict_churn, get_suggestions, get_tree_base64)

app = Flask(__name__)
app.secret_key = 'churnguard_secret_2024'

# In-memory store (per-session state via module-level dict keyed by session id)
_store = {}

def _sid():
    if 'sid' not in session:
        import uuid
        session['sid'] = str(uuid.uuid4())
    return session['sid']


# ─── ROUTES ─────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    if 'csvfile' not in request.files:
        return render_template('index.html', error='No file selected.')

    f = request.files['csvfile']
    if f.filename == '':
        return render_template('index.html', error='No file selected.')
    if not f.filename.lower().endswith('.csv'):
        return render_template('index.html', error='Please upload a CSV file.')

    try:
        df = pd.read_csv(f)

        # ── Preprocess ──────────────────────────────────────────
        X, y, scaler, le_dict, customer_ids, feature_names = preprocess(df)

        # ── Train / Evaluate ─────────────────────────────────────
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)

        models       = train_all_models(X_train, y_train)
        eval_results = evaluate_models(models, X_test, y_test)

        # ── Best model ───────────────────────────────────────────
        best_name = max(eval_results, key=lambda m: eval_results[m]['accuracy'])

        # ── Per-customer predictions AND raw metrics for all models ───
        
        
        all_models_data = {}
        for name, model in eval_results.items():
            preds = predict_churn(model['model'], X, customer_ids, feature_names)
            
            total = len(preds)
            churn_yes = sum(1 for p in preds if p['Churn'] == 'Yes')
            churn_rate = round(churn_yes / total * 100, 1) if total > 0 else 0
            
            # --- Model Specific Extractions ---
            rf_importances = []
            dt_base64 = ""
            if name == 'Random Forest':
                importances = model['model'].feature_importances_
                # Zip and sort top 10
                feat_imp = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:10]
                rf_importances = [{'feature': f, 'importance': round(float(imp), 4)} for f, imp in feat_imp]
            elif name == 'Decision Tree':
                dt_base64 = get_tree_base64(model['model'], feature_names)
            
            all_models_data[name] = {
                'predictions': preds,
                'total': total,
                'churn_yes': churn_yes,
                'churn_no': total - churn_yes,
                'churn_rate': churn_rate,
                'high_risk': [p for p in preds if p['RiskLevel'] == 'High'],
                'suggestions': get_suggestions(churn_rate),
                'accuracy': eval_results[name]['accuracy'],
                'cm': eval_results[name]['cm'].tolist(),
                'roc': {
                    'fpr': eval_results[name]['fpr'].tolist(),
                    'tpr': eval_results[name]['tpr'].tolist(),
                    'roc_auc': eval_results[name]['roc_auc']
                },
                'rf_importances': rf_importances,
                'dt_base64': dt_base64
            }

        # ── Model accuracy list ───────────────────────────────────
        model_accuracies = [
            {'name': name, 'accuracy': eval_results[name]['accuracy']}
            for name in eval_results
        ]
        model_accuracies_sorted = sorted(model_accuracies,
                                         key=lambda x: x['accuracy'],
                                         reverse=True)

        # Store in memory
        sid = _sid()
        _store[sid] = {
            'all_models_data':  all_models_data,
            'model_names':      list(eval_results.keys()),
            'model_accuracies': model_accuracies_sorted,
            'best_name':        best_name,
            'models':           models,
            'scaler':           scaler,
            'le_dict':          le_dict,
            'feature_names':    feature_names,
            'X_train':          X_train,
            'y_train':          y_train,
            'X_test':           X_test,
            'y_test':           y_test
        }

        return redirect(url_for('dashboard'))

    except Exception as e:
        return render_template('index.html', error=f'Error processing file: {str(e)}')


@app.route('/dashboard')
def dashboard():
    sid = _sid()
    store_data = _store.get(sid)
    if not store_data:
        return redirect(url_for('index'))
    
    best_name = store_data['best_name']
    best_data = store_data['all_models_data'][best_name]
    
    data = {
        'best_name': best_name,
        'best_accuracy': best_data['accuracy'],
        'total': best_data['total'],
        'churn_yes': best_data['churn_yes'],
        'churn_no': best_data['churn_no'],
        'churn_rate': best_data['churn_rate'],
        'model_accuracies': store_data['model_accuracies']
    }
    return render_template('dashboard.html', data=data)


@app.route('/results')
def results():
    sid = _sid()
    store_data = _store.get(sid)
    if not store_data:
        return redirect(url_for('index'))
    
    selected_model = request.args.get('model', store_data['best_name'])
    if selected_model not in store_data['all_models_data']:
        selected_model = store_data['best_name']
        
    model_data = store_data['all_models_data'][selected_model]
    
    data = {
        'selected_model': selected_model,
        'best_name': store_data['best_name'],
        'all_model_names': store_data['model_names'],
        'high_risk': model_data['high_risk'],
        'predictions': model_data['predictions'],
        'suggestions': model_data['suggestions'],
        'cm': model_data['cm'],
        'roc': model_data['roc'],
        'churn_yes': model_data['churn_yes'],
        'churn_no': model_data['churn_no'],
        'rf_importances': model_data['rf_importances'],
        'dt_base64': model_data['dt_base64']
    }
    return render_template('results.html', data=data)


@app.route('/download')
def download():
    sid = _sid()
    store_data = _store.get(sid)
    if not store_data:
        return redirect(url_for('index'))
    
    selected_model = request.args.get('model', store_data['best_name'])
    if selected_model not in store_data['all_models_data']:
        selected_model = store_data['best_name']
        
    df = pd.DataFrame(store_data['all_models_data'][selected_model]['predictions'])
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return send_file(
        io.BytesIO(buf.getvalue().encode()),
        mimetype='text/csv',
        as_attachment=True,
        download_name=f'churn_predictions_{selected_model}.csv'
    )


@app.route('/simulator')
def simulator():
    sid = _sid()
    store_data = _store.get(sid)
    if not store_data:
        return redirect(url_for('index'))
    
    feature_meta = []
    for feat in store_data['feature_names']:
        if feat in store_data['le_dict']:
            options = list(store_data['le_dict'][feat].classes_)
            feature_meta.append({'name': feat, 'type': 'categorical', 'options': options})
        else:
            feature_meta.append({'name': feat, 'type': 'numerical'})
            
    return render_template('simulator.html', features=feature_meta)


@app.route('/api/simulate', methods=['POST'])
def api_simulate():
    sid = _sid()
    store_data = _store.get(sid)
    if not store_data:
        return {"error": "Session expired"}, 400
        
    data = request.json
    try:
        features_ordered = []
        for feat in store_data['feature_names']:
            val = data.get(feat, '')
            if feat in store_data['le_dict']:
                encoded = store_data['le_dict'][feat].transform([str(val)])[0]
                features_ordered.append(encoded)
            else:
                features_ordered.append(float(val) if val != '' else 0.0)
                
        X_custom = np.array([features_ordered])
        X_scaled = store_data['scaler'].transform(X_custom)
        
        results = {}
        for name, model_info in store_data['models'].items():
            # In train_all_models, model is either the direct sklearn object or part of a dict
            # Wait, train_all_models returns just the sklearn model objects!
            model = model_info 
            if hasattr(model, 'predict_proba'):
                prob = model.predict_proba(X_scaled)[0, 1]
            else:
                pred = model.predict(X_scaled)[0]
                prob = float(pred)
            results[name] = round(prob * 100, 1)
            
        return {"success": True, "predictions": results}
    except Exception as e:
        return {"error": str(e)}, 500


@app.route('/tuning')
def tuning():
    sid = _sid()
    store_data = _store.get(sid)
    if not store_data:
        return redirect(url_for('index'))
    return render_template('tuning.html', models=store_data['model_names'])


@app.route('/api/tune', methods=['POST'])
def api_tune():
    sid = _sid()
    store_data = _store.get(sid)
    if not store_data:
        return {"error": "Session expired"}, 400
        
    data = request.json
    model_name = data.get('model')
    param_val = float(data.get('value'))
    
    X_train = store_data['X_train']
    y_train = store_data['y_train']
    X_test = store_data['X_test']
    y_test = store_data['y_test']
    
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    
    try:
        if model_name == 'K-Nearest Neighbors':
            new_model = KNeighborsClassifier(n_neighbors=max(1, int(param_val)))
        elif model_name == 'Logistic Regression':
            new_model = LogisticRegression(max_iter=1000, C=param_val, random_state=42)
        elif model_name == 'Decision Tree':
            new_model = DecisionTreeClassifier(max_depth=max(1, int(param_val)), random_state=42)
        elif model_name == 'Random Forest':
            new_model = RandomForestClassifier(n_estimators=max(10, int(param_val)), random_state=42)
        else:
            return {"error": "Model cannot be tuned"}, 400
            
        new_model.fit(X_train, y_train)
        
        eval_dict = evaluate_models({model_name: new_model}, X_test, y_test)
        new_eval = eval_dict[model_name]
        acc = new_eval['accuracy']
        
        graph_data = {}
        if model_name == 'K-Nearest Neighbors':
            graph_data['cm'] = new_eval['cm'].tolist()
        elif model_name == 'Logistic Regression':
            graph_data['roc'] = {
                'fpr': new_eval['fpr'].tolist(),
                'tpr': new_eval['tpr'].tolist(),
                'roc_auc': new_eval['roc_auc']
            }
        elif model_name == 'Decision Tree':
            graph_data['dt_base64'] = get_tree_base64(new_model, store_data['feature_names'])
        elif model_name == 'Random Forest':
            importances = new_model.feature_importances_
            feat_imp = sorted(zip(store_data['feature_names'], importances), key=lambda x: x[1], reverse=True)[:10]
            graph_data['rf_importances'] = [{'feature': f, 'importance': round(float(imp), 4)} for f, imp in feat_imp]
            
        return {"success": True, "new_accuracy": acc, "graph_data": graph_data}
    except Exception as e:
        return {"error": str(e)}, 500


if __name__ == '__main__':
    app.run(debug=True)
