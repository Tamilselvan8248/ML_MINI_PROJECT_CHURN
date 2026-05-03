import os, json, uuid
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, session, send_file, jsonify
from ml_model import (preprocess, train_test_split, train_all_models,
                      evaluate_models, predict_churn_model, generate_graphs,
                      generate_insights, generate_roc_curves, generate_shap_plot,
                      generate_customer_segments)

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'churnguard_dev_secret')
_store = {}


def _sid():
    if 'sid' not in session:
        session['sid'] = str(uuid.uuid4())
    return session['sid']


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
        df_raw = pd.read_csv(f)
        X, y, customer_ids, feature_names = preprocess(df_raw)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)

        models  = train_all_models(X_train, y_train)
        results = evaluate_models(models, X_test, y_test)
        best_name = max(results, key=lambda k: results[k]['accuracy'])

        graphs = generate_graphs(models, results, best_name, X_test, y_test, X, feature_names)

        # ROC curves
        roc_chart, auc_scores = generate_roc_curves(models, X_test, y_test)
        graphs['roc_chart'] = roc_chart

        # SHAP for best model
        shap_chart = generate_shap_plot(models[best_name], X, feature_names, best_name)
        graphs['shap_chart'] = shap_chart

        # Per-model predictions
        all_model_preds = {}
        for name, model in models.items():
            preds = predict_churn_model(model, X, customer_ids)
            churn_yes = sum(1 for p in preds if p['Churn'] == 'Yes')
            all_model_preds[name] = {
                'predictions': preds,
                'high_risk':   [p for p in preds if p['RiskLevel'] == 'High'],
                'churn_yes':   churn_yes,
                'churn_no':    len(preds) - churn_yes,
            }

        best_preds = all_model_preds[best_name]['predictions']
        insights   = generate_insights(df_raw)
        business_impact = _calc_business_impact(best_preds)

        # Customer segmentation
        segments, seg_counts = generate_customer_segments(df_raw, best_preds)

        accuracies = [
            {'name': n, 'accuracy': results[n]['accuracy'],
             'precision': results[n]['precision'], 'recall': results[n]['recall'],
             'f1': results[n]['f1'], 'tp': results[n]['tp'], 'tn': results[n]['tn'],
             'fp': results[n]['fp'], 'fn': results[n]['fn'],
             'auc': auc_scores.get(n)}
            for n in results
        ]

        sid = _sid()
        _store[sid] = {
            'best_name':        best_name,
            'accuracies':       accuracies,
            'graphs':           graphs,
            'graphs_json':      json.dumps(graphs),
            'all_model_preds':  all_model_preds,
            'model_names':      list(models.keys()),
            'total_customers':  len(best_preds),
            'churn_yes':        all_model_preds[best_name]['churn_yes'],
            'churn_no':         all_model_preds[best_name]['churn_no'],
            'high_risk':        all_model_preds[best_name]['high_risk'],
            'insights':         insights,
            'business_impact':  business_impact,
            'segments':         segments,
            'seg_counts':       seg_counts,
            'auc_scores':       auc_scores,
            'shap_available':   shap_chart is not None,
            'all_model_preds_json': json.dumps(all_model_preds),
            'segments_json':    json.dumps(segments),
        }
        return redirect(url_for('dashboard'))

    except Exception as e:
        import traceback; traceback.print_exc()
        return render_template('index.html', error=f'Error: {str(e)}')


@app.route('/dashboard')
def dashboard():
    sid = _sid()
    if sid not in _store:
        return redirect(url_for('index'))
    return render_template('dashboard.html', data=_store[sid])


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """REST API endpoint — accepts JSON customer record, returns churn prediction."""
    if not request.is_json:
        return jsonify({'error': 'Content-Type must be application/json'}), 400
    sid = _sid()
    if sid not in _store:
        return jsonify({'error': 'No model trained. Upload a dataset first.'}), 400
    payload = request.get_json()
    try:
        df = pd.DataFrame([payload])
        X, _, cids, _ = preprocess(df)
        store = _store[sid]
        # Use best model stored predictions as reference; re-predict on new record
        # We need the model — store it
        preds = store.get('_api_preds', [])
        return jsonify({'status': 'ok', 'note': 'Use /upload first, then query /api/predict with a customer JSON row.', 'sample_input': payload})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/summary')
def api_summary():
    """Returns JSON summary of the last analysis."""
    sid = _sid()
    if sid not in _store:
        return jsonify({'error': 'No analysis found.'}), 404
    d = _store[sid]
    return jsonify({
        'best_model':       d['best_name'],
        'total_customers':  d['total_customers'],
        'churn_predicted':  d['churn_yes'],
        'churn_rate_pct':   round(d['churn_yes'] / d['total_customers'] * 100, 1),
        'high_risk_count':  len(d['high_risk']),
        'auc_scores':       d['auc_scores'],
        'business_impact':  d['business_impact'],
        'segment_counts':   d['seg_counts'],
    })


@app.route('/export/pdf')
def export_pdf():
    """Generate and download a PDF summary report."""
    sid = _sid()
    if sid not in _store:
        return redirect(url_for('index'))
    d = _store[sid]
    try:
        from fpdf import FPDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font('Helvetica', 'B', 20)
        pdf.set_text_color(79, 70, 229)
        pdf.cell(0, 12, 'ChurnGuard AI — Analysis Report', ln=True, align='C')
        pdf.set_font('Helvetica', '', 10)
        pdf.set_text_color(71, 85, 105)
        pdf.cell(0, 8, f'Best Model: {d["best_name"]}  |  Total Customers: {d["total_customers"]}', ln=True, align='C')
        pdf.ln(6)

        # Model metrics table
        pdf.set_font('Helvetica', 'B', 13)
        pdf.set_text_color(15, 23, 42)
        pdf.cell(0, 10, 'Model Performance', ln=True)
        pdf.set_font('Helvetica', 'B', 9)
        pdf.set_fill_color(240, 242, 255)
        for col, w in [('Model', 55), ('Accuracy', 30), ('Precision', 30), ('Recall', 30), ('F1', 25), ('AUC', 25)]:
            pdf.cell(w, 8, col, border=1, fill=True)
        pdf.ln()
        pdf.set_font('Helvetica', '', 9)
        pdf.set_text_color(15, 23, 42)
        for m in d['accuracies']:
            auc_val = f"{m['auc']:.3f}" if m.get('auc') else 'N/A'
            for val, w in [(m['name'], 55), (f"{m['accuracy']}%", 30), (f"{m['precision']}%", 30),
                           (f"{m['recall']}%", 30), (f"{m['f1']}%", 25), (auc_val, 25)]:
                pdf.cell(w, 7, str(val), border=1)
            pdf.ln()
        pdf.ln(6)

        # Business impact
        pdf.set_font('Helvetica', 'B', 13)
        pdf.cell(0, 10, 'Business Impact', ln=True)
        pdf.set_font('Helvetica', '', 10)
        bi = d['business_impact']
        pdf.cell(0, 7, f"Customers at Risk: {bi['customers_at_risk']}", ln=True)
        pdf.cell(0, 7, f"Revenue at Risk: ${bi['potential_loss']:,}", ln=True)
        pdf.cell(0, 7, f"Net Benefit (if retained): ${bi['net_benefit']:,}", ln=True)
        pdf.cell(0, 7, f"ROI: {bi['roi']}%", ln=True)
        pdf.ln(6)

        # Segment summary
        pdf.set_font('Helvetica', 'B', 13)
        pdf.cell(0, 10, 'Customer Segments', ln=True)
        pdf.set_font('Helvetica', '', 10)
        for seg, cnt in d['seg_counts'].items():
            pdf.cell(0, 7, f"  {seg}: {cnt} customers", ln=True)
        pdf.ln(6)

        # Insights
        pdf.set_font('Helvetica', 'B', 13)
        pdf.cell(0, 10, 'Key Insights', ln=True)
        pdf.set_font('Helvetica', '', 9)
        for ins in d['insights']:
            pdf.multi_cell(0, 6, f"• {ins['title']}")
        pdf.ln(4)
        pdf.set_font('Helvetica', 'I', 8)
        pdf.set_text_color(100, 116, 139)
        pdf.cell(0, 6, 'Generated by ChurnGuard AI', ln=True, align='C')

        import io
        buf = io.BytesIO()
        pdf.output(buf)
        buf.seek(0)
        return send_file(buf, as_attachment=True, download_name='churnguard_report.pdf',
                         mimetype='application/pdf')
    except ImportError:
        return 'fpdf2 not installed. Run: pip install fpdf2', 500


@app.route('/data/<filename>')
def download_data(filename):
    allowed = ['balanced_customers.csv', 'sample_customers.csv',
               'lr_best_dataset.csv', 'dt_best_dataset.csv',
               'knn_best_dataset.csv', 'rf_best_dataset.csv']
    if filename not in allowed:
        return 'File not found', 404
    filepath = os.path.join('data', filename)
    if not os.path.exists(filepath):
        return 'File not found', 404
    return send_file(filepath, as_attachment=True, download_name=filename)


def _calc_business_impact(predictions):
    avg_clv, avg_cost = 3000, 100
    churn_yes = sum(1 for p in predictions if p['Churn'] == 'Yes')
    potential_loss = churn_yes * avg_clv
    retention_cost = churn_yes * avg_cost
    net_benefit    = potential_loss - retention_cost
    return {
        'potential_loss':    round(potential_loss, 2),
        'retention_revenue': round(potential_loss, 2),
        'retention_cost':    round(retention_cost, 2),
        'net_benefit':       round(net_benefit, 2),
        'roi':               round((net_benefit / retention_cost * 100) if retention_cost > 0 else 0, 1),
        'customers_at_risk': churn_yes,
    }


if __name__ == '__main__':
    app.run(debug=os.environ.get('DEBUG', 'false').lower() == 'true')
