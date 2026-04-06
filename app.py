from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os

app = Flask(__name__)

# ── Chargement des modèles ─────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
model   = joblib.load(os.path.join(BASE, 'models/best_model.pkl'))
scaler = joblib.load(os.path.join(BASE, 'models/scaler.pkl'))
features = joblib.load(os.path.join(BASE, 'models/features.pkl'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = [float(request.form[f]) for f in features]
        X = np.array(data).reshape(1, -1)
        X_sc = scaler.transform(X)

        pred = model.predict(X_sc)[0]
        proba = model.predict_proba(X_sc)[0][1] * 100

        return jsonify({
            'prediction': int(pred),
            'probability': round(proba, 2),
            'status': '🔴 DÉFAUT PROBABLE' if pred == 1
            else '🟢 PAS DE DÉFAUT'
        })
    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True, port=5000)
