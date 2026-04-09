import os
import io
import joblib
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
from utils import preprocess_data, get_prediction_explanation

app = Flask(__name__)

# ================== CONFIGURATION ==================
MODEL_PATH = "titanic_model.pkl"
DATA_PATH = "titanic.csv"

# ================== LOAD MODEL & DATA ==================
try:
    model_data = joblib.load(MODEL_PATH)
    model = model_data['model']
    features = model_data['features']
except Exception as e:
    print(f"Error loading model: {e}")
    model_data = None

df_raw = pd.read_csv(DATA_PATH)

# ================== ROUTES ==================

@app.route('/')
def index():
    # Basic statistics for the dashboard
    stats = {
        'total': len(df_raw),
        'survived_rate': round((df_raw['Survived'].sum() / len(df_raw)) * 100, 1),
        'male_count': int((df_raw['Sex'] == 'male').sum()),
        'female_count': int((df_raw['Sex'] == 'female').sum()),
    }
    return render_template('index.html', stats=stats)

@app.route('/predict', methods=['POST'])
def predict():
    if not model_data:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        # Get data from form
        input_data = {
            'Pclass': int(request.form['pclass']),
            'Sex': request.form['sex'],
            'Age': float(request.form['age']),
            'Fare': float(request.form['fare']),
            # Handle optional/derived fields
            'SibSp': int(request.form.get('sibsp', 0)),
            'Parch': int(request.form.get('parch', 0)),
            'Embarked': request.form.get('embarked', 'S')
        }
        
        # Create a DataFrame for preprocessing
        input_df = pd.DataFrame([input_data])
        
        # Preprocess
        X_processed, processed_df = preprocess_data(input_df, model_data)
        
        # Prediction
        prediction = int(model.predict(X_processed)[0])
        probability = float(model.predict_proba(X_processed)[0][prediction])
        
        # Get explanation
        explanation = get_prediction_explanation(processed_df.iloc[0], model)
        
        return jsonify({
            "survived": prediction,
            "probability": round(probability * 100, 2),
            "explanation": explanation
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/batch-predict', methods=['POST'])
def batch_predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    try:
        # Read CSV
        batch_df = pd.read_csv(io.StringIO(file.stream.read().decode('utf-8')))
        
        # Preprocess the entire batch
        X_batch, processed_batch_df = preprocess_data(batch_df, model_data)
        
        # Predict
        predictions = model.predict(X_batch)
        probabilities = model.predict_proba(X_batch).max(axis=1)
        
        # Add results to dataframe
        batch_df['Survival_Prediction'] = ["Survived" if p == 1 else "Not Survived" for p in predictions]
        batch_df['Confidence'] = [f"{round(prob * 100, 2)}%" for prob in probabilities]
        
        # Return as JSON (or we could return a CSV file download)
        results = batch_df[['Name', 'Survival_Prediction', 'Confidence']].head(10).to_dict(orient='records')
        return jsonify({"results": results, "total": len(batch_df)})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/analytics-data')
def analytics_data():
    """Returns data for Chart.js"""
    # Survival by Gender
    gender_survival = df_raw.groupby('Sex')['Survived'].mean().to_dict()
    
    # Survival by Pclass
    pclass_survival = df_raw.groupby('Pclass')['Survived'].mean().to_dict()
    
    # Age Distribution (binned)
    age_bins = [0, 18, 35, 60, 100]
    age_labels = ['0-18', '19-35', '36-60', '60+']
    df_raw['AgeGroup'] = pd.cut(df_raw['Age'], bins=age_bins, labels=age_labels)
    age_dist = df_raw['AgeGroup'].value_counts().sort_index().to_dict()

    return jsonify({
        "gender_survival": gender_survival,
        "pclass_survival": pclass_survival,
        "age_dist": age_dist
    })


@app.route('/chat', methods=['POST'])
def chat():
    user_msg = request.json.get('message', '').lower()
    
    # Simple rule-based response (can be replaced with Gemini API)
    if 'accuracy' in user_msg:
        return jsonify({"reply": "The model has an accuracy of approximately 82.7%. It uses a Random Forest algorithm."})
    elif 'features' in user_msg:
        return jsonify({"reply": f"The model uses features like: {', '.join(features)}."})
    elif 'survive' in user_msg:
        return jsonify({"reply": "Survival was highly dependent on Sex, Passenger Class, and Age. Females and children had higher survival rates."})
    else:
        return jsonify({"reply": "I'm a specialized Titanic AI. I can tell you about the model's accuracy, features, or general survival trends."})

if __name__ == "__main__":
    app.run(debug=True, port=5000)