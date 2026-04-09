import pandas as pd
import numpy as np
import joblib

def preprocess_data(df, model_data):
    """
    Apply the same preprocessing steps as used in training.
    """
    df = df.copy()
    
    # --- Feature Engineering ---
    # Title extraction
    if 'Name' in df.columns:
        df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
        df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
        df['Title'] = df['Title'].replace('Mlle', 'Miss')
        df['Title'] = df['Title'].replace('Ms', 'Miss')
        df['Title'] = df['Title'].replace('Mme', 'Mrs')
    else:
        # If name is not provided (single prediction), we might need a default or user input
        df['Title'] = 'Mr' # Default or logic based on Sex/Age
        
    # FamilySize and IsAlone
    if 'SibSp' in df.columns and 'Parch' in df.columns:
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        df['IsAlone'] = 0
        df.loc[df['FamilySize'] == 1, 'IsAlone'] = 1
    else:
        # Handle case where SibSp/Parch might be missing in simplified input
        df['FamilySize'] = 1
        df['IsAlone'] = 1

    # --- Handling Missing Values (Using training medians/modes would be better, but for now we simplify) ---
    # In a real production app, we should save and use the training medians.
    df['Age'] = df['Age'].fillna(29) # Approximate median
    df['Fare'] = df['Fare'].fillna(14.45) # Approximate median
    df['Embarked'] = df['Embarked'].fillna('S')
    
    # --- Encoding ---
    # Use the label encoders from model_data
    df['Sex'] = df['Sex'].map({'male': 1, 'female': 0}).fillna(1) # Default to male if unknown
    
    # Title and Embarked need to match the encoders
    # Simple mapping for Title if not using the actual le_title
    title_map = {t: i for i, t in enumerate(model_data['le_title'].classes_)}
    df['Title'] = df['Title'].map(title_map).fillna(title_map.get('Mr', 0))
    
    embarked_map = {e: i for i, e in enumerate(model_data['le_embarked'].classes_)}
    df['Embarked'] = df['Embarked'].map(embarked_map).fillna(embarked_map.get('S', 0))

    # --- Scaling ---
    features = model_data['features']
    X = df[features]
    X_scaled = model_data['scaler'].transform(X)
    
    return X_scaled, df

def get_prediction_explanation(row, model):
    """
    Generate a simple explanation for the prediction.
    Highly simplified logic based on feature importance.
    """
    explanation = []
    if row['Sex'] == 0: # Female
        explanation.append("Being female significantly increased survival chances.")
    if row['Pclass'] == 1:
        explanation.append("First-class passengers had priority during evacuation.")
    if row['Age'] < 12:
        explanation.append("Children were prioritized for lifeboats.")
    if row['FamilySize'] > 1 and row['FamilySize'] < 5:
        explanation.append("Traveling with a small family increased chances of assistance.")
    
    if not explanation:
        explanation.append("Survival was determined by specific circumstances and lifeboat availability.")
    
    return " ".join(explanation)
