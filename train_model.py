import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler

def train_model():
    # Load dataset
    df = pd.read_csv('titanic.csv')
    
    # --- Feature Engineering ---
    # Extract Title from Name
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    # Group Rare Titles
    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    
    # FamilySize and IsAlone
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = 0
    df.loc[df['FamilySize'] == 1, 'IsAlone'] = 1
    
    # --- Handling Missing Values ---
    # Fill Age with median based on Pclass and Title for more accuracy
    df['Age'] = df.groupby(['Pclass', 'Title'])['Age'].transform(lambda x: x.fillna(x.median()))
    
    # Fill Embarked with mode
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    
    # Fill Fare with median
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    
    # --- Encoding Categorical Variables ---
    le_sex = LabelEncoder()
    df['Sex'] = le_sex.fit_transform(df['Sex']) # male: 1, female: 0
    
    le_embarked = LabelEncoder()
    df['Embarked'] = le_embarked.fit_transform(df['Embarked'])
    
    le_title = LabelEncoder()
    df['Title'] = le_title.fit_transform(df['Title'])
    
    # --- Feature Selection ---
    features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'FamilySize', 'IsAlone', 'Title']
    X = df[features]
    y = df['Survived']
    
    # --- Train/Test Split ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # --- Scaling ---
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # --- Model Training ---
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    # --- Evaluation ---
    y_pred = model.predict(X_test)
    print("Accuracy Score:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    
    # --- Saving everything ---
    # We need to save the model, scaler, and label encoders
    model_data = {
        'model': model,
        'scaler': scaler,
        'le_sex': le_sex,
        'le_embarked': le_embarked,
        'le_title': le_title,
        'features': features
    }
    joblib.dump(model_data, 'titanic_model.pkl')
    print("\nModel and preprocessing objects saved to 'titanic_model.pkl'")

if __name__ == "__main__":
    train_model()
