import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, f1_score, recall_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(file_path):
    df = pd.read_csv(file_path)
    print("Data loaded successfully. Shape:", df.shape)
    return df


def preprocess_data(df):
    if 'id' in df.columns:
        df = df.drop('id', axis=1)
    
 
    df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')
    
    X = df.drop('stroke', axis=1)
    y = df['stroke']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

def create_preprocessor():
    numeric_features = ['age', 'avg_glucose_level', 'bmi']
    categorical_features = ['gender', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor


def train_and_evaluate_model(model, X_train, X_test, y_train, y_test, preprocessor):
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', model)])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    
    print(f"\nModel: {type(model).__name__}")
    print(f"ROC AUC: {roc_auc_score(y_test, y_pred_proba):.3f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.3f}")
    print(f"Recall: {recall_score(y_test, y_pred):.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return pipeline

def fine_tune_model(model, param_grid, X_train, y_train, preprocessor):
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', model)])
    
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    print(f"\nBest parameters for {type(model).__name__}:")
    print(grid_search.best_params_)
    print(f"Best ROC AUC score: {grid_search.best_score_:.3f}")
    
    return grid_search.best_estimator_

def plot_feature_importances(model, preprocessor):
    feature_names = (preprocessor.named_transformers_['num'].get_feature_names_out().tolist() +
                     preprocessor.named_transformers_['cat'].get_feature_names_out().tolist())
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        print("Model does not have feature importances or coefficients.")
        return
    
    feature_imp = pd.DataFrame({'feature': feature_names, 'importance': importances})
    feature_imp = feature_imp.sort_values('importance', ascending=False).head(10)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_imp)
    plt.title(f'Top 10 Feature Importances - {type(model).__name__}')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    df = load_data('/content/stroke.csv')
    X_train, X_test, y_train, y_test = preprocess_data(df)
    
    preprocessor = create_preprocessor()
    
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'XGBoost': XGBClassifier(random_state=42)
    }
    
    trained_models = {}
    for name, model in models.items():
        trained_models[name] = train_and_evaluate_model(model, X_train, X_test, y_train, y_test, preprocessor)
    
    param_grids = {
        'Logistic Regression': {'classifier__C': [0.1, 1, 10]},
        'Random Forest': {'classifier__n_estimators': [100, 200], 'classifier__max_depth': [5, 10, None]},
        'XGBoost': {'classifier__n_estimators': [100, 200], 'classifier__max_depth': [3, 5, 7]}
    }
    
    best_models = {}
    for name, model in models.items():
        print(f"\nFine-tuning {name}...")
        best_models[name] = fine_tune_model(model, param_grids[name], X_train, y_train, preprocessor)
    

    for name, model in best_models.items():
        print(f"\nEvaluating best {name} model:")
        train_and_evaluate_model(model.named_steps['classifier'], X_train, X_test, y_train, y_test, preprocessor)
    
    for name, model in best_models.items():
        plot_feature_importances(model.named_steps['classifier'], preprocessor)