import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import make_pipeline
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

class HealthCareAI:
    def __init__(self, data_path):
        self.data_path = data_path
        self.model = None
        self.results = None
    
    def load_data(self):
        """Load dataset from the specified path."""
        self.data = pd.read_csv(self.data_path)
        print(f"Data loaded successfully from {self.data_path}.")
    
    def preprocess_data(self):
        """Preprocess the data: handle missing values, encode categorical variables, and scale features."""
        self.data.fillna(method='ffill', inplace=True)
        
        categorical_columns = self.data.select_dtypes(include=['object']).columns
        label_encoders = {}
        
        for col in categorical_columns:
            le = LabelEncoder()
            self.data[col] = le.fit_transform(self.data[col])
            label_encoders[col] = le
        
        print(f"Categorical columns encoded: {list(label_encoders.keys())}.")
        
        features = self.data.drop('diagnosis', axis=1)
        labels = self.data['diagnosis']
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
    
    def train_model(self):
        """Train the model using Random Forest Classifier."""
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(self.X_train, self.y_train)
        print("Model trained successfully.")
    
    def evaluate_model(self):
        """Evaluate the trained model and print performance metrics."""
        predictions = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, predictions)
        report = classification_report(self.y_test, predictions)
        
        print(f"Model Accuracy: {accuracy:.2f}")
        print("Classification Report:")
        print(report)
        
        self.results = (accuracy, report)
    
    def feature_importance(self):
        """Visualize the feature importance of the model."""
        importances = self.model.feature_importances_
        feature_names = self.X_train.columns
        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(10, 6))
        sns.barplot(x=importances[indices], y=feature_names[indices])
        plt.title("Feature Importance")
        plt.xlabel("Relative Importance")
        plt.ylabel("Features")
        plt.show()
        
    def save_model(self, file_name='healthcare_model.pkl'):
        """Save the trained model to a file."""
        joblib.dump(self.model, file_name)
        joblib.dump(self.scaler, 'scaler.pkl')
        print(f"Model saved as {file_name} and scaler as 'scaler.pkl'.")
    
    def load_model(self, file_name='healthcare_model.pkl'):
        """Load the model from a file."""
        self.model = joblib.load(file_name)
        self.scaler = joblib.load('scaler.pkl')
        print(f"Model loaded from {file_name}.")
    
    def predict(self, input_data):
        """Predict using the loaded model."""
        input_data_scaled = self.scaler.transform(np.array(input_data).reshape(1, -1))
        prediction = self.model.predict(input_data_scaled)
        return prediction[0]

if __name__ == "__main__":
    ai_healthcare = HealthCareAI(data_path='health_data.csv')
    ai_healthcare.load_data()
    ai_healthcare.preprocess_data()
    ai_healthcare.train_model()
    ai_healthcare.evaluate_model()
    ai_healthcare.feature_importance()
    ai_healthcare.save_model()