import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

def get_or_train_model(model_path = 'sentiment_model.pkl', vectorizer_path='vectorizer.pkl', csv_path = 'cleaned_imdb_reviews.csv'):
    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        print("Loading saved model")
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        
        #reload data for test set
        df = pd.read_csv(csv_path)
        X = vectorizer.transform(df['cleaned_review'])
        y = df['sentiment'].map({'positive':1, 'negative':0})
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        return model, vectorizer, X_test, y_test

    print("Training new model and vectorizer")
    
    #loading the csv(dataset)
    df = pd.read_csv("cleaned_imdb_reviews.csv")

    #vectorizing the value
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df['cleaned_review'])
    y = df['sentiment'].map({'positive':1, 'negative':0})

    #splitting the data to test and train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #training the model
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    #saving the model
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    return model, vectorizer, X_test, y_test

def model_prediton_and_report(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

if __name__ == '__main__':
    
    model, vectorizer, X_test, y_test = get_or_train_model()
    
    #option to test the model
    model_prediton_and_report(model, X_test, y_test)
    
    #option for custom review and the ai predicts the answer (postive or negative)
    custom_review = input("enter a moview review : ")
    clean_custom_review = custom_review.lower()
    custom_vec = vectorizer.transform([clean_custom_review])
    prediction = model.predict(custom_vec)
    print("Prediction:", "Positive" if prediction[0] == 1 else "Negative")