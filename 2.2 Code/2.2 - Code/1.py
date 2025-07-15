import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

# Load the trained model and pre-fitted scaler
model = joblib.load("train_model.pkl")
scaler = joblib.load("scaler.pkl")  # Ensure scaler.pkl is available and pre-fitted

def main():
    print("Welcome to the Personality Prediction Program!")
    
    # Input collection from the user
    gender = input("Enter gender (Female/Male): ").strip()
    gender_no = 1 if gender.lower() == "female" else 2

    try:
        age = float(input("Enter age: "))
        openness = float(input("Enter openness score: "))
        neuroticism = float(input("Enter neuroticism score: "))
        conscientiousness = float(input("Enter conscientiousness score: "))
        agreeableness = float(input("Enter agreeableness score: "))
        extraversion = float(input("Enter extraversion score: "))
    except ValueError:
        print("Invalid input! Please enter numerical values where applicable.")
        return

    # Prepare input data
    input_data = np.array([gender_no, age, openness, neuroticism, conscientiousness, agreeableness, extraversion], ndmin=2)
    scaled_data = scaler.transform(input_data)  # Scale the input data

    # Predict the personality
    personality = model.predict(scaled_data)[0]
    print(f"Predicted Personality: {personality}")

if __name__ == "__main__":
    main()
