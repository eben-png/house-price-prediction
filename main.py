import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib

def load_data(data_path, y_data, X_data):
    # Load data
    df = pd.read_csv(data_path)
    df.dropna(inplace=True)
    y = df[y_data]

    X = df[X_data].copy()

    if 'ocean_proximity' in X.columns:
        X = pd.get_dummies(X, columns=["ocean_proximity"], drop_first=True)

    return X, y    

def get_accuracy_full(X, X_test, y, y_test):
    model = RandomForestRegressor(random_state=1)
    model.fit(X, y)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    return r2

def get_accuracy_split(X_train, X_test, y_train, y_test):
    model = RandomForestRegressor(random_state=1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    return r2

def save_full_accuracy(X, y):
    model = RandomForestRegressor(random_state=1)
    model.fit(X, y)
    joblib.dump(model, "house_model.pkl")

if __name__ == "__main__":
    X, y = load_data("data/housing.csv", "median_house_value", ["longitude","latitude","housing_median_age","total_rooms","total_bedrooms","population","households","median_income", "ocean_proximity"])
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

    RED = "\033[91m"
    GREEN = "\033[92m"
    BLUE = "\033[94m"
    RESET = "\033[0m"

    answer = ""

    while answer != "0":
        print(f"{BLUE}1. Get accuracy with split data{RESET}")
        print(f"{BLUE}2. Get accuracy with full data{RESET}")
        print(f"{BLUE}3. Save model to disk{RESET}")
        print(f"{RED}0. Exit{RESET}")
        answer = input("What will you do? ")

        match(answer):
            case "1":
                acc = get_accuracy_split(X_train, X_test, y_train, y_test)
                print(f"{GREEN}Accuracy (split data): {acc}{RESET}")
            case "2":
                acc = get_accuracy_full(X, X_test, y, y_test)
                print(f"{GREEN}Accuracy (full data): {acc}{RESET}")
            case "3":
                save_full_accuracy(X, y)
                print(f"{GREEN}Model saved to disk{RESET}")

    print(f"{RED}Goodbye{RESET}")
