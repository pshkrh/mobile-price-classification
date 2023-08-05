from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import sklearn
import joblib
import argparse
import os
import pandas as pd


def model_fn(model_dir):
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf


if __name__ == '__main__':
    print("[INFO] Extracting arguments")
    parser = argparse.ArgumentParser()

    # Hyperparameters set by the client are passed as command-line arguments
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--random_state", type=int, default=0)

    # Data, model, and output directories
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TEST"))
    parser.add_argument("--train-file", type=str, default="mobile-price-train.csv")
    parser.add_argument("--test-file", type=str, default="mobile-price-test.csv")

    args, _ = parser.parse_known_args()

    print(f"sklearn version: {sklearn.__version__}")
    print(f"Joblib version: {joblib.__version__}")

    print("[INFO] Reading data\n")
    train_df = pd.read_csv(os.path.join(args.train, args.train_file))
    test_df = pd.read_csv(os.path.join(args.test, args.test_file))

    features = list(train_df.columns)
    label = features.pop()

    print("Building training and testing datasets\n")

    X_train = train_df[features]
    X_test = test_df[features]
    y_train = train_df[label]
    y_test = test_df[label]

    print("Training Data Shape\n")
    print(X_train.shape)
    print(y_train.shape)

    print("Testing Data Shape\n")
    print(X_test.shape)
    print(y_test.shape)

    print("Training Random Forest Model\n")
    model = RandomForestClassifier(n_estimators=args.n_estimators, random_state=args.random_state, verbose=True)
    model.fit(X_train, y_train)

    model_path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(model, model_path)
    print(f"Model persisted at {model_path}")

    y_pred_test = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred_test)
    test_report = classification_report(y_test, y_pred_test)

    print("Metrics\n")
    print(f"Total Rows: {X_test.shape[0]}")
    print(f"[TESTING] Model Accuracy: {test_acc}")
    print(f"[TESTING] Testing Report:")
    print(test_report)
