import re

import numpy
import pandas
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


def load_from_csv(f: str) :
    df = pandas.read_csv(f)
    train_data = df[df["collection"] == "train"]
    val_data = df[df["collection"] == "val"]
    test_data = df[df["collection"] == "test"]

    feature_cols = [col for col in df.columns if not re.match("^(Unnamed: \\d+|id|label|collection)$", col)]
    return (
        {
            "id": train_data["id"].values,
            "x": numpy.nan_to_num(train_data[feature_cols].values),
            "y": train_data[["label"]].values.ravel(),
        },
        {
            "id": val_data["id"].values,
            "x": numpy.nan_to_num(val_data[feature_cols].values),
            "y": val_data[["label"]].values.ravel(),
        },
        {
            "id": test_data["id"].values,
            "x": numpy.nan_to_num(test_data[feature_cols].values),
            "y": test_data[["label"]].values.ravel(),
        }
    )

if __name__ == '__main__':
    compares = [
        ("patient_features_with_ff_fd", "compare/v5.1_30,20_10,5_17,5/patient_attrs.csv"),
        ("patient_embeddings_with_time_emb32", "compare/v5.1_30,20_10,5_17,5/patient_embeddings_with_time_emb32.csv"),
        ("patient_embeddings_with_time_emb64", "compare/v5.1_30,20_10,5_17,5/patient_embeddings_with_time_emb64.csv"),
        ("patient_embeddings_with_time_emb128", "compare/v5.1_30,20_10,5_17,5/patient_embeddings_with_time_emb128.csv"),
        ("patient_embeddings_without_time_emb32", "compare/v5.1_30,20_10,5_17,5/patient_embeddings_without_time_emb32.csv"),
        ("patient_embeddings_without_time_emb64", "compare/v5.1_30,20_10,5_17,5/patient_embeddings_without_time_emb64.csv"),
        ("patient_embeddings_without_time_emb128", "compare/v5.1_30,20_10,5_17,5/patient_embeddings_without_time_emb128.csv"),
    ]
    for compare_name, csv in compares:
        print()
        print("------------------------------------------------------------------------------------")
        print(compare_name)

        train_xy, val_xy, test_xy = load_from_csv(csv)

        train_x = train_xy['x']
        train_y = train_xy['y'].astype(numpy.float32)
        val_x = val_xy['x']
        val_y = val_xy['y'].astype(numpy.float32)
        test_x = test_xy['x']
        test_y = test_xy['y'].astype(numpy.float32)

        models = {
            "Logistic": LogisticRegression(random_state=42, max_iter=1000),
            "SVM": SVC(kernel='linear', probability=True, random_state=42, max_iter=1000),
            "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
            "DecisionTree": DecisionTreeClassifier(random_state=42),
            "XGB": XGBClassifier(objective="multi:softmax", num_class=2, eval_metric="mlogloss")
        }

        for name, model in models.items():
            model.fit(train_x, train_y)

            train_pred_y = model.predict(train_x).astype(numpy.float32)
            train_acc = accuracy_score(train_y, train_pred_y)

            val_pred_y = model.predict(val_x).astype(numpy.float32)
            val_acc = accuracy_score(val_y, val_pred_y)

            test_pred_y = model.predict(test_x).astype(numpy.float32)
            test_acc = accuracy_score(test_y, test_pred_y)

            print(f"""
{name}
train accuracy score: {train_acc}
real: {train_y}
pred: {train_pred_y}
val accuracy score: {val_acc}
real: {val_y}
pred: {val_pred_y}
test accuracy score: {test_acc}
real: {test_y}
pred: {test_pred_y}
            """)
