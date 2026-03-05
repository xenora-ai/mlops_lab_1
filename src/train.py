import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import joblib
import yaml
import os

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import mlflow
import mlflow.sklearn


def main():
    # 1. Налаштування CLI аргументів (Крок 5.1)
    parser = argparse.ArgumentParser(description="Train Graduate Salary Model")
    parser.add_argument("--n_estimators", type=int, help="Number of trees")
    parser.add_argument("--max_depth", type=int, help="Max depth of trees")
    parser.add_argument("--random_state", type=int, default=42, help="Random state")
    parser.add_argument("input_dir")  # data/prepared
    parser.add_argument("output_dir")  # data/models
    args = parser.parse_args()

    if args.n_estimators is None or args.max_depth is None:
        with open("params.yaml") as f:
            params = yaml.safe_load(f)
        n_estimators = params["train"]["n_estimators"]
        max_depth = params["train"]["max_depth"]
    else:
        n_estimators = args.n_estimators
        max_depth = args.max_depth

    # 2. Завантаження даних
    data_path = "data/raw/global_graduate_employability_index.csv"
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return

    df = pd.read_csv(data_path)

    # Видаляємо Graduation_Year та цільову зміну
    X = df.drop(columns=['Average_Starting_Salary_USD', 'Graduation_Year'])
    y = df['Average_Starting_Salary_USD']

    # 3. Розділення вибірки (Крок 4.3)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Визначення ознак для обробки
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    ordinal_feature = ['Degree_Level']
    nominal_features = [col for col in categorical_features if col not in ordinal_feature]
    degree_order = [["Bachelor", "Master", "PhD"]]

    # 5. Створення пайплайнів обробки
    ordinal_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ordinal", OrdinalEncoder(categories=degree_order))
    ])

    nominal_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('ord', ordinal_pipeline, ordinal_feature),
            ("nom", nominal_pipeline, nominal_features)
        ],
        remainder='passthrough'
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()

    # 6. Ініціалізація MLflow (Крок 4.4)
    mlflow.set_experiment("Graduate_Salary_Baseline_v2")

    with mlflow.start_run():
        # Додавання тегів (Крок 5.3)
        mlflow.set_tag("model_type", "RandomForestRegressor")
        mlflow.set_tag("developer", "Kate")

        # Навчання моделі (Крок 4.5)
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=args.random_state
        )
        model.fit(X_train_processed, y_train)

        y_train_pred = model.predict(X_train_processed)
        y_test_pred = model.predict(X_test_processed)

        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)

        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)

        # Логування параметрів
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)

        # Логування метрик
        mlflow.log_metric("train_rmse", train_rmse)
        mlflow.log_metric("test_rmse", test_rmse)
        mlflow.log_metric("train_mae", train_mae)
        mlflow.log_metric("test_mae", test_mae)
        mlflow.log_metric("train_r2", train_r2)
        mlflow.log_metric("test_r2", test_r2)

        # 7. Візуалізація Feature Importance (Крок 5.2)
        plt.figure(figsize=(10, 8))
        importances = model.feature_importances_
        indices = np.argsort(importances)[-10:]  # Топ-10 ознак
        plt.barh(range(len(indices)), importances[indices], align='center')
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.title("Top 10 Feature Importances")

        plot_name = "feature_importance.png"
        plt.savefig(plot_name)
        mlflow.log_artifact(plot_name)
        plt.close()

        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=y_test, y=y_test_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Salary')
        plt.ylabel('Predicted Salary')
        plt.title('Predicted vs Actual')

        plot_path = "predicted_vs_actual.png"
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path)
        plt.close()

        # 8. Логування моделі
        # 8a. Збереження моделі для DVC
        os.makedirs(args.output_dir, exist_ok=True)
        model_path = os.path.join(args.output_dir, "random_forest_model.pkl")
        joblib.dump(model, model_path)

        # 8b. Логування моделі в MLflow
        mlflow.sklearn.log_model(model, "random_forest_model")

        print(f"Run completed. Test RMSE: {test_rmse:.2f}, Test MAE: {test_mae:.2f}, R2: {test_r2:.2f}")


if __name__ == "__main__":
    main()
