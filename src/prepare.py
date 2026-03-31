import pandas as pd
import sys
import os
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def main():
    input_file = sys.argv[1]  # data/raw/global_graduate_employability_index.csv
    output_dir = sys.argv[2]  # data/prepared
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(input_file)

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

    # Збереження оброблених даних
    pd.DataFrame(X_train_processed, columns=preprocessor.get_feature_names_out()).to_csv(
        os.path.join(output_dir, "X_train.csv"), index=False)
    pd.DataFrame(X_test_processed, columns=preprocessor.get_feature_names_out()).to_csv(
        os.path.join(output_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)


if __name__ == "__main__":
    main()
