import json
import pandas as pd
import os


def test_data_schema():
    """Перевірка структури та якості вхідних даних."""
    x_train_path = "data/prepared/X_train.csv"
    y_train_path = "data/prepared/y_train.csv"

    assert os.path.exists(x_train_path), f"Файл не знайдено: {x_train_path}"
    assert os.path.exists(y_train_path), f"Файл не знайдено: {y_train_path}"

    X = pd.read_csv(x_train_path)
    y = pd.read_csv(y_train_path)

    assert X.shape[0] >= 50, "Занадто мало даних для навчання (менше 50 рядків)"

    assert y.iloc[:, 0].notna().all(), "Цільова змінна містить порожні значення (NaN)"

    assert (
        X.shape[0] == y.shape[0]
    ), "Кількість ознак не відповідає кількості цільових значень"


def test_artifacts_exist():
    """Перевірка наявності створених артефактів."""
    artifacts = ["models/best_model.pkl", "metrics.json", "confusion_matrix.png"]
    for artifact in artifacts:

        assert os.path.exists(
            artifact
        ), f"Артефакт {artifact} не знайдено після тренування"


def test_quality_gate_r2():
    """Quality Gate: перевірка, чи метрика R^2 не впала нижче порогу."""
    # Встановлюємо поріг через змінну оточення або 0.80 за замовчуванням
    threshold = float(os.getenv("R2_THRESHOLD", "0.80"))

    with open("metrics.json", "r", encoding="utf-8") as f:
        metrics = json.load(f)

    r2 = float(metrics.get("r2", 0))

    assert (
        r2 >= threshold
    ), f"Quality Gate не пройдено: R^2 ({r2:.4f}) нижче за поріг ({threshold:.2f})"
