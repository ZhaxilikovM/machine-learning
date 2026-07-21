import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # TotalCharges иногда содержит пробелы вместо чисел (у клиентов с tenure=0)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0)

    # customerID не несёт предсказательной ценности
    df = df.drop(columns=['customerID'])

    # Таргет в бинарный формат
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    return df
