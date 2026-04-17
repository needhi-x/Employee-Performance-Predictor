import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess(df):
    df = df.dropna()

    le = LabelEncoder()
    df['department'] = le.fit_transform(df['department'])
    df['performance'] = le.fit_transform(df['performance'])

    X = df.drop(['performance', 'performance_score'], axis=1)
    y = df['performance']

    return X, y