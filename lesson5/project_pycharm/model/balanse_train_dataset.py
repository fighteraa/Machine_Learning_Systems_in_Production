from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

def balanse_dataset(X, y):
    """
    Balance dataset
    """
    X = dataset.drop(['user_id', 'is_churned'], axis=1)
    y = dataset['is_churned']

    scaler = StandardScaler()
    X_mm = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_mm,
                                                        y,
                                                        test_size=0.3,
                                                        shuffle=True,
                                                        stratify=y,
                                                        random_state=100)

    # Снизим дизбаланс классов
    X_train_balanced, y_train_balanced = SMOTE(random_state=42, ratio=0.3).fit_sample(X_train, y_train)
    return X_train_balanced, y_train_balanced, X_test, y_test