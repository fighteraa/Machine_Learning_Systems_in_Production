from sklearn.preprocessing import StandardScaler

dataset_test = pd.read_csv('dataset/dataset_test.csv', sep=';')
def xgb_fit_predict_test(X_train, y_train, X_test):
    clf = xgb.XGBClassifier(max_depth=3,
                            n_estimators=100,
                            learning_rate=0.1,
                            nthread=5,
                            subsample=1.,
                            colsample_bytree=0.5,
                            min_child_weight = 3,
                            reg_alpha=1,
                            reg_lambda=1,
                            seed=42,
                            missing=1e10)

    clf.fit(X_train, y_train, eval_metric='aucpr', verbose=10)
    predict_proba_test = clf.predict_proba(X_test)
    predict_test = clf.predict(X_test)
    return predict_proba_test, predict_test

scaler = StandardScaler()
X_mm = scaler.fit_transform(X)

X_test_mm = scaler.transform(dataset_test[COL_NAME_TEST])

predict_proba_test, predict_test = xgb_fit_predict_test(X_train_balanced, y_train_balanced, X_test)
dataset_test['is_churned'] = predict_test

dataset_test.loc[:, ['user_id','is_churned']].to_csv('AAndoskin_predictions_churned.csv', index=None)