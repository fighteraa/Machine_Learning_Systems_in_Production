def feature_seection(X_train_balanced, y_train_balanced):
    perm = PermutationImportance(fitted_clf, random_state=42).fit(X_train_balanced, y_train_balanced)

    res = pd.DataFrame(X.columns, columns=['feature'])
    res['score'] = perm.feature_importances_
    res['std'] = perm.feature_importances_std_
    res = res.sort_values(by='score', ascending=False).reset_index(drop=True)
    return res