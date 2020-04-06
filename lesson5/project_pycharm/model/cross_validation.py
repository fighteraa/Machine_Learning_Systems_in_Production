from sklearn.model_selection import KFold, RandomizedSearchCV

params =  {'n_estimators': [100, 150, 200, 250],
            'learning_rate': [0.01, 0.05, 0.1, 0.15],
            'subsample': [0.3, 0.5, 0.7, 1.],
            'max_depth': [3, 4, 5, 6],
            'colsample_bytree': [0.45, 0.47, 0.5],
            'min_child_weight': [1, 2, 3]
           }

cv = KFold(n_splits=3, random_state=21, shuffle=True)
rs = RandomizedSearchCV(clf, params, n_iter=10, scoring='f1', cv=cv, n_jobs=-1)
rs.fit(X_train_balanced, y_train_balanced, eval_metric='aucpr', verbose=10)