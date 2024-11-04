import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as skl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import lightgbm as lgbm

pd.set_option('display.max_columns',1000)
pd.set_option('display.max_rows',1000)

df = pd.read_csv('df_bp.csv', low_memory=False)
df.drop(['completion_info', 'forfeit_info', 'protest_info', 'ump_LF_id', 'misc_info',
         'ump_LF_name', 'ump_RF_id', 'ump_RF_name', 'pitcher_id_s', 'pitcher_name_s'],
        axis=1, inplace=True)
df.dropna(inplace=True)

hv_mean = df.home_victory.mean()

df = df[df.run_diff != 0]
df_train = df[(df.season >= 1980) & (df.season <= 2018)]
df_valid = df[(df.season == (2019 | 2020))]
df_test = df[df.season >= 2021]


features = ['OBP_162_h', 'OBP_162_v',
            'SLG_162_h', 'SLG_162_v',
            'OBP_30_h', 'OBP_30_v',
            'SLG_30_h', 'SLG_30_v',
            'game_no_h'
            ]
target = 'home_victory'

X_train = df_train.loc[:,features]
X_valid = df_valid.loc[:,features]
X_test = df_test.loc[:,features]

y_train = df_train[target].to_numpy()
y_valid = df_valid[target].to_numpy()
y_test = df_test[target].to_numpy()

lgbm1 = lgbm.LGBMClassifier(n_estimators= 1000, learning_rate= 0.02, max_depth=3)
lgbm1.fit(X_train, y_train, eval_set=(X_valid, y_valid), eval_metric='logloss',
          callbacks=[lgbm.early_stopping(stopping_rounds=50), lgbm.log_evaluation(10)])

preds_lgbm = lgbm1.predict_proba(X_test)[:,1]
accuracty_lgbm = accuracy_score(y_test, preds_lgbm)

model = LogisticRegression(max_iter=200)\

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy_skl = accuracy_score(y_test, y_pred)
classification_report = classification_report(y_test, y_pred)

print(f'SKL Accuracy: {accuracy_skl}')
print(f'SKL Classification Report: {classification_report}')
print(f'LightGBM accuracy: {accuracy_lgbm}')
