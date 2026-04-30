import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
#from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns #matplotlib tabanli daha modern
import joblib

pd.set_option("display.max_columns", None)

df = pd.read_csv("predictive_maintenance.csv")
print(df.info())

le = LabelEncoder()
df["device"] = le.fit_transform(df["device"])

df["date"] = pd.to_datetime(df["date"])
df["day"] = df["date"].dt.day
df["month"] = df["date"].dt.month
df["day_of_week"] = df["date"].dt.dayofweek
df = df.drop(columns=["date"])

print(df.head())
print(df["failure"].value_counts())

y = df["failure"]
x = df.drop(columns=["failure"])

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.88, random_state=42, stratify=y)

#model = LogisticRegression()
#model.fit(x_train, y_train)
#print(model.score(x_test, y_test)) #Accuracy Paradox, zaten verilerin %90'ı 0 sınıfında. O yüzden model tembellik yaptı ve oran 0.99 çıktı

#y_pred = model.predict(x_test)
#cm = confusion_matrix(y_test, y_pred)
#print(cm)

#rate = len(y_train[y_train==0]) / len(y_train[y_train==1]) #sağlam/arızalı
#print(rate)

#model = XGBClassifier(scale_pos_weight = rate, random_state = 42)
#model.fit(x_train, y_train)
#print(model.score(x_test, y_test))

#y_pred = model.predict(x_test)

#print("\nKarmaşıklık Matrisi\n")
#print(confusion_matrix(y_test, y_pred))
#print("\nSınıflandırma Raporu\n")
#print(classification_report(y_test, y_pred))

smote = SMOTE(random_state=42)

print("\nSMOTE ÖNCESİ\n")
print("Arıza: ", sum(y_train == 1))
print("\nSağlam: ", sum(y_train == 0))

x_train_smote, y_train_smote = smote.fit_resample(x_train, y_train)

print("\nSMOTE SONRASI\n")
print("Arıza: ", sum(y_train_smote == 1))
print("\nSağlam: ", sum(y_train_smote == 0))

"""model = XGBClassifier(random_state=42)
model.fit(x_train_smote, y_train_smote)

y_pred = model.predict(x_test)

print("\nKarmaşıklık Matrisi\n")
print(confusion_matrix(y_test, y_pred))
print("\nSınıflandırma Raporu\n")
print(classification_report(y_test, y_pred))"""

parameter_grid = {
    "max_depth": [2, 3, 4],
    "learning_rate":[0.1],
    "n_estimators": [120],
    "min_child_weight": [1, 3],
    "gamma": [0, 0.1],
    "subsample": [0.8]
}

base_model = XGBClassifier(random_state = 42)
grid_search = GridSearchCV(estimator = base_model,
                           param_grid = parameter_grid, 
                           cv = 3, #çapraz doğrulama
                           verbose = 2, #raporla
                           n_jobs = -1 #tüm çekirdekleri kullan
)

grid_search.fit(x_train_smote, y_train_smote)
print("\nBulunan en iyi ayar: ", grid_search.best_params_)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(x_test)

print("\nKarmaşıklık Matrisi\n")
print(confusion_matrix(y_test, y_pred))

print("\nSınıflandırma Raporu\n")
print(classification_report(y_test, y_pred))

y_pred_possibilities = best_model.predict_proba(x_test)[:, 1] #ilk örn olasılıkları, proba ihtimalleri söyler

new_threshold_value = 0.30
y_pred_parano = (y_pred_possibilities >= new_threshold_value).astype(int)

print(f"\nYeni eşik ({new_threshold_value} ile Karmaşıklık Matrisi\n)")
print(confusion_matrix(y_test, y_pred_parano))

print(f"\nYeni eşik ({new_threshold_value} ile Sınıflandırma Raporu\n)")
print(classification_report(y_test, y_pred_parano))

priorities = best_model.feature_importances_
column_names = x_test.columns

priorities_table = pd.DataFrame({"Özellik": column_names, "Önem": priorities})
priorities_table = priorities_table.sort_values(by = "Önem", ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x = "Önem", y = "Özellik", data = priorities_table, palette="viridis")
plt.title("Özellik Önem Düzeyleri")
plt.xlabel("Önem Skoru")
plt.ylabel("Özellikler")
plt.tight_layout()
plt.show()

joblib.dump(best_model, "predictive_model.pkl")











