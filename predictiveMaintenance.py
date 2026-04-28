import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
#from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from xgboost import XGBClassifier

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

rate = len(y_train[y_train==0]) / len(y_train[y_train==1]) #sağlam/arızalı
print(rate)

model = XGBClassifier(scale_pos_weight = rate, random_state = 42)
model.fit(x_train, y_train)
print(model.score(x_test, y_test))

y_pred = model.predict(x_test)

print("\nKarmaşıklık Matrisi\n")
print(confusion_matrix(y_test, y_pred))
print("\nSınıflandırma Raporu\n")
print(classification_report(y_test, y_pred))












