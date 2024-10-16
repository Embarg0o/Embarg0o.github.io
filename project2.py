# Обнаружение болезни паркинсона с помощью XGBoost

# Загрузка файла с помощью wget
!wget https://storage.yandexcloud.net/storage32/parkinsons.data

# Импорт необходимых библиотек 

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv('/content/parkinsons.data')

# Разделим данные на признаки (X) и метки (y):
X = data.drop(['name', 'status'], axis=1)
y = data['status']

# Признак 'name' не несет информации о заболевании, поэтому исключим его из признаков. Метка 'status' указывает на наличие/отсутствие заболевания, поэтому будет нашей целевой переменной.
# Разделим данные на обучающую и тестовую выборки:

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # разделили данные на 80% для обучения и 20% для тестирования

scaler = StandardScaler() # Стандартизируем признаки с помощью StandardScaler
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = XGBClassifier() # Cоздадим модель XGBoost и обучим ее на обучающей выборке
model.fit(X_train_scaled, y_train)

# Оценим точность модели на тестовой выборке

y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
accuracy_percent = round(accuracy * 100)
print("Точность модели: {}%".format(accuracy_percent))

# Визуализация распределения классов меток

import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
data['status'].value_counts().plot(kind='bar')
plt.title("Distribution of Parkinson's Disease")
plt.xlabel("Status")
plt.ylabel("Count")
plt.show()

# Визуализация важности признаков

importances = pd.Series(model.feature_importances_, index=X.columns)
plt.figure(figsize=(10, 6))
importances.sort_values().plot(kind='barh')
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()