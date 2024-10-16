# Обнаружение фальшивых новостей
# Нужно используя библиотеку sklearn построить модель классического машинного обучения, 
# которая может с высокой точностью более 90% определять, является ли новость реальной (REAL） или фальшивой（FAKE).

!pip install scikit-learn

import pandas as pd

# Загрузка файла с помощью wget
!wget https://storage.yandexcloud.net/storage32/fake_news.csv

# Начнем с загрузки данных и предобработки:
df = pd.read_csv(f'/content/fake_news.csv', error_bad_lines=False) # создадим датафрейм из файла метаданных фильмов:
df.shape
df.head()

# Установка меток классов
y = df['label']

# Проверка наличия пропущенных значений
df.isnull().sum()

# Удаление нерелевантных столбцов
df.drop(['Unnamed: 0', 'title'], axis=1, inplace=True)

# Разделение данных на обучающую и тестовую выборки
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df['text'], y, test_size=0.2, random_state = 0)

# Далее, векторизуем текстовые данные используя TfidfVectorizer. 
# TfidfVectorizer создает матрицу TF-IDF признаков на основе входных текстовых данных

from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Инициализация модели классификации
pac = PassiveAggressiveClassifier()

# Обучение модели на обучающих данных
pac.fit(tfidf_train, y_train)

# Предсказание на тестовых данных
y_pred = pac.predict(tfidf_test)

# Узнаем точность предсказания
score = accuracy_score(y_test,y_pred)
print(f'Точность: {round(score*100,2)}%')

# С помощью этой модели мы получили точность более 90%.(благодаря параметку random_state в train_test_split точность будет немного плавующая)

# Оценим качество модели с помощью confusion matrix:
from sklearn.metrics import confusion_matrix

# Получение матрицы ошибок
confusion = confusion_matrix(y_test, y_pred)

print(confusion)

# Визуализация результатов с использованием матрицы ошибок:
import matplotlib.pyplot as plt
import seaborn as sns

# Визуализация матрицы ошибок
plt.figure(figsize=(8,6))
sns.heatmap(confusion, annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Подсчет числа правильно и неправильно классифицированных новостей
correct = (y_pred == y_test).sum()
incorrect = (y_pred != y_test).sum()

# Визуализация доли правильно и неправильно классифицированных новостей
plt.figure(figsize=(8,6))
plt.bar(["Correct", "Incorrect"], [correct, incorrect])
plt.title("Accuracy")
plt.xlabel("Prediction")
plt.ylabel("Count")
plt.show()