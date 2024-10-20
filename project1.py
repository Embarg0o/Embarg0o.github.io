# Цель - создать модель машинного обучения, которая будет классифицировать отзывы к фильмам на положительные и отрицательные.

# Загрузка файла с помощью wget
!wget https://storage.yandexcloud.net/storage32/imdb_master.csv

import pandas as pd

# Загрузка данных
data = pd.read_csv('imdb_master.csv', encoding='ISO-8859-1')

# Просмотр первых 5 записей
print(data.head())

# Просмотр информации о данных
print(data.info())

# Препроцессинг данных
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Инициализация лемматизатора и стоп-слов
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Функция для препроцессинга текста
def preprocess(text):
    # Очистка текста от специальных символов
    text = re.sub(r'\W+','', text)

    # Приведение текста к нижнему регистру
    text = text.lower()

    # Токенизация текста
    tokens = nltk.word_tokenize(text)

    # Удаление стоп-слов и лемматизация токенов
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]

    return''.join(tokens)

# Применение препроцессинга к столбцу с текстом
data['review'] = data['review'].apply(lambda x: preprocess(x))

# Разделение на обучающую и тестовую выборки
from sklearn.model_selection import train_test_split

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(data['review'], data['label'], test_size=0.2, random_state=42)

# Векторизация текста
from sklearn.feature_extraction.text import TfidfVectorizer

# Создание экземпляра векторизатора TF-IDF
vectorizer = TfidfVectorizer()

# Преобразование обучающих данных в числовые векторы
X_train_vectorized = vectorizer.fit_transform(X_train)

# Преобразование тестовых данных в числовые векторы
X_test_vectorized = vectorizer.transform(X_test)

# Обучение модели
from sklearn.linear_model import LogisticRegression

# Создание экземпляра модели логистической регрессии
model = LogisticRegression(max_iter=1000)

# Обучение модели на обучающих данных
model.fit(X_train_vectorized, y_train)

# Оценка модели
from sklearn.metrics import accuracy_score

# Предсказание меток классов для тестовых данных
y_pred = model.predict(X_test_vectorized)

# Оценка производительности модели
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Визуализация 1: Распределение положительных и отрицательных отзывов
import matplotlib.pyplot as plt

# Подсчет количества положительных и отрицательных отзывов
positive_reviews = data[data['label'] == 'pos']
negative_reviews = data[data['label'] == 'neg']

num_positive_reviews = len(positive_reviews)
num_negative_reviews = len(negative_reviews)

# Построение круговой диаграммы
labels = ['Positive', 'Negative']
sizes = [num_positive_reviews, num_negative_reviews]
colors = ['#ff9999','#66b3ff']
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
plt.axis('equal')
plt.title('Distribution of Positive and Negative Reviews')
plt.show()

# Визуализация 2: Важность признаков

import numpy as np

# Получение списка признаков
features = vectorizer.get_feature_names_out()

# Получение коэффициентов логистической регрессии
coefficients = model.coef_[0]

# Отсортировать признаки по их важности
sorted_features = [ feature for _, feature in sorted(zip(coefficients, features))]

# Построение графика баров
plt.figure(figsize=(12, 6))
plt.bar(range(len(sorted_features[:20])), sorted(coefficients[:20]), align='center')
plt.xticks(range(len(sorted_features[:20])), sorted_features[:20], rotation='vertical')
plt.xlabel('Features')
plt.ylabel('Coefficient')
plt.title('Top 20 Important Features')
plt.tight_layout()
plt.xticks(rotation=35)
plt.show()

# Визуализация 3: Матрица ошибок

from sklearn.metrics import confusion_matrix
import seaborn as sns

# Построение матрицы ошибок
confusion = confusion_matrix(y_test, y_pred)

# Визуализация тепловой карты
sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues')

plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Визуализация 4: Кривая обучения

from sklearn.model_selection import learning_curve
import numpy as np

# Вычисление кривой обучения
train_sizes, train_scores, test_scores = learning_curve(model, X_train_vectorized, y_train, cv=5)
train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)

# Построение графика зависимости производительности от размера обучающей выборки
plt.figure(figsize=(8, 6))
plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training score')
plt.plot(train_sizes, test_scores_mean, 'o-', color='g', label='Cross-validation score')
plt.xlabel('Training Examples')
plt.ylabel('Score')
plt.title('Learning Curve')
plt.legend()
plt.show()


