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