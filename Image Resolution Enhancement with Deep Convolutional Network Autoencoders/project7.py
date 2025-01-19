# Тема: "Повышение разрешения изображений с помощью автокодировщиков с глубокими сверточными сетями"
# Установка и импорт необходимых библиотек
!pip install tensorflow==2.15.0
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
import requests

# Подготовка данных
# Загрузка датасета (DIV2K)
!wget https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip
# Распаковка архива
!unzip DIV2K_train_HR.zip -d ./DIV2K_train_HR

# Предобработка
# Функция load_images загружает и предобрабатывает изображения из заданного каталога, а затем возвращает их в виде массива NumPy

# Загрузка и предобработка данных
def load_images(path, size=(128, 128, 3)):
    images = []                                             # Создает пустой список для хранения изображений
    for filename in os.listdir(path):                       # Цикл for перебирает все файлы в указанном каталоге, заданном аргументом path
        img_path = os.path.join(path, filename)             # Создает полный путь к файлу изображения, объединяя путь к каталогу и имя файла
        if os.path.isfile(img_path):                        # Проверяет, является ли img_path обычным файлом
            img = cv2.imread(img_path)                      # Загружает изображение с помощью функции cv2.imread() и сохраняет его в переменной img
            if img is not None:                             # Проверяет, было ли изображение успешно загружено
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Конвертирует изображение из формата BGR (синий, зеленый, красный) в формат RGB (красный, зеленый, синий)
                                                            # с помощью функции cv2.cvtColor(). Это необходимо, поскольку OpenCV по умолчанию использует формат BGR для изображений
                img = cv2.resize(img, (size[0], size[1]))   # Изменяет размер изображения до заданного размера (size[0], size[1]) с помощью функции cv2.resize().
                                                            # Аргумент size - это кортеж, содержащий ширину и высоту, на которые нужно изменить размер изображения
                images.append(img)                          # Добавляет измененное изображение в список images
    return np.array(images)

# Функция augment_images выполняет расширение набора изображений, применяя различные преобразования к исходным изображениям, и возвращает массив расширенных изображений
def augment_images(images):
    augmented_images = []                          # Создаем пустой список для хранения расширенных изображений
    for img in images:                             # Цикл for перебирает каждый исходный образ в списке images
        augmented_images.append(img)               # Добавляем исходное изображение в список augmented_images
        augmented_images.append(cv2.flip(img, 1))  # Применяем горизонтальное отражение к изображению с помощью функции cv2.flip()
                                                   # с аргументом 1 и добавляем отраженное изображение в список augmented_images
        augmented_images.append(cv2.flip(img, 0))  # Применяем вертикальное отражение к изображению с помощью функции cv2.flip()
                                                   # с аргументом 0 и добавляем отраженное изображение в список augmented_images
        augmented_images.append(cv2.flip(cv2.flip(img, 1), 0))  # Сначала применяем горизонтальное отражение к изображению, затем вертикальное отражение
                                                   # к полученному отраженному изображению и добавляем результат в список augmented_images
        augmented_images.append(cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE))  # Поворачиваем изображение на 90 градусов по часовой стрелке с помощью функции cv2.rotate()
                                                   # и добавляем повернутое изображение в список augmented_images
        augmented_images.append(cv2.rotate(img, cv2.ROTATE_180))  # Поворачиваем изображение на 180 градусов с помощью функции cv2.rotate()
                                                   # и добавляем повернутое изображение в список augmented_images
        augmented_images.append(cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE))  # Поворачиваем изображение на 90 градусов против часовой стрелки
                                                   # с помощью функции cv2.rotate() и добавляем повернутое изображение в список augmented_images
    return np.array(augmented_images)

# Загрузка и предобработка данных
hr_images = load_images('/content/DIV2K_train_HR/DIV2K_train_HR') # Загружает изображения высокого разрешения из указанного каталога с помощью функции `load_images`
hr_images = augment_images(hr_images)                             # Расширяет набор изображений высокого разрешения с помощью функции `augment_images`
lr_images = [cv2.resize(img, (32, 32)) for img in hr_images]      # Создает список изображений низкого разрешения, сжимая каждое изображение высокого разрешения до размера 32x32
hr_images = np.array(hr_images) / 255.0                           # Нормализует значения пикселей изображений высокого разрешения, делая их в диапазоне [0, 1]
lr_images = np.array(lr_images) / 255.0 

# Проверка загруженных данных
print(f"Количество изображений в высоком разрешении: {len(hr_images)}")
print(f"Количество изображений в низком разрешении: {len(lr_images)}")

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(lr_images, hr_images, test_size=0.2, random_state=42)

# Модель UNET, которую мы будем использовать, ожидает входные данные размером 64x64 пикселей. Изменение размера 
# входных данных до 64x64 пикселей гарантирует, что данные будут совместимы с архитектурой модели.

# Resize входных данных до 64x64
X_train = tf.image.resize(X_train, (64, 64))
X_test = tf.image.resize(X_test, (64, 64))
y_train = tf.image.resize(y_train, (64, 64))
y_test = tf.image.resize(y_test, (64, 64))

# Проверка размеров данных
print(f"Размер входных данных (обучающая выборка): {X_train.shape}")
print(f"Размер выходных данных (обучающая выборка): {y_train.shape}")
print(f"Размер входных данных (тестовая выборка): {X_test.shape}")
print(f"Размер выходных данных (тестовая выборка): {y_test.shape}")

# Определение архитектуры UNET для повышения разрешения изображений.
def build_unet(input_shape):
    inputs = layers.Input(shape=input_shape) # Создает входной слой с заданным размером input_shape

    # Encoder
    # Первый блок
    conv1 = layers.Conv2D(64, (3, 3), activation='leaky_relu', padding='same')(inputs) # Применяет сверточный слой с 64 фильтрами, размером ядра 3x3, функцией активации Leaky ReLU
                                                                                       # и поддержкой padding, чтобы размер изображения оставался неизменным.
    conv1 = layers.Conv2D(64, (3, 3), activation='leaky_relu', padding='same')(conv1)  # Применяет второй сверточный слой с 64 фильтрами
    pool1 = layers.MaxPooling2D((2, 2))(conv1) # Применяет слой максимального пулинга с размером окна 2x2, уменьшая размер изображения в 2 раза по каждой оси

    # Второй блок
    conv2 = layers.Conv2D(128, (3, 3), activation='leaky_relu', padding='same')(pool1) # Применяет сверточный слой с 128 фильтрами
    conv2 = layers.Conv2D(128, (3, 3), activation='leaky_relu', padding='same')(conv2) # Применяет второй сверточный слой с 128 фильтрами
    pool2 = layers.MaxPooling2D((2, 2))(conv2) # Применяет слой максимального пулинга

    # Третий блок
    conv3 = layers.Conv2D(256, (3, 3), activation='leaky_relu', padding='same')(pool2) # Применяет сверточный слой с 256 фильтрами
    conv3 = layers.Conv2D(256, (3, 3), activation='leaky_relu', padding='same')(conv3) # Применяет второй сверточный слой с 256 фильтрами
    pool3 = layers.MaxPooling2D((2, 2))(conv3) # Применяет слой максимального пулинга

    # Четвертый блок
    conv4 = layers.Conv2D(512, (3, 3), activation='leaky_relu', padding='same')(pool3) # Применяет сверточный слой с 512 фильтрами
    conv4 = layers.Conv2D(512, (3, 3), activation='leaky_relu', padding='same')(conv4) # Применяет второй сверточный слой с 512 фильтрами

    # Decoder
    # Первый блок
    up1 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), activation='leaky_relu')(conv4) # Применяет транспонированный сверточный слой (сверточный слой с upsampling)
                                                                                              # с 256 фильтрами, увеличивая размер изображения в 2 раза
    merge1 = layers.Concatenate()([up1, conv3]) # Объединяет результат транспонированного слоя с соответствующим слоем из энкодера
    conv5 = layers.Conv2D(256, (3, 3), activation='leaky_relu', padding='same')(merge1) # Применяет сверточный слой с 256 фильтрами
    conv5 = layers.Conv2D(256, (3, 3), activation='leaky_relu', padding='same')(conv5)  # Применяет второй сверточный слой с 256 фильтрами
    conv5 = layers.Dropout(0.5)(conv5) #  Применяет слой dropout с вероятностью

    # Второй блок
    up2 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), activation='leaky_relu')(conv5) # Применяет транспонированный сверточный слой с 128 фильтрами
    merge2 = layers.Concatenate()([up2, conv2]) # Объединяет результат транспонированного слоя с соответствующим слоем из энкодера.
    conv6 = layers.Conv2D(128, (3, 3), activation='leaky_relu', padding='same')(merge2) # Применяет сверточный слой с 128 фильтрами
    conv6 = layers.Conv2D(128, (3, 3), activation='leaky_relu', padding='same')(conv6)  # Применяет второй сверточный слой с 128 фильтрами
    conv6 = layers.Dropout(0.5)(conv6) # Применяет слой dropout с вероятностью 0.5

    # Третий блок
    up3 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), activation='relu')(conv6) # Применяет транспонированный сверточный слой с 64 фильтрами
    merge3 = layers.Concatenate()([up3, conv1]) # Объединяет результат транспонированного слоя с соответствующим слоем из энкодера
    conv7 = layers.Conv2D(64, (3, 3), activation='leaky_relu', padding='same')(merge3) # Применяет сверточный слой с 64 фильтрами
    conv7 = layers.Conv2D(64, (3, 3), activation='leaky_relu', padding='same')(conv7)  # Применяет второй сверточный слой с 64 фильтрами
    conv7 = layers.Dropout(0.5)(conv7) # Применяет слой dropout с вероятностью 0.5

    # Четвертый блок
    outputs = layers.Conv2D(3, (3, 3), activation='tanh', padding='same')(conv7) # Применяет сверточный слой с 3 фильтрами (по одному на каждый канал RGB) и
                                                                                 # функцией активации tanh, чтобы выходные значения были в диапазоне [-1, 1]

    model = models.Model(inputs=inputs, outputs=outputs) # Применяет сверточный слой с 3 фильтрами (по одному на каждый канал RGB) и функцией активации tanh, чтобы выходные значения были в диапазоне [-1, 1]
    return model

# Создание модели
unet_model = build_unet((64, 64, 3))
unet_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])

# Создание callback для сохранения чекпоинтов
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='unet_model_checkpoint.h5',
    save_weights_only=True,
    save_best_only=True,
    monitor='val_loss',
    mode='min',
    verbose=1
)

# Обучение модели с сохранением чекпоинтов
history = unet_model.fit(
    X_train,
    y_train,
    epochs=300,
    batch_size=256,
    validation_split=0.2,
    callbacks=[checkpoint_callback]
)

# Визуализация процесса обучения (метрики)
plt.plot(history.history['mse'], label='Тренировочная ошибка MSE')
plt.plot(history.history['val_mse'], label='Валидационная ошибка MSE')
plt.legend()
plt.show()

# Оценка результатов 
# Оценка качества на тестовой выборке
predictions = unet_model.predict(X_test)

# Денормализация предсказаний и тестовых данных
predictions = (predictions + 1.0) * 127.5
y_test_denorm = (y_test + 1.0) * 127.5
X_test_denorm = (X_test + 1.0) * 127.5

# Преобразование данных в numpy-массивы с типом np.float64
predictions = np.array(predictions, dtype=np.float64)
y_test_denorm = np.array(y_test_denorm, dtype=np.float64)

# Расчет метрик PSNR и SSIM
psnr_values = [psnr(y_test_denorm[i], predictions[i], data_range=255.0) for i in range(len(predictions))]
ssim_values = [ssim(y_test_denorm[i], predictions[i], multichannel=True, win_size=3, data_range=255.0) for i in range(len(predictions))]

print(f"Average PSNR: {np.mean(psnr_values)}")
print(f"Average SSIM: {np.mean(ssim_values)}")

# Визуальный анализ результатов
fig, axs = plt.subplots(3, 3, figsize=(10, 10))
indices = np.random.choice(len(X_test_denorm), 3, replace=False)
for i, idx in enumerate(indices):
    axs[i, 0].imshow((X_test_denorm[idx] / 255.0))
    axs[i, 0].set_title('Low-Resolution')
    axs[i, 0].axis('off')

    axs[i, 1].imshow((y_test_denorm[idx] / 255.0))
    axs[i, 1].set_title('High-Resolution')
    axs[i, 1].axis('off')

    axs[i, 2].imshow((predictions[idx] / 255.0))
    axs[i, 2].set_title('Predicted')
    axs[i, 2].axis('off')
plt.show()