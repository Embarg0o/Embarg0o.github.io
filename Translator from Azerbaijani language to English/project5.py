1		-
# Цель: создать переводчик с азербаджанского азыка на английский.
2		-
3		-
## Установка и импорт необходимых библиотек
4		-


5		-
# Установим свежую версию TensorFlow для поддержки слоя `tf.keras.layers.MultiHeadAttention`.
6		-
!apt install --allow-change-held-packages libcudnn8=8.1.0.77-1+cuda11.2
7		-
!pip uninstall -y -q tensorflow keras tensorflow-estimator tensorflow-text
8		-
!pip install protobuf~=3.20.3
9		-
!pip install -q tensorflow_datasets
10		-
!pip install -q -U tensorflow-text tensorflow
11		-
12		-
# Логирование для отладки
13		-
import logging
14		-
15		-
# Замеры времени выполнения
16		-
import time
17		-
18		-
# Линейная алгебра
19		-
import numpy as np
20		-
21		-
# Вывод графиков
22		-
import matplotlib.pyplot as plt
23		-
24		-
# Фреймворк Tensorflow
25		-
import tensorflow_datasets as tfds
26		-
import tensorflow as tf
27		-
28		-
import tensorflow_text as text
29		-
# Регулярные выражения
30		-
import re
31		-
32		-
# Файловая система
33		-
import pathlib
34		-
35		-
# Токенизатор
36		-
from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab
37		-
38		-
# Отключим мешаюшие предупреждения
39		-
import warnings
40		-
warnings.filterwarnings("ignore")
41		-
42		-
## Загрузка датасета
43		-
44		-
examples, metadata = tfds.load('ted_hrlr_translate/az_to_en',
45		-
                               with_info=True,
46		-
                               as_supervised=True)
47		-
48		-
train_examples, val_examples = examples['train'], examples['validation']
49		-
50		-
print('Размер обучающей выборки: ', len(train_examples))
51		-
print('Размер валидационной выборки: ', len(val_examples))
52		-
53		-
# Возьмем из 3-х пакетов по одной записи:
54		-
55		-
for az_examples, en_examples in train_examples.batch(3).take(1):
56		-
  print('Примеры на азейбарджанском языке:')
57		-
  print()
58		-
  for az in az_examples.numpy():
59		-
    print(az.decode('utf-8'))
60		-
  print()
61		-
62		-
  print('Примеры на английском языке:')
63		-
  print()
64		-
  for en in en_examples.numpy():
65		-
    print(en.decode('utf-8'))
66		-
67		-
## Токенизация
68		-
69		-
### Создаем свой токенайзер на базе BERT
70		-
71		-
#### Создаем словари слов
72		-
73		-
VOCAB_SIZE = 8000
74		-
# Параметры токенизатора (lower_case - приводим к нижнему регистру)
75		-
bert_tokenizer_params=dict(lower_case=True)
76		-
77		-
# Определяем токены, с которыми работает токенизатор
78		-
# [START] - начало строки
79		-
# [END]   - конец строки
80		-
# [UNK]   - неизвестное слово
81		-
# [PAD]   - используется для выравнивания длин всех предложений
82		-
reserved_tokens=["[PAD]", "[UNK]", "[START]", "[END]"]
83		-
84		-
bert_vocab_args = dict(
85		-
    # Желаемый размер словаря
86		-
    vocab_size = VOCAB_SIZE,
87		-
    # Токены включаемые в словарь
88		-
    reserved_tokens=reserved_tokens,
89		-
    # Аргументы для `text.BertTokenizer`
90		-
    bert_tokenizer_params=bert_tokenizer_params,
91		-
    # Аргументы для `wordpiece_vocab.wordpiece_tokenizer_learner_lib.learn`
92		-
    learn_params={}, # не используем, но определяем чтобы не было ошибок
93		-
)
94		-
95		-
# Вспомагательные лямба функции, чтобы не писать лишние циклы:
96		-
97		-
train_en = train_examples.map(lambda az, en: en)
98		-
train_az = train_examples.map(lambda az, en: az)
99		-
100		-
# Создаем словарь азербаджанских слов из датасета (засечем время выполнения с помощью `%%time`):
101		-
102		-
%%time
103		-
az_vocab = bert_vocab.bert_vocab_from_dataset(
104		-
    train_az.batch(1000).prefetch(2),
105		-
    **bert_vocab_args
106		-
)
107		-
108		-
# Будем сохранять словари в файл:
109		-
110		-
def write_vocab_file(filepath, vocab):
111		-
  with open(filepath, 'w') as f:
112		-
    for token in vocab:
113		-
      print(token, file=f)
114		-
115		-
write_vocab_file('az_vocab.txt', az_vocab)
116		-
117		-
# Создаем словарь английских слов из датасета:
118		-
119		-
%%time
120		-
en_vocab = bert_vocab.bert_vocab_from_dataset(
121		-
    train_en.batch(1000).prefetch(2),
122		-
    **bert_vocab_args
123		-
)
124		-
125		-
write_vocab_file('en_vocab.txt', en_vocab)
126		-
127		-
#### Загрузка токенайзера из файла
128		-
129		-
az_tokenizer = text.BertTokenizer('az_vocab.txt', **bert_tokenizer_params)
130		-
en_tokenizer = text.BertTokenizer('en_vocab.txt', **bert_tokenizer_params)
131		-
132		-
# Прогоним примеры через `BertTokenizer.tokenize` метод. Он вернет нам объект `tf.RaggedTensor` с осями `(batch, word, word-piece)`:
133		-
134		-
# Токенизируем примеры и получаем тензор формы (batch, word, word-piece)
135		-
token_batch = en_tokenizer.tokenize(en_examples)
136		-
# Объединяем оси word и word-piece и получаем тензор формы (batch, tokens)
137		-
token_batch = token_batch.merge_dims(-2,-1)
138		-
139		-
print('Токенизируем отобранные строки:')
140		-
for ex in token_batch.to_list():
141		-
  print(ex)
142		-
143		-
# Для того, чтобы повторно собрать слова из извлеченных токенов, необходимо использовать BertTokenizer.detokenize метод:
144		-
145		-
words = en_tokenizer.detokenize(token_batch)
146		-
print('Проверим обратное преобразование:')
147		-
148		-
# Объединение полученного тензора в текст, объединяем пробелами
149		-
print(tf.strings.reduce_join(words, separator=' ', axis=-1))
150		-
151		-
#### Длина фраз в датасете
152		-
153		-
# Необходимо оценить как распределяются токены по примерам:
154		-
155		-
lengths = []
156		-
157		-
for az_examples, en_examples in train_examples.batch(1024):
158		-
  az_tokens = az_tokenizer.tokenize(az_examples)
159		-
  lengths.append(az_tokens.row_lengths())
160		-
161		-
  en_tokens = en_tokenizer.tokenize(en_examples)
162		-
  lengths.append(en_tokens.row_lengths())
163		-
164		-
# И отобразим на графике:
165		-
166		-
all_lengths = np.concatenate(lengths)
167		-
168		-
plt.hist(all_lengths, np.linspace(0, 500, 101))
169		-
plt.ylim(plt.ylim())
170		-
max_length = max(all_lengths)
171		-
plt.plot([max_length, max_length], plt.ylim())
172		-
plt.title(f'Максимальное количество токенов в примере: {max_length}');
173		-
174		-
MAX_TOKENS=128
175		-
176		-
#### Добавление токенов [START] и [END]
177		-
178		-
# С помощью `reserved_tokens` мы уже включили токены `[START]` и `[END]` в словарь. 
179		-
# Теперь необходимо добавить токены во все фразы, они имеют одинаковые индексы для обоих языков:
180		-
181		-
START = tf.argmax(tf.constant(reserved_tokens) == "[START]")
182		-
END = tf.argmax(tf.constant(reserved_tokens) == "[END]")
183		-
184		-
def add_start_end(ragged):
185		-
  count = ragged.bounding_shape()[0]
186		-
  starts = tf.fill([count,1], START)
187		-
  ends = tf.fill([count,1], END)
188		-
  return tf.concat([starts, ragged, ends], axis=1)
189		-
190		-
### Очистка текста при детокенизации
191		-
192		-
# Мы хотим, чтобы при детокенизации мы сразу получали строку, а не тензоры, 
193		-
# а также чтобы выходные строки не содержали зарезервированные токены. Для этого напишем вспомогательную функцию:
194		-
195		-
def cleanup_text(reserved_tokens, token_txt):
196		-
  # Удаление токенов, кроме "[UNK]".
197		-
  # Поиск зарезервированных токенов кроме [UNK]
198		-
  bad_tokens = [re.escape(tok) for tok in reserved_tokens if tok != "[UNK]"]
199		-
  # "Плохие" токены для регулярки объединяем знаком ИЛИ (|)
200		-
  bad_token_re = "|".join(bad_tokens)
201		-
202		-
  # Ищем в строке регулярку
203		-
  bad_cells = tf.strings.regex_full_match(token_txt, bad_token_re)
204		-
  # Отсеиваем из исходной строки все найденные включения "плохих" токенов
205		-
  result = tf.ragged.boolean_mask(token_txt, ~bad_cells)
206		-
207		-
  # Сцепление строк.
208		-
  result = tf.strings.reduce_join(result, separator=' ', axis=-1)
209		-
210		-
  return result
211		-
212		-
#### Создание кастомного токенизатора
213		-
214		-
# Создадим свой кастомный класс `CustomTokenizer` на базе `text.BertTokenizer`, 
215		-
# с дополнительной пользовательской логикой, и `@tf.function` обертками, необходимых для экспорта.
216		-
217		-
class CustomTokenizer(tf.Module):
218		-
  def __init__(self, reserved_tokens, vocab_path):
219		-
    # Определяем токенизатор
220		-
    self.tokenizer = text.BertTokenizer(vocab_path, lower_case=True)
221		-
    # Зарезервированные токены
222		-
    self._reserved_tokens = reserved_tokens
223		-
    # Путь к файлу словаря
224		-
    self._vocab_path = tf.saved_model.Asset(vocab_path)
225		-
    # Читаем из файла словарь и делим по строкам
226		-
    vocab = pathlib.Path(vocab_path).read_text().splitlines()
227		-
    self.vocab = tf.Variable(vocab)
228		-
229		-
    # Для экспорта класса необходимо создать так называемые сигнатуры,
230		-
    # чтобы tensorflow понимал с какими данными он работает
231		-
232		-
    # Сигнатура для tokenize (работает с пакетами строк).
233		-
    self.tokenize.get_concrete_function(
234		-
        tf.TensorSpec(shape=[None], dtype=tf.string))
235		-
236		-
    # Сигнатура для `detokenize` и `lookup`
237		-
    # Могут работать как с `Tensors`, так и `RaggedTensors`
238		-
    # с тензорами формы [batch, tokens]
239		-
    self.detokenize.get_concrete_function(
240		-
        tf.TensorSpec(shape=[None, None], dtype=tf.int64))
241		-
    self.detokenize.get_concrete_function(
242		-
          tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int64))
243		-
244		-
    self.lookup.get_concrete_function(
245		-
        tf.TensorSpec(shape=[None, None], dtype=tf.int64))
246		-
    self.lookup.get_concrete_function(
247		-
          tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int64))
248		-
249		-
    # Методы `get_*` не имеют аргументов
250		-
    self.get_vocab_size.get_concrete_function()
251		-
    self.get_vocab_path.get_concrete_function()
252		-
    self.get_reserved_tokens.get_concrete_function()
253		-
254		-
    # После определения сигнатур можно определить и сами методы класса
255		-
256		-
  @tf.function
257		-
  def tokenize(self, strings):
258		-
    enc = self.tokenizer.tokenize(strings)
259		-
    # Объединяем оси `word` и `word-piece` (как в примере выше)
260		-
    enc = enc.merge_dims(-2,-1)
261		-
    enc = add_start_end(enc)
262		-
    return enc
263		-
264		-
  @tf.function
265		-
  def detokenize(self, tokenized):
266		-
    words = self.tokenizer.detokenize(tokenized)
267		-
    return cleanup_text(self._reserved_tokens, words) # очищаем перед выводом
268		-
269		-
  @tf.function
270		-
  def lookup(self, token_ids):
271		-
    return tf.gather(self.vocab, token_ids) # возвращаем явное соответствие словаря токенам
272		-
273		-
  @tf.function
274		-
  def get_vocab_size(self):
275		-
    return tf.shape(self.vocab)[0] # определяем длину словаря по нулевому индексу формы
276		-
277		-
  @tf.function
278		-
  def get_vocab_path(self):
279		-
    return self._vocab_path # получение пути к файлу словаря
280		-
281		-
  @tf.function
282		-
  def get_reserved_tokens(self):
283		-
    return tf.constant(self._reserved_tokens) # получение списка зарезервированных токенов
284		-
285		-
tokenizers = tf.Module()
286		-
tokenizers.az = CustomTokenizer(reserved_tokens, 'az_vocab.txt')
287		-
tokenizers.en = CustomTokenizer(reserved_tokens, 'en_vocab.txt')
288		-
289		-
# Модуль сохраним и будем обращаться к нему, когда он нам понадобиться.
290		-
291		-
model_name = 'ted_hrlr_translate_az_en_converter'
292		-
tf.saved_model.save(tokenizers, model_name)
293		-
294		-
## Подготовка датасета, предобработка данных
295		-
296		-
# Определим вспомогательную функцию `prepare_batch` для подготовки датасета к виду, пригодному для использования в методе обучения `fit`:
297		-
298		-
299		-
def prepare_batch(az, en):
300		-
    az = tokenizers.az.tokenize(az)   # Токенизируем данные
301		-
    az = az[:, :MAX_TOKENS]           # Выравнивание данных по MAX_TOKENS.
302		-
    az = az.to_tensor()               # Преобразуем в тензор с равномерными измерениями
303		-
304		-
    en = tokenizers.en.tokenize(en)
305		-
    en = en[:, :(MAX_TOKENS+1)]
306		-
    en_inputs = en[:, :-1].to_tensor()  # Удаляем [END] токены (вход декодировщика)
307		-
    en_labels = en[:, 1:].to_tensor()   # Удаляем [START] токены (выход декодировщика)
308		-
309		-
    return (az, en_inputs), en_labels
310		-
311		-
# Определим константы для формирования пакетов:
312		-
313		-
# Размер буфера в памяти при подготовке датасета
314		-
BUFFER_SIZE = 20000
315		-
316		-
# Размер пакета
317		-
BATCH_SIZE = 64
318		-
319		-
# Прогоним фразы через подготовленную функцию и сформируем пакеты:
320		-
321		-
def make_batches(ds):
322		-
  return (
323		-
      ds
324		-
      .shuffle(BUFFER_SIZE)                     # перемешиваем данные
325		-
      .batch(BATCH_SIZE)                        # делим датасет на пакеты
326		-
      .map(prepare_batch, tf.data.AUTOTUNE)     # применим функцию prepare_batch
327		-
      .prefetch(buffer_size=tf.data.AUTOTUNE))  # prefetch используется для разделения времени, когда данные подготавливаются и потребляются, что ускоряет обучение сети
328		-
329		-
# Создаем пакеты для обучающей и валидационной выборок:
330		-
331		-
train_batches = make_batches(train_examples)
332		-
val_batches = make_batches(val_examples)
333		-
334		-
## Архитектура модели
335		-
336		-
### Позиционное кодирование и эмбеддинги
337		-
338		-
# Входы как для кодировщика, так и для декодировщика используют одну и ту же логику эмбеддингов и позиционного кодирования.
339		-
340		-
# length - порядковый номер слова в фразе
341		-
# depth - размер пространства эмбеддинга
342		-
def positional_encoding(length, depth):
343		-
  depth = depth/2
344		-
345		-
  positions = np.arange(length)[:, np.newaxis]     # форма (seq, 1)
346		-
  depths = np.arange(depth)[np.newaxis, :]/depth   # форма (1, depth)
347		-
348		-
  angle_rates = 1 / (10000**depths)         # форма (1, depth)
349		-
  angle_rads = positions * angle_rates      # форма (pos, depth)
350		-
351		-
  pos_encoding = np.concatenate(
352		-
      [np.sin(angle_rads), np.cos(angle_rads)],
353		-
      axis=-1)
354		-
355		-
  return tf.cast(pos_encoding, dtype=tf.float32)  # указываем тип возвращаемых данных
356		-
357		-
# Создадим слой `Position Embedding`
358		-
359		-
# Наследуем класс от tf.keras.layers.Layer
360		-
# Теперь наш слой тоже является классом Keras
361		-
class PositionalEmbedding(tf.keras.layers.Layer):
362		-
  def __init__(self, vocab_size, d_model):
363		-
    super().__init__()
364		-
    self.d_model = d_model
365		-
    self.embedding = tf.keras.layers.Embedding(vocab_size, d_model, mask_zero=True) # эмбеддинг
366		-
    self.pos_encoding = positional_encoding(length=2048, depth=d_model) # позиционное кодирование
367		-
368		-
  # Данный метод возвращает маску эмбеддинга
369		-
  # Так как вектора выравниваются до одной длины с помощью pad_sequences,
370		-
  # то метод вернет True для ненулевых токенов, и False для нулевых токенов
371		-
372		-
  def compute_mask(self, *args, **kwargs):
373		-
    return self.embedding.compute_mask(*args, **kwargs)
374		-
375		-
  def call(self, x):
376		-
    length = tf.shape(x)[1]
377		-
    x = self.embedding(x)
378		-
379		-
    # Этот коэффициент задает относительный масштаб встраивания и позиционного кодирования
380		-
    # C этим параметром можно и нужно играться!
381		-
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
382		-
    x = x + self.pos_encoding[tf.newaxis, :length, :]
383		-
    return x
384		-
385		-
### Базовый класс внимания
386		-
387		-
class BaseAttention(tf.keras.layers.Layer):
388		-
  def __init__(self, **kwargs):
389		-
    super().__init__()
390		-
    self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
391		-
    self.layernorm = tf.keras.layers.LayerNormalization()
392		-
    self.add = tf.keras.layers.Add()
393		-
394		-
### Encoder-Decoder Attention
395		-
396		-
class CrossAttention(BaseAttention):
397		-
  def call(self, x, context):
398		-
    # Пропускаем сигнал через многоголовое внимание
399		-
    attn_output, attn_scores = self.mha(
400		-
        query=x,                        # запрос
401		-
        key=context,                    # ключ
402		-
        value=context,                  # значение
403		-
        return_attention_scores=True)   # возвращаем оценки внимания
404		-
405		-
    # Запоминаем оценки на будущее
406		-
    self.last_attn_scores = attn_scores
407		-
408		-
    # Добавляем остаточную связь и нормализацию
409		-
    x = self.add([x, attn_output])
410		-
    x = self.layernorm(x)
411		-
412		-
    return x
413		-
414		-
### Слой GlobalSelfAttention
415		-
416		-
class GlobalSelfAttention(BaseAttention):
417		-
  def call(self, x):
418		-
    # Пропускаем сигнал через многоголовое внимание
419		-
    attn_output = self.mha(
420		-
        query=x,  # запрос
421		-
        value=x,  # ключ
422		-
        key=x)    # значение
423		-
424		-
    # Добавляем остаточную связь и нормализацию
425		-
    x = self.add([x, attn_output])
426		-
    x = self.layernorm(x)
427		-
    return x
428		-
429		-
### CausalSelfAttention слой (с причинно-следственной связью)
430		-
431		-
class CausalSelfAttention(BaseAttention):
432		-
  def call(self, x):
433		-
    attn_output = self.mha(
434		-
        query=x,
435		-
        value=x,
436		-
        key=x,
437		-
        use_causal_mask = True)  # отличается от GlobalSelfAttention одним аргументом
438		-
    x = self.add([x, attn_output])
439		-
    x = self.layernorm(x)
440		-
    return x
441		-
442		-
### Сеть прямого распространения (feed forward network)
443		-
444		-
# Сеть состоит из двух слоев `Dense` (Relu и линейной активации), а также слоем регуляризации `Dropout`. 
445		-
# Как и в случае со слоями внимания, добавляем также остаточную связь и нормализацию:
446		-
447		-
class FeedForward(tf.keras.layers.Layer):
448		-
  def __init__(self, d_model, dff, dropout_rate=0.1):
449		-
    super().__init__()
450		-
    self.seq = tf.keras.Sequential([
451		-
      tf.keras.layers.Dense(dff, activation='relu'),
452		-
      tf.keras.layers.Dense(d_model),
453		-
      tf.keras.layers.Dropout(dropout_rate)
454		-
    ])
455		-
    self.add = tf.keras.layers.Add()
456		-
    self.layer_norm = tf.keras.layers.LayerNormalization()
457		-
458		-
  def call(self, x):
459		-
    x = self.add([x, self.seq(x)])
460		-
    x = self.layer_norm(x)
461		-
    return x
462		-
463		-
## Собираем сеть целиком
464		-
465		-
### Слой кодировщика
466		-
467		-
# Определим один слой кодировщика `EncoderLayer`:
468		-
469		-
class EncoderLayer(tf.keras.layers.Layer):
470		-
  def __init__(self,*, d_model, num_heads, dff, dropout_rate=0.1):
471		-
    super().__init__()
472		-
473		-
    self.self_attention = GlobalSelfAttention(
474		-
        num_heads=num_heads,  # число голов
475		-
        key_dim=d_model,      # размерность ключа
476		-
        dropout=dropout_rate) # уровень регуляризации
477		-
478		-
    self.ffn = FeedForward(d_model, dff) # число нейронов во втором и первом Dense слое, соответственно
479		-
480		-
  def call(self, x):
481		-
    x = self.self_attention(x)
482		-
    x = self.ffn(x)
483		-
    return x
484		-
485		-
### Блок кодировщика
486		-
487		-
# Блок кодировщика состоит из входного `PositionalEmbedding` слоя на входе и стека из `EncoderLayer` слоев:
488		-
489		-
class Encoder(tf.keras.layers.Layer):
490		-
  def __init__(self, *, num_layers, d_model, num_heads,
491		-
               dff, vocab_size, dropout_rate=0.1):
492		-
    super().__init__()
493		-
494		-
    # Инициируем переменные внутри класса
495		-
    self.d_model = d_model
496		-
    self.num_layers = num_layers
497		-
498		-
    # Создаем объект класса позиционного кодирования
499		-
    self.pos_embedding = PositionalEmbedding(
500		-
        vocab_size=vocab_size, d_model=d_model)
501		-
502		-
    # Создаем объект класса для слоя кодировщика
503		-
    self.enc_layers = [
504		-
        EncoderLayer(d_model=d_model,
505		-
                     num_heads=num_heads,
506		-
                     dff=dff,
507		-
                     dropout_rate=dropout_rate)
508		-
        for _ in range(num_layers)]
509		-
510		-
    # Создаем объект класса для слоя регуляризации
511		-
    self.dropout = tf.keras.layers.Dropout(dropout_rate)
512		-
513		-
  def call(self, x):
514		-
    # Форма x токена: (batch, seq_len)
515		-
    # Прогоняем последовательность токенов через слой позиционного кодирования
516		-
    x = self.pos_embedding(x)  # форма на выходе (batch_size, seq_len, d_model)
517		-
518		-
    # Прогоняем последовательность токенов через слой регуляризации
519		-
    x = self.dropout(x)
520		-
521		-
    # Прогоняем последовательность токенов через num_layers слоев кодировщика
522		-
    for i in range(self.num_layers):
523		-
      x = self.enc_layers[i](x)
524		-
525		-
    return x  # форма на выходе (batch_size, seq_len, d_model)
526		-
527		-
### Слой декодировщика
528		-
529		-
# Стек декодировщиков немного сложнее, поскольку каждый слой декодировщика содержит слой внимания 
530		-
# с причинно-следственной связью `CausalSelfAttention`, кросс-внимания `CrossAttention` и `FeedForward` слой:
531		-
532		-
class DecoderLayer(tf.keras.layers.Layer):
533		-
  def __init__(self,
534		-
               *,
535		-
               d_model,
536		-
               num_heads,
537		-
               dff,
538		-
               dropout_rate=0.1):
539		-
    super(DecoderLayer, self).__init__()
540		-
541		-
    # Слой внимания с причинно-следственной связью
542		-
    self.causal_self_attention = CausalSelfAttention(
543		-
        num_heads=num_heads,
544		-
        key_dim=d_model,
545		-
        dropout=dropout_rate)
546		-
547		-
    # Слой с кросс-вниманием
548		-
    self.cross_attention = CrossAttention(
549		-
        num_heads=num_heads,
550		-
        key_dim=d_model,
551		-
        dropout=dropout_rate)
552		-
553		-
    # Слой прямого распространения
554		-
    self.ffn = FeedForward(d_model, dff)
555		-
556		-
  def call(self, x, context):
557		-
    # Пропускаем последовательность токенов через:
558		-
    # Каузальный слой внимания
559		-
    x = self.causal_self_attention(x=x)
560		-
    # Слой кросс-внимания и контекстным вектором из кодировщика
561		-
    x = self.cross_attention(x=x, context=context)
562		-
563		-
    # Запомним оценки внимания на будущее
564		-
    self.last_attn_scores = self.cross_attention.last_attn_scores
565		-
    # Через слой прямого распространения
566		-
    x = self.ffn(x)  # Форма `(batch_size, seq_len, d_model)`.
567		-
    return x
568		-
569		-
### Блок декодировщика
570		-
571		-
# Подобно кодировщику, декодировщик состоит из слоя позиционного встраивания `Positional Embedding` и стека слоев декодировщиков `DecoderLayer`:
572		-
573		-
# Определим декодировщик целиком как наследник класса слоя keras `tf.keras.layers.Layer`:
574		-
575		-
class Decoder(tf.keras.layers.Layer):
576		-
  def __init__(self, *, num_layers, d_model, num_heads, dff, vocab_size,
577		-
               dropout_rate=0.1):
578		-
    super(Decoder, self).__init__()
579		-
580		-
    # Инициируем переменные внутри класса
581		-
    self.d_model = d_model
582		-
    self.num_layers = num_layers
583		-
584		-
    # Создаем объект класса позиционного кодирования
585		-
    self.pos_embedding = PositionalEmbedding(vocab_size=vocab_size,
586		-
                                             d_model=d_model)
587		-
    # Создаем объект класса для слоя регуляризации
588		-
    self.dropout = tf.keras.layers.Dropout(dropout_rate)
589		-
590		-
    # Создаем сразу стек слоев декодировщиков с помощью генератора списков по числу слоев
591		-
    self.dec_layers = [
592		-
        DecoderLayer(d_model=d_model, num_heads=num_heads,
593		-
                     dff=dff, dropout_rate=dropout_rate)
594		-
        for _ in range(num_layers)]
595		-
596		-
    # Сбрасываем оценки внимания
597		-
    self.last_attn_scores = None
598		-
599		-
  def call(self, x, context):
600		-
    # Подаем на вход последовательность токенов x формой (batch, target_seq_len)
601		-
602		-
    # Пропускаем через слой позиционного кодирования (и конечно же эмбеддинг)
603		-
    x = self.pos_embedding(x)  # форма на выходе (batch_size, target_seq_len, d_model)
604		-
605		-
    # Регуляризация
606		-
    x = self.dropout(x)
607		-
608		-
    # Прогоняем через num_layers слоев декодировщиков
609		-
    for i in range(self.num_layers):
610		-
      x  = self.dec_layers[i](x, context)
611		-
612		-
    # Сохраняем оценки внимания из последнего слоя
613		-
    self.last_attn_scores = self.dec_layers[-1].last_attn_scores
614		-
615		-
    # Форма x на выходе (batch_size, target_seq_len, d_model)
616		-
    return x
617		-
618		-
## Трансформер
619		-
620		-
# Чтобы получить модель `Transformer` целиком, нам необходимо их соединить и добавить конечный `Dense` слой.
621		-
622		-
class Transformer(tf.keras.Model):
623		-
  def __init__(self, *, num_layers, d_model, num_heads, dff,
624		-
               input_vocab_size, target_vocab_size, dropout_rate=0.1):
625		-
    super().__init__()
626		-
    # Кодировщик
627		-
    self.encoder = Encoder(num_layers=num_layers, d_model=d_model,
628		-
                           num_heads=num_heads, dff=dff,
629		-
                           vocab_size=input_vocab_size,
630		-
                           dropout_rate=dropout_rate)
631		-
    # Декодировщик
632		-
    self.decoder = Decoder(num_layers=num_layers, d_model=d_model,
633		-
                           num_heads=num_heads, dff=dff,
634		-
                           vocab_size=target_vocab_size,
635		-
                           dropout_rate=dropout_rate)
636		-
    # Конечный слой
637		-
    self.final_layer = tf.keras.layers.Dense(target_vocab_size)
638		-
639		-
  def call(self, inputs):
640		-
    # Чтобы использовать метод `.fit` для обучения модели, необходимо передать
641		-
    # все входные данные в первом аргументе
642		-
    context, x  = inputs
643		-
644		-
    # Передаем контекст в кодировщик
645		-
    context = self.encoder(context)  # форма выходных данных (batch_size, context_len, d_model)
646		-
647		-
    # Передаем контекст и целевой вектор в декодировщик
648		-
    x = self.decoder(x, context)  # форма выходных данных (batch_size, target_len, d_model)
649		-
650		-
    # Прогоняем выходные данные через финальный слой
651		-
    logits = self.final_layer(x)  # форма выходных данных (batch_size, target_len, target_vocab_size)
652		-
653		-
    try:
654		-
      # После прохождения данных через все слои необходимо удалить
655		-
      # маску, чтобы она не масштабировала, потери и метрики
656		-
      # Обработчик ошибок позволяет избежать исключений при повторной попытке удаления
657		-
      del logits._keras_mask
658		-
    except AttributeError: # отлавливаем ошибку отсутствия аттрибута
659		-
      pass
660		-
661		-
    # Возвращаем наши логиты
662		-
    return logits
663		-
664		-
## Обучение модели
665		-
666		-
num_layers = 4
667		-
d_model = 128
668		-
dff = 512
669		-
num_heads = 8
670		-
dropout_rate = 0.1
671		-
672		-
EPOCHS = 20
673		-
674		-
transformer = Transformer(
675		-
    num_layers=num_layers,
676		-
    d_model=d_model,
677		-
    num_heads=num_heads,
678		-
    dff=dff,
679		-
    input_vocab_size=tokenizers.az.get_vocab_size().numpy(),
680		-
    target_vocab_size=tokenizers.en.get_vocab_size().numpy(),
681		-
    dropout_rate=dropout_rate)
682		-
683		-
### Оптимизатор
684		-
685		-
# Создадим класс CustomSchedule для создания собственных параметров оптимизатора:
686		-
687		-
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
688		-
  def __init__(self, d_model, warmup_steps=4000):
689		-
    super().__init__()
690		-
691		-
    self.d_model = d_model
692		-
    self.d_model = tf.cast(self.d_model, tf.float32)
693		-
694		-
    self.warmup_steps = warmup_steps
695		-
696		-
  def __call__(self, step):
697		-
    step = tf.cast(step, dtype=tf.float32)
698		-
    arg1 = tf.math.rsqrt(step)
699		-
    arg2 = step * (self.warmup_steps ** -1.5)
700		-
701		-
    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
702		-
703		-
learning_rate = CustomSchedule(d_model)
704		-
705		-
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
706		-
                                     epsilon=1e-9)
707		-
708		-
# Определим число шагов обучения на эпоху:
709		-
710		-
# Количество батчей для обучения
711		-
num_batches = 0
712		-
for (batch, (_,_)) in enumerate(train_batches):
713		-
  num_batches = batch
714		-
print(num_batches)
715		-
716		-
# Скорость обучения на каждом шагу при учете, что мы планируем обучать на 20 эпохах:
717		-
718		-
plt.plot(learning_rate(tf.range(num_batches*EPOCHS, dtype=tf.float32)))
719		-
plt.ylabel('Скорость обучения')
720		-
plt.xlabel('Шаг обучения')
721		-
722		-
### Функция потерь и метрики
723		-
724		-
# Поскольку целевые последовательности выровнены и заполнены нулями до конца последовательности, важно применять маску заполнения при расчете потерь.
725		-
# В качестве функции потерь мы будем применять разряженную категориальную кросс-энтропию `tf.keras.losses.SparseCategoricalCrossentropy`:
726		-
727		-
# Функция потерь с учетом маски
728		-
def masked_loss(label, pred):
729		-
  # Задаем маску, где метки не равны 0
730		-
  mask = label != 0
731		-
  # Определяем функцию потерь
732		-
  loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
733		-
    from_logits=True, reduction='none')
734		-
  loss = loss_object(label, pred)
735		-
736		-
  # Важно чтобы mask и loss имели одинаковый тип данных
737		-
  mask = tf.cast(mask, dtype=loss.dtype)
738		-
  # Наложение маски на loss
739		-
  loss *= mask
740		-
741		-
  # Масштабирование потерь на маску
742		-
  loss = tf.reduce_sum(loss)/tf.reduce_sum(mask)
743		-
  return loss
744		-
745		-
# Функция точности с учетом маски
746		-
def masked_accuracy(label, pred):
747		-
  pred = tf.argmax(pred, axis=2)
748		-
  label = tf.cast(label, pred.dtype)
749		-
  # Оценка совпадения метки и предсказания
750		-
  match = label == pred
751		-
  # Задаем маску, где метки не равны 0
752		-
  mask = label != 0
753		-
754		-
  # Логическое И
755		-
  match = match & mask
756		-
757		-
  # Преобразуем к одному типу и масштабирование совпадений на маску
758		-
  match = tf.cast(match, dtype=tf.float32)
759		-
  mask = tf.cast(mask, dtype=tf.float32)
760		-
  return tf.reduce_sum(match)/tf.reduce_sum(mask)
761		-
762		-
### Компилируем и обучаем модель
763		-
764		-
transformer.compile(
765		-
    loss=masked_loss,
766		-
    optimizer=optimizer,
767		-
    metrics=[masked_accuracy])
768		-
769		-
transformer.fit(train_batches,
770		-
                epochs=EPOCHS,
771		-
                validation_data=val_batches)
772		-
773		-
## Выполнение модели (Inference)
774		-
775		-
# Теперь мы можем протестировать модель, для этого оформим класс переводчика `Translator`, как модуль tensorflow:
776		-
777		-
class Translator(tf.Module):
778		-
  def __init__(self, tokenizers, transformer):
779		-
    self.tokenizers = tokenizers
780		-
    self.transformer = transformer
781		-
782		-
  def __call__(self, sentence, max_length=MAX_TOKENS):
783		-
784		-
    assert isinstance(sentence, tf.Tensor) # Проверяем, что последовательность является тензором
785		-
    if len(sentence.shape) == 0:
786		-
      sentence = sentence[tf.newaxis]
787		-
788		-
    sentence = self.tokenizers.az.tokenize(sentence).to_tensor()
789		-
    # Введенное предложение написано на португальском языке
790		-
    encoder_input = sentence
791		-
792		-
    # Поскольку языком вывода является английский, инициализируйте вывод с помощью токена [START]
793		-
    start_end = self.tokenizers.en.tokenize([''])[0]
794		-
    start = start_end[0][tf.newaxis]
795		-
    end = start_end[1][tf.newaxis]
796		-
797		-
    # Здесь требуется  tf.TensorArray` (вместо списка Python), чтобы динамический цикл
798		-
    # можно было отследить с помощью `tf.function`.
799		-
    output_array = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
800		-
    output_array = output_array.write(0, start)
801		-
802		-
    for i in tf.range(max_length):
803		-
      # на каждом шаге складываем выходные токены в массив для
804		-
      #  передачи на вход на следующем шаге
805		-
      output = tf.transpose(output_array.stack())
806		-
      # передаем в трансформер для предсказания токены
807		-
      predictions = self.transformer([encoder_input, output], training=False)
808		-
809		-
      # Выбираем последний токен из измерения `seq_len`
810		-
      predictions = predictions[:, -1:, :]  # Форма `(batch_size, 1, vocab_size)`.
811		-
812		-
      # Предсказанный токен
813		-
      predicted_id = tf.argmax(predictions, axis=-1)
814		-
815		-
      # Объединяем `predicted_id` с выходными данными, которые передаются
816		-
      # декодеру в качестве входных данных.
817		-
      output_array = output_array.write(i+1, predicted_id[0])
818		-
819		-
      if predicted_id == end:
820		-
        break
821		-
822		-
    output = tf.transpose(output_array.stack())
823		-
    # Токены в текст
824		-
    text = tokenizers.en.detokenize(output)[0]  # Shape: `()`.
825		-
826		-
    tokens = tokenizers.en.lookup(output)[0]
827		-
828		-
    # `tf.function` не позволяет нам использовать значения attention_weight, которые были
829		-
    # рассчитаны на последней итерации цикла.
830		-
    # Поэтому пересчитаем их вне цикла.
831		-
    self.transformer([encoder_input, output[:,:-1]], training=False)
832		-
    attention_weights = self.transformer.decoder.last_attn_scores
833		-
834		-
    return text, tokens, attention_weights
835		-
836		-
# Создадим экземпляр класса Translator и протестируем на нескольких фразах:
837		-
838		-
translator = Translator(tokenizers, transformer)
839		-
840		-
def print_translation(sentence, tokens, ground_truth):
841		-
  print(f'{"Фраза для перевода:":25s}: {sentence}')
842		-
  print(f'{"Предсказанный перевод:":25s}: {tokens.numpy().decode("utf-8")}')
843		-
  print(f'{"Оригинальный перевод":25s}: {ground_truth}')
844		-
845		-
# Пример 1:
846		-
847		-
sentence = 'bu həll etməli olduğumuz problemdir.'
848		-
ground_truth = 'this is a problem we have to solve .'
849		-
850		-
translated_text, translated_tokens, attention_weights = translator(
851		-
    tf.constant(sentence))
852		-
print_translation(sentence, translated_text, ground_truth)
853		-
854		-
# Пример 2:
855		-
856		-
sentence = 'və mənim qonşu evlərim bu fikri eşitdi.'
857		-
ground_truth = 'and my neighboring homes heard about this idea.'
858		-
859		-
translated_text, translated_tokens, attention_weights = translator(
860		-
    tf.constant(sentence))
861		-
print_translation(sentence, translated_text, ground_truth)