# Задача создать нейро-сотрудника(нейро-турагента)

!pip install openai gradio tiktoken langchain langchain-openai langchain-community chromadb

# Установка ключа для OpenAI
import getpass # для работы с паролями
import os      # для работы с окружением и файловой системой

# Запрос ввода ключа от OpenAI
os.environ["OPENAI_API_KEY"] = getpass.getpass("Введите OpenAI API Key:")

models = [
              {
                "doc": "",
                "prompt": '''Шаблон ''',
                "name": "Шаблон",
                "query": "Шаблон"
              },
              {
                "doc": " https://docs.google.com/document/d/1n9KQZEzCIvv5oVryBrXUFCeBoSJuJjEDcsg3q2KU3AA/edit?usp=sharing ",
                "prompt": '''Ты нейро-турагент и тебе необходимо предоставить исчерпывающую информацию о роли
                        турагента в туристической индустрии. Тебе могут задать вопросы потенциальные клиенты или
                        коллеги из индустрии, заинтересованные в твоих услугах. Твоя задача - дать подробный и точный
                        ответ, чтобы у собеседника не осталось никаких вопросов. При этом важно сохранять профессионализм
                        и лаконичность, не отвлекаясь на лишние эмоции и детали. Помни, что ты нейро-турагент и
                        твоя цель - предоставить только фактическую информацию.''',
                "name": "Нейро-турагент",
                "query": "Какие обязанности и ответственность несет турагент?"
              }
              ]

# Импорт необходимых библиотек

# Блок библиотек фреймворка LangChain

# Работа с документами в langchain
from langchain.docstore.document import Document
# Эмбеддинги для OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
# Доступ к векторной базе данных
from langchain.vectorstores import Chroma
# Разделение текста на куски или чанки (chunk)
from langchain.text_splitter import CharacterTextSplitter

# Отправка запросов
import requests

#Доступ к OpenAI
from openai import OpenAI

# Отприсовка интерфейса с помощью grad
import gradio as gr

# Библиотека подсчёта токенов
# Без запроcов к OpenAI, тем самым не тратим деньги на запросы
import tiktoken

# Для работы с регулярными выражениями
import re
import openai

# Создание базового класса для нейро-сотрудника

# Объявляем класс нейро-сотрудника
class GPT():
    # Объявляем конструктор класса, для передачи имени модели и инициализации атрибутов класса
    def __init__(self, model="gpt-3.5-turbo"):
        self.log = ''               # атрибут для сбора логов (сообщений)
        self.model = model          # атрибут для хранения выбранной модели OpenAI
        self.search_index = None    # атрибут для хранения ссылки на базу знаний (если None, то модель не обучена)
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"]) # при инициализации запрашиваем ключ от OpenAI

    # Метод загрузки текстового документа в векторную базу знаний
    def load_search_indexes(self, url):
        # Извлекаем document ID гугл документа из URL с помощью регулярных выражений
        match_ = re.search('/document/d/([a-zA-Z0-9-_]+)', url)

        # Если ID не найден - генерируем исключение
        if match_ is None:
            raise ValueError('Неверный Google Docs URL')

        # Первый элемент в результате поиска
        doc_id = match_.group(1)

        # Скачиваем гугл документ по его ID в текстовом формате
        response = requests.get(f'https://docs.google.com/document/d/{doc_id}/export?format=txt')

        # При неудачных статусах запроса будет вызвано исключение
        response.raise_for_status()

        # Извлекаем данные как текст
        text = response.text

        # Вызываем метод векторизации текста и сохранения в векторную базу данных
        return self.create_embedding(text)

    # Подсчет числа токенов в строке по имени модели
    def num_tokens_from_string(self, string):
            """Возвращает число токенов в строке"""
            encoding = tiktoken.encoding_for_model(self.model)  # получаем кодировщик по имени модели
            num_tokens = len(encoding.encode(string))           # расчитываем строку с помощью кодировщика
            return num_tokens                                   # возвращаем число токенов

    # Метод разбора текста и его сохранение в векторную базу знаний
    def create_embedding(self, data):
        # Список документов, полученных из фрагментов текста
        source_chunks = []
        # Разделяем текст на строки по \n (перенос на новую строку) или длине фрагмента (chunk_size=1024) с помощью сплитера
        # chunk_overlap=0 - означает, что фрагменты не перекрываются друг с другом.
        # Если больше нуля, то захватываем дополнительное число символов от соседних чанков.
        splitter = CharacterTextSplitter(separator="\n", chunk_size=1024, chunk_overlap=0)

        # Применяем splitter (функцию расщепления) к данным и перебираем все получившиеся чанки (фрагменты)
        for chunk in splitter.split_text(data):
            # LangChain работает с документами, поэтому из текстовых чанков мы создаем фрагменты документов
            source_chunks.append(Document(page_content=chunk, metadata={}))

        # Подсчет числа токенов в документах без запроса к OpenAI (экономим денежные средства)
        count_token = self.num_tokens_from_string(' '.join([x.page_content for x in source_chunks]))
        # Вместо вывода print, мы формируем переменную log для дальнейшего вывода в gradio информации
        self.log += f'Количество токенов в документе : {count_token}\n'

        # Создание индексов документа. Применяем к нашему списку документов эмбеддингов OpenAi и в таком виде загружаем в базу ChromaDB
        self.search_index = Chroma.from_documents(source_chunks, OpenAIEmbeddings(), )
        # Вместо вывода print, мы формируем переменную log для дальнейшего вывода в gradio информации
        self.log += f'Данные из документа загружены в в векторную базу данных\n'

        # Возвращаем ссылку на базу данных
        return self.search_index

    # Демонстрация более аккуратного расчета числа токенов в зависимости от модели
    def num_tokens_from_messages(self, messages, model):
        """Возвращает число токенов из списка сообщений"""
        try:
            encoding = tiktoken.encoding_for_model(model) # получаем кодировщик по имени модели
        except KeyError:
            print("Предупреждение: модель не создана. Используйте cl100k_base кодировку.")
            encoding = tiktoken.get_encoding("cl100k_base") # если по имени не нашли, то используем базовый для моделей OpenAI
        # Выбор модели
        if model in {
            "gpt-3.5-turbo-0613",
            "gpt-3.5-turbo-16k-0613",
            "gpt-4-0314",
            "gpt-4-32k-0314",
            "gpt-4-0613",
            "gpt-4-32k-0613",
            "gpt-4o",
            "gpt-4o-2024-05-13"
            }:
            tokens_per_message = 3 # дополнительное число токенов на сообщение
            tokens_per_name = 1    # токенов на имя
        elif model == "gpt-3.5-turbo-0301":
            tokens_per_message = 4  # каждое сообщение содержит <im_start>{role/name}\n{content}<im_end>\n
            tokens_per_name = -1  # если есть имя, то роль не указывается
        elif "gpt-3.5-turbo" in model:
            self.log += f'Внимание! gpt-3.5-turbo может обновиться в любой момент. Используйте gpt-3.5-turbo-0613. \n'
            return self.num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
        elif "gpt-4" in model:
            self.log += f'Внимание! gpt-4 может обновиться в любой момент. Используйте gpt-4-0613. \n'
            return self.num_tokens_from_messages(messages, model="gpt-4-0613")
        else: # исключение, если модель не поддерживается
            raise NotImplementedError(
                f"""num_tokens_from_messages() не реализован для модели {model}."""
            )

        # Запускаем подсчет токенов
        num_tokens = 0                        # счетчик токенов
        for message in messages:              # цикл по всем сообщениям
            num_tokens += tokens_per_message  # прибовляем число токенов на каждое сообщение
            for key, value in message.items():
                num_tokens += len(encoding.encode(value)) # считаем токены в сообщении с помощью кодировщика
                if key == "name":                     # если встретили имя
                    num_tokens += tokens_per_name     # то добавили число токенов на
        num_tokens += 3                               # каждый ответ оборачивается в <|start|>assistant<|message|>
        return num_tokens                             # возвращаем число токенов

    # Функция фильтрации запросов
    def filter_request(self, request):
        # Список запрещенных ключевых слов
        blacklist = [
        "расист", "сексист", "гомофоб", "трансфоб", "ксенофоб", "антисемит", "фашист", "нацист",
        "расизм", "сексизм", "гомофобия", "трансфобия", "ксенофобия", "антисемитизм",
        "убийство", "самоубийство", "насилие", "похищение", "взрыв", "угрозы", "терроризм", "экстремизм", "радикализм",
        "мошенничество", "обман", "кража", "грабеж", "разбой",
        "порнография", "эротика", "сексуальные намеки", "обнаженность", "секс", "эротика", "порно", "сексуальный", "интимный", "эротический",
        "наркотики", "алкоголь", "марихуана", "кокаин", "героин", "метадон", "амфетамин", "экстази", "ЛСД", "Педофилия", "вино", "водка", "виски",
        "черножопый", "оскорбления", "пидор", "педофил", "нецензурная лексика", "нецензурные выражения", "ниггер", "гомик", "нецензурные слова",
        "терроризм", "экстремизм", "радикализм", "исламистский", "исламский", "джихад", "шариат", "халифат", "террорист", "экстремист", "радикал",
        "пидорас", "мошенничество", "обман", "кража", "грабеж", "разбой", "мошенник", "обманщик", "вор", "грабитель", "растление"
        ]

        # Максимальная длина запроса
        max_length = 1000

        # Проверка на наличие запрещенных ключевых слов
        for word in blacklist:
            if re.search(word, request, re.IGNORECASE):
                return False

        # Проверка длины запроса
        if len(request) > max_length:
            return False

        return True

    # Метод запроса к языковой модели
    def answer_index(self, system, topic, temp = 0.8):
        # Проверка запроса на наличие запрещенных ключевых слов
        if not self.filter_request(topic):
            self.log += 'Запрос заблокирован из-за наличия запрещенных ключевых слов или превышения максимальной длины.\n'
            return ''


        # Проверяем обучена ли наша модель
        if not self.search_index:
            self.log += 'Модель необходимо обучить! \n'
            return ''

        # Выборка документов по схожести с запросом из векторной базы данных, topic- строка запроса, k - число извлекаемых фрагментов
        docs = self.search_index.similarity_search(topic, k=5)

        # Вместо вывода print, мы формируем переменную log для дальнейшего вывода в gradio информации
        self.log += 'Выбираем документы по степени схожести с вопросом из векторной базы данных: \n '
        # Очищаем запрос от двойных пустых строк. Каждый фрагмент подписываем: Отрывок документа № и дальше порядковый номер
        message_content = re.sub(r'\n{2}', ' ', '\n '.join([f'Отрывок документа №{i+1}:\n' + doc.page_content + '\\n' for i, doc in enumerate(docs)]))
        # Вместо вывода print, мы формируем переменную log для дальнейшего вывода в gradio информации
        self.log += f'{message_content} \n'

        # В системную роль помещаем найденные фрагменты и промпт, в пользовательскую - вопрос от пользователя
        messages = [
            {"role": "system", "content": system + f"{message_content}"},
            {"role": "user", "content": topic}
        ]

        # Вместо вывода print, мы формируем переменную log для дальнейшего вывода в gradio информации
        self.log += f"\n\nТокенов использовано на вопрос по версии TikToken: {self.num_tokens_from_messages(messages, self.model)}\n"


        # Запрос к языковой моделе
        completion = self.client.chat.completions.create(
            model=self.model,   # используемая модель
            messages=messages,  # список форматированных сообщений с ролями
            temperature=temp    # точность ответов модели
        )


        # Вместо вывода print, мы формируем переменную log для дальнейшего вывода в gradio информации
        self.log += '\nСтатистика по токенам от языковой модели:\n'
        self.log += f'Токенов использовано всего (вопрос): {completion.usage.prompt_tokens} \n'       # Число токенов на вопрос по расчетам LLM
        self.log += f'Токенов использовано всего (вопрос-ответ): {completion.usage.total_tokens} \n'  # Число токенов на вопрос и ответ по расчетам LLM

        return completion.choices[0].message.content # возвращаем результат предсказания

# Объявляем экземпляр класса GPT (созданный ранее) и передаем ему в конструктор модель LLM, с которой будем работать
gpt = GPT("gpt-3.5-turbo")

# Gradio позволяет объединять элементы в блоки
blocks = gr.Blocks()

# Работаем с блоком
with blocks as demo:
    # Объявляем элемент выбор из списка (с подписью Данные), список выбирает из поля name нашей переменной models
    subject = gr.Dropdown([(elem["name"], index) for index, elem in enumerate(models)], label="Данные")
    # Здесь отобразиться выбранное имя name из списка
    name = gr.Label(show_label=False)
    # Промпт для запроса к LLM (по умолчанию поле prompt из models)
    prompt = gr.Textbox(label="Промт", interactive=True)
    # Ссылка на файл обучения (по умолчанию поле doc из models)
    link = gr.HTML()
    # Поле пользовательского запроса к LLM (по умолчанию поле query из models)
    query = gr.Textbox(label="Запрос к LLM", interactive=True)


    # Функция на выбор нейро-сотрудника в models
    # Ей передается параметр subject - выбранное значение в поле списка
    # А возвращаемые значения извлекаются из models
    def onchange(dropdown):
      return [
          models[dropdown]['name'],                               # имя возвращается без изменения
          re.sub('\t+|\s\s+', ' ', models[dropdown]['prompt']),   # в промте удаляются двойные пробелы \s\s+ и табуляция \t+
          models[dropdown]['query'],                              # запрос возвращается без изменения
          f"<a target='_blank' href = '{models[dropdown]['doc']}'>Документ для обучения</a>" # ссылка на документ оборачивается в html тег <a>  (https://htmlbook.ru/html/a)
          ]

    # При изменении значения в поле списка subject, вызывается функция onchange
    # Ей передается параметр subject - выбранное значение в поле списка
    # А возвращаемые значения устанавливаются в элементы name, prompt, query и link
    subject.change(onchange, inputs = [subject], outputs = [name, prompt, query, link])

    # Строку в gradio можно разделить на столбцы (каждая кнопка в своем столбце)
    with gr.Row():
        train_btn = gr.Button("Обучить модель")       # кнопка запуска обучения
        request_btn = gr.Button("Запрос к модели")    # кнопка отправки запроса к LLM

    # функция обучения
    def train(dropdown):
        # парсим документ и сохраняем его в базу данных
        gpt.load_search_indexes(models[dropdown]['doc'])
        return gpt.log

    # Функция отправки запроса к LLM
    def predict(p, q):
      try:
        result = gpt.answer_index(
            p,
            q,
            0.8
        )
        return [result, gpt.log]
      except Exception as e:
        return [f"Ошибка: {str(e)}", gpt.log]

    # Выводим поля response с ответом от LLM и log (вывод сообщений работы класса GPT) на 2 колонки
    with gr.Row():
        response = gr.Textbox(label="Ответ LLM") # Текстовое поле с ответом от LLM
        log = gr.Textbox(label="Логирование")    # Текстовое поле с выводом сообщений от GPT


    # При нажатии на кнопку train_btn запускается функция обучения train_btn с параметром subject
    # Результат выполнения функции сохраняем в текстовое поле log - лог выполнения
    train_btn.click(train, [subject], log)

    # При нажатии на кнопку request_btn запускается функция отправки запроса к LLM request_btn с параметром prompt, query
    # Результат выполнения функции сохраняем в текстовые поля  response - ответ модели, log - лог выполнения
    request_btn.click(predict, [prompt, query], [response, log])

# Запуск приложения
demo.launch()

# Проведем трассировку работы нейро-сотрудника

# Инициализация
# 1. Нейро-сотрудник инициализируется с помощью класса 'GPT', который принимает в конструкторе имя модели LLM (model="gpt-3.5-turbo").
# 2. В конструкторе класса 'GPT' инициализируются атрибуты: 'log' (для сбора логов), 'model' (имя модели LLM), 'search_index'
# (ссылка на базу знаний, изначально равна None), и 'client' (клиент OpenAI для взаимодействия с LLM).

# Обучение
# 1. Когда пользователь выбирает значение в поле списка 'subject', вызывается функция 'onchange', которая возвращает имя, промпт, запрос и ссылку на документ для обучения.
# 2. При нажатии на кнопку "Обучить модель" ('train_btn') вызывается функция 'train', которая принимает в качестве параметра значение из поля списка 'subject'.
# 3. Функция 'train' вызывает метод 'load_search_indexes' класса 'GPT', который загружает документ для обучения в базу знаний.
# 4. Метод 'load_search_indexes' парсит документ и сохраняет его в базу данных.

# Запрос к LLM
# 1. При нажатии на кнопку "Запрос к модели" ('request_btn') вызывается функция 'predict', которая принимает в качестве параметров промпт ('p') и запрос ('q').
# 2. Функция 'predict' вызывает метод 'answer_index' класса 'GPT', который отправляет запрос к LLM.
# 3. Метод 'answer_index' проверяет, обучена ли модель (т. е., не равна ли 'search_index' 'None'). Если модель не обучена, метод возвращает пустую строку.
# 4. Если модель обучена, метод 'answer_index' выполняет поиск в базе знаний по запросу и возвращает ответ от LLM.