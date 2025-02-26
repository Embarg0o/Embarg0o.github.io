import 'package:flutter/material.dart';

// Модель данных для задачи
class Task {
  String title; // Заголовок задачи
  bool isCompleted; // Статус выполнения

  Task({
    required this.title,
    this.isCompleted = false,
  });
}

void main() {
  runApp(const TaskManagerApp());
}

// Основной виджет приложения
class TaskManagerApp extends StatelessWidget {
  const TaskManagerApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Менеджер Задач',
      theme: ThemeData(
        primarySwatch: Colors.blue,
        useMaterial3: true,
      ),
      home: const TaskListScreen(),
    );
  }
}

// Экран со списком задач
class TaskListScreen extends StatefulWidget {
  const TaskListScreen({super.key});

  @override
  State<TaskListScreen> createState() => _TaskListScreenState();
}

class _TaskListScreenState extends State<TaskListScreen> {
  // Список задач
  final List<Task> _tasks = [];
  // Контроллер для текстового поля
  final TextEditingController _textController = TextEditingController();

  @override
  void dispose() {
    _textController.dispose();
    super.dispose();
  }

  // Метод для добавления новой задачи
  void _addTask() {
    if (_textController.text.isNotEmpty) {
      setState(() {
        _tasks.add(Task(title: _textController.text));
        _textController.clear();
      });
    }
  }

  // Метод для редактирования задачи
  void _editTask(int index) {
    showDialog(
      context: context,
      builder: (context) {
        final TextEditingController editController =
            TextEditingController(text: _tasks[index].title);
        return AlertDialog(
          title: const Text('Редактировать задачу'),
          content: TextField(
            controller: editController,
            decoration: const InputDecoration(
              labelText: 'Название задачи',
            ),
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(context),
              child: const Text('Отмена'),
            ),
            TextButton(
              onPressed: () {
                if (editController.text.isNotEmpty) {
                  setState(() {
                    _tasks[index].title = editController.text;
                  });
                  Navigator.pop(context);
                }
              },
              child: const Text('Сохранить'),
            ),
          ],
        );
      },
    );
  }

  // Метод для удаления задачи
  void _deleteTask(int index) {
    setState(() {
      _tasks.removeAt(index);
    });
  }

  // Метод для изменения статуса выполнения задачи
  void _toggleTaskStatus(int index) {
    setState(() {
      _tasks[index].isCompleted = !_tasks[index].isCompleted;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Список задач'),
        backgroundColor: Theme.of(context).colorScheme.primary,
        foregroundColor: Colors.white,
      ),
      body: Column(
        children: [
          // Панель добавления новой задачи
          Padding(
            padding: const EdgeInsets.all(16.0),
            child: Row(
              children: [
                Expanded(
                  child: TextField(
                    controller: _textController,
                    decoration: const InputDecoration(
                      labelText: 'Новая задача',
                      border: OutlineInputBorder(),
                    ),
                  ),
                ),
                const SizedBox(width: 16.0),
                ElevatedButton(
                  onPressed: _addTask,
                  child: const Text('Добавить'),
                ),
              ],
            ),
          ),
          // Список задач
          Expanded(
            child: ListView.builder(
              itemCount: _tasks.length,
              itemBuilder: (context, index) {
                final task = _tasks[index];
                return Dismissible(
                  // Свайп для удаления
                  key: Key(task.title + index.toString()),
                  background: Container(
                    color: Colors.red,
                    alignment: Alignment.centerRight,
                    padding: const EdgeInsets.only(right: 16.0),
                    child: const Icon(Icons.delete, color: Colors.white),
                  ),
                  direction: DismissDirection.endToStart,
                  onDismissed: (direction) => _deleteTask(index),
                  child: ListTile(
                    leading: Checkbox(
                      value: task.isCompleted,
                      onChanged: (value) => _toggleTaskStatus(index),
                    ),
                    title: Text(
                      task.title,
                      style: TextStyle(
                        decoration: task.isCompleted
                            ? TextDecoration.lineThrough
                            : null,
                      ),
                    ),
                    // Кнопки управления задачей (редактирование и удаление)
                    trailing: Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        // Кнопка редактирования задачи
                        IconButton(
                          icon: const Icon(Icons.edit),
                          onPressed: () => _editTask(index),
                        ),
                        // Кнопка удаления задачи
                        IconButton(
                          icon: const Icon(Icons.delete),
                          onPressed: () => _deleteTask(index),
                          color: Colors.red,
                        ),
                      ],
                    ),
                  ),
                );
              },
            ),
          ),
        ],
      ),
    );
  }
}
