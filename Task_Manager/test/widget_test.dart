import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:my_first_app/main.dart';

void main() {
  testWidgets('Task management smoke test', (WidgetTester tester) async {
    // Build our app and trigger a frame.
    await tester.pumpWidget(const TaskManagerApp());

    // Verify that the app title is displayed
    expect(find.text('Список задач'), findsOneWidget);

    // Find the text field and add button
    expect(find.byType(TextField), findsOneWidget);
    expect(find.text('Добавить'), findsOneWidget);

    // Enter text and add a task
    await tester.enterText(find.byType(TextField), 'Тестовая задача');
    await tester.tap(find.text('Добавить'));
    await tester.pump();

    // Verify that the task was added
    expect(find.text('Тестовая задача'), findsOneWidget);
  });
}
