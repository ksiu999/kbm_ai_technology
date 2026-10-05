### Отчет о выполнении домашнего задания: Создание RAG-приложения

**Выполнила:** Абрамова Оксана Витальевна
**Тема:** Разработка RAG-системы для анализа технологических карт предприятия КБМ.

#### 1. Промпт для генерации ТЗ (Vibe-coding подход)

Для постановки задачи ИИ-ассистенту я использовала следующий структурированный промпт, чтобы задать архитектуру без написания кода вручную:

> «Действуй как Senior AI Architect. Сформулируй подробное ТЗ для создания RAG-приложения.
> **Стек:** Python, Streamlit (для UI), FAISS (или легковесная альтернатива для векторного поиска в памяти).
> **Безопасность:** Управление секретами (API-ключи) должно осуществляться через SOPS (Secrets OPerationS) с шифрованием age.
> **Интеграции:**
>
> 1. Embedding: использование модели `text-embedding-3-small` через API учебной платформы.
> 2. Генерация: использование LLM через API платформы (с fallback-маршрутизацией через RouterAI на модели Qwen/DeepSeek для обеспечения отказоустойчивости).
>    **Функционал:** Загрузка текстовой базы знаний, разбивка на чанки, векторный поиск (cosine similarity), вывод топ-5 релевантных чанков и структурированный JSON-ответ от AI в интерфейсе чата.»

#### 2. Выбор инструментов и эволюция разработки

Изначально разработка велась в соответствии с заданием через браузерные no-code/low-code среды:

1. **Replit:** Проект был успешно инициирован, но в процессе работы с объемными файлами сервис стал недоступен (см. Скриншоты из таблицы ниже).
2. **Bolt.new:** При попытке перенести ТЗ туда, среда зависла на этапе парсинга.
   **Причина:** В качестве базы знаний я использовала не синтетические данные, а реальные технологические карты и ГОСТы производственной компании. Браузерные ИИ-билдеры столкнулись со сложностями при интерпретации специфичных промышленных CSV-схем и форматирования.

**Решение:** Для сохранения темпа и качества я перешла на проверенный стек: **VS Code + плагин AI-агента (Cline) + локальный терминал**. Это позволило сохранить zero-code подход к написанию логики (весь код генерировался агентом по моим промптам), но дало полный контроль над окружением. Подключенные через RouterAI модели (Qwen) отлично справились с задачей, так как среда была привычной и стабильной.

#### 3. Алгоритм работы RAG-пайплайна (реализовано ИИ-агентом)

1. **Инжестия и чанкинг:** Приложение считывает `.txt` файл и разбивает его на смысловые блоки (чанки) по двойным переносам строк, игнорируя фрагменты короче 20 символов.
2. **Векторизация (Embedding):** Каждый чанк отправляется в API учебной платформы (`text-embedding-3-small`), где преобразуется в числовой вектор. Результат сохраняется в оперативной памяти (`session_state`).
3. **Поиск (Retrieval):** При запросе пользователя его текст также векторизуется. Алгоритм вычисляет косинусное сходство (Cosine Similarity) между запросом и всеми чанками, отбирая топ-5 наиболее релевантных фрагментов.
4. **Генерация (Generation):** Топ-5 чанков передаются в LLM со строгим системным промптом вернуть ответ в формате JSON (резюме + таблица оборудования). Это гарантирует отсутствие галлюцинаций и структурированный вывод.

#### 4. Результаты тестирования диалогов

#### 4. Результаты тестирования диалогов

| №   | Визуализация и ссылка                                                                                                                                                                             | Описание сценария тестирования                                                                                                                                                                                                                                                                |
| --- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | [![Блок проекта](https://raw.githubusercontent.com/ksiu999/kbm_ai_technology/main/img/1_Block.png)](https://github.com/ksiu999/kbm_ai_technology/blob/main/img/1_Block.png)                       | **Блок от Replit** Я зарегистрирована, авторизовалась, закинула первый промпт и получила блок. <br>🔗 _[Открыть скриншот на GitHub](https://github.com/ksiu999/kbm_ai_technology/blob/main/img/1_Block.png)_                                                                                  |
| 2   | [![Старт Bolt](https://raw.githubusercontent.com/ksiu999/kbm_ai_technology/main/img/2_bolt_start.png)](https://github.com/ksiu999/kbm_ai_technology/blob/main/img/2_bolt_start.png)               | **Инициализация в Bolt.** Попытка начала работы с браузерной средой разработки. <br>🔗 _[Открыть скриншот на GitHub](https://github.com/ksiu999/kbm_ai_technology/blob/main/img/2_bolt_start.png)_                                                                                            |
| 3   | [![Ошибка Bolt](https://raw.githubusercontent.com/ksiu999/kbm_ai_technology/main/img/3_bolt_err.png)](https://github.com/ksiu999/kbm_ai_technology/blob/main/img/3_bolt_err.png)                  | **Ошибка в Bolt.** Подвис на обработке реальной производственной технологической базе. <br>🔗 _[Открыть скриншот на GitHub](https://github.com/ksiu999/kbm_ai_technology/blob/main/img/3_bolt_err.png)_                                                                                       |
| 4   | [![Ошибки моделей](https://raw.githubusercontent.com/ksiu999/kbm_ai_technology/main/img/4_models-err.png)](https://github.com/ksiu999/kbm_ai_technology/blob/main/img/4_models-err.png)           | **Ошибки API моделей.** Визуализация проблем с HTTP 400 при обращении к школьному API. Поюзав bolt, добилась сгенерированной страницы, но начали отваливаться модели openAI<br>🔗 _[Открыть скриншот на GitHub](https://github.com/ksiu999/kbm_ai_technology/blob/main/img/4_models-err.png)_ |
| 5   | [![Работа с Qwen](https://raw.githubusercontent.com/ksiu999/kbm_ai_technology/main/img/5_qwen.png)](https://github.com/ksiu999/kbm_ai_technology/blob/main/img/5_qwen.png)                        | **Интеграция Qwen.** Промпт для bolt от qwen. <br>🔗 _[Открыть скриншот на GitHub](https://github.com/ksiu999/kbm_ai_technology/blob/main/img/5_qwen.png)_                                                                                                                                    |
| 6   | [![Промпт RAG](https://raw.githubusercontent.com/ksiu999/kbm_ai_technology/main/img/6_prompt-RAG.png)](https://github.com/ksiu999/kbm_ai_technology/blob/main/img/6_prompt-RAG.png)               | **Промпт для RAG.** Структурированный промпт от qwen для cline, используемый для генерации ТЗ и настройки системы. <br>🔗 _[Открыть скриншот на GitHub](https://github.com/ksiu999/kbm_ai_technology/blob/main/img/6_prompt-RAG.png)_                                                         |
| 7   | [![Работа Cline 1](https://raw.githubusercontent.com/ksiu999/kbm_ai_technology/main/img/7_cline_worker.png)](https://github.com/ksiu999/kbm_ai_technology/blob/main/img/7_cline_worker.png)       | **AI-агент Cline.** Демонстрация работы агента в VS Code (cline c моделью qwen) при реализации RAG-пайплайна. <br>🔗 _[Открыть скриншот на GitHub](https://github.com/ksiu999/kbm_ai_technology/blob/main/img/7_cline_worker.png)_                                                            |
| 8   | [![Работа Cline 2](https://raw.githubusercontent.com/ksiu999/kbm_ai_technology/main/img/8_cline_worker1.png)](https://github.com/ksiu999/kbm_ai_technology/blob/main/img/8_cline_worker1.png)     | **Продолжение работы Cline.** Визуализация процесса генерации кода агентом (еще один процесс). <br>🔗 _[Открыть скриншот на GitHub](https://github.com/ksiu999/kbm_ai_technology/blob/main/img/8_cline_worker1.png)_                                                                          |
| 9   | [![Подготовка RAG](https://raw.githubusercontent.com/ksiu999/kbm_ai_technology/main/img/9-prepare-RAG.png)](https://github.com/ksiu999/kbm_ai_technology/blob/main/img/9-prepare-RAG.png)         | **Подготовка RAG-системы.** Этап загрузки и векторизации базы знаний. <br>🔗 _[Открыть скриншот на GitHub](https://github.com/ksiu999/kbm_ai_technology/blob/main/img/9-prepare-RAG.png)_                                                                                                     |
| 10  | [![Результат чанкинга](https://raw.githubusercontent.com/ksiu999/kbm_ai_technology/main/img/10-chunk_result.png)](https://github.com/ksiu999/kbm_ai_technology/blob/main/img/10-chunk_result.png) | **Результат чанкинга.** Визуализация успешно загруженных 214 чанков и их эмбеддингов. <br>🔗 _[Открыть скриншот на GitHub](https://github.com/ksiu999/kbm_ai_technology/blob/main/img/10-chunk_result.png)_                                                                                   |
| 11  | [![Тестовые вопросы](https://raw.githubusercontent.com/ksiu999/kbm_ai_technology/main/img/11_questions.png)](https://github.com/ksiu999/kbm_ai_technology/blob/main/img/11_questions.png)         | **Набор тестовых вопросов.** Список вопросов для проверки работы RAG-системы. <br>🔗 _[Открыть скриншот на GitHub](https://github.com/ksiu999/kbm_ai_technology/blob/main/img/11_questions.png)_                                                                                              |
| 12  | [![Ответ 1](https://raw.githubusercontent.com/ksiu999/kbm_ai_technology/main/img/12_answer1.png)](https://github.com/ksiu999/kbm_ai_technology/blob/main/img/12_answer1.png)                      | **Ответ AI на вопрос 1.** Структурированный вывод: резюме, таблица оборудования, источники. <br>🔗 _[Открыть скриншот на GitHub](https://github.com/ksiu999/kbm_ai_technology/blob/main/img/12_answer1.png)_                                                                                  |
| 13  | [![Ответ 2](https://raw.githubusercontent.com/ksiu999/kbm_ai_technology/main/img/13_answer2.png)](https://github.com/ksiu999/kbm_ai_technology/blob/main/img/13_answer2.png)                      | **Ответ AI на вопрос 2.** Демонстрация работы косинусного сходства и формирования ответа. <br>🔗 _[Открыть скриншот на GitHub](https://github.com/ksiu999/kbm_ai_technology/blob/main/img/13_answer2.png)_                                                                                    |
| 14  | [![Ответ 3](https://raw.githubusercontent.com/ksiu999/kbm_ai_technology/main/img/14_answer3.png)](https://github.com/ksiu999/kbm_ai_technology/blob/main/img/14_answer3.png)                      | **Ответ AI на вопрос 3.** Проверка точности извлечения числовых данных из техкарт. <br>🔗 _[Открыть скриншот на GitHub](https://github.com/ksiu999/kbm_ai_technology/blob/main/img/14_answer3.png)_                                                                                           |
| 15  | [![Поиск элементов](https://raw.githubusercontent.com/ksiu999/kbm_ai_technology/main/img/15_find_elements.png)](https://github.com/ksiu999/kbm_ai_technology/blob/main/img/15_find_elements.png)  | **Поиск элементов оборудования.** Финальный тест работы RAG-системы с раскрытыми источниками. <br>🔗 _[Открыть скриншот на GitHub](https://github.com/ksiu999/kbm_ai_technology/blob/main/img/15_find_elements.png)_                                                                          |

#### 5. Итог

Задание выполнено. RAG-приложение успешно обрабатывает реальную базу знаний, использует API учебной платформы для эмбеддингов, а секреты защищены с помощью SOPS. Демонстрация проведена с акцентом на оркестрацию ИИ-инструментов, а не на ручное написание кода.
