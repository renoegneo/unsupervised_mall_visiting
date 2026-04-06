# Mall Customer Unsupervised Learning App

## Описание

Проект обучает модель K-Means для сегментации посетителей торгового центра по их характеристикам:
- пол
- возраст
- годовой доход (k$)
- оценка трат (Spending Score)

Веб-приложение показывает результаты кластеризации и визуализирует сегменты.

## Установка

1. **Клонируйте репозиторий:**
   ```bash
   git clone https://github.com/your-username/house-rent-prediction.git
   cd house-rent-prediction
   ```

2. **Создайте виртуальное окружение:**
   ```bash
   python -m venv venv
   venv\Scripts\activate     # Windows
   ```

3. **Установите зависимости:**
   ```bash
   pip install -r requirements.txt
   ```

## Обучение модели

```bash
python train_model.py
```

## Запуск приложения

```bash
python app.py
```

## Примечания

- Убедитесь, что файл `Mall_Customers.csv` находится в корне проекта.
- После обучения модель и артефакты сохраняются в `kmeans_artifacts.joblib`.
- График кластеров сохраняется в `kmeans_clusters.png`.
