

"""
predict_nbeats.py - Скрипт для прогнозирования с использованием обученных N-BEATS моделей
"""

import os
import pickle
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

from utils import FishSalesDataset, safe_filename
from utils import NBeatsModel, predict_nbeats


def load_nbeats_model(model_name, models_dir='models_nbeats', device='cpu'):
    """
    Загружает обученную N-BEATS модель с конфигурацией и scaler.

    Параметры:
    - model_name: безопасное имя номенклатуры
    - models_dir: директория с моделями
    - device: CPU или GPU

    Возвращает:
    - model: загруженная модель
    - scaler: масштабировщик
    - config: конфигурация модели
    - features: список признаков
    """

    model_path = os.path.join(models_dir, f"{model_name}_nbeats.pth")
    scaler_path = os.path.join(models_dir, f"{model_name}_nbeats_scaler.pkl")
    config_path = os.path.join(models_dir, f"{model_name}_nbeats_config.txt")
    features_path = os.path.join(models_dir, f"{model_name}_nbeats_features.npy")

    # Проверяем наличие всех файлов
    if not all(os.path.exists(p) for p in [model_path, scaler_path, config_path, features_path]):
        return None, None, None, None

    # Загружаем конфигурацию
    config = {}
    with open(config_path, 'r') as f:
        for line in f:
            if ':' in line:
                key, value = line.split(':')
                key = key.strip()
                value = value.strip()
                # Пытаемся преобразовать в нужный тип
                try:
                    config[key] = int(value)
                except:
                    try:
                        config[key] = eval(value)
                    except:
                        config[key] = value

    # Создаем модель с загруженной конфигурацией
    model = NBeatsModel(
        input_size=config.get('input_size', 23),
        output_size=config.get('forecast_horizon', 7),
        num_stacks=config.get('num_stacks', 3),
        num_blocks=config.get('num_blocks', 3),
        hidden_layers=config.get('hidden_layers', [512, 512]),
        dropout=config.get('dropout', 0.1)
    )

    # Загружаем веса
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Загружаем scaler
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    # Загружаем список признаков
    features = np.load(features_path, allow_pickle=True).tolist()

    return model, scaler, config, features


def predict_for_date_range(models_dir='models_nbeats', 
                           data_path='data/DATA.csv',
                           days_forward=30,
                           device='cpu'):
    """
    Делает прогнозы для всех номенклатур на заданное количество дней.

    Параметры:
    - models_dir: директория с моделями
    - data_path: путь к данным
    - days_forward: количество дней для прогноза
    - device: CPU или GPU

    Возвращает:
    - results_df: DataFrame с прогнозами
    """

    # Загружаем данные
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date']).dt.normalize()
    df = df.dropna().reset_index(drop=True)

    results = []

    for nomenklatura in df['Номенклатура'].unique():
        safe_name = safe_filename(nomenklatura)

        # Загружаем модель
        model, scaler, config, features = load_nbeats_model(
            safe_name, 
            models_dir=models_dir, 
            device=device
        )

        if model is None:
            print(f"⚠️  Модель не найдена для: {nomenklatura}")
            continue

        # Фильтруем данные
        df_sub = df[df['Номенклатура'] == nomenklatura].reset_index(drop=True)

        if len(df_sub) < 100:
            print(f"⚠️  Недостаточно данных для: {nomenklatura}")
            continue

        try:
            # Создаем датасет
            dataset = FishSalesDataset(
                df_sub,
                sequence_length=config.get('sequence_length', 30),
                target_column='Количество'
            )

            # Делаем прогноз
            predictions = predict_nbeats(
                model=model,
                dataset=dataset,
                num_steps=days_forward,
                device=device
            )

            # Суммируем прогнозы
            total_pred = sum(predictions)

            results.append({
                'Номенклатура': nomenklatura,
                'Прогноз_общий': total_pred,
                'Среднее_значение': total_pred / len(predictions),
                'Количество_дней': len(predictions)
            })

            print(f"✅ Прогноз для {nomenklatura}: {total_pred:.2f}")

        except Exception as e:
            print(f"❌ Ошибка при прогнозировании {nomenklatura}: {str(e)}")
            continue

    results_df = pd.DataFrame(results).sort_values(
        'Прогноз_общий', 
        ascending=False
    ).reset_index(drop=True)

    return results_df


def main():
    """
    Главная функция для интерактивного прогнозирования.
    """

    print("╔" + "="*58 + "╗")
    print("║" + " "*58 + "║")
    print("║" + "N-BEATS: Система прогнозирования продаж рыбной продукции".center(58) + "║")
    print("║" + " "*58 + "║")
    print("╚" + "="*58 + "╝")
    print()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Устройство: {device}\n")

    while True:
        print("Выберите режим:")
        print("1 - Прогноз на 30 дней")
        print("2 - Прогноз на пользовательское количество дней")
        print("3 - Выход")
        print()

        choice = input("Ваш выбор (1-3): ").strip()

        if choice == '1':
            print("\nДелаю прогнозы на 30 дней...")
            results = predict_for_date_range(days_forward=30, device=device)
            print("\n" + "="*60)
            print("РЕЗУЛЬТАТЫ ПРОГНОЗА:")
            print("="*60)
            print(results.to_string())
            print("="*60 + "\n")

        elif choice == '2':
            try:
                days = int(input("Количество дней для прогноза: ").strip())
                if days <= 0:
                    print("Количество дней должно быть положительным\n")
                    continue

                print(f"\nДелаю прогнозы на {days} дней...")
                results = predict_for_date_range(days_forward=days, device=device)
                print("\n" + "="*60)
                print("РЕЗУЛЬТАТЫ ПРОГНОЗА:")
                print("="*60)
                print(results.to_string())
                print("="*60 + "\n")

            except ValueError:
                print("Ошибка: введите целое число\n")

        elif choice == '3':
            print("До свидания!")
            break

        else:
            print("Неверный выбор\n")


if __name__ == "__main__":
    main()