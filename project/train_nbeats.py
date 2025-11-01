import os
import re
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split, Dataset

from utils import FishSalesDataset, safe_filename
from utils import NBeatsModel, train_nbeats_model


class FishSalesDatasetFlattened(Dataset):
    """
    Модификация FishSalesDataset для N-BEATS.

    Вместо возврата (sequence_length, num_features) возвращает
    (sequence_length * num_features,) - полностью флаттенированный вектор
    """

    def __init__(self, dataframe, sequence_length=30, target_column='Количество'):
        # Используем оригинальный датасет
        self.original_dataset = FishSalesDataset(
            dataframe, 
            sequence_length=sequence_length, 
            target_column=target_column
        )

        self.sequence_length = sequence_length
        self.num_features = len(self.original_dataset.features_columns)
        self.flattened_size = self.sequence_length * self.num_features

        # Копируем важные атрибуты
        self.features_columns = self.original_dataset.features_columns
        self.scaler = self.original_dataset.scaler
        self.scaled_features = self.original_dataset.scaled_features
        self.scaled_target = self.original_dataset.scaled_target

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        # Получаем оригинальный батч
        x_orig, y = self.original_dataset[idx]

        # Флаттенируем входные данные
        # x_orig имеет форму (sequence_length, num_features) = (30, 23)
        x_flat = x_orig.flatten()  # Получаем форму (690,)

        return x_flat, y


def main():
    """
    Главная функция для обучения N-BEATS моделей с исправлениями.
    """

    # ==========================================
    # КОНФИГУРАЦИЯ
    # ==========================================

    data_path = 'data/DATA.csv'
    models_dir = 'models_nbeats'
    os.makedirs(models_dir, exist_ok=True)

    # Гиперпараметры N-BEATS
    SEQUENCE_LENGTH = 30           # Lookback window (сколько дней смотрим назад)
    FORECAST_HORIZON = 7           # Forecast horizon (на сколько дней вперед)
    NUM_STACKS = 3                 # Количество стеков
    NUM_BLOCKS = 3                 # Количество блоков в каждом стеке
    HIDDEN_LAYERS = [512, 512]     # Размеры скрытых слоев
    DROPOUT = 0.1                  # Dropout вероятность

    BATCH_SIZE = 32
    EPOCHS = 100
    LEARNING_RATE = 0.001
    EARLY_STOPPING_PATIENCE = 10

    TRAIN_TEST_RATIO = 0.8
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Используется устройство: {DEVICE}")
    print(f"\nПараметры модели:")
    print(f"  - Sequence Length (lookback): {SEQUENCE_LENGTH}")
    print(f"  - Forecast Horizon: {FORECAST_HORIZON}")
    print(f"  - Количество стеков: {NUM_STACKS}")
    print(f"  - Количество блоков в стеке: {NUM_BLOCKS}")
    print(f"  - Скрытые слои: {HIDDEN_LAYERS}")
    print()

    # ==========================================
    # ЗАГРУЗКА ДАННЫХ
    # ==========================================

    print("Загрузка данных...")
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date']).dt.normalize()
    df = df.dropna().reset_index(drop=True)

    nomenklatura_list = df['Номенклатура'].unique()
    print(f"Найдено {len(nomenklatura_list)} номенклатур\n")

    # ==========================================
    # ОБУЧЕНИЕ МОДЕЛЕЙ
    # ==========================================

    successful_models = 0
    failed_models = 0

    for idx, nomenklatura in enumerate(nomenklatura_list):
        print(f"[{idx+1}/{len(nomenklatura_list)}] Обучаем модель для: {nomenklatura}")

        # Фильтруем данные для текущей номенклатуры
        df_sub = df[df['Номенклатура'] == nomenklatura].reset_index(drop=True)

        # Проверяем минимальный размер данных
        if len(df_sub) < 100:
            print(f"  ⚠️  Слишком мало данных ({len(df_sub)} < 100), пропускаем\n")
            failed_models += 1
            continue

        try:
            # ==========================================
            # ПОДГОТОВКА ДАТАСЕТА (ИСПРАВЛЕНО)
            # ==========================================

            # Используем FLATTENED датасет вместо оригинального
            dataset = FishSalesDatasetFlattened(
                df_sub, 
                sequence_length=SEQUENCE_LENGTH, 
                target_column='Количество'
            )

            # КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ: правильный input_size
            input_size = SEQUENCE_LENGTH * dataset.num_features

            print(f"  - Размер входа (flattened): {input_size} "
                  f"({SEQUENCE_LENGTH} дней × {dataset.num_features} признаков)")

            if len(dataset) < 10:
                print(f"  ⚠️  Недостаточно примеров в датасете, пропускаем\n")
                failed_models += 1
                continue

            # Разделяем на обучающий и валидационный наборы
            size = len(dataset)
            train_size = int(TRAIN_TEST_RATIO * size)
            val_size = size - train_size

            train_set, val_set = random_split(
                dataset, 
                [train_size, val_size], 
                generator=torch.Generator().manual_seed(42)
            )

            # Создаем DataLoaders
            train_loader = DataLoader(
                train_set, 
                batch_size=BATCH_SIZE, 
                shuffle=True
            )
            val_loader = DataLoader(
                val_set, 
                batch_size=BATCH_SIZE, 
                shuffle=False
            )

            print(f"  - Размер обучающего набора: {train_size}")
            print(f"  - Размер валидационного набора: {val_size}")

            # ==========================================
            # СОЗДАНИЕ И ОБУЧЕНИЕ МОДЕЛИ
            # ==========================================

            # Создаем N-BEATS модель с ПРАВИЛЬНЫМ input_size
            model = NBeatsModel(
                input_size=input_size,              # ← ИСПРАВЛЕНО!
                output_size=FORECAST_HORIZON,
                num_stacks=NUM_STACKS,
                num_blocks=NUM_BLOCKS,
                hidden_layers=HIDDEN_LAYERS,
                dropout=DROPOUT
            )

            print(f"  - Количество параметров: "
                  f"{sum(p.numel() for p in model.parameters()):,}")

            # Обучаем модель
            print("  - Обучение запущено...")
            model, history = train_nbeats_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=EPOCHS,
                lr=LEARNING_RATE,
                device=DEVICE,
                early_stopping_patience=EARLY_STOPPING_PATIENCE
            )

            # ==========================================
            # СОХРАНЕНИЕ МОДЕЛИ
            # ==========================================

            safe_name = safe_filename(nomenklatura)

            model_path = os.path.join(models_dir, f"{safe_name}_nbeats.pth")
            scaler_path = os.path.join(models_dir, f"{safe_name}_nbeats_scaler.pkl")
            features_path = os.path.join(models_dir, f"{safe_name}_nbeats_features.npy")
            config_path = os.path.join(models_dir, f"{safe_name}_nbeats_config.txt")

            # Сохраняем веса модели
            torch.save(model.state_dict(), model_path)

            # Сохраняем scaler
            import pickle
            with open(scaler_path, 'wb') as f:
                pickle.dump(dataset.scaler, f)

            # Сохраняем список признаков
            np.save(features_path, np.array(dataset.features_columns, dtype=object))

            # Сохраняем конфигурацию модели (ИСПРАВЛЕНО!)
            config = {
                'input_size': input_size,           # ← ИСПРАВЛЕНО!
                'forecast_horizon': FORECAST_HORIZON,
                'num_stacks': NUM_STACKS,
                'num_blocks': NUM_BLOCKS,
                'hidden_layers': HIDDEN_LAYERS,
                'dropout': DROPOUT,
                'sequence_length': SEQUENCE_LENGTH,
                'num_features': dataset.num_features,  # ← НОВОЕ: для десериализации
                'is_flattened': True                    # ← НОВОЕ: флаг
            }

            with open(config_path, 'w') as f:
                for key, value in config.items():
                    f.write(f"{key}: {value}\n")

            print(f"  ✅ Модель сохранена: {model_path}")
            print(f"  ✅ Scaler сохранен: {scaler_path}")
            print(f"  ✅ Признаки сохранены: {features_path}")
            print()

            successful_models += 1

        except Exception as e:
            print(f"  ❌ Ошибка при обучении: {str(e)}\n")
            failed_models += 1
            import traceback
            traceback.print_exc()
            continue

    # ==========================================
    # ИТОГИ
    # ==========================================

    print("="*60)
    print(f"ОБУЧЕНИЕ ЗАВЕРШЕНО")
    print(f"  ✅ Успешно обучено моделей: {successful_models}")
    print(f"  ❌ Ошибок при обучении: {failed_models}")
    print(f"  📁 Модели сохранены в: {models_dir}")
    print("="*60)


if __name__ == "__main__":
    main()