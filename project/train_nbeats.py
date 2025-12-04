"""
train_nbeats_fixed.py - Новый скрипт обучения с исправленной архитектурой
"""
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from utils import (
    TimeSeriesDataset, NBeatsModel, train_nbeats,
    safe_filename, save_model_complete
)

def main():
    # ==========================================
    # КОНФИГУРАЦИЯ
    # ==========================================
    
    DATA_PATH = 'data/DATA.csv'
    MODELS_DIR = 'models_nbeats_v2'
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Гиперпараметры
    SEQUENCE_LENGTH = 30        # Окно истории (дней)
    FORECAST_HORIZON = 20       # Горизонт прогноза (дней)
    NUM_STACKS = 4              # Стеков в модели
    NUM_BLOCKS = 4              # Блоков на стек
    HIDDEN_SIZES = [512, 512, 512]  # Размеры скрытых слоёв
    DROPOUT = 0.15
    
    BATCH_SIZE = 16
    EPOCHS = 200
    LEARNING_RATE = 0.001
    EARLY_STOPPING = 20
    TRAIN_RATIO = 0.8
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*70)
    print("N-BEATS v2: ИСПРАВЛЕННАЯ АРХИТЕКТУРА")
    print("="*70)
    print(f"\n📊 Параметры:")
    print(f"  Sequence Length: {SEQUENCE_LENGTH} дней")
    print(f"  Forecast Horizon: {FORECAST_HORIZON} дней")
    print(f"  Stacks: {NUM_STACKS}, Blocks: {NUM_BLOCKS}")
    print(f"  Hidden Sizes: {HIDDEN_SIZES}")
    print(f"  Batch Size: {BATCH_SIZE}, Epochs: {EPOCHS}")
    print(f"  Device: {DEVICE}\n")
    
    # ==========================================
    # ЗАГРУЗКА ДАННЫХ
    # ==========================================
    
    print("📂 Загрузка данных...")
    df = pd.read_csv(DATA_PATH)
    df['date'] = pd.to_datetime(df['date']).dt.normalize()
    df = df.dropna().reset_index(drop=True)
    
    nomenclatures = sorted(df['Номенклатура'].unique())
    print(f"✅ Найдено {len(nomenclatures)} номенклатур\n")
    
    # ==========================================
    # ОБУЧЕНИЕ
    # ==========================================
    
    successful = 0
    failed = 0
    
    for idx, nomen in enumerate(nomenclatures, 1):
        print(f"[{idx}/{len(nomenclatures)}] {nomen}")
        
        df_sub = df[df['Номенклатура'] == nomen].reset_index(drop=True)
        
        # Минимальный размер
        min_required = SEQUENCE_LENGTH + FORECAST_HORIZON + 50
        if len(df_sub) < min_required:
            print(f"  ⚠️  Недостаточно данных ({len(df_sub)} < {min_required})\n")
            failed += 1
            continue
        
        try:
            # ===== ДАТАСЕТ =====
            dataset = TimeSeriesDataset(
                df_sub,
                sequence_length=SEQUENCE_LENGTH,
                target_column='Количество',
                use_features=True
            )
            
            if len(dataset) < 20:
                print(f"  ⚠️  Слишком мало примеров после создания датасета\n")
                failed += 1
                continue
            
            # Флаттенированный размер входа
            # КЛЮЧЕВОЙ МОМЕНТ: одно значение целевой переменной + признаки
            input_dim = 1 + len(dataset.feature_columns)  # целевая + признаки
            input_size = SEQUENCE_LENGTH * input_dim
            
            print(f"  📈 Input Size: {input_size} ({SEQUENCE_LENGTH} × {input_dim})")
            print(f"  📊 Dataset Size: {len(dataset)} примеров")
            
            # ===== SPLIT =====
            size = len(dataset)
            train_size = int(TRAIN_RATIO * size)
            val_size = size - train_size
            
            train_set, val_set = random_split(
                dataset,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42)
            )
            
            train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
            
            print(f"  🔄 Train: {train_size}, Val: {val_size}")
            
            # ===== МОДЕЛЬ =====
            model = NBeatsModel(
                input_size=input_size,
                output_size=FORECAST_HORIZON,
                num_stacks=NUM_STACKS,
                num_blocks_per_stack=NUM_BLOCKS,
                hidden_sizes=HIDDEN_SIZES,
                dropout=DROPOUT
            )
            
            num_params = sum(p.numel() for p in model.parameters())
            print(f"  🧠 Parameters: {num_params:,}")
            
            # ===== ОБУЧЕНИЕ =====
            print(f"  ⏳ Обучение...")
            model, history = train_nbeats(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=EPOCHS,
                lr=LEARNING_RATE,
                device=DEVICE,
                early_stopping_patience=EARLY_STOPPING,
                verbose=False
            )
            
            final_train_loss = history['train_loss'][-1]
            final_val_loss = history['val_loss'][-1]
            print(f"  ✅ Final Loss - Train: {final_train_loss:.6f}, Val: {final_val_loss:.6f}")
            
            # ===== СОХРАНЕНИЕ =====
            safe_name = safe_filename(nomen)
            save_dir = os.path.join(MODELS_DIR, safe_name)
            
            config = {
                'model_params': {
                    'input_size': input_size,
                    'output_size': FORECAST_HORIZON,
                    'num_stacks': NUM_STACKS,
                    'num_blocks_per_stack': NUM_BLOCKS,
                    'hidden_sizes': HIDDEN_SIZES,
                    'dropout': DROPOUT
                },
                'sequence_length': SEQUENCE_LENGTH,
                'forecast_horizon': FORECAST_HORIZON,
                'feature_columns': dataset.feature_columns
            }
            
            save_model_complete(model, dataset, config, save_dir)
            print(f"  💾 Модель сохранена: {save_dir}\n")
            
            successful += 1
            
        except Exception as e:
            print(f"  ❌ Ошибка: {str(e)}\n")
            failed += 1
            continue
    
    # ===== ИТОГИ =====
    print("="*70)
    print(f"✅ УСПЕШНО:    {successful}")
    print(f"❌ ОШИБОК:     {failed}")
    print(f"📁 ДИРЕКТОРИЯ: {MODELS_DIR}")
    print("="*70)

if __name__ == "__main__":
    # ПРОВЕРКА GPU
    print("🔍 Проверка GPU...")
    print(f"CUDA доступна: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Память: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"Выделено: {torch.cuda.memory_allocated(0) / 1e9:.1f} GB")
    else:
        print("⚠️  GPU недоступна, используется CPU")
    print()
    main()