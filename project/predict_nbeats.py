"""
predict_nbeats_fixed.py - Прогнозирование, оценка и сохранение графиков
"""
import os
import pickle
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from utils import (
    TimeSeriesDataset, NBeatsModel, predict_nbeats,
    safe_filename, load_model_complete
)

plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

# ========================
# МЕТРИКИ
# ========================

def calculate_metrics(y_true, y_pred):
    """Вычисляет MAE, MAPE, RMSE."""
    y_true = np.array(y_true, dtype=np.float32)
    y_pred = np.array(y_pred, dtype=np.float32)
    
    mae = np.mean(np.abs(y_true - y_pred))
    
    # MAPE с защитой от деления на ноль
    mape_vals = np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8))
    mape = np.mean(mape_vals) * 100
    
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    # R² Score
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    return {
        'mae': mae,
        'mape': mape,
        'rmse': rmse,
        'r2': r2
    }

def evaluate_nomenclature(nomen, df, models_dir='models_nbeats_v2',
                          forecast_days=20, device='cpu'):
    """
    Оценивает точность для одной номенклатуры.
    
    Возвращает:
    - metrics: словарь с MAE, MAPE, RMSE, R²
    - dates_eval: даты для графика
    - y_true: реальные значения
    - y_pred: прогнозы
    """
    
    # Фильтруем данные
    df_sub = df[df['Номенклатура'] == nomen].sort_values('date').reset_index(drop=True)
    
    if len(df_sub) < 100:
        return None, None, None, None
    
    safe_name = safe_filename(nomen)
    model_path = os.path.join(models_dir, safe_name)
    
    if not os.path.exists(model_path):
        return None, None, None, None
    
    try:
        # Загружаем модель
        model, target_scaler, feature_scaler, config = load_model_complete(
            NBeatsModel, model_path, device=device
        )
        
        seq_len = config['sequence_length']
        forecast_h = config['forecast_horizon']
        
        # Проверка размера
        if len(df_sub) <= forecast_days + seq_len + 10:
            return None, None, None, None
        
        # Делим на историю и тест
        df_hist = df_sub.iloc[:-forecast_days].reset_index(drop=True)
        df_test = df_sub.iloc[-forecast_days:].reset_index(drop=True)
        
        # Создаём датасет на истории
        dataset = TimeSeriesDataset(
            df_hist,
            sequence_length=seq_len,
            target_column='Количество',
            use_features=True
        )
        
        # Прогнозируем
        preds = predict_nbeats(
            model=model,
            dataset=dataset,
            num_steps=forecast_days,
            device=device
        )
        
        # Берём реальные значения
        y_true = df_test['Количество'].values.astype(np.float32)
        y_pred = np.array(preds[:len(y_true)], dtype=np.float32)
        
        # Метрики
        metrics = calculate_metrics(y_true, y_pred)
        
        return metrics, df_test['date'].values, y_true, y_pred
    
    except Exception as e:
        print(f"  ⚠️  Ошибка: {e}")
        return None, None, None, None

def plot_forecast(nomen, dates, y_true, y_pred, metrics, save_dir='plots'):
    """Рисует и сохраняет график прогноза."""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Определяем тип оси X
    if isinstance(dates[0], np.datetime64):
        dates_dt = pd.to_datetime(dates)
    else:
        dates_dt = dates
    
    # Графики
    ax.plot(dates_dt, y_true, 'o-', label='Реальные значения', 
            linewidth=2, markersize=5, color='#2E86AB')
    ax.plot(dates_dt, y_pred, 's--', label='Прогнозы',
            linewidth=2, markersize=5, color='#A23B72')
    
    # Форматирование
    ax.set_xlabel('Дата', fontsize=12, fontweight='bold')
    ax.set_ylabel('Количество', fontsize=12, fontweight='bold')
    ax.set_title(f'Прогноз: {nomen}', fontsize=14, fontweight='bold', pad=20)
    
    # Легенда с метриками
    metrics_text = (f"MAE: {metrics['mae']:.2f}\n"
                    f"MAPE: {metrics['mape']:.2f}%\n"
                    f"RMSE: {metrics['rmse']:.2f}\n"
                    f"R²: {metrics['r2']:.4f}")
    ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Ротация дат
    fig.autofmt_xdate(rotation=45, ha='right')
    plt.tight_layout()
    
    # Сохраняем
    safe_name = safe_filename(nomen)
    filename = f"{safe_name}_forecast.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    return filepath

# ========================
# ГЛОБАЛЬНАЯ ОЦЕНКА
# ========================

def evaluate_global(df, models_dir='models_nbeats_v2', forecast_days=20, device='cpu'):
    """
    Оценивает точность по всем номенклатурам.
    
    Возвращает:
    - results_df: таблица с результатами
    """
    
    nomenclatures = sorted(df['Номенклатура'].unique())
    results = []
    
    all_true = []
    all_pred = []
    
    for idx, nomen in enumerate(nomenclatures, 1):
        print(f"  [{idx:3d}/{len(nomenclatures)}] {nomen[:50]:<50} ", end='', flush=True)
        
        metrics, dates, y_true, y_pred = evaluate_nomenclature(
            nomen, df, models_dir, forecast_days, device
        )
        
        if metrics is None:
            print("⚠️")
            continue
        
        # Сохраняем график
        try:
            plot_path = plot_forecast(nomen, dates, y_true, y_pred, metrics)
            plot_status = "✅"
        except:
            plot_path = None
            plot_status = "❌"
        
        # Собираем статистику
        results.append({
            'Номенклатура': nomen,
            'MAE': metrics['mae'],
            'MAPE_%': metrics['mape'],
            'RMSE': metrics['rmse'],
            'R²': metrics['r2'],
            'График': plot_status
        })
        
        all_true.extend(y_true)
        all_pred.extend(y_pred)
        
        print(f"MAE={metrics['mae']:.2f}, MAPE={metrics['mape']:.1f}%")
    
    # Глобальные метрики
    global_metrics = calculate_metrics(all_true, all_pred)
    
    results_df = pd.DataFrame(results).sort_values('MAE')
    
    return results_df, global_metrics

# ========================
# ИНТЕРАКТИВНЫЙ РЕЖИМ
# ========================

def interactive_evaluation(data_path='data/DATA.csv', models_dir='models_nbeats_v2'):
    """Интерактивный режим оценки."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n📱 Устройство: {device}\n")
    
    # Загружаем данные
    print("📂 Загрузка данных...")
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date']).dt.normalize()
    df = df.dropna().reset_index(drop=True)
    print(f"✅ Загружено {len(df)} записей\n")
    
    while True:
        print("\n" + "="*70)
        print("РЕЖИМЫ ОЦЕНКИ")
        print("="*70)
        print("1️⃣  - Глобальная оценка по всем номенклатурам")
        print("2️⃣  - Оценка по выбранной номенклатуре")
        print("3️⃣  - Выход")
        print("="*70)
        
        choice = input("\n👉 Выбор (1-3): ").strip()
        
        if choice == '1':
            forecast_days_input = input("Горизонт прогноза в днях (по умолчанию 20): ").strip()
            forecast_days = int(forecast_days_input) if forecast_days_input.isdigit() else 20
            
            print(f"\n🔄 Оценка для {forecast_days} дней...\n")
            
            results_df, global_metrics = evaluate_global(
                df, models_dir, forecast_days, device
            )
            results_df.sort_values(by="R²")
            # Вывод таблицы
            print("\n" + "="*90)
            print("РЕЗУЛЬТАТЫ ОЦЕНКИ")
            print("="*90)
            print(results_df.to_string(index=False))

            print("="*90)
            
            # Глобальные метрики
            print("\n📊 ГЛОБАЛЬНЫЕ МЕТРИКИ:")
            print(f"  MAE (ср. абс. ошибка):  {global_metrics['mae']:.3f}")
            print(f"  MAPE (ср. % ошибка):    {global_metrics['mape']:.2f}%")
            print(f"  RMSE:                   {global_metrics['rmse']:.3f}")
            print(f"  R² Score:               {global_metrics['r2']:.4f}")
            print()
            
            # Сохраняем результаты
            results_df.to_csv('evaluation_results.csv', index=False)
            print("💾 Результаты сохранены в evaluation_results.csv")
            
        elif choice == '2':
            # Список номенклатур
            noms = sorted(df['Номенклатура'].unique().tolist())
            print(f"\n📋 Доступные номенклатуры ({len(noms)} шт):")
            for i, n in enumerate(noms[:20], 1):
                print(f"  {i:2d}. {n}")
            if len(noms) > 20:
                print(f"  ... и ещё {len(noms) - 20}")
            
            try:
                idx_input = input("\n👉 Введите номер: ").strip()
                idx = int(idx_input)
                if not (1 <= idx <= len(noms)):
                    print("❌ Неверный номер")
                    continue
                
                nomen = noms[idx - 1]
                forecast_days_input = input("Горизонт прогноза в днях (по умолчанию 20): ").strip()
                forecast_days = int(forecast_days_input) if forecast_days_input.isdigit() else 20
                
                print(f"\n🔄 Оценка для '{nomen}'...")
                
                metrics, dates, y_true, y_pred = evaluate_nomenclature(
                    nomen, df, models_dir, forecast_days, device
                )
                
                if metrics is None:
                    print("❌ Не удалось загрузить или обработать модель")
                    continue
                
                # Вывод метрик
                print("\n" + "="*70)
                print(f"РЕЗУЛЬТАТЫ: {nomen}")
                print("="*70)
                print(f"MAE (средняя абсолютная ошибка):  {metrics['mae']:.3f}")
                print(f"MAPE (средняя процентная ошибка): {metrics['mape']:.2f}%")
                print(f"RMSE (корень из MSE):             {metrics['rmse']:.3f}")
                print(f"R² Score:                         {metrics['r2']:.4f}")
                print("="*70)
                
                # График
                plot_path = plot_forecast(nomen, dates, y_true, y_pred, metrics)
                print(f"\n📊 График сохранён: {plot_path}")
                
                # Таблица значений
                print("\n📋 Значения:")
                comp_df = pd.DataFrame({
                    'Дата': pd.to_datetime(dates).strftime('%Y-%m-%d'),
                    'Реально': y_true,
                    'Прогноз': y_pred,
                    'Ошибка': np.abs(y_true - y_pred)
                })
                print(comp_df.to_string(index=False))
                
            except (ValueError, IndexError):
                print("❌ Ошибка ввода")
                continue
        
        elif choice == '3':
            print("\n👋 До свидания!\n")
            break
        
        else:
            print("❌ Неверный выбор")

# ========================
# БЫСТРЫЙ ПРОГНОЗ
# ========================

def quick_forecast(data_path='data/DATA.csv', models_dir='models_nbeats_v2',
                   forecast_days=20):
    """Быстрый прогноз по всем моделям."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date']).dt.normalize()
    df = df.dropna().reset_index(drop=True)
    
    nomenclatures = sorted(df['Номенклатура'].unique())
    results = []
    
    for nomen in nomenclatures:
        df_sub = df[df['Номенклатура'] == nomen].sort_values('date').reset_index(drop=True)
        
        if len(df_sub) < 100:
            continue
        
        safe_name = safe_filename(nomen)
        model_path = os.path.join(models_dir, safe_name)
        
        if not os.path.exists(model_path):
            continue
        
        try:
            model, target_scaler, feature_scaler, config = load_model_complete(
                NBeatsModel, model_path, device=device
            )
            
            dataset = TimeSeriesDataset(
                df_sub,
                sequence_length=config['sequence_length'],
                target_column='Количество',
                use_features=True
            )
            
            preds = predict_nbeats(model, dataset, forecast_days, device)
            total = sum(preds)
            
            results.append({
                'Номенклатура': nomen,
                'Сумма_30дней': total,
                'Среднее': total / len(preds)
            })
        except:
            continue
    
    results_df = pd.DataFrame(results).sort_values('Сумма_30дней', ascending=False)
    return results_df

# ========================
# MAIN
# ========================

if __name__ == "__main__":
    interactive_evaluation()