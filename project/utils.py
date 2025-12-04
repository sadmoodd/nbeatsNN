"""
utils_fixed.py - Исправленная версия с правильной N-BEATS архитектурой
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import re
import os
import pickle

pd.set_option('future.no_silent_downcasting', True)

# ========================
# ТРАНСЛИТЕРАЦИЯ
# ========================



RUSLAT = {
    u'а': 'a', u'б': 'b', u'в': 'v', u'г': 'g', u'д': 'd', u'е': 'e', u'ё': 'e',
    u'ж': 'zh', u'з': 'z', u'и': 'i', u'й': 'j', u'к': 'k', u'л': 'l', u'м': 'm',
    u'н': 'n', u'о': 'o', u'п': 'p', u'р': 'r', u'с': 's', u'т': 't', u'у': 'u',
    u'ф': 'f', u'х': 'h', u'ц': 'c', u'ч': 'ch', u'ш': 'sh', u'щ': 'shh',
    u'ъ': '', u'ы': 'y', u'ь': '', u'э': 'e', u'ю': 'yu', u'я': 'ya'
}

def translit_ru(txt):
    res = ''
    for c in txt:
        up = c.isupper()
        c_l = c.lower()
        c_t = RUSLAT.get(c_l, c_l)
        if up: c_t = c_t.capitalize()
        res += c_t
    return res

def safe_filename(name: str) -> str:
    name = translit_ru(name)
    name = re.sub(r'[^a-zA-Z0-9-_]', '_', name)
    name = re.sub(r'_+', '_', name)
    name = name.strip('_')
    return name

# ========================
# ДАТАСЕТ - ИСПРАВЛЕННЫЙ
# ========================

class TimeSeriesDataset(Dataset):
    """
    Правильный датасет для временных рядов.
    Работает ТОЛЬКО с целевой переменной (Количество).
    Признаки (погода, календарь) используются как дополнительный контекст.
    """
    
    def __init__(self, dataframe, sequence_length=30, target_column='Количество', use_features=True):
        self.sequence_length = sequence_length
        self.target_column = target_column
        self.use_features = use_features
        self.df = dataframe.copy()
        self.df = self.df.sort_values('date').reset_index(drop=True)
        self.df['date'] = pd.to_datetime(self.df['date'])
        
        # Целевая переменная
        self.target_values = self.df[target_column].values.astype(np.float32)
        
        # Признаки (если используются)
        if use_features:
            feature_cols = [
                'Максимальная температура', 'Минимальная температура', 'Средняя температура',
                'Атмосферное давление, гПа', 'Скорость ветра, м/с', 'Осадки, мм',
                'Эффективная температура', 'year', 'month', 'day', 'day_of_week',
                'is_weekend', 'is_working', 'is_holiday', 'is_pre_holiday',
                'is_new_year', 'is_spring_holiday', 'is_may_holiday', 'season',
                'is_monday', 'is_friday', 'is_month_end', 'is_quarter_end', 'salary_week'
            ]
            self.feature_columns = [col for col in feature_cols if col in self.df.columns]
            
            # Масштабируем ТОЛЬКО признаки (целевая переменная масштабируется отдельно)
            feature_data = self.df[self.feature_columns].fillna(0).values
            self.feature_scaler = StandardScaler()
            self.scaled_features = self.feature_scaler.fit_transform(feature_data).astype(np.float32)
        else:
            self.feature_columns = []
            self.feature_scaler = None
            self.scaled_features = None
        
        # Масштабируем целевую переменную (ОТДЕЛЬНО!)
        self.target_scaler = StandardScaler()
        self.scaled_target = self.target_scaler.fit_transform(
            self.target_values.reshape(-1, 1)
        ).flatten().astype(np.float32)
    
    def __len__(self):
        return len(self.scaled_target) - self.sequence_length
    
    def __getitem__(self, idx):
        # История целевой переменной (последовательность)
        target_seq = self.scaled_target[idx:idx + self.sequence_length]  # shape: (seq_len,)
        
        # Цель (следующее значение целевой переменной)
        target_next = self.scaled_target[idx + self.sequence_length]  # scalar
        
        if self.use_features and self.scaled_features is not None:
            # Признаки для этого временного окна
            feature_seq = self.scaled_features[idx:idx + self.sequence_length]  # (seq_len, num_features)
            
            # ✅ КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ: объединяем и ФЛАТТЕНИМ!
            x_combined = np.concatenate([
                target_seq.reshape(-1, 1),      # (30, 1)
                feature_seq                     # (30, 24)
            ], axis=1)                          # (30, 25)
            
            x_flat = x_combined.flatten()       # (750,) ← ОТВЕТ!
            
        else:
            x_flat = target_seq                 # (30,)
        
        return torch.FloatTensor(x_flat), torch.FloatTensor([target_next])


# ========================
# N-BEATS АРХИТЕКТУРА (ИСПРАВЛЕННАЯ)
# ========================

class NBeatsBlock(nn.Module):
    """
    N-BEATS блок с правильными residual connections.
    
    Входные данные: (batch_size, seq_len, input_dim)
    где input_dim = 1 (целевая) + num_features
    """
    
    def __init__(self, 
                 input_size,  # seq_len * input_dim (flattened)
                 output_size,  # forecast horizon
                 hidden_sizes=None,
                 dropout=0.1):
        super(NBeatsBlock, self).__init__()
        
        if hidden_sizes is None:
            hidden_sizes = [512, 512, 512]
        
        self.input_size = input_size
        self.output_size = output_size
        
        # Стек полносвязных слоев
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        self.fc_stack = nn.Sequential(*layers)
        
        # Backcast head: восстанавливает входную последовательность
        self.backcast_head = nn.Linear(hidden_sizes[-1], input_size)
        
        # Forecast head: генерирует прогноз
        self.forecast_head = nn.Linear(hidden_sizes[-1], output_size)
    
    def forward(self, x):
        """
        x: (batch_size, input_size) - флаттенированная последовательность
        Returns:
            backcast: (batch_size, input_size)
            forecast: (batch_size, output_size)
        """
        hidden = self.fc_stack(x)
        backcast = self.backcast_head(hidden)
        forecast = self.forecast_head(hidden)
        return backcast, forecast

class NBeatsStack(nn.Module):
    """
    Стек N-BEATS блоков с residual connections.
    """
    
    def __init__(self,
                 num_blocks,
                 input_size,
                 output_size,
                 hidden_sizes=None,
                 dropout=0.1):
        super(NBeatsStack, self).__init__()
        
        if hidden_sizes is None:
            hidden_sizes = [512, 512, 512]
        
        self.blocks = nn.ModuleList([
            NBeatsBlock(input_size, output_size, hidden_sizes, dropout)
            for _ in range(num_blocks)
        ])
    
    def forward(self, x):
        """
        x: (batch_size, input_size)
        Returns:
            forecast_sum: (batch_size, output_size) - суммированные прогнозы всех блоков
            residual: (batch_size, input_size) - остаток после всех блоков
        """
        residual = x.clone()
        forecast_sum = None
        
        for block in self.blocks:
            backcast, forecast = block(residual)
            residual = residual - backcast  # Residual connection
            
            if forecast_sum is None:
                forecast_sum = forecast
            else:
                forecast_sum = forecast_sum + forecast
        
        return forecast_sum, residual

class NBeatsModel(nn.Module):
    """
    Полная N-BEATS архитектура с несколькими стеками.
    """
    
    def __init__(self,
                 input_size,
                 output_size,
                 num_stacks=3,
                 num_blocks_per_stack=3,
                 hidden_sizes=None,
                 dropout=0.1):
        super(NBeatsModel, self).__init__()
        
        if hidden_sizes is None:
            hidden_sizes = [512, 512, 512]
        
        self.stacks = nn.ModuleList([
            NBeatsStack(num_blocks_per_stack, input_size, output_size, hidden_sizes, dropout)
            for _ in range(num_stacks)
        ])
    
    def forward(self, x):
        """
        x: (batch_size, input_size)
        Returns:
            forecast: (batch_size, output_size)
        """
        forecasts = []
        for stack in self.stacks:
            forecast, _ = stack(x)
            forecasts.append(forecast)
        
        # Суммируем прогнозы всех стеков
        final_forecast = torch.stack(forecasts, dim=0).sum(dim=0)
        return final_forecast

# ========================
# ОБУЧЕНИЕ
# ========================

def train_nbeats(model, train_loader, val_loader,
                 epochs=150, lr=0.001, device='cpu',
                 early_stopping_patience=15, verbose=True):
    """
    Обучение N-BEATS модели.
    """
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(epochs):
        # ===== ОБУЧЕНИЕ =====
        model.train()
        train_losses = []
        
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            predictions = model(x_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
        
        train_loss = np.mean(train_losses)
        
        # ===== ВАЛИДАЦИЯ =====
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val = x_val.to(device)
                y_val = y_val.to(device)
                
                predictions = model(x_val)
                loss = criterion(predictions, y_val)
                val_losses.append(loss.item())
        
        val_loss = np.mean(val_losses)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        scheduler.step(val_loss)
        
        if verbose and (epoch + 1) % 20 == 0:
            print(f'  Epoch {epoch+1:3d}/{epochs} | Train: {train_loss:.6f} | Val: {val_loss:.6f}')
        
        # ===== EARLY STOPPING =====
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), '_best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                if verbose:
                    print(f'  Early stopping at epoch {epoch+1}')
                break
    
    # Загружаем лучшую модель
    if os.path.exists('_best_model.pth'):
        model.load_state_dict(torch.load('_best_model.pth'))
        os.remove('_best_model.pth')
    
    return model, history

# ========================
# ПРОГНОЗИРОВАНИЕ
# ========================

def predict_nbeats(model, dataset, num_steps=30, device='cpu'):
    """
    Прогнозирование на num_steps шагов вперёд.
    
    Возвращает прогнозы в ОРИГИНАЛЬНОЙ шкале (денормализованные).
    """
    model.eval()
    
    # Получаем последнюю последовательность
    seq_len = dataset.sequence_length
    last_target = dataset.scaled_target[-seq_len:].copy()  # (seq_len,)
    
    if dataset.use_features:
        last_features = dataset.scaled_features[-seq_len:].copy()  # (seq_len, num_features)
        # Объединяем: целевая + признаки
        current_seq = np.concatenate([
            last_target.reshape(-1, 1),
            last_features
        ], axis=1)  # (seq_len, 1 + num_features)
    else:
        current_seq = last_target.reshape(-1, 1)  # (seq_len, 1)
    
    predictions_scaled = []
    
    with torch.no_grad():
        for step in range(num_steps):
            # Флаттенируем для модели
            seq_flat = current_seq.flatten()  # (seq_len * (1 + num_features),)
            seq_tensor = torch.FloatTensor(seq_flat).unsqueeze(0).to(device)
            
            # Предсказываем
            pred_scaled = model(seq_tensor).cpu().numpy()[0, 0]
            predictions_scaled.append(pred_scaled)
            
            # Обновляем последовательность: сдвигаемся на шаг
            if dataset.use_features:
                # Используем последний вектор признаков (они меняются редко)
                next_features = current_seq[-1, 1:].copy()
                new_row = np.concatenate([[pred_scaled], next_features])
            else:
                new_row = np.array([pred_scaled])
            
            current_seq = np.vstack([current_seq[1:], new_row])
    
    # ===== ДЕНОРМАЛИЗАЦИЯ =====
    predictions_original = []
    for pred_scaled in predictions_scaled:
        # Используем target_scaler для денормализации
        pred_original = dataset.target_scaler.inverse_transform(
            np.array([[pred_scaled]])
        )[0, 0]
        predictions_original.append(float(pred_original))
    
    return predictions_original

# ========================
# СОХРАНЕНИЕ / ЗАГРУЗКА
# ========================

def save_model_complete(model, dataset, config, save_path):
    """Сохраняет модель и всю необходимую информацию."""
    os.makedirs(save_path, exist_ok=True)
    
    # Модель
    torch.save(model.state_dict(), os.path.join(save_path, 'model.pth'))
    
    # Scalers
    with open(os.path.join(save_path, 'target_scaler.pkl'), 'wb') as f:
        pickle.dump(dataset.target_scaler, f)
    
    if dataset.feature_scaler is not None:
        with open(os.path.join(save_path, 'feature_scaler.pkl'), 'wb') as f:
            pickle.dump(dataset.feature_scaler, f)
    
    # Конфигурация
    with open(os.path.join(save_path, 'config.pkl'), 'wb') as f:
        pickle.dump(config, f)
    
    # Список признаков
    if dataset.feature_columns:
        np.save(os.path.join(save_path, 'feature_columns.npy'), 
                np.array(dataset.feature_columns, dtype=object))

def load_model_complete(model_class, save_path, device='cpu'):
    """Загружает модель со всеми параметрами."""
    # Конфигурация
    with open(os.path.join(save_path, 'config.pkl'), 'rb') as f:
        config = pickle.load(f)
    
    # Создаём и загружаем модель
    model = model_class(**config['model_params'])
    model.load_state_dict(torch.load(os.path.join(save_path, 'model.pth'), 
                                     map_location=device))
    model.to(device)
    model.eval()
    
    # Scalers
    with open(os.path.join(save_path, 'target_scaler.pkl'), 'rb') as f:
        target_scaler = pickle.load(f)
    
    feature_scaler = None
    feature_path = os.path.join(save_path, 'feature_scaler.pkl')
    if os.path.exists(feature_path):
        with open(feature_path, 'rb') as f:
            feature_scaler = pickle.load(f)
    
    return model, target_scaler, feature_scaler, config