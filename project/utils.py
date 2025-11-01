import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import re

# ------------------------
# Транслитерация и очистка
# ------------------------
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

    
class FishSalesDataset(Dataset):
    def __init__(self, dataframe, sequence_length=30, target_column='Количество'):
        self.sequence_length = sequence_length
        self.target_column = target_column
        self.df = dataframe.copy()
        self.df = self.df.sort_values('date').reset_index(drop=True)

        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df['year'] = self.df['date'].dt.year
        self.df['month'] = self.df['date'].dt.month
        self.df['day'] = self.df['date'].dt.day
        self.df['day_of_week'] = self.df['date'].dt.dayofweek
        self.df['is_weekend'] = (self.df['day_of_week'] >= 5).astype(int)

        numeric_columns = [
            'Максимальная температура', 'Минимальная температура', 'Средняя температура',
            'Атмосферное давление, гПа', 'Скорость ветра, м/с', 'Осадки, мм',
            'Эффективная температура', 'year', 'month', 'day', 'day_of_week',
            'is_weekend', 'is_working', 'is_holiday', 'is_pre_holiday',
            'is_new_year', 'is_spring_holiday', 'is_may_holiday', 'season',
            'is_monday', 'is_friday', 'is_month_end', 'is_quarter_end', 'salary_week'
        ]
        self.features_columns = [col for col in numeric_columns if col in self.df.columns]

        self.df[self.features_columns + [self.target_column]] = self.df[self.features_columns + [self.target_column]].fillna(0)

        self.scaler = StandardScaler()
        scaled = self.scaler.fit_transform(self.df[self.features_columns + [self.target_column]])

        self.scaled_features = scaled[:, :-1]
        self.scaled_target = scaled[:, -1]

    def __len__(self):
        return len(self.scaled_features) - self.sequence_length

    def __getitem__(self, idx):
        x = self.scaled_features[idx:idx + self.sequence_length]
        y = self.scaled_target[idx + self.sequence_length]
        return torch.FloatTensor(x), torch.FloatTensor([y])

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

def train_model(model, train_loader, val_loader, epochs=100, lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    best_loss = float('inf')
    patience = 10
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            out = model(x_batch)
            loss = criterion(out, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        train_loss = np.mean(train_losses)

        model.eval()
        val_losses = []
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                out_val = model(x_val)
                loss_val = criterion(out_val, y_val)
                val_losses.append(loss_val.item())
        val_loss = np.mean(val_losses)
        scheduler.step(val_loss)

        print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping on epoch {epoch+1}')
                break

    model.load_state_dict(torch.load('best_model.pth'))
    return model

def predict_future(model, dataset, days=30):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    data = dataset.scaled_features
    seq_len = dataset.sequence_length
    scaler = dataset.scaler

    sequence = data[-seq_len:].copy()
    predictions = []

    for _ in range(days):
        seq_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(device)
        with torch.no_grad():
            pred_scaled = model(seq_tensor).cpu().numpy()[0, 0]

        dummy = np.zeros((1, len(dataset.features_columns) + 1))
        dummy[0, -1] = pred_scaled
        pred_original = scaler.inverse_transform(dummy)[0, -1]
        pred_original = max(0, pred_original)

        predictions.append(pred_original)

        next_features = sequence[-1].copy()
        next_features[-1] = pred_scaled
        sequence = np.vstack([sequence[1:], next_features])

    return predictions

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
import os


# ================================
# 1. N-BEATS BLOCK (исправленный)
# ================================

class NBeatsBlock(nn.Module):
    """
    Базовый блок N-BEATS с residual connections.

    ИСПРАВЛЕНИЕ: теперь правильно работает с флаттенированными входами

    Входные данные должны быть формы: (batch_size, flattened_input_size)
    где flattened_input_size = sequence_length * num_features
    """

    def __init__(self, 
                 input_size, 
                 output_size, 
                 hidden_layers=None,
                 dropout=0.1,
                 activation='relu'):
        super(NBeatsBlock, self).__init__()

        if hidden_layers is None:
            hidden_layers = [512, 512]

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout

        # Активационная функция
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.ReLU()

        # ==========================================
        # ОСНОВНОЙ МОДУЛЬ: Стек полносвязных слоев
        # ==========================================

        layers = []
        prev_size = input_size

        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(self.activation)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_size = hidden_size

        self.fc_layers = nn.Sequential(*layers)

        # ==========================================
        # ВЫХОДНЫЕ СЛОИ
        # ==========================================

        # 1. Backcast head - восстанавливает входную последовательность
        self.backcast_fc = nn.Linear(hidden_layers[-1], input_size)

        # 2. Forecast head - генерирует прогноз
        self.forecast_fc = nn.Linear(hidden_layers[-1], output_size)

    def forward(self, x):
        """
        Входные параметры:
        - x: (batch_size, flattened_input_size) - флаттенированная входная последовательность

        Выходные параметры:
        - backcast: (batch_size, flattened_input_size) - восстановленный вход
        - forecast: (batch_size, output_size) - прогноз
        """

        # Проверяем размеры (для debug)
        if x.shape[-1] != self.input_size:
            raise ValueError(
                f"Ошибка размера: ожидается input_size={self.input_size}, "
                f"получено {x.shape[-1]}. "
                f"Полная форма батча: {x.shape}"
            )

        # Пропускаем через стек полносвязных слоев
        hidden = self.fc_layers(x)

        # Генерируем backcast и forecast
        backcast = self.backcast_fc(hidden)
        forecast = self.forecast_fc(hidden)

        return backcast, forecast


# ================================
# 2. N-BEATS STACK (исправленный)
# ================================

class NBeatsStack(nn.Module):
    """
    Стек блоков N-BEATS.
    """

    def __init__(self, 
                 num_blocks,
                 input_size,
                 output_size,
                 hidden_layers=None,
                 dropout=0.1):
        super(NBeatsStack, self).__init__()

        if hidden_layers is None:
            hidden_layers = [512, 512]

        self.num_blocks = num_blocks
        self.input_size = input_size
        self.output_size = output_size

        # Создаем несколько блоков в стеке
        self.blocks = nn.ModuleList([
            NBeatsBlock(
                input_size=input_size,
                output_size=output_size,
                hidden_layers=hidden_layers,
                dropout=dropout
            )
            for _ in range(num_blocks)
        ])

    def forward(self, x):
        """
        Входные параметры:
        - x: (batch_size, input_size) - входная последовательность

        Выходные параметры:
        - forecast_sum: (batch_size, output_size) - суммированные прогнозы
        - residual: (batch_size, input_size) - остаток после всех блоков
        """

        # Инициализируем остаток первоначально равным входу
        residual = x
        forecast_sum = None

        for i, block in enumerate(self.blocks):
            try:
                # Пропускаем остаток через блок
                backcast, forecast = block(residual)

                # Обновляем остаток: вычитаем backcast из текущего остатка
                residual = residual - backcast

                # Суммируем прогнозы
                if forecast_sum is None:
                    forecast_sum = forecast
                else:
                    forecast_sum = forecast_sum + forecast
            except Exception as e:
                print(f"Ошибка в блоке {i} стека:")
                print(f"  Форма residual: {residual.shape}")
                print(f"  Форма backcast: {backcast.shape}")
                print(f"  Форма forecast: {forecast.shape}")
                raise e

        return forecast_sum, residual


# ================================
# 3. ПОЛНАЯ N-BEATS АРХИТЕКТУРА
# ================================

class NBeatsModel(nn.Module):
    """
    Полная архитектура N-BEATS с несколькими стеками.
    """

    def __init__(self, 
                 input_size,
                 output_size,
                 num_stacks=3,
                 num_blocks=3,
                 hidden_layers=None,
                 dropout=0.1):
        super(NBeatsModel, self).__init__()

        if hidden_layers is None:
            hidden_layers = [512, 512]

        self.input_size = input_size
        self.output_size = output_size
        self.num_stacks = num_stacks

        # Создаем несколько стеков
        self.stacks = nn.ModuleList([
            NBeatsStack(
                num_blocks=num_blocks,
                input_size=input_size,
                output_size=output_size,
                hidden_layers=hidden_layers,
                dropout=dropout
            )
            for _ in range(num_stacks)
        ])

    def forward(self, x):
        """
        Входные параметры:
        - x: (batch_size, input_size) - входная последовательность

        Выходные параметры:
        - forecast: (batch_size, output_size) - итоговый прогноз
        """

        forecasts = []

        for stack in self.stacks:
            forecast, _ = stack(x)
            forecasts.append(forecast)

        # Суммируем прогнозы всех стеков
        final_forecast = sum(forecasts)

        return final_forecast


# ================================
# 4. ФУНКЦИИ ОБУЧЕНИЯ
# ================================

def train_nbeats_model(model, train_loader, val_loader, 
                       epochs=100, lr=0.001, device='cpu',
                       early_stopping_patience=10):
    """
    Функция обучения N-BEATS модели с обработкой ошибок.
    """

    model.to(device)

    # Оптимизатор
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Loss функция
    criterion = nn.MSELoss()

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=5
    )

    best_val_loss = float('inf')
    patience_counter = 0
    training_history = {'train_loss': [], 'val_loss': []}

    for epoch in range(epochs):
        # ==================
        # ОБУЧЕНИЕ
        # ==================
        model.train()
        train_losses = []

        try:
            for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                # Проверяем размеры
                if x_batch.dim() != 2:
                    raise ValueError(
                        f"Ожидается 2D батч (batch_size, input_size), "
                        f"получено {x_batch.dim()}D с формой {x_batch.shape}"
                    )

                # Прямой проход
                optimizer.zero_grad()
                predictions = model(x_batch)

                # Вычисляем loss
                loss = criterion(predictions, y_batch)

                # Обратный проход
                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())

        except Exception as e:
            print(f"\n❌ Ошибка в батче {batch_idx} эпохи {epoch+1}:")
            print(f"   Форма x_batch: {x_batch.shape}")
            print(f"   Форма y_batch: {y_batch.shape}")
            print(f"   Сообщение ошибки: {str(e)}")
            raise e

        train_loss = np.mean(train_losses)

        # ==================
        # ВАЛИДАЦИЯ
        # ==================
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

        # Сохраняем историю
        training_history['train_loss'].append(train_loss)
        training_history['val_loss'].append(val_loss)

        # Learning rate scheduler step
        scheduler.step(val_loss)

        # Вывод прогресса
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs} | '
                  f'Train Loss: {train_loss:.6f} | '
                  f'Val Loss: {val_loss:.6f}')

        # ==================
        # EARLY STOPPING
        # ==================
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Сохраняем лучшую модель
            torch.save(model.state_dict(), 'best_nbeats_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f'Early stopping at epoch {epoch+1}')
                break

    # Загружаем лучшую модель
    if os.path.exists('best_nbeats_model.pth'):
        model.load_state_dict(torch.load('best_nbeats_model.pth'))

    return model, training_history


def predict_nbeats(model, dataset, num_steps=30, device='cpu'):
    """
    Функция для прогнозирования будущих значений N-BEATS моделью.

    ИСПРАВЛЕНО: работает с флаттенированными данными
    """

    model.eval()

    # Получаем последовательность из данных
    sequence = dataset.scaled_features[-dataset.sequence_length:].copy()
    scaler = dataset.scaler

    predictions = []

    with torch.no_grad():
        for _ in range(num_steps):
            # Флаттенируем последовательность
            # sequence.shape = (sequence_length, num_features)
            sequence_flat = sequence.flatten()  # Shape: (sequence_length * num_features,)
            sequence_tensor = torch.FloatTensor(sequence_flat).unsqueeze(0).to(device)

            # Пропускаем через модель
            pred_flat = model(sequence_tensor).cpu().numpy()[0]
            pred_scaled = pred_flat[0]  # Берем первый элемент прогноза (7 дней)

            predictions.append(pred_scaled)

            # Обновляем последовательность: убираем первый элемент, добавляем новый
            next_features = sequence[-1].copy()
            next_features[-1] = pred_scaled  # Последний элемент - целевая переменная

            sequence = np.vstack([sequence[1:], next_features])

    # Денормализуем прогнозы
    denorm_predictions = []
    for pred_scaled in predictions:
        dummy = np.zeros((1, len(dataset.features_columns) + 1))
        dummy[0, -1] = pred_scaled
        pred_original = scaler.inverse_transform(dummy)[0, -1]
        denorm_predictions.append(max(0, pred_original))

    return denorm_predictions