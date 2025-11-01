import os
import re
import pandas as pd
import numpy as np
import torch
from utils import FishSalesDataset, LSTMModel, predict_future, safe_filename

# --- Простая транслитерация (как в train_models.py) ---
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

# --- Загрузка моделей и скалеров ---
def load_all_models(models_dir='models'):
    models = {}
    scalers = {}
    features = {}

    for file in os.listdir(models_dir):
        if file.endswith('.pth'):
            base_name = file.replace('.pth', '')
            model_path = os.path.join(models_dir, file)
            scaler_path = os.path.join(models_dir, f"{base_name}_scaler.npy")
            features_path = os.path.join(models_dir, f"{base_name}_features.npy")

            # Проверяем наличие сопутствующих файлов
            if os.path.exists(scaler_path) and os.path.exists(features_path):
                scaler = np.load(scaler_path, allow_pickle=True).item()
                feat_cols = np.load(features_path, allow_pickle=True)
                if isinstance(feat_cols, np.ndarray):  # если это массив
                    feat_cols = feat_cols.tolist()
                # Создаем модель с нужным размером входа
                model = LSTMModel(input_size=len(feat_cols))
                model.load_state_dict(torch.load(model_path, map_location='cpu'))
                model.eval()
                models[base_name] = model
                scalers[base_name] = scaler
                features[base_name] = feat_cols
    return models, scalers, features

# --- Основной цикл меню ---
def predict_menu_loop(models, scalers, features, df):
    print("Система прогноза запущена. Введите команды.")

    while True:
        print("\nВыберите режим прогноза:\n1 - Одна дата\n2 - Интервал дат")
        choice = input("Ввод: ").strip()

        if choice == '1':
            date_str = input("Введите дату (ГГГГ-ММ-ДД): ").strip()
            try:
                pred_date = pd.to_datetime(date_str)
            except:
                print("Неверный формат даты")
                continue
            result_df = run_prediction(models, scalers, features, df, pred_date, pred_date)
            print("\nПрогноз на дату:", pred_date.date())
            print(result_df)

        elif choice == '2':
            start_str = input("Введите начало интервала (ГГГГ-ММ-ДД): ").strip()
            end_str = input("Введите конец интервала (ГГГГ-ММ-ДД): ").strip()
            try:
                start_date = pd.to_datetime(start_str)
                end_date = pd.to_datetime(end_str)
                if end_date < start_date:
                    print("Конец интервала меньше начала")
                    continue
            except:
                print("Ошибка в формате даты")
                continue

            result_df = run_prediction(models, scalers, features, df, start_date, end_date)
            print(f"\nПрогноз на период: {start_date.date()} - {end_date.date()}")
            print(result_df)

        else:
            print("Неверный выбор")

# --- Прогноз для одной или интервала дат ---
def run_prediction(models, scalers, features, df, start_date, end_date):
    results = []
    last_date = df['date'].max()

    days = (end_date - last_date).days + 1
    if days <= 0:
        print("Даты прогноза должны быть позже последней даты в данных")
        return pd.DataFrame()

    for nomenklatura in df['Номенклатура'].unique():
        safe_name = safe_filename(nomenklatura)
        model = models.get(safe_name)
        scaler = scalers.get(safe_name)
        features_cols = features.get(safe_name)
        if model is None or scaler is None or features_cols is None:
            print(f"Нет модели для номенклатуры {nomenklatura}")
            continue

        df_sub = df[df['Номенклатура'] == nomenklatura].reset_index(drop=True)
        if len(df_sub) < 100:
            continue

        df_sub['date'] = pd.to_datetime(df_sub['date'])
        # Временные признаки
        df_sub['year'] = df_sub['date'].dt.year
        df_sub['month'] = df_sub['date'].dt.month
        df_sub['day'] = df_sub['date'].dt.day
        df_sub['day_of_week'] = df_sub['date'].dt.dayofweek
        df_sub['is_weekend'] = (df_sub['day_of_week'] >= 5).astype(int)
        df_sub[features_cols + ['Количество']] = df_sub[features_cols + ['Количество']].fillna(0)

        scaled_data = scaler.transform(df_sub[features_cols + ['Количество']])
        sequence = scaled_data[-30:, :-1]

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        seq_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(device)

        preds_scaled = []
        with torch.no_grad():
            for _ in range(days):
                pred = model(seq_tensor).cpu().numpy().flatten()[0]
                preds_scaled.append(pred)
                last_features = seq_tensor[0, -1, :-1].cpu().numpy()
                next_row = np.append(last_features, pred)
                seq_tensor = torch.cat((seq_tensor[:, 1:, :], torch.tensor(next_row, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)), dim=1)

        preds = []
        for p in preds_scaled:
            dummy = np.zeros(len(features_cols)+1)
            dummy[-1] = p
            pred_orig = scaler.inverse_transform([dummy])[0, -1]
            preds.append(max(0, pred_orig))

        total_pred = sum(preds)
        results.append({'Номенклатура': nomenklatura, 'Количество_прогноз': total_pred})

    result_df = pd.DataFrame(results).sort_values('Количество_прогноз', ascending=False).reset_index(drop=True)
    return result_df



def main():
    print("Загрузка данных для прогнозов...")
    df = pd.read_csv('data/DATA.csv')
    df['date'] = pd.to_datetime(df['date']).dt.normalize()
    df = df.dropna().reset_index(drop=True)

    print("Загрузка моделей из папки models...")
    models, scalers, features = load_all_models('models')
    print(f"Загружено моделей: {len(models)}")

    predict_menu_loop(models, scalers, features, df)

if __name__ == "__main__":
    main()