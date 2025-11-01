import os
import re
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
# Импорт этих классов из utils.py
from utils import FishSalesDataset, LSTMModel, train_model, safe_filename

# --- Простая транслитерация (без сторонних библиотек) ---
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
        if up:
            c_t = c_t.capitalize()
        res += c_t
    return res

def safe_filename(name: str) -> str:
    name = translit_ru(name)
    name = re.sub(r'[^a-zA-Z0-9-_]', '_', name)
    name = re.sub(r'_+', '_', name)
    name = name.strip('_')
    return name

def main():
    data_path = 'data/DATA.csv'
    models_dir = 'models'
    os.makedirs(models_dir, exist_ok=True)

    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date']).dt.normalize()
    df = df.dropna().reset_index(drop=True)

    nomenklatura_list = df['Номенклатура'].unique()

    for nomenklatura in nomenklatura_list:
        print(f"Обучаем модель для номенклатуры: {nomenklatura}")
        df_sub = df[df['Номенклатура'] == nomenklatura].reset_index(drop=True)
        if len(df_sub) < 100:
            print("Слишком мало данных для обучения, пропускаем")
            continue

        dataset = FishSalesDataset(df_sub, sequence_length=30, target_column='Количество')
        size = len(dataset)
        train_size = int(0.8 * size)
        val_size = size - train_size

        train_set, val_set = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
        train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=32, shuffle=False)

        model = LSTMModel(input_size=len(dataset.features_columns), hidden_size=128, num_layers=2, dropout=0.3)
        model = train_model(model, train_loader, val_loader, epochs=100, lr=0.001)

        safe_name = safe_filename(nomenklatura)

        model_path = os.path.join(models_dir, f"{safe_name}.pth")
        scaler_path = os.path.join(models_dir, f"{safe_name}_scaler.npy")
        features_path = os.path.join(models_dir, f"{safe_name}_features.npy")

        torch.save(model.state_dict(), model_path)
        np.save(scaler_path, dataset.scaler)
        np.save(features_path, np.array(dataset.features_columns, dtype="object"))

        print(f"Модель для '{nomenklatura}' сохранена как: {model_path}")

if __name__ == "__main__":
    main()
