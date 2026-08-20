import os
import numpy as np
import pandas as pd
from urllib import request
import shutil
import zipfile
import json
from generate_mask import generate_mask

DATA_DIR = 'datasets'


NAME_URL_DICT_UCI = {
    'adult': 'https://archive.ics.uci.edu/static/public/2/adult.zip',
    'default': 'https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip',
    'shoppers': 'https://archive.ics.uci.edu/static/public/468/online+shoppers+purchasing+intention+dataset.zip',
    'news': 'https://archive.ics.uci.edu/static/public/332/online+news+popularity.zip',
    'bike': 'https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip'
}

def unzip_file(zip_filepath, dest_path):
    with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
        zip_ref.extractall(dest_path)


def download_from_uci(name):

    print(f'Start processing dataset {name} from UCI.')
    save_dir = f'{DATA_DIR}/{name}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

        url = NAME_URL_DICT_UCI[name]
        request.urlretrieve(url, f'{save_dir}/{name}.zip')
        print(f'Finish downloading dataset from {url}, data has been saved to {save_dir}.')
        
        unzip_file(f'{save_dir}/{name}.zip', save_dir)
        print(f'Finish unzipping {name}.')
    
    else:
        print('Aready downloaded.')

def process_adult():
    data_dir = f'{DATA_DIR}/adult'
    df = pd.read_csv(f'{data_dir}/adult.data', header=None)
    df.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', ]
                  
def process_gesture():

    file_names = ['a1_va3', 'a2_va3', 'a3_va3', 'b1_va3', 'b1_va3', 'c1_va3', 'c3_va3']
    datas = []
    for name in file_names:
        df = pd.read_csv(f'{DATA_DIR}/gesture/{name}.csv')
        data = df.to_numpy()
        datas.append(data)

    data = np.concatenate(datas, axis=0)
    data_df = pd.DataFrame(data)
    data_df.to_csv(f'{DATA_DIR}/gesture/data.csv', index = False)


def process_letter():
    dataname = 'letter'
    path = f'{DATA_DIR}/{dataname}/{dataname}-recognition.data'
    save_path = f'{DATA_DIR}/{dataname}/data.csv'
    df = pd.read_csv(path, header = None)

    cols = df.columns.tolist()
    cols = cols[1:] + cols[:1]

    df = df[cols]
    df.to_csv(save_path, index = False, header = True)

def process_news():
    path = f'{DATA_DIR}/news/OnlineNewsPopularity/OnlineNewsPopularity.csv'
    save_path = f'{DATA_DIR}/news/data.csv'

    data_df = pd.read_csv(path)
    data_df = data_df.drop('url', axis=1)

    columns = np.array(data_df.columns.tolist())

    cat_columns1 = columns[list(range(12,18))]
    cat_columns2 = columns[list(range(30,38))]

    cat_col1 = data_df[cat_columns1].astype(int).to_numpy().argmax(axis = 1)
    cat_col2 = data_df[cat_columns2].astype(int).to_numpy().argmax(axis = 1)

    data_df = data_df.drop(cat_columns2, axis=1)
    data_df = data_df.drop(cat_columns1, axis=1)

    data_df['data_channel'] = cat_col1
    data_df['weekday'] = cat_col2
    
    data_df.to_csv(f'{save_path}', index = False)

def process_adult():
    path = f'{DATA_DIR}/adult/adult.data'
    save_path = f'{DATA_DIR}/adult/data.csv'
    data_df = pd.read_csv(path, header=None)

    df_cleaned = data_df.dropna()
    df_cleaned.to_csv(save_path, index = False)

def process_bike():
    path = f'{DATA_DIR}/bike/hour.csv'
    save_path = f'{DATA_DIR}/bike/data.csv'
    data_df = pd.read_csv(path)
    data_df = data_df.drop(['instant', 'dteday'], axis=1)

    df_cleaned = data_df.dropna()
    df_cleaned.to_csv(save_path, index = False)

def process_shoppers():
    path = f'{DATA_DIR}/shoppers/online_shoppers_intention.csv'
    save_path = f'{DATA_DIR}/shoppers/data.csv'
    data_df = pd.read_csv(path)

    df_cleaned = data_df.dropna()
    df_cleaned.to_csv(save_path, index = False)

def process_default():
    path = f'{DATA_DIR}/default/default of credit card clients.xls'
    save_path = f'{DATA_DIR}/default/data.csv'
    data_df = pd.read_excel(path, sheet_name='Data', header=1)
    data_df = data_df.drop('ID', axis=1)

    df_cleaned = data_df.dropna()
    df_cleaned.to_csv(save_path, index = False)

def process_magic():

    path = f'{DATA_DIR}/magic/magic04.data'
    save_path = f'{DATA_DIR}/magic/data.csv'
    data_df = pd.read_csv(path, header=None)
    columns = data_df.columns
    df_cleaned = data_df.dropna()
    df_cleaned.to_csv(save_path, index = False)

def process_bean():

    path = f'{DATA_DIR}/bean/DryBeanDataset/Dry_Bean_Dataset.xlsx'
    save_path = f'{DATA_DIR}/bean/data.csv'
    data_df = pd.read_excel(path, sheet_name='Dry_Beans_Dataset', header=1)

    df_cleaned = data_df.dropna()
    df_cleaned.to_csv(save_path, index = False)

# =============================================================================
# [MODIFIKASI] train_test_split -> train_val_test_split
#
# Alasan perubahan:
#   Dulu data cuma dibagi 2 (train/test). Sekarang dibagi 3 (train/val/test)
#   supaya ada validation set khusus buat early stopping / checkpoint
#   selection saat training model diffusion (lihat main_mrmd.py).
#
#   Komposisi default 60:10:30 sengaja dipilih supaya PORSI TEST TETAP 30%,
#   sama persis dengan skema 70:30 di paper acuan (DiffPuter). Yang berubah
#   cuma porsi "70% train" di paper itu, sekarang dipecah lagi jadi
#   60% train-inner + 10% val-inner secara internal. Total train+val = 70%,
#   identik dengan pool training di paper -> hasil out-of-sample (test) tetap
#   apple-to-apple buat dibandingkan.
#
#   seed tetap SAMA (1234) supaya split ini persis reproducible dan bisa
#   dipakai bareng-bareng oleh baseline DiffPuter maupun model MRmD kamu --
#   dua-duanya WAJIB baca dari train.csv/val.csv/test.csv yang sama, bukan
#   split ulang sendiri-sendiri, biar perbandingannya fair.
# =============================================================================
def train_val_test_split(dataname, train_ratio = 0.6, val_ratio = 0.1, test_ratio = 0.3, seed = 1234):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        f'train_ratio + val_ratio + test_ratio harus = 1.0, sekarang = {train_ratio + val_ratio + test_ratio}'

    data_dir = f'{DATA_DIR}/{dataname}'
    path = f'{DATA_DIR}/{dataname}/data.csv'
    info_path = f'{DATA_DIR}/Info/{dataname}.json'

    with open(info_path, 'r') as f:
        info = json.load(f)

    cat_idx = info['cat_col_idx']
    num_idx = info['num_col_idx']

    data_df = pd.read_csv(path)
    total_num = data_df.shape[0]

    if len(cat_idx) == 0:
        data_values = data_df.values[:, :-1].astype(np.float32)

        nan_idx = np.isnan(data_values).nonzero()[0]

        keep_idx = list(set(np.arange(data_values.shape[0])) - set(list(nan_idx)))
        keep_idx = np.array(keep_idx)
    else:
        keep_idx = np.arange(total_num)

    n_keep = keep_idx.shape[0]
    num_train = int(n_keep * train_ratio)
    num_val   = int(n_keep * val_ratio)
    # [MODIFIKASI] sisa dialokasikan ke test (bukan dihitung ulang dari ratio)
    # supaya total train+val+test selalu tepat n_keep, menghindari kehilangan
    # baris akibat pembulatan int().
    num_test  = n_keep - num_train - num_val

    np.random.seed(seed)
    np.random.shuffle(keep_idx)

    train_idx = keep_idx[:num_train]
    val_idx   = keep_idx[num_train : num_train + num_val]
    test_idx  = keep_idx[num_train + num_val :]

    train_df = data_df.loc[train_idx]
    val_df   = data_df.loc[val_idx]
    test_df  = data_df.loc[test_idx]

    train_path = f'{data_dir}/train.csv'
    val_path   = f'{data_dir}/val.csv'
    test_path  = f'{data_dir}/test.csv'

    train_df.to_csv(train_path, index = False)
    val_df.to_csv(val_path, index = False)
    test_df.to_csv(test_path, index = False)

    print(f'Splitting Train/Val/Test data for {dataname} is done.')
    print(f'Train data shape: {train_df.shape}, Val data shape: {val_df.shape}, Test data shape: {test_df.shape}')
    print(f'Ratio realisasi -> train: {len(train_idx)/n_keep:.3f}, '
          f'val: {len(val_idx)/n_keep:.3f}, test: {len(test_idx)/n_keep:.3f}')
    print(f'Saved at {train_path}, {val_path}, {test_path}.')

    # Catatan: generate mask (train_mask/val_mask/test_mask) sekarang
    # dilakukan lewat generate_mask() di generate_mask.py (dipanggil di
    # blok __main__ di bawah), bukan di sini.


if __name__ == '__main__':

    # Downloading dataset
    for name in NAME_URL_DICT_UCI.keys():
        download_from_uci(name)

    for name in NAME_URL_DICT_UCI.keys():
        eval(f'process_{name}()')
        # [MODIFIKASI] train_test_split(ratio=0.7) -> train_val_test_split(60:10:30)
        train_val_test_split(name, train_ratio = 0.65, val_ratio = 0.15, test_ratio = 0.2)
        for mask_type in ['MCAR', 'MAR', 'MNAR_logistic_T2']:
            for mask_p in [0.3]:
                
                generate_mask(dataname = name,
                                mask_type = mask_type,
                                mask_num = 10,
                                p = mask_p,
                                )