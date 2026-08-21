import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import os
import json

DATA_DIR = 'datasets'

def load_dataset(dataname, idx = 0, mask_type = 'MCAR', ratio = '30'):
    data_dir = f'datasets/{dataname}'
    info_path = f'datasets/Info/{dataname}.json'

    with open(info_path, 'r') as f:
        info = json.load(f)

    num_col_idx = info['num_col_idx']
    cat_col_idx = info['cat_col_idx']
    target_col_idx = info['target_col_idx']

    train_path = f'{data_dir}/train.csv'
    test_path = f'{data_dir}/test.csv'

    train_mask_path = f'{data_dir}/masks/rate{ratio}/{mask_type}/train_mask_{idx}.npy'
    test_mask_path = f'{data_dir}/masks/rate{ratio}/{mask_type}/test_mask_{idx}.npy'

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)


    train_mask = np.load(train_mask_path)
    test_mask = np.load(test_mask_path)

    cols = train_df.columns

    train_num = train_df[cols[num_col_idx]].values.astype(np.float32)
    train_cat = train_df[cols[cat_col_idx]].astype(str)
    train_y = train_df[cols[target_col_idx]]


    test_num = test_df[cols[num_col_idx]].values.astype(np.float32)
    test_cat = test_df[cols[cat_col_idx]].astype(str)
    test_y = test_df[cols[target_col_idx]]
    
    cat_columns = train_cat.columns

    train_cat_idx, test_cat_idx = None, None
    extend_train_mask = None
    extend_test_mask = None
    cat_bin_num = None


    # only contain numerical features

    if len(cat_col_idx) == 0:
        train_X = train_num
        test_X = test_num

        extend_train_mask = train_mask[:, num_col_idx]
        extend_test_mask = test_mask[:, num_col_idx]

    # Contain both numerical and categorical features

    else:

        if not os.path.exists(f'{data_dir}/{cat_columns[0]}_map.json'):

            for column in cat_columns:
                map_path_bin = f'{data_dir}/{column}_map_bin.json'
                map_path_idx = f'{data_dir}/{column}_map_idx.json'

                # ANTI DATA-LEAKAGE: vocabulary kategori HANYA dibangun dari train_cat
                # (bukan dari data.csv / gabungan train+test). Dengan begitu urutan indeks,
                # jumlah kategori, dan jumlah bit encoding tidak pernah "mengintip" test set.
                categories = sorted(train_cat[column].unique().tolist())
                num_categories = len(categories)

                # Kategori yang muncul di test tapi TIDAK pernah muncul di train dianggap
                # out-of-vocabulary (OOV) dan dipetakan ke satu slot cadangan '__UNK__',
                # bukan diberi index baru berdasarkan test set (yang akan jadi leakage).
                category_to_idx = {category: index for index, category in enumerate(categories)}
                category_to_idx['__UNK__'] = num_categories

                total_slots = num_categories + 1  # +1 untuk slot OOV/'__UNK__'
                num_bits = (total_slots - 1).bit_length()

                category_to_binary = {
                    category: format(index, '0' + str(num_bits) + 'b')
                    for category, index in category_to_idx.items()
                }

                with open(map_path_bin, 'w') as f:
                    json.dump(category_to_binary, f)
                with open(map_path_idx, 'w') as f:
                    json.dump(category_to_idx, f)

        train_cat_bin = []
        test_cat_bin = []

        train_cat_idx = []
        test_cat_idx = []
        cat_bin_num = []
                
        for column in cat_columns:
            map_path_bin = f'{data_dir}/{column}_map_bin.json'
            map_path_idx = f'{data_dir}/{column}_map_idx.json'
            
            with open(map_path_bin, 'r') as f:
                category_to_binary = json.load(f)
            with open(map_path_idx, 'r') as f:
                category_to_idx = json.load(f)

            unk_bin = category_to_binary.get('__UNK__')
            unk_idx = category_to_idx.get('__UNK__')

            train_cat_enc_i = train_cat[column].map(category_to_binary)
            train_cat_idx_i = train_cat[column].map(category_to_idx)

            test_cat_enc_i = test_cat[column].map(category_to_binary)
            test_cat_idx_i = test_cat[column].map(category_to_idx)

            # Kategori di test yang tidak ada di vocabulary train (unseen/OOV) dipetakan
            # ke slot '__UNK__' alih-alih menghasilkan NaN. Ini BUKAN leakage karena
            # vocabulary/index-nya tetap murni berasal dari train, hanya nilai test yang
            # tidak dikenal "dijatuhkan" ke bucket generik.
            if unk_bin is not None:
                n_oov = int(test_cat_enc_i.isna().sum())
                if n_oov > 0:
                    print(f'[INFO] Kolom "{column}": {n_oov} nilai di test set adalah kategori '
                          f'yang tidak pernah muncul di train (unseen/OOV) -> dipetakan ke "__UNK__".')
                test_cat_enc_i = test_cat_enc_i.fillna(unk_bin)
                test_cat_idx_i = test_cat_idx_i.fillna(unk_idx)

            train_cat_enc_i = train_cat_enc_i.to_numpy()
            train_cat_idx_i = train_cat_idx_i.to_numpy().astype(np.int64)
            train_cat_bin_i = np.array([list(map(int, binary)) for binary in train_cat_enc_i])

            test_cat_enc_i = test_cat_enc_i.to_numpy()
            test_cat_idx_i = test_cat_idx_i.to_numpy().astype(np.int64)
            test_cat_bin_i = np.array([list(map(int, binary)) for binary in test_cat_enc_i])

            train_cat_bin.append(train_cat_bin_i)
            test_cat_bin.append(test_cat_bin_i)
            
            train_cat_idx.append(train_cat_idx_i)
            test_cat_idx.append(test_cat_idx_i)
            cat_bin_num.append(train_cat_bin_i.shape[1])
                
        train_cat_bin = np.concatenate(train_cat_bin, axis = 1).astype(np.float32)
        test_cat_bin = np.concatenate(test_cat_bin, axis = 1).astype(np.float32)

        train_cat_idx = np.stack(train_cat_idx, axis = 1)
        test_cat_idx = np.stack(test_cat_idx, axis = 1)

        cat_bin_num = np.array(cat_bin_num)

        train_X = np.concatenate([train_num, train_cat_bin], axis = 1)
        test_X = np.concatenate([test_num, test_cat_bin], axis = 1)

        train_num_mask = train_mask[:, num_col_idx]
        train_cat_mask = train_mask[:, cat_col_idx]
        test_num_mask = test_mask[:, num_col_idx]
        test_cat_mask = test_mask[:, cat_col_idx]

        def extend_mask(mask, bin_num):

            num_rows, num_cols = mask.shape
            cum_sum = bin_num.cumsum()
            cum_sum = np.insert(cum_sum, 0, 0)
            result = np.zeros((num_rows, bin_num.sum() ), dtype=bool)
            
            for idx in range(num_cols):
                res = np.tile(mask[:, idx][:, np.newaxis], bin_num[idx])
                result[:, cum_sum[idx]:cum_sum[idx + 1]] = res
                
            return result

        train_cat_mask = extend_mask(train_cat_mask, cat_bin_num)
        test_cat_mask = extend_mask(test_cat_mask, cat_bin_num)

        extend_train_mask = np.concatenate([train_num_mask, train_cat_mask], axis = 1)
        extend_test_mask = np.concatenate([test_num_mask, test_cat_mask], axis = 1)

    return train_X, test_X, train_mask, test_mask, train_num, test_num, train_cat_idx, test_cat_idx, extend_train_mask, extend_test_mask, cat_bin_num

def mean_std(data, mask):
    mask = ~mask
    mask = mask.astype(np.float32)
    mask_sum = mask.sum(0)
    mask_sum[mask_sum == 0] = 1
    mean = (data * mask).sum(0) / mask_sum
    var = ((data - mean) ** 2 * mask).sum(0) / mask_sum
    std = np.sqrt(var)
    std[std == 0] = 1  # hindari divide by zero jika kolom konstan
    return mean, std


def _bits_to_int(bits):
    """
    Konversi array binary bits ke integer.
    Ekuivalen dengan argmax pada one-hot, tapi untuk binary encoding.

    Contoh:
        bits = [0, 1, 1]  →  0*4 + 1*2 + 1*1 = 3
        bits = [1, 0, 0]  →  1*4 + 0*2 + 0*1 = 4

    Parameter:
        bits : np.ndarray, shape (N, b) — nilai kontinu hasil prediksi model
                                          (belum di-round, range bebas)

    Return:
        idx  : np.ndarray, shape (N,) — integer label hasil decoding
    """
    b = bits.shape[1]
    # Round ke 0/1 terlebih dahulu (sesuai semangat argmax: pilih nilai terbesar/terkecil)
    bits_rounded = (bits > 0.5).astype(np.int32)

    # Bobot posisi bit: [2^(b-1), 2^(b-2), ..., 2^0]
    powers = (2 ** np.arange(b - 1, -1, -1)).astype(np.int32)  # shape (b,)

    # Dot product → integer index per baris
    idx = bits_rounded.dot(powers)  # shape (N,)
    return idx


def get_eval(dataname, X_recon, X_true, truth_cat_idx, num_num, cat_bin_num, mask, oos=False):
    """
    Menghitung MAE, RMSE (untuk kolom numerik), dan Accuracy (untuk kolom kategorik)
    hanya pada posisi missing (mask == True).

    Logika Accuracy:
    ----------------
    Paper menyebutkan "argmax" setelah one-hot decoding. Karena implementasi ini
    memakai binary encoding (bukan one-hot), maka padanannya adalah:

        1. Round prediksi bit ke 0/1  →  binary string hasil prediksi
        2. Konversi binary → integer  →  predicted label index
        3. Bandingkan dengan ground-truth label index (truth_cat_idx)

    Ini sepenuhnya deterministik dan tidak bergantung pada distribusi prediksi,
    sehingga konsisten dengan semangat argmax di paper.
    """

    info_path = f'datasets/Info/{dataname}.json'
    with open(info_path, 'r') as f:
        info = json.load(f)

    num_col_idx = info['num_col_idx']
    cat_col_idx = info['cat_col_idx']

    # True(1) = missing, False(0) = observed
    num_mask = mask[:, num_col_idx].astype(bool)
    cat_mask = mask[:, cat_col_idx].astype(bool) if len(cat_col_idx) > 0 else None

    num_pred = X_recon[:, :num_num]
    cat_pred_bits = X_recon[:, num_num:]

    num_true = X_true[:, :num_num]

    # Special-case: buang 1 baris di news oos agar dimensi align
    if dataname == 'news' and oos is True:
        drop = 6265
        num_mask = np.delete(num_mask, drop, axis=0)
        num_pred = np.delete(num_pred, drop, axis=0)
        num_true = np.delete(num_true, drop, axis=0)
        if cat_mask is not None:
            cat_mask = np.delete(cat_mask, drop, axis=0)
        if truth_cat_idx is not None:
            truth_cat_idx = np.delete(truth_cat_idx, drop, axis=0)
        cat_pred_bits = np.delete(cat_pred_bits, drop, axis=0)

    # ===== Continuous metrics: hanya pada posisi missing =====
    div = num_pred[num_mask] - num_true[num_mask]
    mae  = np.abs(div).mean()
    rmse = np.sqrt((div ** 2).mean())

    # ===== Discrete metric: Accuracy hanya pada posisi missing =====
    acc = np.nan
    if (truth_cat_idx is not None) and (len(cat_col_idx) > 0) and (cat_bin_num is not None):

        cat_bin_num = np.array(cat_bin_num).astype(int)
        ends   = np.cumsum(cat_bin_num)
        starts = np.concatenate(([0], ends[:-1]))

        correct_total = 0
        total_missing = 0

        for j, (s, e) in enumerate(zip(starts, ends)):

            rows_miss = cat_mask[:, j]          # boolean mask baris yang missing
            if rows_miss.sum() == 0:
                continue

            # Prediksi bit untuk kolom kategorik ke-j
            pred_bits = cat_pred_bits[:, s:e]           # shape (N, b)

            # Ground-truth label index
            true_idx = truth_cat_idx[:, j].astype(int)  # shape (N,)

            # ===========================================================
            # ARGMAX via binary decoding (pengganti argmax one-hot):
            #   round bit prediksi → 0/1, lalu ubah ke integer
            # ===========================================================
            pred_idx = _bits_to_int(pred_bits)           # shape (N,)

            # Clamp: jika hasil decoding melebihi jumlah kelas valid,
            # anggap sebagai prediksi salah (tidak di-assign ke kelas manapun)
            nclass = int(true_idx.max()) + 1
            pred_idx = np.clip(pred_idx, 0, nclass - 1)

            # Hitung correct hanya pada baris yang missing
            correct = ((pred_idx == true_idx) & rows_miss).sum()
            total   = rows_miss.sum()

            correct_total += int(correct)
            total_missing += int(total)

        if total_missing > 0:
            acc = correct_total / total_missing

    return mae, rmse, acc


def save_imputed_csv(dataname, pred_X, num_num, cat_bin_num, mask, split_df_path,
                      save_path, oos=False):
    """
    Simpan hasil imputasi ke file CSV dengan struktur kolom asli dataset.

    Aturan penyusunan nilai:
      - Posisi yang OBSERVED (mask == False) -> diambil dari nilai ASLI (data.csv/test.csv/train.csv).
      - Posisi yang MISSING  (mask == True)  -> diambil dari hasil rekonstruksi model:
            * kolom numerik   -> langsung dipakai nilai float hasil imputasi (sudah didenormalisasi).
            * kolom kategorik -> bit hasil imputasi di-decode dulu (binary -> index -> label kategori
                                   asli) memakai mapping '{column}_map_idx.json'.

    Parameters
    ----------
    dataname : str
        Nama dataset (dipakai untuk membuka Info/{dataname}.json dan mapping kategorik).
    pred_X : np.ndarray, shape (N, num_num + sum(cat_bin_num))
        Hasil rekonstruksi model (sudah didenormalisasi), format sama seperti X_recon pada get_eval.
    num_num : int
        Jumlah kolom numerik (banyaknya kolom di num_col_idx).
    cat_bin_num : np.ndarray atau None
        Jumlah bit binary-encoding untuk tiap kolom kategorik.
    mask : np.ndarray boolean/0-1, shape (N, len(num_col_idx) + len(cat_col_idx))
        Mask ASLI per-kolom (bukan versi extended/binary) - True/1 berarti nilai tsb hilang (missing).
        Ini sama persis dengan argumen `mask` pada get_eval (mis. ori_train_mask / ori_test_mask).
    split_df_path : str
        Path ke csv asli (data.csv / train.csv / test.csv) yang menjadi acuan struktur kolom & nilai observed.
    save_path : str
        Path tujuan penyimpanan file csv hasil imputasi.
    oos : bool
        Sama seperti pada get_eval, dipakai untuk menangani kasus khusus dataset 'news' pada out-of-sample
        (ada 1 baris yang perlu dibuang agar dimensi tetap align).

    Return
    ------
    result_df : pd.DataFrame
        DataFrame hasil gabungan (observed asli + missing hasil imputasi) yang juga sudah disimpan ke `save_path`.
    """

    info_path = f'datasets/Info/{dataname}.json'
    with open(info_path, 'r') as f:
        info = json.load(f)

    num_col_idx = info['num_col_idx']
    cat_col_idx = info['cat_col_idx']

    orig_df = pd.read_csv(split_df_path)
    cols = orig_df.columns

    mask = np.asarray(mask).astype(bool)
    num_mask = mask[:, num_col_idx]
    cat_mask = mask[:, cat_col_idx] if len(cat_col_idx) > 0 else None

    num_pred = pred_X[:, :num_num]
    cat_pred_bits = pred_X[:, num_num:]

    result_df = orig_df.copy()

    # Special-case sama seperti get_eval: buang 1 baris di news oos agar dimensi align
    if dataname == 'news' and oos is True:
        drop = 6265
        if drop < len(result_df):
            result_df = result_df.drop(index=drop).reset_index(drop=True)
        num_mask = np.delete(num_mask, drop, axis=0)
        num_pred = np.delete(num_pred, drop, axis=0)
        if cat_mask is not None:
            cat_mask = np.delete(cat_mask, drop, axis=0)
        cat_pred_bits = np.delete(cat_pred_bits, drop, axis=0)

    # ===== Kolom numerik: observed = asli, missing = hasil imputasi =====
    num_cols = cols[num_col_idx]
    for i, col in enumerate(num_cols):
        col_values = result_df[col].values.astype(np.float32).copy()
        miss_rows = num_mask[:, i]
        col_values[miss_rows] = num_pred[miss_rows, i]
        result_df[col] = col_values

    # ===== Kolom kategorik: observed = asli, missing = hasil decode imputasi =====
    if len(cat_col_idx) > 0 and cat_bin_num is not None:
        data_dir = f'datasets/{dataname}'
        cat_cols = cols[cat_col_idx]

        cat_bin_num_arr = np.array(cat_bin_num).astype(int)
        ends = np.cumsum(cat_bin_num_arr)
        starts = np.concatenate(([0], ends[:-1]))

        for j, col in enumerate(cat_cols):
            s, e = starts[j], ends[j]

            miss_rows = cat_mask[:, j]
            if miss_rows.sum() == 0:
                continue

            map_path_idx = f'{data_dir}/{col}_map_idx.json'
            with open(map_path_idx, 'r') as f:
                category_to_idx = json.load(f)
            idx_to_category = {v: k for k, v in category_to_idx.items()}
            nclass = len(idx_to_category)

            pred_bits = cat_pred_bits[:, s:e]
            pred_idx = _bits_to_int(pred_bits)
            pred_idx = np.clip(pred_idx, 0, nclass - 1)

            decoded_col = np.array([idx_to_category[int(i)] for i in pred_idx], dtype=object)

            col_values = result_df[col].astype(object).values.copy()
            col_values[miss_rows] = decoded_col[miss_rows]
            result_df[col] = col_values

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    result_df.to_csv(save_path, index=False)
    print(f'[INFO] Hasil imputasi disimpan ke: {save_path}')

    return result_df


def select_best_iteration(maes, rmses, accs):
    """
    Memilih iterasi terbaik berdasarkan kombinasi MAE, RMSE, dan Accuracy (out-of-sample).

    Karena ketiga metrik punya skala/arah yang berbeda (MAE & RMSE: makin kecil makin baik,
    Accuracy: makin besar makin baik), pemilihan dilakukan dengan cara ranking:
        1. Ranking tiap metrik di semua iterasi (rank 1 = paling baik).
        2. Jumlahkan rank ketiga metrik -> total_rank.
        3. Iterasi dengan total_rank terkecil dipilih sebagai iterasi terbaik.

    Jika Accuracy tidak tersedia (semua NaN, misal dataset tanpa kolom kategorik),
    Accuracy diabaikan dari perhitungan (dianggap seri di semua iterasi).

    Parameters
    ----------
    maes, rmses, accs : list atau np.ndarray
        Nilai metrik out-of-sample per iterasi (index sejajar dengan urutan iterasi).

    Return
    ------
    best_idx : int
        Index iterasi terbaik (0-based, sesuai urutan pada `maes`/`rmses`/`accs`).
    """

    maes = pd.Series(np.asarray(maes, dtype=np.float64))
    rmses = pd.Series(np.asarray(rmses, dtype=np.float64))
    accs = pd.Series(np.asarray(accs, dtype=np.float64))

    mae_rank = maes.rank(method='min')
    rmse_rank = rmses.rank(method='min')

    if accs.isna().all():
        acc_rank = pd.Series(np.zeros(len(accs)))
    else:
        # Accuracy makin besar makin baik -> rank berdasarkan nilai negatifnya
        acc_rank = (-accs).rank(method='min')

    total_rank = mae_rank + rmse_rank + acc_rank
    best_idx = int(total_rank.idxmin())

    return best_idx


def denormalize_numeric_for_csv(pred_X, mean_X, std_X, num_num):
    """
    (FUNGSI TAMBAHAN - terpisah, tidak mengubah alur/fungsi lain yang sudah ada)

    Mengembalikan bagian NUMERIK dari `pred_X` ke skala asli dataset (undo normalisasi
    z-score), KHUSUS untuk keperluan penyimpanan hasil imputasi ke CSV.

    Latar belakang
    --------------
    Pada alur utama, seluruh kolom `train_X`/`test_X` (numerik + bit kategorik digabung)
    dinormalisasi bersama sebagai:
            X = (data - mean_X) / std_X / 2
    Saat rekonstruksi (di main_base.py), yang di-denormalisasi kembali (dikali std_X,
    ditambah mean_X) HANYA bagian kategorik-nya saja (dipakai untuk decoding bit -> label
    kategori). Bagian numerik pada array hasil rekonstruksi (`pred_X`, mis. isi file
    'oos_pred_{iteration}.npy') masih dalam skala (x - mean) / std, BUKAN skala asli.

    Ini TIDAK memengaruhi perhitungan MAE/RMSE/Accuracy yang sudah berjalan (alur tersebut
    sengaja tidak diubah/disentuh). Fungsi ini dipanggil terpisah, hanya sesaat sebelum
    data ditulis ke CSV lewat `save_imputed_csv`, agar nilai numerik yang tersimpan di CSV
    benar-benar dalam satuan/skala asli dataset - konsisten dengan nilai observed yang
    diambil langsung dari train.csv/test.csv asli.

    Parameters
    ----------
    pred_X : np.ndarray, shape (N, num_num + sum(cat_bin_num))
        Array hasil rekonstruksi seperti yang disimpan pada 'oos_pred_{iteration}.npy'
        (bagian numerik MASIH ternormalisasi, bagian kategorik SUDAH didenormalisasi
        seperti pada alur yang sudah ada).
    mean_X, std_X : np.ndarray, shape (num_num + sum(cat_bin_num),)
        Mean & std yang sama persis dipakai saat normalisasi awal di main_base.py
        (hasil dari `mean_std(train_X, train_mask)`).
    num_num : int
        Jumlah kolom numerik (banyaknya kolom pada num_col_idx).

    Return
    ------
    pred_X_fixed : np.ndarray
        SALINAN dari `pred_X` (array input tidak diubah/in-place) dengan bagian numerik
        sudah dalam skala asli. Bagian kategorik dibiarkan apa adanya (sudah benar).
    """

    pred_X_fixed = np.array(pred_X, copy=True)

    mean_X = np.asarray(mean_X)
    std_X = np.asarray(std_X)

    pred_X_fixed[:, :num_num] = pred_X_fixed[:, :num_num] * std_X[:num_num] + mean_X[:num_num]

    return pred_X_fixed


def round_numeric_for_csv(result_df, dataname, split_df_path, num_col_idx=None,
                           decimals=4, save_path=None):
    """
    (FUNGSI TAMBAHAN - terpisah, tidak mengubah save_imputed_csv ataupun fungsi lain)

    Membulatkan kolom numerik pada hasil imputasi (`result_df`, keluaran save_imputed_csv)
    supaya rapi dibaca, dengan deteksi OTOMATIS per kolom:
        - Kalau SEMUA nilai di kolom itu pada file CSV ASLI (train.csv/test.csv) adalah
          bilangan bulat (mis. jumlah, tahun, kode numerik) -> dibulatkan ke integer.
        - Kalau tidak (memang mengandung desimal, mis. harga/berat/ukuran) -> dibulatkan
          ke `decimals` angka di belakang koma.

    Deteksi integer/bukan dilakukan dari file CSV ASLI (bukan dari result_df), karena:
        1. File CSV asli (train.csv/test.csv) sudah berisi nilai LENGKAP tanpa NaN -
           "missing" di pipeline ini murni disimulasikan lewat mask untuk evaluasi,
           jadi nilai asli di posisi manapun (observed maupun yang nanti disimulasikan
           hilang) tetap valid dipakai sebagai referensi.
        2. Hasil imputasi mentah (sebelum dibulatkan) hampir pasti TIDAK bulat walau
           kolomnya sebenarnya integer, sehingga tidak bisa dipakai untuk deteksi.

    Parameters
    ----------
    result_df : pd.DataFrame
        DataFrame hasil dari save_imputed_csv (observed asli + missing hasil imputasi).
    dataname : str
        Nama dataset, dipakai membaca Info/{dataname}.json kalau num_col_idx tidak diberikan.
    split_df_path : str
        Path ke CSV asli (train.csv/test.csv) yang jadi referensi deteksi integer/bukan.
    num_col_idx : list[int] atau None
        Index kolom numerik. Kalau None, diambil dari Info/{dataname}.json.
    decimals : int
        Jumlah desimal untuk kolom yang TIDAK terdeteksi sebagai integer (default 4).
    save_path : str atau None
        Kalau diisi, hasil setelah pembulatan ditulis ulang (overwrite) ke path ini.

    Return
    ------
    rounded_df : pd.DataFrame
        SALINAN result_df dengan kolom numerik sudah dibulatkan (result_df asli tidak diubah).
    """

    if num_col_idx is None:
        info_path = f'datasets/Info/{dataname}.json'
        with open(info_path, 'r') as f:
            info = json.load(f)
        num_col_idx = info['num_col_idx']

    orig_df = pd.read_csv(split_df_path)
    cols = orig_df.columns
    num_cols = cols[num_col_idx]

    rounded_df = result_df.copy()

    for col in num_cols:
        orig_vals = orig_df[col].values.astype(np.float64)
        # PENTING: pakai selisih absolut MURNI (bukan np.allclose dengan rtol default),
        # karena rtol ikut menyesuaikan skala nilai - untuk kolom bernilai besar (mis.
        # ratusan ribu), rtol default membuat toleransi jadi sangat longgar sehingga
        # nilai desimal seperti 199999.99 salah terdeteksi sebagai "bulat".
        is_int_col = np.max(np.abs(orig_vals - np.round(orig_vals))) < 1e-6

        result_vals = rounded_df[col].values.astype(np.float64)

        if is_int_col:
            rounded_df[col] = np.round(result_vals).astype(np.int64)
        else:
            rounded_df[col] = np.round(result_vals, decimals)

    if save_path is not None:
        rounded_df.to_csv(save_path, index=False)
        print(f'[INFO] Hasil imputasi (sudah dibulatkan) disimpan ke: {save_path}')

    return rounded_df