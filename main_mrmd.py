import os
import json
import argparse
import warnings

import numpy as np

from dataset_mrmd import load_dataset_mrmd_only

warnings.filterwarnings('ignore')

# ===========================================================================
#  main_mrmd.py — VERSI MRmD-ONLY
#
#  [HIDE] Tahap embedding (SupervisedLearnableEmbeddingModel) dan EM/diffusion
#  (M-step density estimation + E-step imputation) SENGAJA di-nonaktifkan
#  dulu. Fokus script ini: menjalankan & memvalidasi alur MRmD discretization
#  dengan validation set eksternal yang persisten (TRAIN_NEW + VALIDATION),
#  lalu transform-only ke TRAIN_FULL & TEST.
#
#  Kode versi lengkap (embedding + diffusion + EM loop, GPU, dst.) TIDAK
#  DIHAPUS — cukup tidak dipanggil dari sini (lihat main_mrmd_ORIGINAL_FULL_backup.py).
#  Untuk mengaktifkan kembali, ganti import & pemanggilan di bawah dari
#  `load_dataset_mrmd_only` ke `load_dataset` (versi lengkap di
#  dataset_mrmd.py), lalu kembalikan E-step/M-step loop (torch,
#  diffusion_utils, model.py, dst).
# ===========================================================================

parser = argparse.ArgumentParser(
    description='MRmD Discretization Only (embedding & EM di-hide sementara)'
)

parser.add_argument('--dataname',     type=str,   default='shoppers', help='Nama dataset.')
parser.add_argument('--split_idx',    type=int,   default=0,          help='Index mask split.')
parser.add_argument('--ratio',        type=str,   default='30',       help='Masking ratio.')
parser.add_argument('--mask',         type=str,   default='MCAR',     help='Masking mechanism.')
parser.add_argument('--random_state', type=int,   default=42,         help='Seed untuk MRmDDiscretizer (fallback internal split saja).')

args = parser.parse_args()


if __name__ == '__main__':

    dataname   = args.dataname
    split_idx  = args.split_idx
    ratio      = args.ratio
    mask_type  = args.mask

    if mask_type == 'MNAR':
        mask_type = 'MNAR_logistic_T2'

    print(f'{"="*60}')
    print(f'[MRmD-ONLY] dataset={dataname}, ratio={ratio}, mask={mask_type}, '
          f'split_idx={split_idx}')
    print(f'{"="*60}\n')

    # =========================================================================
    #  Jalankan tahap MRmD saja
    #  - TRAIN_NEW (56%) & VALIDATION (14%) di-LOAD (sudah disiapkan manual,
    #    TIDAK di-generate/split oleh kode ini)
    #  - MRmD fit dari TRAIN_NEW + VALIDATION
    #  - Transform-only ke TRAIN_FULL & TEST
    #  - Mean/std & LabelEncoder tetap di-fit dari TRAIN_FULL (70%)
    # =========================================================================
    result = load_dataset_mrmd_only(
        dataname     = dataname,
        idx          = split_idx,
        mask_type    = mask_type,
        ratio        = ratio,
        random_state = args.random_state,
    )

    if result['mrmd'] is None:
        print('\n[MRmD-ONLY] Tidak ada kolom numerik pada dataset ini, '
              'tidak ada yang perlu didiskritisasi.')
        exit(0)

    mrmd          = result['mrmd']
    n_num_cols    = result['n_num_cols']
    t_mrmd        = result['t_mrmd']
    bin_midpoints = result['bin_midpoints']

    print(f'\n{"="*60}')
    print(f'[TIMING] Waktu komputasi MRmD discretization: {t_mrmd:.4f}s')
    print(f'{"="*60}')

    # =========================================================================
    #  Ringkasan hasil per subset — sanity check ukuran & shape
    # =========================================================================
    print(f'\n[Ringkasan Shape]')
    print(f'  TRAIN_FULL num_bin : {result["train_full_num_bin"].shape}')
    print(f'  TEST       num_bin : {result["test_num_bin"].shape}')
    print(f'  TRAIN_NEW  num_bin : {result["train_new_num_bin"].shape}')
    print(f'  VALIDATION num_bin : {result["val_num_bin"].shape}')

    # =========================================================================
    #  Simpan hasil MRmD-only run untuk inspeksi/lanjutan (tanpa embedding)
    # =========================================================================
    result_dir = f'results/{dataname}/rate{ratio}/{mask_type}/{split_idx}'
    os.makedirs(result_dir, exist_ok=True)

    summary = {
        'dataname':    dataname,
        'ratio':       ratio,
        'mask_type':   mask_type,
        'split_idx':   split_idx,
        'n_num_cols':  n_num_cols,
        'n_bins':      [int(b) for b in mrmd.n_bins_],
        't_mrmd_sec':  t_mrmd,
        'n_train_full': len(result['train_full_df']),
        'n_test':       len(result['test_df']),
        'n_train_new':  len(result['train_new_df']),
        'n_val':        len(result['val_df']),
    }

    with open(f'{result_dir}/mrmd_only_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f'\n[MRmD-ONLY] Ringkasan disimpan ke '
          f'{result_dir}/mrmd_only_summary.json')
    print(f'\n{"="*60}')
    print('[DONE] Tahap MRmD selesai. Embedding & EM/diffusion masih di-hide.')
    print(f'{"="*60}')