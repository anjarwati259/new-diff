import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
import os
import json
import time
import pickle

# MRmD helper functions (inline dari mrmd_discretizer.py)
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

DATA_DIR = 'datasets'

# ===========================================================================
#  VARIAN 1 : "DISKRIT tanpa EMBEDDING"
#  ---------------------------------------------------------------------
#  - Fitur numerik      : tetap didiskritisasi dengan MRmD (Max-Relevance-
#                          Min-Divergence) -> integer bin index, SAMA PERSIS
#                          seperti dataset_mrmd.py.
#  - Fitur kategorikal   : tetap di-label-encode (LabelEncoder), SAMA PERSIS
#                          seperti dataset_mrmd.py.
#  - TIDAK ADA nn.Embedding / SupervisedLearnableEmbeddingModel sama sekali.
#    Sebagai gantinya, bin index numerik + label index kategorikal langsung
#    di-ONE-HOT ENCODE dan digabung menjadi satu vektor per baris, gaya
#    "binary-encoding" pada DiffPuter original.
#  - Vektor one-hot inilah yang menjadi input untuk model diffusion (bukan
#    embedding hasil belajar).
#  - Decoding saat evaluasi/imputasi CSV dilakukan lewat argmax (kategorikal)
#    atau softmax-weighted-midpoint (numerik) pada segmen one-hot masing-
#    masing kolom hasil rekonstruksi diffusion -- TANPA linear decoder yang
#    dilatih, karena memang tidak ada model embedding yang dilatih.
# ===========================================================================


# ===========================================================================
#  MRmD Discretizer (implementasi Max-Relevance-Min-Divergence)
#  [TIDAK BERUBAH dari dataset_mrmd.py]
#  Berdasarkan: Wang et al., Pattern Recognition 149 (2024) 110236
# ===========================================================================

def _mutual_information(a_discrete: np.ndarray, c: np.ndarray) -> float:
    """
    Hitung Mutual Information I(A; C).
    Persamaan (3) di paper: I(A;C) = Σ_{a,c} P(a,c)*log[P(a,c)/(P(a)*P(c))]
    """
    n = len(a_discrete)
    bins_a = np.unique(a_discrete)
    bins_c = np.unique(c)
    mi = 0.0
    for a_val in bins_a:
        mask_a = (a_discrete == a_val)
        p_a = mask_a.sum() / n
        for c_val in bins_c:
            p_ac = ((mask_a) & (c == c_val)).sum() / n
            p_c  = (c == c_val).sum() / n
            if p_ac > 0 and p_a > 0 and p_c > 0:
                mi += p_ac * np.log(p_ac / (p_a * p_c))
    return max(mi, 0.0)


def _js_divergence(p_t: np.ndarray, p_v: np.ndarray) -> float:
    """
    Hitung Jensen-Shannon Divergence D_JS(P_t ‖ P_v).
    Persamaan (4)-(6) di paper. D_JS ∈ [0, 1].
    """
    eps    = 1e-10
    p_t    = np.clip(p_t, eps, 1.0)
    p_v    = np.clip(p_v, eps, 1.0)
    p_star = 0.5 * (p_t + p_v)
    kl_t   = np.sum(p_t * np.log(p_t / p_star))
    kl_v   = np.sum(p_v * np.log(p_v / p_star))
    return float(np.clip(0.5 * (kl_t + kl_v), 0.0, 1.0))


def _get_distributions(a_train: np.ndarray, a_val: np.ndarray):
    """Hitung P_t dan P_v dari label bin diskrit (training & validation)."""
    all_bins = np.union1d(np.unique(a_train), np.unique(a_val))
    p_t = np.array([(a_train == b).sum() for b in all_bins], dtype=float)
    p_v = np.array([(a_val   == b).sum() for b in all_bins], dtype=float)
    if p_t.sum() > 0: p_t /= p_t.sum()
    if p_v.sum() > 0: p_v /= p_v.sum()
    return p_t, p_v


def _make_bins(cut_points: np.ndarray, x_min: float, x_max: float) -> np.ndarray:
    """Bangun array edges bins: [x_min-ε, cp1, cp2, ..., x_max+ε]."""
    lo = x_min - 1e-10
    hi = x_max + 1e-10
    if len(cut_points) == 0:
        return np.array([lo, hi])
    return np.concatenate([[lo], np.sort(cut_points), [hi]])


def _discretize_mrmd(x: np.ndarray, cut_points: np.ndarray,
                     x_min: float, x_max: float) -> np.ndarray:
    """Diskritisasi array x menggunakan cut_points → label bin integer [0,1,2,...]."""
    if len(cut_points) == 0:
        return np.zeros(len(x), dtype=int)
    bins = _make_bins(cut_points, x_min, x_max)
    return (np.digitize(x, bins[1:-1])).astype(int)


class MRmDDiscretizer(BaseEstimator, TransformerMixin):
    """
    MRmD (Max-Relevance-Min-Divergence) Discretizer.
    [TIDAK BERUBAH dari dataset_mrmd.py]

    Implementasi Algorithm 1 dari:
      Wang et al., Pattern Recognition 149 (2024) 110236

    Optimasi dua kriteria secara bersamaan (Persamaan 13):
      Ψ(Aj; C) = λ * I(Aj; C)  −  D_JS(P_t(aj) ‖ P_v(aj))

    di mana:
      • I(Aj; C)         = Mutual Information atribut-diskrit vs kelas
      • D_JS(P_t ‖ P_v)  = Jensen-Shannon Divergence distribusi train vs val
      • λ = exp(-|D*_j| / N_D) (bobot adaptif, Persamaan 14)

    Kompatibel dengan scikit-learn API (fit / transform / fit_transform).

    Parameters
    ----------
    val_size     : float, default=0.125  — proporsi data untuk validasi internal
    N_D          : int,   default=50     — parameter λ
    random_state : int or None           — seed untuk split train/val
    verbose      : bool,  default=False
    """

    def __init__(self, val_size: float = 0.125, N_D: int = 50,
                 random_state=None, verbose: bool = False):
        self.val_size     = val_size
        self.N_D          = N_D
        self.random_state = random_state
        self.verbose      = verbose

    def fit(self, X, y, mask=None, X_val=None, y_val=None, mask_val=None):
        """
        Fit MRmD: temukan cut point optimal untuk tiap fitur.
        X    : [N, n_cols] float — fitur numerik (TRAIN)
        y    : [N]         int   — label kelas (TRAIN)
        mask : [N, n_cols] bool, opsional — True = nilai HILANG (missing) di TRAIN.
               [FIX-LEAKAGE] Jika diberikan, posisi missing DIKECUALIKAN dari
               fitting (cut point, MI, JS-divergence) per kolom.

        X_val, y_val, mask_val : [FIX-LEAKAGE] VALIDASI EKSTERNAL. Jika
               diberikan, dipakai LANGSUNG sebagai pembanding distribusi
               (JS-divergence) — MENGGANTIKAN random split internal dari X.
               Jika X_val tidak diberikan, fallback ke random split internal
               (val_size, random_state) seperti versi lama.
        """
        if hasattr(X, 'columns'):
            self.feature_names_in_ = np.array(X.columns)
            X = np.array(X, dtype=float)
        else:
            X = np.array(X, dtype=float)

        y = np.array(y)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        if mask is not None:
            mask = np.array(mask, dtype=bool)

        if X_val is not None:
            # [FIX-LEAKAGE] Validasi eksternal — tidak ada split internal.
            X_tr, y_tr = X, y
            X_vl       = np.array(X_val, dtype=float)
            mask_tr    = mask
            mask_vl    = np.array(mask_val, dtype=bool) if mask_val is not None else None

            if self.verbose:
                print(f'[MRmD] Validasi eksternal: n_train={len(X_tr)}, '
                      f'n_val={len(X_vl)}, n_features={n_features}')
        else:
            # Fallback: split internal (perilaku lama)
            rng       = np.random.RandomState(self.random_state)
            val_n     = max(1, int(n_samples * self.val_size))
            val_idx   = rng.choice(n_samples, size=val_n, replace=False)
            train_idx = np.setdiff1d(np.arange(n_samples), val_idx)

            X_tr, y_tr = X[train_idx], y[train_idx]
            X_vl       = X[val_idx]
            mask_tr    = mask[train_idx] if mask is not None else None
            mask_vl    = mask[val_idx]   if mask is not None else None

            if self.verbose:
                print(f'[MRmD] n_train={len(train_idx)}, n_val={len(val_idx)}, '
                      f'n_features={n_features}')

        self.cut_points_ = []
        self.x_min_      = []
        self.x_max_      = []
        # Alias agar kompatibel dengan kode lama yang pakai .n_bins_
        self.n_bins_     = []

        for j in range(n_features):
            x_tr_j = X_tr[:, j]
            x_vl_j = X_vl[:, j]

            # [FIX-LEAKAGE] Exclude posisi missing dari fitting kolom ini
            if mask_tr is not None or mask_vl is not None:
                obs_tr_j = (~mask_tr[:, j]) if mask_tr is not None else np.ones(len(x_tr_j), dtype=bool)
                obs_vl_j = (~mask_vl[:, j]) if mask_vl is not None else np.ones(len(x_vl_j), dtype=bool)
                x_tr_j_obs = x_tr_j[obs_tr_j]
                x_vl_j_obs = x_vl_j[obs_vl_j]
                y_tr_obs   = y_tr[obs_tr_j]
            else:
                x_tr_j_obs = x_tr_j
                x_vl_j_obs = x_vl_j
                y_tr_obs   = y_tr

            if len(x_tr_j_obs) == 0:
                x_tr_j_obs = x_tr_j
                y_tr_obs   = y_tr
            if len(x_vl_j_obs) == 0:
                x_vl_j_obs = x_vl_j

            x_min = float(x_tr_j_obs.min())
            x_max = float(x_tr_j_obs.max())
            self.x_min_.append(x_min)
            self.x_max_.append(x_max)

            unique_tr = np.unique(x_tr_j_obs)
            if len(unique_tr) <= 1:
                self.cut_points_.append(np.array([]))
                self.n_bins_.append(1)
                if self.verbose:
                    print(f'  [MRmD] Col {j}: konstan, skip.')
                continue

            cp = self._fit_one_attribute(x_tr_j_obs, x_vl_j_obs, y_tr_obs,
                                         unique_tr, x_min, x_max, j)
            self.cut_points_.append(cp)
            n_bins = len(cp) + 1
            self.n_bins_.append(n_bins)
            print(f'  [MRmD] Col {j}: {len(cp)} cut points → {n_bins} bins')

        return self

    def _fit_one_attribute(self, x_tr, x_vl, c_tr,
                            unique_all, x_min, x_max, j_idx):
        """Algorithm 1 untuk satu atribut. Return cut points optimal (sorted)."""
        D_star_j = np.array([])
        S_j      = unique_all.copy()
        psi_max  = -np.inf

        while len(S_j) > 0:
            best_psi = -np.inf
            best_dk  = None

            for dk in S_j:
                D_k_j     = np.append(D_star_j, dk)
                a_tr_disc = _discretize_mrmd(x_tr, D_k_j, x_min, x_max)
                a_vl_disc = _discretize_mrmd(x_vl, D_k_j, x_min, x_max)

                n_cuts  = len(D_star_j) + 1
                lam     = np.exp(-n_cuts / self.N_D)
                mi_val  = _mutual_information(a_tr_disc, c_tr)
                p_t, p_v = _get_distributions(a_tr_disc, a_vl_disc)
                jsd_val = _js_divergence(p_t, p_v)
                psi_k   = lam * mi_val - jsd_val

                if psi_k > best_psi:
                    best_psi = psi_k
                    best_dk  = dk

            if best_dk is None or best_psi <= psi_max:
                break

            psi_max  = best_psi
            D_star_j = np.append(D_star_j, best_dk)
            S_j      = S_j[S_j != best_dk]

        result = np.sort(D_star_j)
        if self.verbose:
            print(f'  Fitur [{j_idx}]: {len(result)} cut points '
                  f'→ {np.round(result, 4).tolist()}')
        return result

    def transform(self, X) -> np.ndarray:
        """
        Transform nilai kontinu → integer bin index [0, n_bins-1].
        X : [N, n_cols]
        Return : [N, n_cols]  int64
        """
        check_is_fitted(self, 'cut_points_')

        if hasattr(X, 'values'):
            X = X.values
        X = np.array(X, dtype=float)

        out = np.empty(X.shape, dtype=np.int64)
        for j, cp in enumerate(self.cut_points_):
            out[:, j] = _discretize_mrmd(
                X[:, j], cp, self.x_min_[j], self.x_max_[j]
            ).astype(np.int64)

        return out

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y).transform(X)

    def get_n_bins(self) -> np.ndarray:
        """Jumlah bin per fitur setelah fit."""
        check_is_fitted(self, 'cut_points_')
        return np.array(self.n_bins_)

    def get_bin_midpoints(self, X_norm: np.ndarray,
                          X_norm_binned: np.ndarray,
                          missing_mask: np.ndarray = None) -> list:
        """
        Hitung nilai tengah (midpoint) setiap bin dalam skala normalisasi.

        Dipakai saat decoding: bin index → nilai kontinu dalam skala (X-mean)/std.

        X_norm        : [N, n_cols]  — data normalisasi (skala (X-mean)/std)
        X_norm_binned : [N, n_cols]  — hasil transform (integer bin index)
        missing_mask  : [N, n_cols] bool, opsional — True = nilai HILANG.
                        [FIX-LEAKAGE] Jika diberikan, posisi missing
                        DIKECUALIKAN dari perhitungan midpoint per bin.

        Return : list[n_cols] of np.ndarray, tiap elemen panjang n_bins_[col]
        """
        n_cols    = X_norm.shape[1]
        midpoints = []

        if missing_mask is not None:
            missing_mask = np.array(missing_mask, dtype=bool)

        for col in range(n_cols):
            n_bins = self.n_bins_[col]
            mids   = np.zeros(n_bins, dtype=np.float32)

            if missing_mask is not None:
                obs_col = ~missing_mask[:, col]
            else:
                obs_col = np.ones(X_norm.shape[0], dtype=bool)

            for b in range(n_bins):
                mask = (X_norm_binned[:, col] == b) & obs_col
                if mask.sum() > 0:
                    mids[b] = float(X_norm[mask, col].mean())
                else:
                    # Bin kosong (setelah exclude missing) → fallback ke
                    # semua baris di bin tsb, lalu interpolasi linear.
                    mask_all = (X_norm_binned[:, col] == b)
                    if mask_all.sum() > 0:
                        mids[b] = float(X_norm[mask_all, col].mean())
                    else:
                        mids[b] = float(b) / max(n_bins - 1, 1)

            midpoints.append(mids)

        return midpoints

    def summary(self):
        """Cetak tabel ringkasan cut points."""
        check_is_fitted(self, 'cut_points_')
        print('=' * 60)
        print(f"{'MRmD Discretizer — Summary':^60}")
        print('=' * 60)
        print(f"  {'Fitur':<20} {'# Bins':>7}   Cut Points")
        print('  ' + '-' * 56)
        for j, cp in enumerate(self.cut_points_):
            name   = str(self.feature_names_in_[j])[:20] if hasattr(self, 'feature_names_in_') else f'fitur_{j}'
            cp_str = np.round(cp, 4).tolist() if len(cp) > 0 else '[ ]'
            print(f'  {name:<20} {len(cp)+1:>7}   {cp_str}')
        print('=' * 60)
        print(f'  Total cut points: {sum(len(c) for c in self.cut_points_)}')
        print(f'  N_D={self.N_D}, val_size={self.val_size}')
        print('=' * 60)


# ===========================================================================
#  [BARU] One-Hot Encoding helpers (PENGGANTI SupervisedLearnableEmbeddingModel)
#  ---------------------------------------------------------------------
#  Tidak ada parameter yang dilatih di sini -- one-hot adalah representasi
#  tetap (fixed representation), bukan hasil belajar. Ini yang membedakan
#  varian ini dari dataset_mrmd.py (yang memakai nn.Embedding terlatih).
# ===========================================================================

def one_hot_sizes(dims: list) -> list:
    """Ukuran dimensi one-hot per kolom = jumlah kategori/bin kolom itu sendiri
    (TIDAK dikompresi seperti rumus Guo & Berkhahn pada versi embedding)."""
    return list(dims)


def one_hot_encode(idx_array: np.ndarray, dims: list) -> np.ndarray:
    """
    Encode integer index tiap kolom → one-hot lalu concat semua kolom.
    idx_array : [N, n_cols] int
    dims      : list[n_cols] — jumlah kategori/bin per kolom
    Return    : [N, sum(dims)] float32
    """
    idx_array = np.asarray(idx_array)
    N, n_cols = idx_array.shape
    dims_arr  = np.asarray(dims, dtype=int)
    total_dim = int(dims_arr.sum())
    cum       = np.concatenate(([0], np.cumsum(dims_arr)))

    out = np.zeros((N, total_dim), dtype=np.float32)
    rows = np.arange(N)
    for j in range(n_cols):
        col_vals = np.clip(idx_array[:, j], 0, dims_arr[j] - 1)
        out[rows, cum[j] + col_vals] = 1.0
    return out


def decode_onehot_argmax(emb_array: np.ndarray, dims: list) -> np.ndarray:
    """
    Decode hasil rekonstruksi diffusion (skala one-hot asli, sudah
    didenormalisasi) → prediksi kelas tiap kolom (argmax pada segmen
    one-hot masing-masing kolom).

    emb_array : [N, sum(dims)]
    Return    : [N, n_cols]  — predicted integer index
    """
    dims_arr = np.asarray(dims, dtype=int)
    cum      = np.concatenate(([0], np.cumsum(dims_arr)))
    N        = emb_array.shape[0]
    n_cols   = len(dims_arr)

    out = np.zeros((N, n_cols), dtype=np.int64)
    for j in range(n_cols):
        seg = emb_array[:, cum[j]:cum[j + 1]]
        out[:, j] = np.argmax(seg, axis=1)
    return out


def decode_num_onehot_softmax(emb_array: np.ndarray, bin_midpoints: list,
                              n_num_cols: int, dims: list) -> np.ndarray:
    """
    Decode segmen one-hot kolom numerik → nilai kontinu (skala normalisasi)
    memakai Weighted-Sum/Soft-Max Decode, analog dengan
    decode_num_from_embedding pada dataset_mrmd.py, hanya saja di sini logit
    per bin diambil LANGSUNG dari nilai rekonstruksi one-hot (tanpa linear
    decoder terlatih):
        p_i  = softmax(recon_segment_i)     # [N, n_bins_i]
        pred = p_i @ mids_i

    Return : np.ndarray [N, n_num_cols]  — nilai kontinu skala normalisasi
    """
    dims_arr = np.asarray(dims, dtype=int)
    cum      = np.concatenate(([0], np.cumsum(dims_arr)))
    N        = emb_array.shape[0]

    preds = np.zeros((N, n_num_cols), dtype=np.float32)
    for col in range(n_num_cols):
        seg       = emb_array[:, cum[col]:cum[col + 1]].astype(np.float64)  # [N, n_bins_col]
        seg_shift = seg - seg.max(axis=1, keepdims=True)
        exp_seg   = np.exp(seg_shift)
        probs     = exp_seg / exp_seg.sum(axis=1, keepdims=True)
        mids      = np.asarray(bin_midpoints[col], dtype=np.float64)
        preds[:, col] = (probs @ mids).astype(np.float32)

    return preds


def extend_mask_onehot(mask: np.ndarray, sizes) -> np.ndarray:
    """
    Perluas mask [N, n_cols] → [N, sum(sizes)].
    Kolom ke-j diperluas ke sizes[j] dimensi (sama seperti extend_mask_emb
    pada dataset_mrmd.py, dipakai di sini untuk one-hot dims).
    """
    sizes  = np.asarray(sizes, dtype=int)
    N      = mask.shape[0]
    cum    = np.concatenate(([0], sizes.cumsum()))
    result = np.zeros((N, int(sizes.sum())), dtype=bool)
    for j in range(len(sizes)):
        col_mask = mask[:, j][:, np.newaxis]
        result[:, cum[j]:cum[j + 1]] = np.tile(col_mask, sizes[j])
    return result


# ===========================================================================
#  Load Dataset
# ===========================================================================

def load_dataset(dataname, idx=0, mask_type='MCAR', ratio='30', noise_std=0.01):
    """
    Load dataset dengan MRmD discretization untuk numerik + ONE-HOT ENCODING
    (bukan learnable embedding) untuk SEMUA kolom (numerik-bin + kategorikal).

    [PENTING - pemisahan FIT vs TRANSFORM]
    - FITTING MRmD (mencari cut point) TETAP memakai split validasi internal
      ('validation/train.csv' 56% sebagai training, 'validation/val.csv' 14%
      sebagai validasi eksternal untuk kriteria JS-divergence). Ini WAJIB
      karena Algorithm 1 MRmD membutuhkan validasi eksternal yang genuinely
      held-out dari data fitting.
    - TRANSFORM (hasil MRmD.transform(), lalu one-hot encode) diterapkan ke
      dataset FULL yang sesungguhnya dipakai untuk pipeline diffusion:
          train_full : 'datasets/{dataname}/train.csv'
          test       : 'datasets/{dataname}/test.csv'
      beserta mask masing-masing di
          'datasets/{dataname}/masks/rate{ratio}/{mask_type}/train_mask_{idx}.npy'
          'datasets/{dataname}/masks/rate{ratio}/{mask_type}/test_mask_{idx}.npy'
      (TANPA subfolder 'validation').
    - One-hot TIDAK memerlukan training (fixed representation), jadi bisa
      langsung diterapkan pada train_full & test setelah cut points MRmD
      didapat.
    - Normalisasi numerik (num_mean/num_std) & bin_midpoints DIHITUNG dari
      train_full (bukan dari split fit 56%).

    `noise_std` dipertahankan pada signature HANYA untuk kompatibilitas
    pemanggilan dengan main_discrete_noemb.py (tidak dipakai di sini karena
    tidak ada model embedding yang dilatih / diberi noise).

    Return
    ------
    train_X            : [N_train_full, total_onehot_dim]        float32
    test_X             : [N_test,       total_onehot_dim]        float32
    ori_train_mask      : mask asli train_full [N_train_full, total_cols]
    ori_test_mask       : mask asli test       [N_test,       total_cols]
    train_num           : [N_train_full, n_num_cols]  — float asli (ternormalisasi)
    test_num            : [N_test,       n_num_cols]
    train_all_idx       : [N_train_full, n_num_cols + n_cat_cols]  — semua bin/label idx
    test_all_idx        : [N_test,       n_num_cols + n_cat_cols]
    extend_train_mask   : [N_train_full, total_onehot_dim]
    extend_test_mask    : [N_test,       total_onehot_dim]
    cat_bin_num         : None  (legacy)
    emb_model           : None (TIDAK ADA embedding model pada varian ini)
    onehot_sizes        : list[int] — ukuran one-hot tiap kolom (num_bin+cat)
    mrmd                : MRmDDiscretizer  (atau None jika tidak ada fitur numerik)
    bin_midpoints       : list[n_num_cols] of np.ndarray  (atau None)
    n_num_cols          : int
    t_mrmd              : float — waktu komputasi MRmD discretization (detik)
    t_emb               : float — SELALU 0.0 (tidak ada training embedding)
    num_mean, num_std   : mean/std numerik skala asli, atau None
    encoders            : dict {nama_kolom: LabelEncoder} kategorikal, atau {}
    """
    ratio = str(ratio)

    data_dir  = f'datasets/{dataname}'
    info_path = f'datasets/Info/{dataname}.json'

    with open(info_path, 'r') as f:
        info = json.load(f)

    num_col_idx    = info['num_col_idx']
    cat_col_idx    = info['cat_col_idx']
    target_col_idx = info['target_col_idx']

    # ── [FIT] Data untuk fitting MRmD (split validasi) ──────────────────────
    fit_train_path      = f'{data_dir}/validation/train.csv'
    fit_val_path         = f'{data_dir}/validation/val.csv'
    fit_train_mask_path  = f'{data_dir}/masks/validation/rate{ratio}/{mask_type}/train_mask_{idx}.npy'
    fit_val_mask_path    = f'{data_dir}/masks/validation/rate{ratio}/{mask_type}/val_mask_{idx}.npy'

    fit_train_df   = pd.read_csv(fit_train_path)
    fit_val_df     = pd.read_csv(fit_val_path)
    fit_train_mask = np.load(fit_train_mask_path)
    fit_val_mask   = np.load(fit_val_mask_path)

    # ── [TRANSFORM] Data FULL sesungguhnya untuk pipeline diffusion ────────
    full_train_path      = f'{data_dir}/train.csv'
    full_test_path        = f'{data_dir}/test.csv'
    full_train_mask_path  = f'{data_dir}/masks/rate{ratio}/{mask_type}/train_mask_{idx}.npy'
    full_test_mask_path   = f'{data_dir}/masks/rate{ratio}/{mask_type}/test_mask_{idx}.npy'

    full_train_df   = pd.read_csv(full_train_path)
    full_test_df    = pd.read_csv(full_test_path)
    full_train_mask = np.load(full_train_mask_path)
    full_test_mask  = np.load(full_test_mask_path)

    cols = full_train_df.columns

    # ── Fitur numerik (nilai float asli) ─────────────────────────────────
    fit_train_num_raw  = fit_train_df[cols[num_col_idx]].values.astype(np.float32)
    fit_val_num_raw    = fit_val_df[cols[num_col_idx]].values.astype(np.float32)
    full_train_num_raw = full_train_df[cols[num_col_idx]].values.astype(np.float32)
    full_test_num_raw  = full_test_df[cols[num_col_idx]].values.astype(np.float32)

    # ── Labels [HANYA untuk kriteria relevansi MRmD] ────────────────────────
    fit_train_y_str = fit_train_df[cols[target_col_idx]].values.ravel().astype(str)
    fit_val_y_str   = fit_val_df[cols[target_col_idx]].values.ravel().astype(str)

    label_encoder = LabelEncoder()
    label_encoder.fit(fit_train_y_str)

    fit_train_labels = label_encoder.transform(fit_train_y_str)

    unseen_label_mask = ~np.isin(fit_val_y_str, label_encoder.classes_)
    if unseen_label_mask.any():
        n_unseen = int(unseen_label_mask.sum())
        unseen_vals = np.unique(fit_val_y_str[unseen_label_mask])
        print(f'[Dataset][WARNING] {n_unseen} label pada validasi (fit_val) tidak '
              f'dikenal saat fit LabelEncoder (fit_train): {unseen_vals.tolist()}. '
              f'Label tsb dipetakan sementara ke kelas pertama HANYA untuk '
              f'validasi eksternal MRmD.')
        fit_val_y_str_safe = fit_val_y_str.copy()
        fit_val_y_str_safe[unseen_label_mask] = label_encoder.classes_[0]
    else:
        fit_val_y_str_safe = fit_val_y_str

    fit_val_labels = label_encoder.transform(fit_val_y_str_safe)

    # ── Normalisasi numerik — DIHITUNG DARI TRAIN_FULL (bukan split fit) ───
    n_num_cols = len(num_col_idx)

    if n_num_cols > 0:
        fit_train_num_mask  = fit_train_mask[:, num_col_idx].astype(bool)
        fit_val_num_mask    = fit_val_mask[:, num_col_idx].astype(bool)
        full_train_num_mask = full_train_mask[:, num_col_idx].astype(bool)

        mask_obs = (~full_train_num_mask).astype(np.float32)
        mask_sum = mask_obs.sum(0)
        mask_sum[mask_sum == 0] = 1.0

        num_mean = (full_train_num_raw * mask_obs).sum(0) / mask_sum
        num_var  = ((full_train_num_raw - num_mean) ** 2 * mask_obs).sum(0) / mask_sum
        num_std  = np.sqrt(num_var)
        num_std[num_std == 0] = 1.0

        full_train_num_norm = (full_train_num_raw - num_mean) / num_std
        full_test_num_norm  = (full_test_num_raw  - num_mean) / num_std

        train_num = full_train_num_norm.astype(np.float32)
        test_num  = full_test_num_norm.astype(np.float32)

        # ── MRmD Discretization: FIT pada split validasi (dengan cache) ──
        mrmd_cache_path = f'cache/{dataname}/rate{ratio}/{mask_type}/mrmd_discrete_noemb_{idx}.pkl'
        os.makedirs(os.path.dirname(mrmd_cache_path), exist_ok=True)

        if os.path.exists(mrmd_cache_path):
            print(f'[MRmD] Cache ditemukan di {mrmd_cache_path}, skip fitting.')
            with open(mrmd_cache_path, 'rb') as f:
                mrmd = pickle.load(f)
            t_mrmd = 0.0
            print(f'[MRmD] Cut points di-load. n_bins per kolom: {mrmd.n_bins_}')
        else:
            print(f'[MRmD] Cache belum ada. Menjalankan MRmD discretization '
                  f'pada {n_num_cols} kolom numerik (fit: validation/train+val) ...')
            t_mrmd_start = time.time()
            mrmd = MRmDDiscretizer(N_D=50, random_state=42, verbose=False)
            mrmd.fit(
                fit_train_num_raw, fit_train_labels,
                mask=fit_train_num_mask,
                X_val=fit_val_num_raw, y_val=fit_val_labels, mask_val=fit_val_num_mask,
            )
            t_mrmd = time.time() - t_mrmd_start

            with open(mrmd_cache_path, 'wb') as f:
                pickle.dump(mrmd, f)
            print(f'[MRmD] Cache disimpan ke {mrmd_cache_path}')
            print(f'[MRmD] Waktu komputasi diskritisasi: {t_mrmd:.4f}s')

        # [TRANSFORM] Terapkan cut points hasil fit ke TRAIN_FULL & TEST
        full_train_num_bin = mrmd.transform(full_train_num_raw)
        full_test_num_bin  = mrmd.transform(full_test_num_raw)

        # Bin midpoints dihitung dari TRAIN_FULL (bukan dari split fit)
        bin_midpoints = mrmd.get_bin_midpoints(
            full_train_num_norm, full_train_num_bin, missing_mask=full_train_num_mask
        )

        print(f'[MRmD] n_bins per kolom: {mrmd.n_bins_}')
        print(f'[MRmD] Total bins: {sum(mrmd.n_bins_)}')

    else:
        train_num           = np.zeros((len(full_train_df), 0), dtype=np.float32)
        test_num            = np.zeros((len(full_test_df),  0), dtype=np.float32)
        full_train_num_bin  = np.zeros((len(full_train_df), 0), dtype=np.int64)
        full_test_num_bin   = np.zeros((len(full_test_df),  0), dtype=np.int64)
        bin_midpoints       = []
        mrmd                = None
        t_mrmd              = 0.0
        num_mean = None
        num_std  = None

    # ── Encoding kolom kategorikal ────────────────────────────────────────
    # [FIT] LabelEncoder di-fit LANGSUNG pada TRAIN_FULL (bukan fit_train /
    # split validasi) -- karena one-hot TIDAK melalui training apa pun
    # (fixed representation), tidak ada alasan vocabulary-nya harus
    # dibatasi ke split validasi seperti pada model embedding yang dilatih.
    # [TRANSFORM] Diterapkan (dengan token 'unknown') ke TEST.
    cat_dims_cat            = []
    full_train_cat_idx_list = []
    full_test_cat_idx_list  = []

    if len(cat_col_idx) > 0:
        cat_columns    = cols[cat_col_idx]
        full_train_cat = full_train_df[cat_columns].astype(str)
        full_test_cat  = full_test_df[cat_columns].astype(str)

        UNKNOWN_TOKEN = '__unknown__'

        encoders = {}
        for col in cat_columns:
            le = LabelEncoder()
            le.fit(full_train_cat[col])

            full_train_vals = full_train_cat[col].values
            full_test_vals  = full_test_cat[col].values

            # Kategori pada test yang TIDAK PERNAH muncul di train_full (unseen)
            unseen_test_mask = ~np.isin(full_test_vals, le.classes_)

            if unseen_test_mask.any():
                n_unseen    = int(unseen_test_mask.sum())
                unseen_vals = np.unique(full_test_vals[unseen_test_mask])
                print(f"[Dataset][WARNING] Kolom '{col}': {n_unseen} nilai pada "
                      f"test tidak dikenal saat fit (train_full): "
                      f"{unseen_vals.tolist()}. Menambahkan 1 token khusus "
                      f"'{UNKNOWN_TOKEN}' ke vocabulary kolom ini.")
                le.classes_ = np.append(le.classes_, UNKNOWN_TOKEN)

            encoders[col] = le
            cat_dims_cat.append(len(le.classes_))

            full_train_cat_idx_list.append(
                le.transform(full_train_vals).astype(np.int64)
            )

            if unseen_test_mask.any():
                full_test_vals_safe = full_test_vals.copy()
                full_test_vals_safe[unseen_test_mask] = UNKNOWN_TOKEN
            else:
                full_test_vals_safe = full_test_vals

            full_test_cat_idx_list.append(
                le.transform(full_test_vals_safe).astype(np.int64)
            )

        full_train_cat_idx = np.stack(full_train_cat_idx_list, axis=1)
        full_test_cat_idx  = np.stack(full_test_cat_idx_list,  axis=1)
    else:
        full_train_cat_idx = np.zeros((len(full_train_df), 0), dtype=np.int64)
        full_test_cat_idx  = np.zeros((len(full_test_df),  0), dtype=np.int64)
        encoders = {}

    # ── Gabungkan: [num_bin | cat_idx] → satu array idx (untuk one-hot) ────
    if n_num_cols > 0 and len(cat_col_idx) > 0:
        train_all_idx = np.concatenate([full_train_num_bin, full_train_cat_idx], axis=1)
        test_all_idx  = np.concatenate([full_test_num_bin,  full_test_cat_idx],  axis=1)
    elif n_num_cols > 0:
        train_all_idx = full_train_num_bin
        test_all_idx  = full_test_num_bin
    else:
        train_all_idx = full_train_cat_idx
        test_all_idx  = full_test_cat_idx

    # ── Dimensi one-hot: dim kolom = jumlah bin/kategori kolom itu sendiri ──
    all_dims      = (mrmd.n_bins_ if mrmd is not None else []) + cat_dims_cat
    onehot_sizes  = one_hot_sizes(all_dims)

    print(f'[OneHot] all_dims (num_bin+cat)={all_dims}')
    print(f'[OneHot] onehot_sizes={onehot_sizes}, total_onehot_dim={sum(onehot_sizes)}')

    # ── [TIDAK ADA training embedding] Langsung one-hot encode TRAIN_FULL & TEST ─
    t_emb = 0.0
    print('[OneHot] Meng-encode TRAIN_FULL & TEST (num_bin+cat) menjadi one-hot '
          '(TIDAK ADA model/parameter yang dilatih pada tahap ini).')

    train_all_emb = one_hot_encode(train_all_idx, onehot_sizes)
    test_all_emb  = one_hot_encode(test_all_idx,  onehot_sizes)

    train_X = train_all_emb
    test_X  = test_all_emb

    onehot_sizes_arr = np.array(onehot_sizes, dtype=int)

    # ── Mask untuk TRAIN_FULL & TEST ─────────────────────────────────────
    full_train_num_mask_ = full_train_mask[:, num_col_idx].astype(bool) if n_num_cols > 0 else np.zeros((len(full_train_df), 0), dtype=bool)
    full_train_cat_mask_ = full_train_mask[:, cat_col_idx].astype(bool) if len(cat_col_idx) > 0 else np.zeros((len(full_train_df), 0), dtype=bool)
    full_test_num_mask_  = full_test_mask[:, num_col_idx].astype(bool)  if n_num_cols > 0 else np.zeros((len(full_test_df),  0), dtype=bool)
    full_test_cat_mask_  = full_test_mask[:, cat_col_idx].astype(bool)  if len(cat_col_idx) > 0 else np.zeros((len(full_test_df),  0), dtype=bool)

    if n_num_cols > 0 and len(cat_col_idx) > 0:
        train_all_mask = np.concatenate([full_train_num_mask_, full_train_cat_mask_], axis=1)
        test_all_mask  = np.concatenate([full_test_num_mask_,  full_test_cat_mask_],  axis=1)
    elif n_num_cols > 0:
        train_all_mask = full_train_num_mask_
        test_all_mask  = full_test_num_mask_
    else:
        train_all_mask = full_train_cat_mask_
        test_all_mask  = full_test_cat_mask_

    extend_train_mask = extend_mask_onehot(train_all_mask, onehot_sizes_arr)
    extend_test_mask  = extend_mask_onehot(test_all_mask,  onehot_sizes_arr)

    return (train_X, test_X,
            full_train_mask, full_test_mask,
            train_num, test_num,
            train_all_idx, test_all_idx,
            extend_train_mask, extend_test_mask,
            None,           # cat_bin_num (legacy)
            None,           # emb_model -> TIDAK ADA pada varian ini
            onehot_sizes,   # menggantikan emb_sizes
            mrmd,
            bin_midpoints,
            n_num_cols,
            t_mrmd,
            t_emb,          # selalu 0.0 (tidak ada training embedding)
            num_mean,
            num_std,
            encoders)



def mean_std(data, mask):
    mask      = (~mask).astype(np.float32)
    mask_sum  = mask.sum(0)
    mask_sum[mask_sum == 0] = 1
    mean      = (data * mask).sum(0) / mask_sum
    var       = ((data - mean) ** 2 * mask).sum(0) / mask_sum
    std       = np.sqrt(var)
    return mean, std


# ===========================================================================
#  Evaluasi
# ===========================================================================

def get_eval(dataname, X_recon, X_true, truth_all_idx,
             num_num, onehot_sizes, mask,
             device='cpu', oos=False,
             bin_midpoints=None, n_num_cols=0,
             num_true_norm=None):
    """
    Hitung MAE, RMSE (numerik) dan Accuracy (kategorikal).

    [VARIAN 1 - diskrit tanpa embedding] Decoding TIDAK memakai linear
    decoder terlatih (karena tidak ada model embedding) -- melainkan
    langsung argmax / softmax-weighted-midpoint pada segmen one-hot hasil
    rekonstruksi diffusion.

    Konvensi input:
    ---------------
    X_recon / X_true : [N, total_onehot_dim]
        Seluruh dimensi adalah one-hot. Tidak ada kolom raw numerik.

    Parameter
    ---------
    onehot_sizes   : list[n_num_cols+n_cat_cols] — ukuran one-hot per kolom
                     (menggantikan emb_sizes+emb_model pada dataset_mrmd.py)
    bin_midpoints  : list[n_num_cols] of np.ndarray  — midpoint per bin, skala norm
    n_num_cols     : int — jumlah kolom numerik
    num_num        : int — DIABAIKAN (legacy, dipertahankan untuk kompatibilitas signature)
    truth_all_idx  : [N, n_num_cols + n_cat_cols]  integer index (bin + label)
    num_true_norm  : [N, n_num_cols] float — nilai numerik asli ternormalisasi
    """
    info_path = f'datasets/Info/{dataname}.json'
    with open(info_path, 'r') as f:
        info = json.load(f)

    num_col_idx = info['num_col_idx']
    cat_col_idx = info['cat_col_idx']

    num_mask = mask[:, num_col_idx].astype(bool) if len(num_col_idx) > 0 else None
    cat_mask = mask[:, cat_col_idx].astype(bool) if len(cat_col_idx) > 0 else None

    if dataname == 'news' and oos:
        drop = 6265
        if num_mask is not None:
            num_mask = np.delete(num_mask, drop, axis=0)
        if cat_mask is not None:
            cat_mask = np.delete(cat_mask, drop, axis=0)
        if truth_all_idx is not None:
            truth_all_idx = np.delete(truth_all_idx, drop, axis=0)
        if num_true_norm is not None:
            num_true_norm = np.delete(num_true_norm, drop, axis=0)
        X_recon = np.delete(X_recon, drop, axis=0)
        X_true  = np.delete(X_true,  drop, axis=0)

    # ── Numerik: MAE & RMSE di skala normalisasi ─────────────────────────
    mae  = np.nan
    rmse = np.nan

    if (n_num_cols > 0
            and num_mask is not None
            and bin_midpoints is not None
            and onehot_sizes is not None):

        num_pred_norm = decode_num_onehot_softmax(
            X_recon, bin_midpoints, n_num_cols, onehot_sizes
        )

        if num_true_norm is not None:
            gt_norm = num_true_norm
        else:
            N = X_true.shape[0]
            gt_norm = np.zeros((N, n_num_cols), dtype=np.float32)
            for col in range(n_num_cols):
                mids     = bin_midpoints[col]
                true_bin = truth_all_idx[:, col].astype(int)
                true_bin = np.clip(true_bin, 0, len(mids) - 1)
                gt_norm[:, col] = mids[true_bin]

        diff = num_pred_norm[num_mask] - gt_norm[num_mask]
        mae  = float(np.abs(diff).mean())
        rmse = float(np.sqrt((diff ** 2).mean()))

    # ── Kategorikal: Akurasi via argmax one-hot ───────────────────────────
    acc = np.nan
    if (truth_all_idx is not None
            and len(cat_col_idx) > 0
            and onehot_sizes is not None
            and cat_mask is not None):

        pred_all_idx = decode_onehot_argmax(X_recon, onehot_sizes)

        n_cat_cols    = len(cat_col_idx)
        correct_total = 0
        total_missing = 0

        for j in range(n_cat_cols):
            rows_miss = cat_mask[:, j]
            if rows_miss.sum() == 0:
                continue

            col_offset = n_num_cols + j

            pred_j = pred_all_idx[:, col_offset]
            true_j = truth_all_idx[:, col_offset].astype(int)

            correct = (pred_j[rows_miss] == true_j[rows_miss]).sum()
            correct_total += int(correct)
            total_missing += int(rows_miss.sum())

        if total_missing > 0:
            acc = correct_total / total_missing

    return mae, rmse, acc


# ===========================================================================
#  [BARU] Penyimpanan hasil imputasi ke CSV
# ===========================================================================

def select_best_iteration(maes, rmses, accs):
    """
    Memilih iterasi terbaik berdasarkan kombinasi MAE, RMSE, dan Accuracy
    (out-of-sample). [SAMA PERSIS dengan versi pada dataset_mrmd.py]
    """
    maes  = pd.Series(np.asarray(maes,  dtype=np.float64))
    rmses = pd.Series(np.asarray(rmses, dtype=np.float64))
    accs  = pd.Series(np.asarray(accs,  dtype=np.float64))

    mae_rank  = maes.rank(method='min')
    rmse_rank = rmses.rank(method='min')

    if accs.isna().all():
        acc_rank = pd.Series(np.zeros(len(accs)))
    else:
        acc_rank = (-accs).rank(method='min')

    total_rank = mae_rank + rmse_rank + acc_rank
    best_idx   = int(total_rank.idxmin())

    return best_idx


def _decode_and_denormalize_numeric_onehot(pred_X_emb, bin_midpoints,
                                           n_num_cols, onehot_sizes,
                                           num_mean, num_std):
    """
    Decode segmen one-hot numerik -> nilai kontinu, lalu kembalikan ke skala
    asli dataset.

    pred_X_emb : [N, total_onehot_dim] — hasil rekonstruksi diffusion
                 (SUDAH didenormalisasi ke skala one-hot asli)
    Return : np.ndarray [N, n_num_cols] skala ASLI dataset.
    """
    N = pred_X_emb.shape[0]
    if n_num_cols == 0 or bin_midpoints is None or onehot_sizes is None:
        return np.zeros((N, 0), dtype=np.float32)

    num_pred_norm = decode_num_onehot_softmax(
        pred_X_emb, bin_midpoints, n_num_cols, onehot_sizes
    )

    num_mean_arr = np.asarray(num_mean)
    num_std_arr  = np.asarray(num_std)
    num_pred_orig = num_pred_norm * num_std_arr + num_mean_arr

    return num_pred_orig.astype(np.float32)


def save_imputed_csv_discrete_noemb(dataname, pred_X, mask, split_df_path, save_path,
                                    onehot_sizes, bin_midpoints, n_num_cols,
                                    num_mean, num_std, cat_encoders, oos=False):
    """
    Simpan hasil imputasi (pipeline MRmD + one-hot, TANPA embedding) ke file
    CSV dengan struktur kolom asli dataset. Analog dengan
    `save_imputed_csv_mrmd` pada dataset_mrmd.py, disesuaikan: decoding
    numerik & kategorikal langsung dari segmen one-hot (argmax /
    softmax-weighted-midpoint), tanpa linear decoder terlatih.

    Aturan penyusunan nilai:
      - Posisi yang OBSERVED (mask == False) -> diambil dari nilai ASLI
        (train.csv / val.csv).
      - Posisi yang MISSING  (mask == True)  -> diambil dari hasil decode
        one-hot hasil imputasi.

    Return
    ------
    result_df : pd.DataFrame
    """

    info_path = f'datasets/Info/{dataname}.json'
    with open(info_path, 'r') as f:
        info = json.load(f)

    num_col_idx = info['num_col_idx']
    cat_col_idx = info['cat_col_idx']

    orig_df = pd.read_csv(split_df_path)
    cols = orig_df.columns

    mask = np.asarray(mask).astype(bool)
    num_mask = mask[:, num_col_idx] if len(num_col_idx) > 0 else None
    cat_mask = mask[:, cat_col_idx] if len(cat_col_idx) > 0 else None

    pred_X_emb = np.array(pred_X, copy=True)

    result_df = orig_df.copy()

    if dataname == 'news' and oos is True:
        drop = 6265
        if drop < len(result_df):
            result_df = result_df.drop(index=drop).reset_index(drop=True)
        if num_mask is not None:
            num_mask = np.delete(num_mask, drop, axis=0)
        if cat_mask is not None:
            cat_mask = np.delete(cat_mask, drop, axis=0)
        pred_X_emb = np.delete(pred_X_emb, drop, axis=0)

    # ===== Kolom numerik =====
    if len(num_col_idx) > 0:
        num_pred_orig = _decode_and_denormalize_numeric_onehot(
            pred_X_emb, bin_midpoints, n_num_cols, onehot_sizes,
            num_mean, num_std
        )

        num_cols = cols[num_col_idx]
        for i, col in enumerate(num_cols):
            col_values = result_df[col].values.astype(np.float32).copy()
            miss_rows = num_mask[:, i]
            col_values[miss_rows] = num_pred_orig[miss_rows, i]
            result_df[col] = col_values

    # ===== Kolom kategorik =====
    if len(cat_col_idx) > 0 and onehot_sizes is not None:
        cat_cols = cols[cat_col_idx]

        pred_all_idx = decode_onehot_argmax(pred_X_emb, onehot_sizes)

        for j, col in enumerate(cat_cols):
            miss_rows = cat_mask[:, j]
            if miss_rows.sum() == 0:
                continue

            col_offset = n_num_cols + j
            pred_idx_col = pred_all_idx[:, col_offset]

            le = cat_encoders[col]
            nclass = len(le.classes_)
            pred_idx_col = np.clip(pred_idx_col, 0, nclass - 1)
            decoded_col = le.inverse_transform(pred_idx_col)

            col_values = result_df[col].astype(object).values.copy()
            col_values[miss_rows] = decoded_col[miss_rows]
            result_df[col] = col_values

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    result_df.to_csv(save_path, index=False)
    print(f'[INFO] Hasil imputasi disimpan ke: {save_path}')

    return result_df


def round_numeric_for_csv(result_df, dataname, split_df_path, num_col_idx=None,
                          decimals=4, save_path=None):
    """
    Membulatkan kolom numerik pada hasil imputasi. [SAMA PERSIS dengan versi
    pada dataset_mrmd.py]
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