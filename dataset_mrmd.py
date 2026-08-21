import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import os
import json
import time
import pickle

# MRmD helper functions (inline dari mrmd_discretizer.py)
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

DATA_DIR = 'datasets'

# ===========================================================================
#  Supervised Learnable Embedding Model dengan Label
#  Arsitektur: nn.Embedding per kolom kategorikal + MLP Classifier untuk prediksi label
#  Konsep: Neural Network Embedding yang dilatih secara supervised
#
#  [DISESUAIKAN] Alur encoding/decoding disejajarkan dengan versi unsupervised
#  agar perbandingan adil (apple-to-apple):
#    - Tambah 1 hidden layer opsional (Linear → SiLU → Linear) setelah concat
#    - Tambah LayerNorm setelah concat/MLP (stabilisasi skala sebelum diffusion)
#    - Tambah Gaussian noise kecil (σ≈0.01) sebelum decoding saat training
#    - Freeze seluruh parameter embedding setelah pretraining
#  Classification loss (alpha * class_loss) TETAP dipertahankan.
#
#  [BARU] Fitur numerik di-diskritisasi dengan MRmD lalu di-embed bersama
#  fitur kategorikal menggunakan SupervisedLearnableEmbeddingModel yang sama.
#  Pipeline dari embedding → imputasi TIDAK BERUBAH.
# ===========================================================================



# ===========================================================================
#  MRmD Discretizer (implementasi Max-Relevance-Min-Divergence)
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
        X    : [N, n_cols] float — fitur numerik (data TRAIN)
        y    : [N]         int   — label kelas (data TRAIN)
        mask : [N, n_cols] bool, opsional — True = nilai HILANG (missing) di TRAIN.
               [FIX - LEAKAGE] Jika diberikan, posisi missing DIKECUALIKAN
               dari proses fitting (perhitungan cut point, MI, JS-divergence)
               per kolom, agar MRmD tidak "melihat" nilai yang seharusnya
               diimputasi.

        X_val, y_val, mask_val : opsional — data VALIDASI EKSTERNAL.
               [BARU] Jika diberikan, dipakai LANGSUNG sebagai P_v (distribusi
               validasi untuk JS-divergence) — MENGGANTIKAN random split
               internal dari X. y_val saat ini tidak dipakai dalam
               perhitungan (JS-divergence hanya butuh distribusi bin,
               bukan label), tapi tetap diterima untuk konsistensi API /
               keperluan tahap selanjutnya.

               Jika X_val TIDAK diberikan, fallback ke perilaku LAMA:
               random split internal dari X (val_size, random_state).
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
            # [BARU] Pakai validasi EKSTERNAL — tidak ada split internal.
            # Seluruh X, y (dan mask) dipakai sebagai "train" untuk fitting;
            # X_val (dan mask_val) dipakai sebagai pembanding distribusi.
            X_tr, y_tr = X, y
            X_vl       = np.array(X_val, dtype=float)
            mask_tr    = mask
            mask_vl    = np.array(mask_val, dtype=bool) if mask_val is not None else None

            if self.verbose:
                print(f'[MRmD] Menggunakan VALIDASI EKSTERNAL: '
                      f'n_train={len(X_tr)}, n_val={len(X_vl)}, '
                      f'n_features={n_features}')
        else:
            # [FALLBACK - LAMA] Split internal: training vs validation
            print('[MRmD] PERINGATAN: X_val tidak diberikan, fallback ke '
                  'random split internal (val_size). Sebaiknya sertakan '
                  'validasi eksternal untuk alur MRmD yang baru.')
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
            y_tr_j = y_tr

            # [FIX - LEAKAGE] Kecualikan posisi missing (per kolom) dari
            # data yang dipakai untuk fitting, agar cut point tidak
            # dihitung dari nilai yang seharusnya "tidak diketahui".
            if mask_tr is not None or mask_vl is not None:
                obs_tr_j = (~mask_tr[:, j]) if mask_tr is not None else np.ones(len(x_tr_j), dtype=bool)
                obs_vl_j = (~mask_vl[:, j]) if mask_vl is not None else np.ones(len(x_vl_j), dtype=bool)
                if obs_tr_j.sum() == 0:
                    # Fallback: jika seluruh train observed kosong (kasus ekstrem),
                    # tetap pakai semua data agar tidak error.
                    obs_tr_j = np.ones_like(obs_tr_j, dtype=bool)
                if obs_vl_j.sum() == 0:
                    obs_vl_j = np.ones_like(obs_vl_j, dtype=bool)
                x_tr_j = x_tr_j[obs_tr_j]
                x_vl_j = x_vl_j[obs_vl_j]
                y_tr_j = y_tr[obs_tr_j]

            x_min = float(x_tr_j.min())
            x_max = float(x_tr_j.max())
            self.x_min_.append(x_min)
            self.x_max_.append(x_max)

            unique_tr = np.unique(x_tr_j)
            if len(unique_tr) <= 1:
                self.cut_points_.append(np.array([]))
                self.n_bins_.append(1)
                if self.verbose:
                    print(f'  [MRmD] Col {j}: konstan, skip.')
                continue

            cp = self._fit_one_attribute(x_tr_j, x_vl_j, y_tr_j,
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
        missing_mask  : [N, n_cols] bool, opsional — True = nilai HILANG (missing).
                        [FIX - LEAKAGE] Jika diberikan, baris yang di-mask missing
                        DIKECUALIKAN dari perhitungan midpoint per kolom, agar
                        midpoint (yang jadi acuan skor MAE/RMSE saat evaluasi)
                        tidak dihitung dari nilai asli di posisi yang seharusnya
                        diimputasi / tidak diketahui.

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
                    # semua baris di bin tsb (walau termasuk missing), dan
                    # jika tetap kosong, interpolasi linear.
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
#  Supervised Learnable Embedding Model (TIDAK BERUBAH)
# ===========================================================================

def compute_embedding_size(n_categories: int) -> int:
    """
    Hitung ukuran embedding optimal berdasarkan jumlah kategori.
    Rumus: min(600, round(1.6 * n_categories^0.56))
    Referensi: Guo & Berkhahn (2016)
    """
    return min(600, round(1.6 * n_categories ** 0.56))


class SupervisedLearnableEmbeddingModel(nn.Module):
    """
    Model Supervised Learnable Embedding untuk fitur kategorikal tabular.

    [TIDAK BERUBAH] — sama persis dengan versi sebelumnya.
    Sekarang juga dipakai untuk fitur numerik yang sudah di-diskritisasi MRmD.

    Alur (setelah penyesuaian arsitektur):
      cat_idx [batch, n_cols]          ← termasuk numerik yg sudah jadi bin index
        → nn.Embedding per kolom → concat → [batch, total_emb_dim]
        → (opsional) Linear → SiLU → Linear   (1 hidden layer, jika use_mlp=True)
        → LayerNorm                            (stabilisasi skala sebelum diffusion)
        → z [batch, total_emb_dim]
        → MLP Classifier → [batch, n_classes]  (supervised signal, TETAP)
        → (+ noise σ=noise_std saat training)
        → Linear Decoder per kolom → logits rekonstruksi
    """

    def __init__(self, cat_dims: list, emb_sizes: list, n_classes: int,
                 dropout: float = 0.1, hidden_dim: int = 256,
                 use_mlp: bool = True, mlp_ratio: float = 1.5,
                 noise_std: float = 0.1,
                 cat_dims_decode: list = None):
        """
        cat_dims        : jumlah baris tabel embedding (nn.Embedding) per kolom.
                           [FIX - LEAKAGE] Bisa berisi +1 token khusus 'missing'
                           per kolom (lihat train_supervised_embedding_model),
                           supaya posisi yang di-mask missing bisa di-encode
                           dengan token netral, bukan nilai aslinya.
        cat_dims_decode : jumlah kelas asli (TANPA token 'missing') untuk output
                           decoder rekonstruksi. Jika None, sama dengan cat_dims
                           (perilaku lama / tidak ada token missing).
        """
        super().__init__()

        cat_dims_decode = cat_dims_decode if cat_dims_decode is not None else cat_dims

        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=n_cat, embedding_dim=emb_dim)
            for n_cat, emb_dim in zip(cat_dims, emb_sizes)
        ])

        self.total_emb_dim = sum(emb_sizes)
        self.n_cols        = len(cat_dims)
        self.cat_dims      = cat_dims
        self.emb_sizes     = emb_sizes
        self.n_classes     = n_classes
        self.noise_std     = noise_std
        self.use_mlp       = use_mlp

        if use_mlp:
            hidden_dim_mlp = max(self.total_emb_dim, int(self.total_emb_dim * mlp_ratio)) # max(11, int(11 × 1.5)) = 16
            self.mlp = nn.Sequential(
                nn.Linear(self.total_emb_dim, hidden_dim_mlp),
                nn.SiLU(),
                nn.Linear(hidden_dim_mlp, self.total_emb_dim),
            )
        else:
            self.mlp = None

        self.layer_norm = nn.LayerNorm(self.total_emb_dim)
        self.out_dim    = self.total_emb_dim

        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(self.total_emb_dim, hidden_dim), #hidden_dim 256
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), #hidden_dim 256/2
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_classes) # hidden_dim 128
        )

        self.decoders = nn.ModuleList([
            nn.Linear(emb_size, n_cat)
            for n_cat, emb_size in zip(cat_dims_decode, emb_sizes)
        ])

    def encode(self, x_cat: torch.Tensor) -> torch.Tensor:
        """
        Encode integer index → vektor embedding dense + LayerNorm.
        x_cat  : [batch, n_cols]  — integer index tiap kolom
        return : [batch, total_emb_dim]
        """
        embedded = [
            self.embeddings[i](x_cat[:, i])
            for i in range(self.n_cols)
        ]
        z = torch.cat(embedded, dim=1)

        if self.mlp is not None:
            z = self.mlp(z)

        z = self.layer_norm(z)
        return z

    def classify(self, z: torch.Tensor) -> torch.Tensor:
        return self.classifier(z)

    def decode(self, z: torch.Tensor) -> list:
        """
        Linear Decoder: embedding → logit tiap kolom.
        z      : [batch, total_emb_dim]
        return : list[n_cols] of [batch, vocab_size_i]
        """
        per_col = torch.split(z, self.emb_sizes, dim=1)
        return [self.decoders[i](per_col[i]) for i in range(self.n_cols)]

    def forward(self, x_cat: torch.Tensor, add_noise: bool = False):
        z            = self.encode(x_cat)
        class_logits = self.classify(z)

        if add_noise and self.training and self.noise_std > 0:
            z_noisy = z + torch.randn_like(z) * self.noise_std
        else:
            z_noisy = z

        recon_logits = self.decode(z_noisy)
        return z, class_logits, recon_logits


# ===========================================================================
#  Training Supervised Embedding (TIDAK BERUBAH)
# ===========================================================================

def train_supervised_embedding_model(cat_idx_array: np.ndarray,
                                     labels: np.ndarray,
                                     cat_dims: list,
                                     emb_sizes: list,
                                     n_classes: int,
                                     device: str,
                                     n_epochs: int = 50,
                                     batch_size: int = 1024,
                                     lr: float = 1e-3,
                                     dropout: float = 0.1,
                                     hidden_dim: int = 256,
                                     use_mlp: bool = True,
                                     mlp_ratio: float = 1.5,
                                     noise_std: float = 0.01,
                                     patience: int = 30,
                                     mask_array: np.ndarray = None) -> SupervisedLearnableEmbeddingModel:
    """
    Latih SupervisedLearnableEmbeddingModel.
    Sekarang cat_idx_array berisi SEMUA kolom (numerik bin + kategorikal).

    mask_array : [N, n_cols] bool, opsional — True = nilai HILANG (missing).
        [FIX - LEAKAGE] Jika diberikan, posisi missing DIKECUALIKAN dari
        reconstruction loss per kolom, agar embedding tidak "menghafal"
        nilai asli di posisi yang seharusnya diimputasi (yang nanti dipakai
        sebagai ground-truth evaluasi in-sample).
    """
    # Fix random seed agar hasil embedding reproducible setiap run
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # [FIX - LEAKAGE] Jika mask_array diberikan, tambahkan 1 token khusus
    # 'missing' per kolom pada tabel embedding (num_embeddings = n_cat + 1).
    # Saat fit, posisi yang di-mask missing akan di-encode memakai token ini
    # (bukan nilai aslinya), sehingga nilai asli di posisi missing TIDAK ikut
    # membentuk embedding vector z — baik untuk cabang classifier maupun
    # cabang reconstruction. Decoder tetap memprediksi ke ruang kelas ASLI
    # (tanpa token missing) lewat cat_dims_decode.
    use_missing_token = mask_array is not None
    if use_missing_token:
        cat_dims_embed = [d + 1 for d in cat_dims]
        # index token 'missing' tiap kolom = jumlah kelas asli kolom itu
        # (posisi paling akhir pada tabel embedding yang sudah diperbesar)
        missing_idx = torch.tensor(cat_dims, dtype=torch.long, device=device)
    else:
        cat_dims_embed = cat_dims
        missing_idx = None

    model = SupervisedLearnableEmbeddingModel(
        cat_dims_embed, emb_sizes, n_classes,
        dropout=dropout,
        hidden_dim=hidden_dim,
        use_mlp=use_mlp,
        mlp_ratio=mlp_ratio,
        noise_std=noise_std,
        cat_dims_decode=cat_dims,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    ce_loss   = nn.CrossEntropyLoss()
    ce_loss_noreduce = nn.CrossEntropyLoss(reduction='none')

    cat_tensor   = torch.tensor(cat_idx_array, dtype=torch.long, device=device)
    label_tensor = torch.tensor(labels, dtype=torch.long, device=device)

    if mask_array is not None:
        # observed_tensor: True = observed (bukan missing)
        observed_tensor = torch.tensor(~np.array(mask_array, dtype=bool),
                                       dtype=torch.bool, device=device)
        dataset = torch.utils.data.TensorDataset(cat_tensor, label_tensor, observed_tensor)
    else:
        dataset = torch.utils.data.TensorDataset(cat_tensor, label_tensor)

    cpu_gen      = torch.Generator(device='cpu')
    loader       = torch.utils.data.DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = 0,
        pin_memory  = False,
        generator   = cpu_gen,
    )

    best_loss        = float('inf')
    patience_counter = 0
    best_model_state = None

    alpha = 1.0
    beta  = 0.25

    model.train()
    for epoch in range(n_epochs):
        total_loss       = 0.0
        total_class_loss = 0.0
        total_recon_loss = 0.0
        n_batches        = 0

        for batch in loader:
            if mask_array is not None:
                batch_cat, batch_labels, batch_observed = batch
            else:
                batch_cat, batch_labels = batch
                batch_observed = None

            optimizer.zero_grad()

            if batch_observed is not None:
                # [FIX - LEAKAGE] Ganti index pada posisi MISSING dengan token
                # khusus 'missing' SEBELUM masuk ke encode(). Dengan begitu,
                # nilai asli di posisi tsb tidak ikut membentuk z, sehingga
                # tidak bisa "bocor" ke classifier maupun ke rekonstruksi
                # kolom lain lewat representasi bersama z.
                batch_cat_in = batch_cat.clone()
                miss_pos     = ~batch_observed
                batch_cat_in[miss_pos] = missing_idx.unsqueeze(0).expand_as(batch_cat)[miss_pos]
            else:
                batch_cat_in = batch_cat

            z, class_logits, recon_logits = model(batch_cat_in, add_noise=True)

            class_loss = ce_loss(class_logits, batch_labels)

            if batch_observed is not None:
                # [FIX - LEAKAGE] recon_loss hanya dihitung pada posisi
                # OBSERVED (bukan missing) per kolom, per sample.
                col_losses = []
                for i in range(model.n_cols):
                    obs_i = batch_observed[:, i]
                    if obs_i.any():
                        per_elem = ce_loss_noreduce(recon_logits[i], batch_cat[:, i])
                        col_losses.append(per_elem[obs_i].mean())
                if len(col_losses) > 0:
                    recon_loss = sum(col_losses) / len(col_losses)
                else:
                    recon_loss = torch.tensor(0.0, device=device)
            else:
                recon_loss = sum(
                    ce_loss(recon_logits[i], batch_cat[:, i])
                    for i in range(model.n_cols)
                ) / model.n_cols

            loss = alpha * class_loss + beta * recon_loss

            loss.backward()
            optimizer.step()

            total_loss       += loss.item()
            total_class_loss += class_loss.item()
            total_recon_loss += recon_loss.item()
            n_batches        += 1

        avg_loss       = total_loss       / n_batches
        avg_class_loss = total_class_loss / n_batches
        avg_recon_loss = total_recon_loss / n_batches

        if (epoch + 1) % 10 == 0:
            print(f'[Embedding] Epoch {epoch+1}/{n_epochs} - '
                  f'Loss: {avg_loss:.4f} (Class: {avg_class_loss:.4f}, '
                  f'Recon: {avg_recon_loss:.4f})')

        if avg_loss < best_loss:
            best_loss        = avg_loss
            patience_counter = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f'[Embedding] Early stopping triggered at epoch {epoch+1}')
            print(f'[Embedding] Best loss: {best_loss:.4f}')
            break

    if best_model_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
        print(f'[Embedding] Loaded best model from epoch {epoch + 1 - patience_counter}')

    model.eval()

    with torch.no_grad():
        sample_cat = cat_tensor[:min(2048, len(cat_tensor))]
        z_sample   = model.encode(sample_cat)
        print(f'[Embedding] Distribusi embedding (N={z_sample.shape[0]}):')
        print(f'  mean={z_sample.mean().item():.4f}  '
              f'std={z_sample.std().item():.4f}  '
              f'norm_mean={z_sample.norm(dim=1).mean().item():.4f}')

    for param in model.parameters():
        param.requires_grad_(False)
    print('[Embedding] Seluruh parameter embedding di-freeze untuk training diffusion.')

    return model


# ===========================================================================
#  Encode / Decode helpers (TIDAK BERUBAH)
# ===========================================================================

def encode_with_embedding(model: SupervisedLearnableEmbeddingModel,
                          cat_idx_array: np.ndarray,
                          device: str,
                          batch_size: int = 4096) -> np.ndarray:
    """
    Encode integer index → embedding numpy array.
    [TIDAK BERUBAH]
    """
    model.eval()
    cat_tensor = torch.tensor(cat_idx_array, dtype=torch.long, device=device)
    dataset    = torch.utils.data.TensorDataset(cat_tensor)
    loader     = torch.utils.data.DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = 0,
        pin_memory  = False,
    )

    all_z = []
    with torch.no_grad():
        for (batch,) in loader:
            z, _, _ = model(batch, add_noise=False)
            all_z.append(z.cpu().numpy())

    return np.concatenate(all_z, axis=0).astype(np.float32)


def decode_cat_from_embedding(model: SupervisedLearnableEmbeddingModel,
                              emb_array: np.ndarray,
                              device: str,
                              batch_size: int = 4096) -> np.ndarray:
    """
    Decode embedding → prediksi kelas tiap kolom (argmax logits).
    [TIDAK BERUBAH] — dipakai untuk kolom kategorikal (dan bisa juga untuk
    numerik-bin jika diperlukan, tapi evaluasi numerik pakai bin_midpoints).

    emb_array : [N, total_emb_dim]
    Return    : [N, n_cols]  — predicted integer index
    """
    model.eval()
    emb_tensor = torch.tensor(emb_array, dtype=torch.float32, device=device)
    dataset    = torch.utils.data.TensorDataset(emb_tensor)
    loader     = torch.utils.data.DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = 0,
        pin_memory  = False,
    )

    all_pred = []
    with torch.no_grad():
        for (batch,) in loader:
            recon_logits = model.decode(batch)
            pred_idx = torch.stack([
                torch.argmax(logits, dim=1)
                for logits in recon_logits
            ], dim=1)
            all_pred.append(pred_idx.cpu().numpy())

    return np.concatenate(all_pred, axis=0).astype(np.int64)


def decode_num_from_embedding(model: SupervisedLearnableEmbeddingModel,
                              emb_array: np.ndarray,
                              bin_midpoints: list,
                              n_num_cols: int,
                              device: str,
                              batch_size: int = 4096) -> np.ndarray:
    """
    Decode embedding → nilai numerik kontinu (dalam skala normalisasi).

    Alur (Weighted Sum / Soft-Max Decode):
      embedding → logits → softmax (probabilitas per bin) → weighted sum midpoints

    Metode ini lebih halus daripada argmax karena mempertimbangkan distribusi
    probabilitas seluruh bin, bukan hanya bin dengan logit tertinggi.
    Untuk kolom ke-i:
        p_i  = softmax(decoder_i(emb_i))      # [N, n_bins_i]
        pred = p_i @ mids_i                   # [N] — dot product = weighted sum

    Kolom numerik diasumsikan berada di AWAL emb_model (indeks 0..n_num_cols-1),
    diikuti kolom kategorikal.

    Parameter
    ---------
    model         : SupervisedLearnableEmbeddingModel
    emb_array     : [N, total_emb_dim]
    bin_midpoints : list[n_num_cols] of np.ndarray  — midpoint per bin, skala norm
    n_num_cols    : int — jumlah kolom numerik (embedding pertama)
    device        : str

    Return : np.ndarray [N, n_num_cols]  — nilai kontinu skala normalisasi
    """
    model.eval()
    emb_tensor = torch.tensor(emb_array, dtype=torch.float32, device=device)
    dataset    = torch.utils.data.TensorDataset(emb_tensor)
    loader     = torch.utils.data.DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = 0,
        pin_memory  = False,
    )

    all_preds = []
    with torch.no_grad():
        for (batch,) in loader:
            recon_logits = model.decode(batch)  # list[n_cols] of [B, vocab_size_i]

            batch_num_preds = []
            for col in range(n_num_cols):
                logits  = recon_logits[col]                          # [B, n_bins_col]
                probs   = torch.softmax(logits, dim=1)               # [B, n_bins_col]
                mids_t  = torch.tensor(
                    bin_midpoints[col], dtype=torch.float32, device=device
                )                                                     # [n_bins_col]
                # Weighted sum: probs @ mids → [B]
                pred_col = (probs * mids_t.unsqueeze(0)).sum(dim=1)  # [B]
                batch_num_preds.append(pred_col.unsqueeze(1))         # [B, 1]

            # Stack semua kolom numerik → [B, n_num_cols]
            batch_num_preds = torch.cat(batch_num_preds, dim=1)
            all_preds.append(batch_num_preds.cpu().numpy())

    return np.concatenate(all_preds, axis=0).astype(np.float32)


# ===========================================================================
#  Load TRAIN_NEW + VALIDATION (sudah disiapkan manual, TIDAK di-generate)
#  [DIPERBAIKI] Sebelumnya fungsi ini melakukan random split sendiri — itu
#  KELIRU, karena file TRAIN_NEW/VALIDATION beserta mask-nya SUDAH disiapkan
#  di luar (manual), jadi di sini murni LOAD, tidak ada proses split lagi.
#
#  Struktur folder aktual (sesuai konfirmasi):
#    datasets/{dataname}/validation/train.csv                              (TRAIN_NEW, 56%)
#    datasets/{dataname}/validation/val.csv                                (VALIDATION, 14%)
#    datasets/{dataname}/masks/validation/rate{ratio}/{mask_type}/train_mask_{idx}.npy
#    datasets/{dataname}/masks/validation/rate{ratio}/{mask_type}/val_mask_{idx}.npy
# ===========================================================================

def load_validation_split(dataname, mask_type='MCAR', ratio='30', idx=0):
    """
    Load TRAIN_NEW & VALIDATION (data + mask) yang SUDAH disiapkan manual.
    Tidak melakukan split apapun — murni membaca file dari disk.

    Return
    ------
    train_new_df, val_df     : pd.DataFrame
    train_new_mask, val_mask : np.ndarray [N, total_cols] bool (True=missing)
    """
    data_dir = f'datasets/{dataname}'

    train_new_csv = f'{data_dir}/validation/train.csv'
    val_csv       = f'{data_dir}/validation/val.csv'

    val_mask_dir        = f'{data_dir}/masks/validation/rate{ratio}/{mask_type}'
    train_new_mask_path = f'{val_mask_dir}/train_mask_{idx}.npy'
    val_mask_path       = f'{val_mask_dir}/val_mask_{idx}.npy'

    for p in [train_new_csv, val_csv, train_new_mask_path, val_mask_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f'[load_validation_split] File tidak ditemukan: {p}\n'
                f'Pastikan TRAIN_NEW/VALIDATION (data & mask) sudah disiapkan '
                f'di lokasi tsb sebelum menjalankan MRmD.'
            )

    train_new_df = pd.read_csv(train_new_csv)
    val_df       = pd.read_csv(val_csv)
    train_new_mask = np.load(train_new_mask_path)
    val_mask       = np.load(val_mask_path)

    print(f'[Validation] Loaded TRAIN_NEW ({len(train_new_df)} baris) & '
          f'VALIDATION ({len(val_df)} baris) dari {data_dir}/validation/')
    print(f'[Validation] Mask di-load dari {val_mask_dir}/')

    assert train_new_mask.shape[0] == len(train_new_df), (
        f'Baris train_new.csv ({len(train_new_df)}) tidak cocok dengan '
        f'train_mask ({train_new_mask.shape[0]}).'
    )
    assert val_mask.shape[0] == len(val_df), (
        f'Baris val.csv ({len(val_df)}) tidak cocok dengan '
        f'val_mask ({val_mask.shape[0]}).'
    )

    return train_new_df, val_df, train_new_mask, val_mask


# ===========================================================================
#  Load Dataset — VERSI MRmD-ONLY [BARU]
#  [HIDE] Tahap embedding (SupervisedLearnableEmbeddingModel) dan EM/diffusion
#  SENGAJA di-nonaktifkan dulu. Fokus: validasi alur MRmD dengan validation
#  set eksternal yang persisten. Untuk versi lengkap (dengan embedding),
#  lihat fungsi load_dataset() di bawah — TIDAK DIHAPUS, hanya tidak dipanggil.
# ===========================================================================

def load_dataset_mrmd_only(dataname, idx=0, mask_type='MCAR', ratio='30',
                            random_state=42):
    """
    Load dataset & jalankan HANYA tahap MRmD discretization.

    Alur scope fitting (sesuai desain yang disepakati):
      - MRmD cut points      : fit dari TRAIN_NEW (56%) + VALIDATION (14%)
                                (transform-only ke TRAIN_FULL & TEST)
      - Mean/std, LabelEncoder : fit dari TRAIN_FULL (70%)

    Return : dict berisi semua objek/array yang relevan untuk tahap MRmD.
    """
    ratio = str(ratio)
    data_dir  = f'datasets/{dataname}'
    info_path = f'datasets/Info/{dataname}.json'

    with open(info_path, 'r') as f:
        info = json.load(f)

    num_col_idx    = info['num_col_idx']
    cat_col_idx    = info['cat_col_idx']
    target_col_idx = info['target_col_idx']

    train_path      = f'{data_dir}/train.csv'
    test_path       = f'{data_dir}/test.csv'
    train_mask_path = f'{data_dir}/masks/rate{ratio}/{mask_type}/train_mask_{idx}.npy'
    test_mask_path  = f'{data_dir}/masks/rate{ratio}/{mask_type}/test_mask_{idx}.npy'

    # ── Load TRAIN_FULL (70%) & TEST (30%) — TIDAK BERUBAH dari sebelumnya ──
    train_full_df = pd.read_csv(train_path)
    test_df       = pd.read_csv(test_path)
    train_full_mask = np.load(train_mask_path)
    test_mask       = np.load(test_mask_path)

    # ── Load TRAIN_NEW (56%) + VALIDATION (14%) — SUDAH disiapkan manual ───
    train_new_df, val_df, train_new_mask, val_mask = load_validation_split(
        dataname, mask_type, ratio, idx
    )

    cols = train_full_df.columns

    # ── Fitur numerik (nilai float asli) ─────────────────────────────────
    train_full_num_raw = train_full_df[cols[num_col_idx]].values.astype(np.float32)
    test_num_raw       = test_df[cols[num_col_idx]].values.astype(np.float32)
    train_new_num_raw  = train_new_df[cols[num_col_idx]].values.astype(np.float32)
    val_num_raw        = val_df[cols[num_col_idx]].values.astype(np.float32)

    n_num_cols = len(num_col_idx)

    # ── Label (untuk MI di MRmD) — LabelEncoder fit dari TRAIN_FULL (70%) ──
    # [Scope fitting] Sesuai kesepakatan: mean/std & LabelEncoder fit dari
    # TRAIN_FULL, BUKAN TRAIN_NEW. TRAIN_NEW & VALIDATION adalah subset dari
    # TRAIN_FULL, jadi encoder yang sama bisa langsung dipakai (tidak ada
    # risiko kategori tak dikenal di subset-nya sendiri).
    train_full_y = train_full_df[cols[target_col_idx]]
    label_encoder = LabelEncoder()
    label_encoder.fit(train_full_y.values.ravel().astype(str))

    def _encode_labels(df):
        return label_encoder.transform(df[cols[target_col_idx]].values.ravel().astype(str))

    train_new_labels = _encode_labels(train_new_df)
    val_labels        = _encode_labels(val_df)

    print(f'[Dataset-MRmD] n_classes={len(label_encoder.classes_)}, '
          f'n_train_full={len(train_full_df)}, n_test={len(test_df)}, '
          f'n_train_new={len(train_new_df)}, n_val={len(val_df)}')

    if n_num_cols == 0:
        print('[Dataset-MRmD] Tidak ada kolom numerik, MRmD dilewati.')
        return {
            'mrmd': None, 'n_num_cols': 0,
            'label_encoder': label_encoder,
            'num_col_idx': num_col_idx, 'cat_col_idx': cat_col_idx,
            'target_col_idx': target_col_idx, 'cols': cols,
            'train_full_mask': train_full_mask, 'test_mask': test_mask,
            'train_new_mask': train_new_mask, 'val_mask': val_mask,
            'train_full_df': train_full_df, 'test_df': test_df,
            'train_new_df': train_new_df, 'val_df': val_df,
            'train_full_num_bin': np.zeros((len(train_full_df), 0), dtype=np.int64),
            'test_num_bin':       np.zeros((len(test_df), 0), dtype=np.int64),
            'train_new_num_bin':  np.zeros((len(train_new_df), 0), dtype=np.int64),
            'val_num_bin':        np.zeros((len(val_df), 0), dtype=np.int64),
            'bin_midpoints': [],
        }

    # ── Mask level kolom numerik, per subset ────────────────────────────
    train_full_num_mask = train_full_mask[:, num_col_idx].astype(bool)
    test_num_mask        = test_mask[:, num_col_idx].astype(bool)
    train_new_num_mask   = train_new_mask[:, num_col_idx].astype(bool)
    val_num_mask         = val_mask[:, num_col_idx].astype(bool)

    # ── Mean/std normalisasi — DIHITUNG DARI TRAIN_FULL (70%) ───────────
    # [Scope fitting] TIDAK dari TRAIN_NEW — supaya konsisten dipakai
    # normalisasi TEST juga (sesuai kesepakatan).
    mask_obs = (~train_full_num_mask).astype(np.float32)
    mask_sum = mask_obs.sum(0)
    mask_sum[mask_sum == 0] = 1.0
    num_mean = (train_full_num_raw * mask_obs).sum(0) / mask_sum
    num_var  = ((train_full_num_raw - num_mean) ** 2 * mask_obs).sum(0) / mask_sum
    num_std  = np.sqrt(num_var)
    num_std[num_std == 0] = 1.0

    train_full_num_norm = (train_full_num_raw - num_mean) / num_std
    test_num_norm       = (test_num_raw       - num_mean) / num_std
    train_new_num_norm  = (train_new_num_raw  - num_mean) / num_std
    val_num_norm        = (val_num_raw        - num_mean) / num_std

    # ── MRmD Discretization — fit dari TRAIN_NEW (56%) + VALIDATION (14%) ─
    # [Cache key FIX] Sertakan ratio & mask_type — cut points SPESIFIK
    # untuk kombinasi ratio+mask_type tertentu (posisi missing berbeda).
    mrmd_cache_path = f'cache/{dataname}/rate{ratio}/{mask_type}/mrmd_{idx}.pkl'
    os.makedirs(os.path.dirname(mrmd_cache_path), exist_ok=True)

    if os.path.exists(mrmd_cache_path):
        print(f'[MRmD] Cache ditemukan di {mrmd_cache_path}, skip fitting.')
        with open(mrmd_cache_path, 'rb') as f:
            mrmd = pickle.load(f)
        t_mrmd = 0.0
    else:
        print(f'[MRmD] Menjalankan MRmD discretization pada {n_num_cols} '
              f'kolom numerik (fit: TRAIN_NEW+VALIDATION) ...')
        t_mrmd_start = time.time()
        mrmd = MRmDDiscretizer(N_D=50, random_state=random_state, verbose=False)
        mrmd.fit(
            train_new_num_raw, train_new_labels,
            mask=train_new_num_mask,
            X_val=val_num_raw, y_val=val_labels, mask_val=val_num_mask,
        )
        t_mrmd = time.time() - t_mrmd_start

        with open(mrmd_cache_path, 'wb') as f:
            pickle.dump(mrmd, f)
        print(f'[MRmD] Cache disimpan ke {mrmd_cache_path}')
        print(f'[MRmD] Waktu komputasi diskritisasi: {t_mrmd:.4f}s')

    mrmd.summary()

    # ── Transform-ONLY (bukan fit ulang) ke semua subset ────────────────
    train_full_num_bin = mrmd.transform(train_full_num_raw)
    test_num_bin        = mrmd.transform(test_num_raw)
    train_new_num_bin   = mrmd.transform(train_new_num_raw)
    val_num_bin         = mrmd.transform(val_num_raw)

    # ── Bin midpoints (skala normalisasi TRAIN_FULL) ────────────────────
    bin_midpoints = mrmd.get_bin_midpoints(
        train_full_num_norm, train_full_num_bin,
        missing_mask=train_full_num_mask
    )

    print(f'[MRmD] n_bins per kolom: {mrmd.n_bins_}')
    print(f'[MRmD] Total bins: {sum(mrmd.n_bins_)}')

    return {
        'mrmd':            mrmd,
        'n_num_cols':      n_num_cols,
        't_mrmd':          t_mrmd,
        'label_encoder':   label_encoder,
        'num_mean':        num_mean,
        'num_std':         num_std,
        'bin_midpoints':   bin_midpoints,

        'num_col_idx':     num_col_idx,
        'cat_col_idx':     cat_col_idx,
        'target_col_idx':  target_col_idx,
        'cols':            cols,

        'train_full_num_bin': train_full_num_bin,
        'test_num_bin':       test_num_bin,
        'train_new_num_bin':  train_new_num_bin,
        'val_num_bin':        val_num_bin,

        'train_full_num_norm': train_full_num_norm,
        'test_num_norm':       test_num_norm,
        'train_new_num_norm':  train_new_num_norm,
        'val_num_norm':        val_num_norm,

        'train_full_num_mask': train_full_num_mask,
        'test_num_mask':       test_num_mask,
        'train_new_num_mask':  train_new_num_mask,
        'val_num_mask':        val_num_mask,

        'train_full_mask': train_full_mask, 'test_mask': test_mask,
        'train_new_mask':  train_new_mask,  'val_mask':  val_mask,

        'train_full_df': train_full_df, 'test_df': test_df,
        'train_new_df':  train_new_df,  'val_df':  val_df,
    }


# ===========================================================================
#  [HIDE] Load Dataset — VERSI LENGKAP (dengan Embedding + siap untuk EM)
#  Fungsi ini TIDAK DIHAPUS, hanya TIDAK DIPANGGIL dari main_mrmd.py untuk
#  sementara. Nanti saat tahap embedding & EM diaktifkan lagi, fungsi ini
#  perlu disesuaikan agar memakai split TRAIN_NEW/VALIDATION + transform-only
#  ke TRAIN_FULL, mengikuti pola yang sudah diimplementasikan di
#  load_dataset_mrmd_only() di atas.
# ===========================================================================

def load_dataset(dataname, idx=0, mask_type='MCAR', ratio='30', noise_std=0.01):
    """
    Load dataset dengan MRmD discretization untuk numerik +
    Supervised Embedding untuk SEMUA kolom (numerik-bin + kategorikal).

    Perubahan dari versi MDLP ke MRmD:
    - Fitur numerik di-diskritisasi dengan MRmD → integer bin index
    - Bin index numerik di-embed BERSAMA kolom kategorikal (posisi pertama)
    - Pipeline embedding → normalisasi → diffusion → imputasi TIDAK BERUBAH
    - train_num / test_num tetap dikembalikan (nilai float asli, ternormalisasi)
      untuk keperluan evaluasi MAE/RMSE di skala normalisasi

    Output tambahan (dibanding versi sebelumnya):
    - mrmd          : MRmDDiscretizer  (untuk transform test & decode)
    - bin_midpoints : list[n_num_cols] — midpoint bin dalam skala normalisasi
    - n_num_cols    : int
    - t_mrmd        : float — waktu komputasi MRmD discretization (detik)
    - t_emb         : float — waktu komputasi embedding training (detik)

    Return
    ------
    train_X           : [N_train, total_emb_dim]           float32
    test_X            : [N_test,  total_emb_dim]           float32
    ori_train_mask    : mask asli train [N_train, total_cols]
    ori_test_mask     : mask asli test  [N_test,  total_cols]
    train_num         : [N_train, n_num_cols]  — float asli (ternormalisasi)
    test_num          : [N_test,  n_num_cols]
    train_all_idx     : [N_train, n_num_cols + n_cat_cols]  — semua bin/label idx
    test_all_idx      : [N_test,  n_num_cols + n_cat_cols]
    extend_train_mask : [N_train, total_emb_dim]
    extend_test_mask  : [N_test,  total_emb_dim]
    cat_bin_num       : None  (legacy)
    emb_model         : SupervisedLearnableEmbeddingModel
    emb_sizes         : list[int]
    mrmd              : MRmDDiscretizer  (atau None jika tidak ada fitur numerik)
    bin_midpoints     : list[n_num_cols] of np.ndarray  (atau None)
    n_num_cols        : int
    t_mrmd            : float — waktu komputasi MRmD discretization (detik)
    t_emb             : float — waktu komputasi embedding training (detik)
    """
    ratio = str(ratio)

    data_dir  = f'datasets/{dataname}'
    info_path = f'datasets/Info/{dataname}.json'

    with open(info_path, 'r') as f:
        info = json.load(f)

    num_col_idx    = info['num_col_idx']
    cat_col_idx    = info['cat_col_idx']
    target_col_idx = info['target_col_idx']

    data_path       = f'{data_dir}/data.csv'
    train_path      = f'{data_dir}/train.csv'
    test_path       = f'{data_dir}/test.csv'
    train_mask_path = f'{data_dir}/masks/rate{ratio}/{mask_type}/train_mask_{idx}.npy'
    test_mask_path  = f'{data_dir}/masks/rate{ratio}/{mask_type}/test_mask_{idx}.npy'

    data_df  = pd.read_csv(data_path)
    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)

    train_mask = np.load(train_mask_path)
    test_mask  = np.load(test_mask_path)

    cols = train_df.columns

    # ── Fitur numerik (nilai float asli) ─────────────────────────────────
    data_num  = data_df[cols[num_col_idx]].values.astype(np.float32)
    train_num_raw = train_df[cols[num_col_idx]].values.astype(np.float32)
    test_num_raw  = test_df[cols[num_col_idx]].values.astype(np.float32)

    # ── Labels untuk supervised learning ─────────────────────────────────
    train_y = train_df[cols[target_col_idx]]
    test_y  = test_df[cols[target_col_idx]]

    # [FIX - LEAKAGE] LabelEncoder untuk target HANYA di-fit pada train_y,
    # bukan gabungan train+test. Fit pada data gabungan berarti informasi
    # test set (kelas apa saja yang ada di test) ikut "dilihat" saat
    # menyiapkan preprocessing untuk train — itu train/test leakage.
    #
    # Jika ada nilai target di test yang TIDAK muncul di train, nilai
    # tersebut dipetakan ke bucket khusus 'unknown' (index = n_classes)
    # alih-alih menyebabkan error atau diam-diam "mengintip" test.
    label_encoder = LabelEncoder()
    label_encoder.fit(train_y.values.ravel().astype(str))
    n_classes = len(label_encoder.classes_)
    known_label_classes = set(label_encoder.classes_)

    def _safe_label_transform(series, encoder, known_classes, unknown_idx):
        vals = series.values.ravel().astype(str)
        out  = np.empty(len(vals), dtype=np.int64)
        unknown_count = 0
        for i, v in enumerate(vals):
            if v in known_classes:
                out[i] = encoder.transform([v])[0]
            else:
                out[i] = unknown_idx
                unknown_count += 1
        return out, unknown_count

    train_labels = label_encoder.transform(train_y.values.ravel().astype(str))
    test_labels, n_unknown_test_labels = _safe_label_transform(
        test_y, label_encoder, known_label_classes, unknown_idx=n_classes
    )
    if n_unknown_test_labels > 0:
        print(f'[Dataset] PERINGATAN: {n_unknown_test_labels} label target di '
              f'test tidak muncul di train, dipetakan ke bucket "unknown".')

    print(f'[Dataset] Detected {n_classes} classes for supervised learning')
    print(f'[Dataset] Classes: {label_encoder.classes_}')

    # ── Normalisasi numerik (untuk evaluasi MAE/RMSE & bin midpoints) ─────
    # Normalisasi dihitung dari observed entries train (mask=False → observed)
    n_num_cols = len(num_col_idx)

    if n_num_cols > 0:
        num_mask_train = train_mask[:, num_col_idx].astype(bool)
        mask_obs       = (~num_mask_train).astype(np.float32)
        mask_sum       = mask_obs.sum(0)
        mask_sum[mask_sum == 0] = 1.0

        num_mean = (train_num_raw * mask_obs).sum(0) / mask_sum
        num_var  = ((train_num_raw - num_mean) ** 2 * mask_obs).sum(0) / mask_sum
        num_std  = np.sqrt(num_var)
        num_std[num_std == 0] = 1.0

        # Skala normalisasi: (X - mean) / std
        train_num_norm = (train_num_raw - num_mean) / num_std
        test_num_norm  = (test_num_raw  - num_mean) / num_std

        # Simpan untuk dikembalikan (dipakai get_eval)
        train_num = train_num_norm.astype(np.float32)
        test_num  = test_num_norm.astype(np.float32)

        # ── MRmD Discretization (dengan cache) ──────────────────────────
        mrmd_cache_path = f'cache/{dataname}/mrmd.pkl'
        os.makedirs(f'cache/{dataname}', exist_ok=True)

        if os.path.exists(mrmd_cache_path):
            # Load cut points dari cache, skip fitting
            print(f'[MRmD] Cache ditemukan di {mrmd_cache_path}, skip fitting.')
            with open(mrmd_cache_path, 'rb') as f:
                mrmd = pickle.load(f)
            t_mrmd = 0.0
            print(f'[MRmD] Cut points di-load. n_bins per kolom: {mrmd.n_bins_}')
        else:
            # Fit MRmD pada data RAW train (bukan normalisasi) dengan label
            print(f'[MRmD] Cache belum ada. Menjalankan MRmD discretization '
                  f'pada {n_num_cols} kolom numerik ...')
            t_mrmd_start = time.time()
            mrmd = MRmDDiscretizer(val_size=0.125, N_D=50, random_state=42, verbose=False)
            # [FIX - LEAKAGE] Sertakan num_mask_train agar MRmD hanya fit
            # pada nilai yang OBSERVED, bukan posisi yang di-mask missing.
            mrmd.fit(train_num_raw, train_labels, mask=num_mask_train)
            t_mrmd = time.time() - t_mrmd_start

            # Simpan objek mrmd (berisi cut_points_, x_min_, x_max_, n_bins_)
            with open(mrmd_cache_path, 'wb') as f:
                pickle.dump(mrmd, f)
            print(f'[MRmD] Cache disimpan ke {mrmd_cache_path}')
            print(f'[MRmD] Waktu komputasi diskritisasi: {t_mrmd:.4f}s')

        # Transform train & test pakai cut points yang sama (fit atau cache)
        train_num_bin = mrmd.transform(train_num_raw)   # [N_train, n_num_cols] int64
        test_num_bin  = mrmd.transform(test_num_raw)    # [N_test,  n_num_cols] int64

        # Hitung bin midpoints dalam skala NORMALISASI
        # (dipakai saat decoding: bin index → nilai kontinu untuk MAE/RMSE)
        # [FIX - LEAKAGE] Sertakan num_mask_train agar baris yang di-mask
        # missing tidak ikut menyumbang ke rata-rata midpoint bin — kalau
        # tidak, nilai asli di posisi yang mau diimputasi bisa "bocor" ke
        # midpoint yang justru dipakai sebagai acuan skor MAE/RMSE.
        bin_midpoints = mrmd.get_bin_midpoints(train_num_norm, train_num_bin,
                                               missing_mask=num_mask_train)

        print(f'[MRmD] n_bins per kolom: {mrmd.n_bins_}')
        print(f'[MRmD] Total bins: {sum(mrmd.n_bins_)}')

    else:
        # Tidak ada fitur numerik
        train_num     = np.zeros((len(train_df), 0), dtype=np.float32)
        test_num      = np.zeros((len(test_df),  0), dtype=np.float32)
        train_num_bin = np.zeros((len(train_df), 0), dtype=np.int64)
        test_num_bin  = np.zeros((len(test_df),  0), dtype=np.int64)
        bin_midpoints = []
        mrmd          = None
        t_mrmd        = 0.0

    # ── Encoding kolom kategorikal ────────────────────────────────────────
    cat_dims_cat           = []
    train_cat_idx_list     = []
    test_cat_idx_list      = []

    if len(cat_col_idx) > 0:
        cat_columns = cols[cat_col_idx]
        train_cat   = train_df[cat_columns].astype(str)
        test_cat    = test_df[cat_columns].astype(str)

        # [FIX - LEAKAGE] LabelEncoder per kolom kategorikal SEKARANG
        # HANYA di-fit pada train_cat, bukan data_df (gabungan train+test).
        # Fit pada gabungan train+test berarti vocabulary kategori dibangun
        # dengan "mengintip" nilai-nilai yang ada di test set — itu
        # train/test leakage.
        #
        # Konsekuensinya: kalau ada kategori di test yang TIDAK pernah
        # muncul di train, kategori tsb dipetakan ke 1 bucket khusus
        # 'unknown' (index terakhir = n_kategori_train) alih-alih error.
        # Ini juga konsisten dengan pola token 'missing' yang sudah ada di
        # train_supervised_embedding_model — sama-sama menambah 1 slot
        # embedding ekstra per kolom untuk kasus "nilai tak dikenal".
        encoders = {}
        for col in cat_columns:
            le = LabelEncoder()
            le.fit(train_cat[col])
            encoders[col] = le

            known_classes = set(le.classes_)
            unknown_idx   = len(le.classes_)
            # +1 slot untuk bucket 'unknown' (kategori test yang tak
            # muncul di train)
            cat_dims_cat.append(len(le.classes_) + 1)

            train_idx_col = le.transform(train_cat[col]).astype(np.int64)

            test_vals   = test_cat[col].values
            test_idx_col = np.empty(len(test_vals), dtype=np.int64)
            n_unknown   = 0
            for i, v in enumerate(test_vals):
                if v in known_classes:
                    test_idx_col[i] = le.transform([v])[0]
                else:
                    test_idx_col[i] = unknown_idx
                    n_unknown += 1
            if n_unknown > 0:
                print(f'[Dataset] PERINGATAN: kolom "{col}" — {n_unknown} '
                      f'nilai di test tidak muncul di train, dipetakan ke '
                      f'bucket "unknown".')

            train_cat_idx_list.append(train_idx_col)
            test_cat_idx_list.append(test_idx_col)

        train_cat_idx = np.stack(train_cat_idx_list, axis=1)
        test_cat_idx  = np.stack(test_cat_idx_list,  axis=1)
    else:
        train_cat_idx = np.zeros((len(train_df), 0), dtype=np.int64)
        test_cat_idx  = np.zeros((len(test_df),  0), dtype=np.int64)

    # ── Gabungkan: [num_bin | cat_idx] → satu array idx untuk embedding ──
    # Urutan: numerik (bin) DULU, lalu kategorikal — konsisten di seluruh pipeline
    if n_num_cols > 0 and len(cat_col_idx) > 0:
        train_all_idx = np.concatenate([train_num_bin, train_cat_idx], axis=1)
        test_all_idx  = np.concatenate([test_num_bin,  test_cat_idx],  axis=1)
    elif n_num_cols > 0:
        train_all_idx = train_num_bin
        test_all_idx  = test_num_bin
    else:
        train_all_idx = train_cat_idx
        test_all_idx  = test_cat_idx

    # ── Dimensi embedding ─────────────────────────────────────────────────
    # Numerik: n_bins per kolom; kategorikal: n_unique per kolom
    all_dims = (mrmd.n_bins_ if mrmd is not None else []) + cat_dims_cat
    emb_sizes = [compute_embedding_size(n) for n in all_dims]

    print(f'[Embedding] all_dims (num_bin+cat)={all_dims}')
    print(f'[Embedding] emb_sizes={emb_sizes}, total_emb_dim={sum(emb_sizes)}')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ── Mask level-kolom [num_mask | cat_mask] ─────────────────────────────
    # [FIX - LEAKAGE] Dihitung DI SINI (sebelum training embedding) agar bisa
    # dipakai untuk mengecualikan posisi missing dari training embedding.
    # Urutan kolom sama dengan all_dims / train_all_idx: numerik dulu, lalu kategorikal.
    train_num_mask = train_mask[:, num_col_idx].astype(bool) if n_num_cols > 0 else np.zeros((len(train_df), 0), dtype=bool)
    train_cat_mask = train_mask[:, cat_col_idx].astype(bool) if len(cat_col_idx) > 0 else np.zeros((len(train_df), 0), dtype=bool)
    test_num_mask  = test_mask[:, num_col_idx].astype(bool)  if n_num_cols > 0 else np.zeros((len(test_df),  0), dtype=bool)
    test_cat_mask  = test_mask[:, cat_col_idx].astype(bool)  if len(cat_col_idx) > 0 else np.zeros((len(test_df),  0), dtype=bool)

    if n_num_cols > 0 and len(cat_col_idx) > 0:
        train_all_mask = np.concatenate([train_num_mask, train_cat_mask], axis=1)
        test_all_mask  = np.concatenate([test_num_mask,  test_cat_mask],  axis=1)
    elif n_num_cols > 0:
        train_all_mask = train_num_mask
        test_all_mask  = test_num_mask
    else:
        train_all_mask = train_cat_mask
        test_all_mask  = test_cat_mask

    # ── Latih SupervisedLearnableEmbeddingModel ───────────────────────────
    # Input: semua kolom (numerik bin + kategorikal) sebagai integer index
    print('[Embedding] Melatih SupervisedLearnableEmbeddingModel '
          '(classification + reconstruction loss) ...')
    t_emb_start = time.time()
    print(noise_std)
    emb_model = train_supervised_embedding_model(
        cat_idx_array = train_all_idx,
        labels        = train_labels,
        cat_dims      = all_dims,
        emb_sizes     = emb_sizes,
        n_classes     = n_classes,
        device        = device,
        n_epochs      = 1000,
        batch_size    = 1024,
        lr            = 1e-3,
        dropout       = 0.1,
        hidden_dim    = 256,
        use_mlp       = True,
        mlp_ratio     = 1.5,
        noise_std     = noise_std,
        patience      = 40,
        mask_array    = train_all_mask,   # [FIX - LEAKAGE] posisi missing dikecualikan dari recon loss
    )
    t_emb_end = time.time()
    t_emb = t_emb_end - t_emb_start
    print('[Embedding] Training selesai. Parameter di-freeze untuk diffusion.')
    print(f'[Embedding] Waktu komputasi embedding: {t_emb:.4f}s')

    # ── Encode semua kolom → embedding vector ────────────────────────────
    # [TIDAK BERUBAH] — encode_with_embedding sama persis
    train_all_emb = encode_with_embedding(emb_model, train_all_idx, device)
    test_all_emb  = encode_with_embedding(emb_model, test_all_idx,  device)
    # shape: [N, total_emb_dim]

    # ── train_X / test_X sekarang HANYA embedding (tidak ada kolom raw num) ─
    # Karena numerik sudah masuk embedding, len_num = 0 di main
    train_X = train_all_emb
    test_X  = test_all_emb

    # ── Buat extended mask ────────────────────────────────────────────────
    # Mask asli: [N, total_original_cols]
    # Extended mask: [N, total_emb_dim] — diperluas sesuai emb_sizes
    # (train_num_mask, train_cat_mask, train_all_mask, dst. sudah dihitung
    #  di atas, sebelum training embedding — lihat blok [FIX - LEAKAGE])

    emb_sizes_arr = np.array(emb_sizes, dtype=int)

    def extend_mask_emb(mask: np.ndarray, sizes: np.ndarray) -> np.ndarray:
        """
        Perluas mask [N, n_cols] → [N, total_emb_dim].
        Kolom ke-j diperluas ke sizes[j] dimensi.
        [TIDAK BERUBAH]
        """
        N      = mask.shape[0]
        cum    = np.concatenate(([0], sizes.cumsum()))
        result = np.zeros((N, sizes.sum()), dtype=bool)
        for j in range(len(sizes)):
            col_mask = mask[:, j][:, np.newaxis]
            result[:, cum[j]:cum[j + 1]] = np.tile(col_mask, sizes[j])
        return result

    extend_train_mask = extend_mask_emb(train_all_mask, emb_sizes_arr)
    extend_test_mask  = extend_mask_emb(test_all_mask,  emb_sizes_arr)

    # Hitung bin_midpoints dalam skala normalisasi (dibutuhkan get_eval)
    # Sudah dihitung di atas, disimpan di mrmd.bin_midpoints_ & bin_midpoints

    return (train_X, test_X,
            train_mask, test_mask,
            train_num, test_num,
            train_all_idx, test_all_idx,
            extend_train_mask, extend_test_mask,
            None,          # cat_bin_num (legacy)
            emb_model,
            emb_sizes,
            mrmd,          # [BARU] MRmDDiscretizer
            bin_midpoints, # [BARU] list[n_num_cols] midpoint per bin, skala norm
            n_num_cols,    # [BARU] jumlah kolom numerik
            t_mrmd,        # [BARU] waktu komputasi MRmD discretization (detik)
            t_emb)         # [BARU] waktu komputasi embedding training (detik)


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
             num_num, emb_model, emb_sizes, mask,
             device='cpu', oos=False,
             bin_midpoints=None, n_num_cols=0,
             num_true_norm=None):
    """
    Hitung MAE, RMSE (numerik) dan Accuracy (kategorikal).

    [MODIFIKASI] Numerik sekarang di-embed bersama kategorikal.
    MAE/RMSE dihitung di skala normalisasi menggunakan ground truth
    nilai asli (bukan midpoint bin) yang dipass via num_true_norm.

    Konvensi input:
    ---------------
    X_recon / X_true : [N, total_emb_dim]
        Seluruh dimensi adalah embedding. Tidak ada kolom raw numerik.

    Numerik (MAE/RMSE):
        decode_num_from_embedding → bin index → midpoint (skala norm) [prediksi]
        Ground truth: num_true_norm — nilai float asli ternormalisasi (skala norm)
        MAE/RMSE dihitung di skala (X-mean)/std (normalisasi).

    Kategorikal (Accuracy):
        decode_cat_from_embedding → argmax logits → dibandingkan truth_all_idx
        Sama persis dengan versi sebelumnya, hanya offset kolom bergeser
        karena kolom numerik (bin) ada di awal.

    Parameter
    ---------
    bin_midpoints  : list[n_num_cols] of np.ndarray  — midpoint per bin, skala norm
                     (dipakai untuk decode prediksi)
    n_num_cols     : int — jumlah kolom numerik
    num_num        : int — DIABAIKAN (legacy, selalu 0 di pipeline baru ini)
                          dipertahankan untuk kompatibilitas signature
    truth_all_idx  : [N, n_num_cols + n_cat_cols]  integer index (bin + label)
    num_true_norm  : [N, n_num_cols] float — nilai numerik asli ternormalisasi
                     (skala (X-mean)/std). Jika None, fallback ke midpoint bin.
    """
    info_path = f'datasets/Info/{dataname}.json'
    with open(info_path, 'r') as f:
        info = json.load(f)

    num_col_idx = info['num_col_idx']
    cat_col_idx = info['cat_col_idx']

    # mask: True(1) = missing, False(0) = observed
    num_mask = mask[:, num_col_idx].astype(bool) if len(num_col_idx) > 0 else None
    cat_mask = mask[:, cat_col_idx].astype(bool) if len(cat_col_idx) > 0 else None

    # ── Special case: news dataset ────────────────────────────────────────
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
            and emb_model is not None):

        # Decode embedding → nilai kontinu prediksi (skala normalisasi) via bin midpoints
        num_pred_norm = decode_num_from_embedding(
            emb_model, X_recon, bin_midpoints, n_num_cols, device
        )  # [N, n_num_cols]

        # Ground truth: gunakan nilai asli ternormalisasi (num_true_norm) jika tersedia.
        # Ini adalah nilai float asli (X - mean) / std, bukan midpoint bin.
        # Fallback ke midpoint bin hanya jika num_true_norm tidak dipass.
        if num_true_norm is not None:
            # Pastikan shape cocok (news dataset bisa ada row yang di-drop)
            gt_norm = num_true_norm
        else:
            # Fallback (legacy): lookup midpoint dari true bin index
            N = X_true.shape[0]
            gt_norm = np.zeros((N, n_num_cols), dtype=np.float32)
            for col in range(n_num_cols):
                mids     = bin_midpoints[col]
                true_bin = truth_all_idx[:, col].astype(int)
                true_bin = np.clip(true_bin, 0, len(mids) - 1)
                gt_norm[:, col] = mids[true_bin]

        # Hitung MAE & RMSE hanya pada posisi missing
        diff = num_pred_norm[num_mask] - gt_norm[num_mask]
        mae  = float(np.abs(diff).mean())
        rmse = float(np.sqrt((diff ** 2).mean()))

    # ── Kategorikal: Akurasi via Linear Decoder ───────────────────────────
    # [TIDAK BERUBAH] — logika sama, hanya offset kolom bergeser
    acc = np.nan
    if (truth_all_idx is not None
            and len(cat_col_idx) > 0
            and emb_model is not None
            and emb_sizes is not None
            and cat_mask is not None):

        # Decode semua kolom → predicted index
        pred_all_idx = decode_cat_from_embedding(
            emb_model, X_recon, device
        )  # [N, n_num_cols + n_cat_cols]

        # Kolom kategorikal berada di offset n_num_cols (setelah numerik)
        n_cat_cols    = len(cat_col_idx)
        correct_total = 0
        total_missing = 0

        for j in range(n_cat_cols):
            rows_miss = cat_mask[:, j]
            if rows_miss.sum() == 0:
                continue

            col_offset = n_num_cols + j      # offset di array all_idx

            pred_j = pred_all_idx[:, col_offset]
            true_j = truth_all_idx[:, col_offset].astype(int)

            correct = (pred_j[rows_miss] == true_j[rows_miss]).sum()
            correct_total += int(correct)
            total_missing += int(rows_miss.sum())

        if total_missing > 0:
            acc = correct_total / total_missing

    return mae, rmse, acc