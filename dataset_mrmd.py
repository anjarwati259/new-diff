import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
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
                           [FIX-LEAKAGE] Bisa berisi +1 token khusus 'missing'
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
        [FIX-LEAKAGE] Jika diberikan, posisi missing di-encode memakai TOKEN
        KHUSUS 'missing' (bukan nilai aslinya), dan DIKECUALIKAN dari
        reconstruction loss per kolom — supaya embedding tidak "menghafal"
        nilai asli di posisi yang seharusnya diimputasi.
    """
    # Fix random seed agar hasil embedding reproducible setiap run
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # [FIX-LEAKAGE] Jika mask_array diberikan, tambahkan 1 token khusus
    # 'missing' per kolom pada tabel embedding (num_embeddings = n_cat + 1).
    # Posisi yang di-mask missing di-encode memakai token ini (bukan nilai
    # aslinya), sehingga nilai asli di posisi missing TIDAK ikut membentuk
    # embedding vector z. Decoder tetap memprediksi ke ruang kelas ASLI
    # (tanpa token missing) lewat cat_dims_decode.
    use_missing_token = mask_array is not None
    if use_missing_token:
        cat_dims_embed = [d + 1 for d in cat_dims]
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
                # [FIX-LEAKAGE] Ganti index pada posisi MISSING dengan token
                # khusus 'missing' SEBELUM masuk ke encode().
                batch_cat_in = batch_cat.clone()
                miss_pos     = ~batch_observed
                batch_cat_in[miss_pos] = missing_idx.unsqueeze(0).expand_as(batch_cat)[miss_pos]
            else:
                batch_cat_in = batch_cat

            z, class_logits, recon_logits = model(batch_cat_in, add_noise=True)

            class_loss = ce_loss(class_logits, batch_labels)

            if batch_observed is not None:
                # [FIX-LEAKAGE] recon_loss HANYA dihitung pada posisi OBSERVED
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
#  Load Dataset
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
    # [GANTI] 'train' = TRAIN_NEW (56%), 'test' = VALIDATION (14%) —
    # HANYA pengambilan dataset yang berubah, logic lain di bawah tetap sama.
    train_path      = f'{data_dir}/validation/train.csv'
    test_path       = f'{data_dir}/validation/val.csv'
    train_mask_path = f'{data_dir}/masks/validation/rate{ratio}/{mask_type}/train_mask_{idx}.npy'
    test_mask_path  = f'{data_dir}/masks/validation/rate{ratio}/{mask_type}/val_mask_{idx}.npy'

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

    # [FIX-LEAKAGE] LabelEncoder untuk label/kelas HANYA di-fit pada TRAIN.
    # Sebelumnya di-fit pada gabungan train+test (all_labels), yang berarti
    # proses fit "melihat" label test set — ini kebocoran informasi.
    train_y_str   = train_y.values.ravel().astype(str)
    test_y_str    = test_y.values.ravel().astype(str)

    label_encoder = LabelEncoder()
    label_encoder.fit(train_y_str)
    n_classes    = len(label_encoder.classes_)

    train_labels = label_encoder.transform(train_y_str)

    # [FIX-LEAKAGE] Tangani label pada test/validation yang TIDAK PERNAH
    # muncul di train (unseen). test_labels di sini HANYA dipakai sebagai
    # y_val eksternal untuk MRmDDiscretizer (MI & JS-divergence), BUKAN untuk
    # melatih classifier — sehingga label unseen aman dipetakan sementara ke
    # kelas pertama (index 0) hanya untuk keperluan perhitungan validasi itu.
    unseen_label_mask = ~np.isin(test_y_str, label_encoder.classes_)
    if unseen_label_mask.any():
        n_unseen = int(unseen_label_mask.sum())
        unseen_vals = np.unique(test_y_str[unseen_label_mask])
        print(f'[Dataset][WARNING] {n_unseen} label pada test/validation tidak '
              f'dikenal saat fit LabelEncoder (train): {unseen_vals.tolist()}. '
              f'Label tsb dipetakan sementara ke kelas pertama HANYA untuk '
              f'validasi eksternal MRmD (tidak memengaruhi training classifier).')
        test_y_str_safe = test_y_str.copy()
        test_y_str_safe[unseen_label_mask] = label_encoder.classes_[0]
    else:
        test_y_str_safe = test_y_str

    test_labels  = label_encoder.transform(test_y_str_safe)

    print(f'[Dataset] Detected {n_classes} classes for supervised learning (fit: train only)')
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
        # [FIX-LEAKAGE] Cache key sekarang menyertakan ratio & mask_type —
        # cut points SPESIFIK untuk kombinasi ratio+mask_type (posisi missing
        # berbeda antar kombinasi, jadi tidak boleh saling reuse cache).
        mrmd_cache_path = f'cache/{dataname}/rate{ratio}/{mask_type}/mrmd_{idx}.pkl'
        os.makedirs(os.path.dirname(mrmd_cache_path), exist_ok=True)

        if os.path.exists(mrmd_cache_path):
            # Load cut points dari cache, skip fitting
            print(f'[MRmD] Cache ditemukan di {mrmd_cache_path}, skip fitting.')
            with open(mrmd_cache_path, 'rb') as f:
                mrmd = pickle.load(f)
            t_mrmd = 0.0
            print(f'[MRmD] Cut points di-load. n_bins per kolom: {mrmd.n_bins_}')
        else:
            # [FIX-LEAKAGE] Fit MRmD dari 'train' (TRAIN_NEW) + validasi
            # eksternal 'test' (VALIDATION) — MENGGANTIKAN random split
            # internal. Posisi missing (mask) DIKECUALIKAN dari fitting.
            print(f'[MRmD] Cache belum ada. Menjalankan MRmD discretization '
                  f'pada {n_num_cols} kolom numerik (fit: train+test) ...')
            t_mrmd_start = time.time()
            mrmd = MRmDDiscretizer(N_D=50, random_state=42, verbose=False)
            mrmd.fit(
                train_num_raw, train_labels,
                mask=num_mask_train,
                X_val=test_num_raw, y_val=test_labels, mask_val=test_mask[:, num_col_idx].astype(bool),
            )
            t_mrmd = time.time() - t_mrmd_start

            # Simpan objek mrmd (berisi cut_points_, x_min_, x_max_, n_bins_)
            with open(mrmd_cache_path, 'wb') as f:
                pickle.dump(mrmd, f)
            print(f'[MRmD] Cache disimpan ke {mrmd_cache_path}')
            print(f'[MRmD] Waktu komputasi diskritisasi: {t_mrmd:.4f}s')

        # Transform 'train' & 'test' pakai cut points yang sama (TRANSFORM-ONLY,
        # tidak fit ulang di sini)
        train_num_bin = mrmd.transform(train_num_raw)   # [N_train, n_num_cols] int64
        test_num_bin  = mrmd.transform(test_num_raw)    # [N_test,  n_num_cols] int64

        # Hitung bin midpoints dalam skala NORMALISASI
        # [FIX-LEAKAGE] Posisi missing (num_mask_train) dikecualikan dari
        # perhitungan midpoint per bin.
        bin_midpoints = mrmd.get_bin_midpoints(
            train_num_norm, train_num_bin, missing_mask=num_mask_train
        )

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
        # [BARU - untuk CSV export] Tidak ada kolom numerik -> tidak ada mean/std
        num_mean = None
        num_std  = None

    # ── Encoding kolom kategorikal ────────────────────────────────────────
    cat_dims_cat           = []
    train_cat_idx_list     = []
    test_cat_idx_list      = []

    if len(cat_col_idx) > 0:
        cat_columns = cols[cat_col_idx]
        train_cat   = train_df[cat_columns].astype(str)
        test_cat    = test_df[cat_columns].astype(str)

        UNKNOWN_TOKEN = '__unknown__'

        encoders = {}
        for col in cat_columns:
            le = LabelEncoder()
            # [FIX-LEAKAGE] Fit HANYA pada TRAIN, bukan pada data_df (full
            # dataset) seperti sebelumnya. Fit di full dataset berarti
            # vocabulary sudah "melihat" kategori yang hanya ada di test.
            le.fit(train_cat[col])

            train_vals = train_cat[col].values
            test_vals  = test_cat[col].values

            # Kategori pada test yang TIDAK PERNAH muncul di train (unseen)
            unseen_mask = ~np.isin(test_vals, le.classes_)

            if unseen_mask.any():
                n_unseen    = int(unseen_mask.sum())
                unseen_vals = np.unique(test_vals[unseen_mask])
                print(f"[Dataset][WARNING] Kolom '{col}': {n_unseen} nilai pada "
                      f"test/validation tidak dikenal saat fit (train): "
                      f"{unseen_vals.tolist()}. Menambahkan 1 token khusus "
                      f"'{UNKNOWN_TOKEN}' ke vocabulary kolom ini untuk "
                      f"menampung kategori baru tsb.")
                # [FIX-LEAKAGE] Tambahkan 1 kategori khusus 'unknown' di akhir
                # vocabulary (bukan memaksa kategori baru cocok ke kategori
                # train yang salah / meng-crash saat transform).
                le.classes_ = np.append(le.classes_, UNKNOWN_TOKEN)

            encoders[col] = le
            cat_dims_cat.append(len(le.classes_))

            train_cat_idx_list.append(
                le.transform(train_vals).astype(np.int64)
            )

            if unseen_mask.any():
                test_vals_safe = test_vals.copy()
                test_vals_safe[unseen_mask] = UNKNOWN_TOKEN
            else:
                test_vals_safe = test_vals

            test_cat_idx_list.append(
                le.transform(test_vals_safe).astype(np.int64)
            )

        train_cat_idx = np.stack(train_cat_idx_list, axis=1)
        test_cat_idx  = np.stack(test_cat_idx_list,  axis=1)
    else:
        train_cat_idx = np.zeros((len(train_df), 0), dtype=np.int64)
        test_cat_idx  = np.zeros((len(test_df),  0), dtype=np.int64)
        # [BARU - untuk CSV export] Tidak ada kolom kategorikal -> tidak ada encoder
        encoders = {}

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

    # ── Mask untuk embedding leakage-fix (dipindah ke sini, sebelum training,
    #    supaya bisa dilewatkan sebagai mask_array) ──────────────────────────
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
        mask_array    = train_all_mask,   # [FIX-LEAKAGE] posisi missing dikecualikan dari recon loss
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
            t_emb,         # [BARU] waktu komputasi embedding training (detik)
            # ─────────────────────────────────────────────────────────────
            # [BARU - untuk CSV export] Ditambahkan di AKHIR tuple supaya
            # TIDAK mengubah urutan/isi nilai yang sudah ada di atas.
            # Dipakai HANYA oleh save_imputed_csv_mrmd() untuk mengembalikan
            # numerik ke skala asli & mendekode label kategorikal asli.
            # Tidak menyentuh training/evaluasi yang sudah berjalan.
            num_mean,      # mean numerik (skala asli) per kolom, atau None
            num_std,       # std  numerik (skala asli) per kolom, atau None
            encoders)      # dict {nama_kolom: LabelEncoder} kategorikal, atau {}


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


# ===========================================================================
#  [BARU] Penyimpanan hasil imputasi ke CSV
#
#  Fungsi-fungsi di bawah ini TERPISAH dari pipeline training/evaluasi di
#  atas (load_dataset, train_supervised_embedding_model, get_eval, dst).
#  Ditambahkan mengikuti pola yang sama seperti pada dataset.py (versi
#  binary-encoding/DiffPuter original), disesuaikan untuk pipeline
#  embedding + MRmD:
#    - Numerik   : embedding -> decode_num_from_embedding (bin midpoint,
#                  skala normalisasi) -> denormalisasi ke skala asli
#                  memakai num_mean/num_std.
#    - Kategorik : embedding -> decode_cat_from_embedding (argmax logits,
#                  integer index) -> inverse_transform via LabelEncoder
#                  (cat_encoders) -> label kategori asli.
#  Tidak mengubah/menyentuh fungsi atau alur lain yang sudah ada di atas.
# ===========================================================================

def select_best_iteration(maes, rmses, accs):
    """
    Memilih iterasi terbaik berdasarkan kombinasi MAE, RMSE, dan Accuracy
    (out-of-sample). [SAMA PERSIS dengan versi pada dataset.py]

    Karena ketiga metrik punya skala/arah yang berbeda (MAE & RMSE: makin
    kecil makin baik, Accuracy: makin besar makin baik), pemilihan
    dilakukan dengan cara ranking:
        1. Ranking tiap metrik di semua iterasi (rank 1 = paling baik).
        2. Jumlahkan rank ketiga metrik -> total_rank.
        3. Iterasi dengan total_rank terkecil dipilih sebagai iterasi terbaik.

    Jika Accuracy tidak tersedia (semua NaN, misal dataset tanpa kolom
    kategorik), Accuracy diabaikan dari perhitungan (dianggap seri di
    semua iterasi).

    Parameters
    ----------
    maes, rmses, accs : list atau np.ndarray
        Nilai metrik out-of-sample per iterasi (index sejajar dengan
        urutan iterasi).

    Return
    ------
    best_idx : int
        Index iterasi terbaik (0-based, sesuai urutan pada
        `maes`/`rmses`/`accs`).
    """

    maes  = pd.Series(np.asarray(maes,  dtype=np.float64))
    rmses = pd.Series(np.asarray(rmses, dtype=np.float64))
    accs  = pd.Series(np.asarray(accs,  dtype=np.float64))

    mae_rank  = maes.rank(method='min')
    rmse_rank = rmses.rank(method='min')

    if accs.isna().all():
        acc_rank = pd.Series(np.zeros(len(accs)))
    else:
        # Accuracy makin besar makin baik -> rank berdasarkan nilai negatifnya
        acc_rank = (-accs).rank(method='min')

    total_rank = mae_rank + rmse_rank + acc_rank
    best_idx   = int(total_rank.idxmin())

    return best_idx


def _decode_and_denormalize_numeric(pred_X_emb, emb_model, bin_midpoints,
                                    n_num_cols, num_mean, num_std, device):
    """
    Decode embedding -> nilai numerik kontinu, lalu kembalikan ke skala asli.

    pred_X_emb : [N, total_emb_dim] — embedding hasil rekonstruksi
                 (SUDAH didenormalisasi ke skala embedding asli, sama
                 seperti `pred_X` yang dipakai sebagai `X_recon` di get_eval).
    Return : np.ndarray [N, n_num_cols] skala ASLI dataset, atau array
             kosong shape (N, 0) jika n_num_cols == 0.
    """
    N = pred_X_emb.shape[0]
    if n_num_cols == 0 or bin_midpoints is None or emb_model is None:
        return np.zeros((N, 0), dtype=np.float32)

    num_pred_norm = decode_num_from_embedding(
        emb_model, pred_X_emb, bin_midpoints, n_num_cols, device
    )  # skala normalisasi (X - mean) / std

    num_mean_arr = np.asarray(num_mean)
    num_std_arr  = np.asarray(num_std)
    num_pred_orig = num_pred_norm * num_std_arr + num_mean_arr

    return num_pred_orig.astype(np.float32)


def save_imputed_csv_mrmd(dataname, pred_X, mask, split_df_path, save_path,
                          emb_model, emb_sizes, bin_midpoints, n_num_cols,
                          num_mean, num_std, cat_encoders, device, oos=False):
    """
    Simpan hasil imputasi (pipeline MRmD + embedding) ke file CSV dengan
    struktur kolom asli dataset. Analog dengan `save_imputed_csv` pada
    dataset.py (versi binary-encoding), disesuaikan untuk embedding.

    Aturan penyusunan nilai:
      - Posisi yang OBSERVED (mask == False) -> diambil dari nilai ASLI
        (train.csv / val.csv).
      - Posisi yang MISSING  (mask == True)  -> diambil dari hasil decode
        embedding:
            * kolom numerik   -> decode_num_from_embedding (bin midpoint,
                                  skala normalisasi) lalu didenormalisasi
                                  ke skala asli dengan num_mean/num_std.
            * kolom kategorik -> decode_cat_from_embedding (argmax logits,
                                  integer index) lalu di-inverse_transform
                                  memakai LabelEncoder pada `cat_encoders`
                                  untuk mendapat label kategori asli.

    Parameters
    ----------
    dataname : str
        Nama dataset (dipakai untuk membuka Info/{dataname}.json).
    pred_X : np.ndarray, shape (N, total_emb_dim)
        Hasil rekonstruksi embedding (SUDAH didenormalisasi ke skala
        embedding asli) — sama seperti `X_recon`/`pred_X` yang dipakai
        sebagai input `get_eval` di main_mrmd.py.
    mask : np.ndarray boolean/0-1, shape (N, len(num_col_idx)+len(cat_col_idx))
        Mask ASLI per-kolom (bukan versi extended/embedding) — sama
        persis dengan argumen `mask` pada get_eval (ori_train_mask /
        ori_test_mask).
    split_df_path : str
        Path ke csv asli (mis. 'datasets/{dataname}/validation/train.csv'
        atau 'datasets/{dataname}/validation/val.csv') yang jadi acuan
        struktur kolom & nilai observed.
    save_path : str
        Path tujuan penyimpanan file csv hasil imputasi.
    emb_model, emb_sizes, bin_midpoints, n_num_cols, num_mean, num_std,
    cat_encoders, device : lihat load_dataset() — dikembalikan langsung
        dari load_dataset dan diteruskan apa adanya ke sini.
    oos : bool
        Sama seperti pada get_eval, dipakai untuk menangani kasus khusus
        dataset 'news' (ada 1 baris yang perlu dibuang agar dimensi
        tetap align dengan data validasi/out-of-sample).

    Return
    ------
    result_df : pd.DataFrame
        DataFrame hasil gabungan (observed asli + missing hasil imputasi)
        yang juga sudah disimpan ke `save_path`.
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

    # Special-case sama seperti get_eval: buang 1 baris di news oos agar
    # dimensi align.
    if dataname == 'news' and oos is True:
        drop = 6265
        if drop < len(result_df):
            result_df = result_df.drop(index=drop).reset_index(drop=True)
        if num_mask is not None:
            num_mask = np.delete(num_mask, drop, axis=0)
        if cat_mask is not None:
            cat_mask = np.delete(cat_mask, drop, axis=0)
        pred_X_emb = np.delete(pred_X_emb, drop, axis=0)

    # ===== Kolom numerik: observed = asli, missing = hasil decode imputasi =====
    if len(num_col_idx) > 0:
        num_pred_orig = _decode_and_denormalize_numeric(
            pred_X_emb, emb_model, bin_midpoints, n_num_cols,
            num_mean, num_std, device
        )  # [N, n_num_cols], skala asli

        num_cols = cols[num_col_idx]
        for i, col in enumerate(num_cols):
            col_values = result_df[col].values.astype(np.float32).copy()
            miss_rows = num_mask[:, i]
            col_values[miss_rows] = num_pred_orig[miss_rows, i]
            result_df[col] = col_values

    # ===== Kolom kategorik: observed = asli, missing = hasil decode imputasi ===
    if len(cat_col_idx) > 0 and emb_model is not None:
        cat_cols = cols[cat_col_idx]

        pred_all_idx = decode_cat_from_embedding(
            emb_model, pred_X_emb, device
        )  # [N, n_num_cols + n_cat_cols]

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
    Membulatkan kolom numerik pada hasil imputasi (`result_df`, keluaran
    save_imputed_csv_mrmd) supaya rapi dibaca, dengan deteksi OTOMATIS per
    kolom. [SAMA PERSIS dengan versi pada dataset.py, generik terhadap
    pipeline apapun — hanya butuh result_df/dataname/split_df_path]

        - Kalau SEMUA nilai di kolom itu pada file CSV ASLI (train.csv/
          val.csv) adalah bilangan bulat (mis. jumlah, tahun, kode
          numerik) -> dibulatkan ke integer.
        - Kalau tidak (memang mengandung desimal, mis. harga/berat/ukuran)
          -> dibulatkan ke `decimals` angka di belakang koma.

    Deteksi integer/bukan dilakukan dari file CSV ASLI (bukan dari
    result_df), karena file CSV asli sudah berisi nilai LENGKAP tanpa NaN.

    Parameters
    ----------
    result_df : pd.DataFrame
        DataFrame hasil dari save_imputed_csv_mrmd.
    dataname : str
        Nama dataset, dipakai membaca Info/{dataname}.json kalau
        num_col_idx tidak diberikan.
    split_df_path : str
        Path ke CSV asli (train.csv/val.csv) yang jadi referensi deteksi
        integer/bukan.
    num_col_idx : list[int] atau None
        Index kolom numerik. Kalau None, diambil dari Info/{dataname}.json.
    decimals : int
        Jumlah desimal untuk kolom yang TIDAK terdeteksi sebagai integer
        (default 4).
    save_path : str atau None
        Kalau diisi, hasil setelah pembulatan ditulis ulang (overwrite)
        ke path ini.

    Return
    ------
    rounded_df : pd.DataFrame
        SALINAN result_df dengan kolom numerik sudah dibulatkan
        (result_df asli tidak diubah).
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
        # PENTING: pakai selisih absolut MURNI (bukan np.allclose dengan
        # rtol default), karena rtol ikut menyesuaikan skala nilai - untuk
        # kolom bernilai besar, rtol default membuat toleransi jadi sangat
        # longgar sehingga nilai desimal bisa salah terdeteksi sebagai
        # "bulat".
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