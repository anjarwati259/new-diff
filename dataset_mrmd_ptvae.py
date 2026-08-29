import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
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
#  PT-VAE Embedding Model — MENGGANTIKAN DAE (Denoising Autoencoder)
#  Berdasarkan: Liu, Z., Liu, Y., Yu, Z., et al. (2025). "PT-VAE: Variational
#  autoencoder with prior concept transformation." Neurocomputing, 638,
#  130129. https://doi.org/10.1016/j.neucom.2025.130129
#
#  Arsitektur mengikuti paper (Fig. 1, Fig. 2, Algorithm 1):
#    [A] Prior Concept Encoder : x_prior → (mu_prior, log_var_prior, T_prior)
#    [B] Main Encoder          : x       → (mu, log_var, T_concept)
#    [C] Gumbel-Softmax        : (T_concept, T_prior) → c_concept, q_c_prior
#    [D] Reparameterization    : z = mu + sigma*eps ; z_fused = z + c_concept
#    [E] Main Decoder          : z_fused → split per emb_sizes → logits per kolom
#    [F] Prior Concept Decoder : c_concept → logits per kolom (utk L_recon)
#    [G] Loss Total (Eq. 14)   : L_ELBO + L_recon + L_KL (+ L_class auxiliary)
#
#  nn.Embedding per kolom dipakai sebagai input lookup (BUKAN one-hot seperti
#  DAE), karena PT-VAE — sesuai paper aslinya — memakai representasi dense
#  embedding sebagai input encoder, bukan vektor biner.
#
#  [FIX-LEAKAGE — KHUSUS FILE INI, TIDAK ADA DI VERSI PT-VAE DASAR]
#  Dataset di sini punya nilai yang MEMANG hilang (mask_array, real missing
#  value untuk task imputasi). Karena PT-VAE memakai nn.Embedding (lookup
#  tabel, BUKAN one-hot seperti DAE), mekanisme "zero one-hot" DAE tidak bisa
#  dipakai langsung — sebagai gantinya dipakai TOKEN KHUSUS 'missing' per
#  kolom (+1 baris di tabel embedding), PERSIS seperti trik anti-leakage pada
#  versi nn.Embedding sebelumnya (SupervisedLearnableEmbeddingModel):
#
#    1) TOKEN MISSING (+1 vocab per kolom): posisi yang BENAR-BENAR missing
#       di-substitusi dengan index token khusus SEBELUM masuk ke embedding
#       lookup (baik jalur Main Encoder maupun Prior Concept Encoder, karena
#       keduanya berbagi embedding & input x yang sama — Section 3.1 paper).
#       Encoder TIDAK PERNAH melihat nilai asli di posisi itu. Decoder tetap
#       memprediksi ke ruang kelas ASLI (tanpa token missing) lewat
#       cat_dims_decode, persis seperti versi nn.Embedding sebelumnya.
#
#    2) RECONSTRUCTION LOSS (CE, komponen L_ELBO) hanya dihitung pada posisi
#       OBSERVED per kolom — posisi missing DIKECUALIKAN, sama seperti
#       prinsip fix-leakage pada versi nn.Embedding sebelumnya.
#
#    3) Komponen loss lain (KL(z), KL(c), L_recon, L_KL, L_class) TIDAK
#       di-mask — semuanya adalah regularizer atas representasi laten /
#       target label per-baris, bukan perbandingan langsung ke nilai fitur
#       x di posisi tertentu, sehingga tidak berisiko leakage per-sel.
#
#  Classification loss (auxiliary, TETAP dipertahankan sesuai desain PT-VAE
#  original — lihat dataset_mrmdwith_ptvae.py) tidak diubah oleh mask_array.
#
#  Fitur numerik (hasil diskritisasi MRmD) tetap di-embed BERSAMA fitur
#  kategorikal memakai model PT-VAE yang sama. Pipeline embedding → imputasi
#  TIDAK BERUBAH.
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


class PTVAEEmbeddingModel(nn.Module):
    """
    PT-VAE Embedding Model — MENGGANTIKAN DAEEmbeddingModel.

    Berdasarkan Liu et al. (2025) "PT-VAE: Variational autoencoder with
    prior concept transformation." Neurocomputing 638, 130129.
    https://doi.org/10.1016/j.neucom.2025.130129

    Arsitektur (sama persis dengan dataset_mrmdwith_ptvae.py — lihat
    docstring di sana untuk rincian tiap komponen [A]-[G]):
      [A] Prior Concept Encoder : x_prior → (mu_prior, log_var_prior, T_prior)
      [B] Main Encoder          : x       → (mu, log_var, T_concept)
      [C] Gumbel-Softmax        : (T_concept, T_prior) → c_concept, q_c_prior
      [D] Reparameterization    : z = mu + sigma*eps ; z_fused = z + c_concept
      [E] Main Decoder          : z_fused → split per emb_sizes → logits per kolom
      [F] Prior Concept Decoder : c_concept → logits per kolom (utk L_recon)

    [FIX-LEAKAGE — tambahan dibanding dataset_mrmdwith_ptvae.py]
    cat_dims yang diterima BISA berisi +1 token khusus 'missing' per kolom
    (lihat train_supervised_embedding_model / train_vae_embedding_model),
    sehingga tabel nn.Embedding punya 1 baris ekstra untuk merepresentasikan
    "nilai tidak diketahui" secara netral. Substitusi index ke token ini
    (untuk posisi yang benar-benar missing) dilakukan DI LUAR model, sebelum
    x_cat masuk ke forward()/encode_with_params() — persis pola
    SupervisedLearnableEmbeddingModel (nn.Embedding) versi sebelumnya.
    Decoder (main & prior) tetap memprediksi ke ruang kelas ASLI (tanpa
    token missing) lewat parameter `cat_dims_decode`.
    """

    def __init__(self, cat_dims: list, emb_sizes: list, n_classes: int,
                 dropout: float = 0.1, hidden_dim: int = 256,
                 latent_dim: int = None,
                 encoder_ratio: float = 1.5,
                 tau: float = 0.5,
                 cat_dims_decode: list = None):
        """
        Parameter
        ---------
        cat_dims        : list[int] — jumlah baris tabel nn.Embedding per kolom.
                           [FIX-LEAKAGE] Bisa berisi +1 token khusus 'missing'
                           per kolom (lihat train_vae_embedding_model), supaya
                           posisi yang di-mask missing bisa di-encode dengan
                           token netral, bukan nilai aslinya.
        emb_sizes       : list[int] — ukuran embedding per kolom
        n_classes       : int       — jumlah kelas untuk supervised classification
        dropout         : float     — dropout rate
        hidden_dim      : int       — hidden dim untuk MLP classifier
        latent_dim      : int|None  — dimensi ruang laten z (default = total_emb_dim)
        encoder_ratio   : float     — rasio hidden dim encoder/decoder
        tau             : float     — temperature tau untuk Gumbel-Softmax
        cat_dims_decode : list[int] — jumlah kelas ASLI (TANPA token 'missing')
                           untuk output decoder rekonstruksi (main & prior).
                           Jika None, sama dengan cat_dims (perilaku lama /
                           tidak ada token missing).
        """
        super().__init__()

        cat_dims_decode = cat_dims_decode if cat_dims_decode is not None else cat_dims

        self.total_emb_dim = sum(emb_sizes)
        self.n_cols        = len(cat_dims)
        self.cat_dims      = cat_dims            # vocab embedding (bisa +1 token missing)
        self.cat_dims_decode = cat_dims_decode   # vocab decoder (ASLI, tanpa token missing)
        self.emb_sizes     = emb_sizes
        self.n_classes     = n_classes
        self.tau           = tau

        # latent_dim = total_emb_dim (default) agar z_fused bisa langsung
        # di-split per emb_sizes untuk decoding — sesuai paper Section 4.1
        # "The decoder are symmetric with the encoder"
        self.latent_dim = latent_dim if latent_dim is not None else self.total_emb_dim
        self.out_dim    = self.latent_dim   # = total_emb_dim (ruang diffusion)

        # ── [A+B] Input embedding lookup per kolom ────────────────────────
        # Shared embeddings untuk main encoder dan prior concept encoder
        # (x dan x_prior adalah data yang sama, paper Section 3.1)
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=n_cat, embedding_dim=emb_dim)
            for n_cat, emb_dim in zip(cat_dims, emb_sizes)
        ])

        enc_hidden = max(self.total_emb_dim, int(self.total_emb_dim * encoder_ratio))

        # ── [B] MAIN ENCODER MLP ──────────────────────────────────────────
        self.encoder_mlp = nn.Sequential(
            nn.Linear(self.total_emb_dim, enc_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(enc_hidden, enc_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.fc_mu      = nn.Linear(enc_hidden, self.latent_dim)
        self.fc_log_var = nn.Linear(enc_hidden, self.latent_dim)

        # ── [A] PRIOR CONCEPT ENCODER MLP ────────────────────────────────
        self.prior_encoder_mlp = nn.Sequential(
            nn.Linear(self.total_emb_dim, enc_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(enc_hidden, enc_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.fc_mu_prior      = nn.Linear(enc_hidden, self.latent_dim)
        self.fc_log_var_prior = nn.Linear(enc_hidden, self.latent_dim)

        # ── [E] MAIN DECODER — Linear per kolom (simetris encoder) ──────
        # [FIX-LEAKAGE] Output size pakai cat_dims_decode (ASLI, tanpa
        # token missing), meskipun embedding lookup pakai cat_dims (+1).
        self.decoders = nn.ModuleList([
            nn.Linear(emb_size, n_cat)
            for n_cat, emb_size in zip(cat_dims_decode, emb_sizes)
        ])

        # ── [F] PRIOR CONCEPT DECODER MLP — c_concept → total_emb_dim ────
        dec_hidden = max(self.latent_dim, int(self.latent_dim * encoder_ratio))
        self.prior_decoder_mlp = nn.Sequential(
            nn.Linear(self.latent_dim, dec_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dec_hidden, self.total_emb_dim),
        )

        # ── MLP Classifier (auxiliary, dipertahankan) ─────────────────────
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(self.latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_classes)
        )

        # ── LayerNorm pada z_fused [batch, latent_dim] ───────────────────
        self.layer_norm = nn.LayerNorm(self.latent_dim)

    # ── Embedding input ───────────────────────────────────────────────────

    def _embed_input(self, x_cat: torch.Tensor) -> torch.Tensor:
        """
        Lookup embedding per kolom dan concat.
        x_cat : [batch, n_cols] — index SUDAH disubstitusi token missing
                (jika ada) oleh pemanggil (lihat train_vae_embedding_model).
        return: [batch, total_emb_dim]
        """
        return torch.cat([
            self.embeddings[i](x_cat[:, i]) for i in range(self.n_cols)
        ], dim=1)

    # ── [B] Main Encoder ──────────────────────────────────────────────────

    def _encode_to_params(self, x_emb: torch.Tensor):
        """Main Encoder q_phi(z|x). Return: (mu, log_var, T_concept)."""
        h       = self.encoder_mlp(x_emb)
        mu      = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        log_var = torch.clamp(log_var, min=-10.0, max=10.0)
        T_concept = torch.exp(0.5 * log_var)
        return mu, log_var, T_concept

    # ── [A] Prior Concept Encoder ─────────────────────────────────────────

    def _encode_prior_to_params(self, x_emb: torch.Tensor):
        """Prior Concept Encoder. Return: (mu_prior, log_var_prior, T_prior)."""
        h             = self.prior_encoder_mlp(x_emb)
        mu_prior      = self.fc_mu_prior(h)
        log_var_prior = self.fc_log_var_prior(h)
        log_var_prior = torch.clamp(log_var_prior, min=-10.0, max=10.0)
        T_prior = torch.exp(0.5 * log_var_prior)
        return mu_prior, log_var_prior, T_prior

    # ── [C] Gumbel-Softmax Reparameterization ─────────────────────────────

    @staticmethod
    def _sample_gumbel(shape, device, eps: float = 1e-6) -> torch.Tensor:
        """Sampling Gumbel(0, 1): g = -log(-log(U)), U ~ Uniform(0,1)."""
        U = torch.rand(shape, device=device).clamp(eps, 1.0 - eps)
        return -torch.log(-torch.log(U))

    def _gumbel_softmax_concept(
        self,
        T_concept: torch.Tensor,
        T_prior:   torch.Tensor,
    ):
        """
        Gumbel-Softmax Reparameterization Trick. Paper Eq. 3 & 4.
        Return: (c_concept, q_c_prior)
        """
        device = T_concept.device
        T_c = torch.clamp(T_concept, min=1e-8)
        T_p = torch.clamp(T_prior,   min=1e-8)

        if self.training:
            g_concept = self._sample_gumbel(T_c.shape, device)
            g_prior   = self._sample_gumbel(T_p.shape, device)
        else:
            g_concept = torch.zeros_like(T_c)
            g_prior   = torch.zeros_like(T_p)

        logit_c = (torch.log(T_c) + g_concept) / self.tau
        logit_p = (torch.log(T_p) + g_prior)   / self.tau

        denom_log = torch.logaddexp(logit_c, logit_p)
        c_concept  = torch.exp(logit_c - denom_log)   # Eq. 3
        q_c_prior  = torch.exp(logit_p - denom_log)   # Eq. 4

        return c_concept, q_c_prior

    # ── [D] Normal Reparameterization ─────────────────────────────────────

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """z = mu + sigma * eps (training); z = mu (inference, deterministik)."""
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    # ── [E+F] Decode ──────────────────────────────────────────────────────

    def _decode_prior_concept(self, c_concept: torch.Tensor) -> torch.Tensor:
        """Prior Concept Decoder: c_concept → [batch, total_emb_dim]. Paper Eq. 12."""
        return self.prior_decoder_mlp(c_concept)

    def _logits_from_recon(self, recon_emb: torch.Tensor) -> list:
        """Split recon embedding [batch, total_emb_dim] → list per-kolom logits."""
        per_col = torch.split(recon_emb, self.emb_sizes, dim=1)
        return [self.decoders[i](per_col[i]) for i in range(self.n_cols)]

    # ── Public API ────────────────────────────────────────────────────────

    def encode(self, x_cat: torch.Tensor) -> torch.Tensor:
        """
        Encode integer index → z_fused (deterministik saat inference).

        [PENTING] x_cat di sini TIDAK disubstitusi token missing — dipakai
        untuk membangun embedding ground-truth (X_true) di load_dataset(),
        lihat komentar di encode_with_embedding().

        Output shape: [N, latent_dim] = [N, total_emb_dim]
        """
        x_emb = self._embed_input(x_cat)
        mu, log_var, T_concept = self._encode_to_params(x_emb)
        _, _, T_prior          = self._encode_prior_to_params(x_emb)
        c_concept, _           = self._gumbel_softmax_concept(T_concept, T_prior)
        z_fused = mu + c_concept              # [N, latent_dim] — ⊕ = ADDITION (Eq.5)
        return self.layer_norm(z_fused)       # [N, latent_dim]

    def encode_with_params(self, x_cat: torch.Tensor):
        """
        Encode dan kembalikan semua parameter untuk PT-VAE loss.
        x_cat : index yang SUDAH disubstitusi token missing oleh pemanggil
                (jika training dengan mask_array — fix-leakage).

        Return: (z_fused, mu, log_var, c_concept, q_c_prior,
                 mu_prior, log_var_prior)
        """
        x_emb = self._embed_input(x_cat)
        mu, log_var, T_concept             = self._encode_to_params(x_emb)
        mu_prior, log_var_prior, T_prior   = self._encode_prior_to_params(x_emb)
        c_concept, q_c_prior               = self._gumbel_softmax_concept(T_concept, T_prior)
        z                                  = self.reparameterize(mu, log_var)
        z_fused                            = z + c_concept          # ⊕ = ADDITION (Eq.5)
        z_normed                           = self.layer_norm(z_fused)
        return (z_normed, mu, log_var, c_concept, q_c_prior, mu_prior, log_var_prior)

    def classify(self, z: torch.Tensor) -> torch.Tensor:
        """Auxiliary classifier: z_fused → logit kelas."""
        return self.classifier(z)

    def decode(self, z: torch.Tensor) -> list:
        """
        Main Decoder: z_fused → per-kolom logits.
        z      : [batch, latent_dim] — z_fused dari encoder atau pred_X dari diffusion
        return : list[n_cols] of [batch, vocab_size_i]  (vocab_size_i = cat_dims_decode[i])
        """
        per_col = torch.split(z, self.emb_sizes, dim=1)
        return [self.decoders[i](per_col[i]) for i in range(self.n_cols)]

    def decode_prior(self, c_concept: torch.Tensor) -> list:
        """Prior Concept Decoder: c_concept → per-kolom logits. Utk L_recon (Eq. 12)."""
        return self._logits_from_recon(self._decode_prior_concept(c_concept))

    def forward(self, x_cat: torch.Tensor, add_noise: bool = False):
        """
        Forward pass PT-VAE untuk training. Algorithm 1, Lines 3-7.

        x_cat : index yang SUDAH disubstitusi token missing oleh pemanggil
                (jika mask_array diberikan saat training — fix-leakage).
                Target reconstruction loss TETAP memakai index ASLI
                (dilakukan di training loop, BUKAN di sini).

        Return: (z_fused, mu, log_var, c_concept, q_c_prior, mu_prior,
                 log_var_prior, class_logits, recon_logits, recon_prior_logits)
        """
        (z_fused, mu, log_var, c_concept, q_c_prior,
         mu_prior, log_var_prior) = self.encode_with_params(x_cat)

        class_logits       = self.classify(z_fused)
        recon_logits       = self.decode(z_fused)
        recon_prior_logits = self._logits_from_recon(
            self._decode_prior_concept(c_concept)
        )

        return (z_fused, mu, log_var, c_concept, q_c_prior,
                mu_prior, log_var_prior, class_logits,
                recon_logits, recon_prior_logits)

    # ── PT-VAE Loss Functions (Section 3.3) ──────────────────────────────

    @staticmethod
    def kl_divergence(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """KL(q(z|x) || p(z)). Paper Eq. 9 & 10. Closed-form (Kingma & Welling 2013)."""
        log_var_c = torch.clamp(log_var, min=-10.0, max=10.0)
        mu_c      = torch.clamp(mu,      min=-10.0, max=10.0)
        kl = -0.5 * torch.sum(
            1 + log_var_c - mu_c.pow(2) - log_var_c.exp(), dim=1
        )
        return kl.mean()

    @staticmethod
    def kl_divergence_c(c_concept: torch.Tensor, K: int) -> torch.Tensor:
        """KL(q(c|x) || p(c)). Paper Eq. 11. Prior p(c) = Bernoulli(0.5) per dimensi."""
        c_s = torch.clamp(c_concept, min=1e-7, max=1.0 - 1e-7)
        H = -(c_s * torch.log(c_s) + (1.0 - c_s) * torch.log(1.0 - c_s))
        log2 = torch.log(torch.tensor(2.0, device=c_concept.device))
        kl_per_dim = log2 - H
        latent_dim = c_concept.shape[1]
        return kl_per_dim.sum(dim=1).mean() / latent_dim

    @staticmethod
    def reconstruction_loss_concept(
        recon_prior_logits: list,
        recon_logits: list,
        n_cols: int
    ) -> torch.Tensor:
        """L_recon = ||x'_concept - x'||^2. Paper Eq. 12 (MSE antar logit dua decoder)."""
        loss = torch.tensor(0.0, device=recon_logits[0].device)
        for i in range(n_cols):
            loss = loss + F.mse_loss(recon_prior_logits[i], recon_logits[i])
        return loss / n_cols

    @staticmethod
    def kl_divergence_concept_prior(
        q_c_prior: torch.Tensor,
        c_concept: torch.Tensor
    ) -> torch.Tensor:
        """L_KL = KL(q(c_prior|x) || q(c_concept|x)). Paper Eq. 13."""
        eps = 1e-7
        p   = torch.clamp(q_c_prior, min=eps, max=1.0 - eps)
        q   = torch.clamp(c_concept, min=eps, max=1.0 - eps)
        kl  = p * torch.log(p / q) + (1.0 - p) * torch.log((1.0 - p) / (1.0 - q))
        latent_dim = p.shape[1]
        return kl.sum(dim=1).mean() / latent_dim


# Alias kompatibilitas — beberapa docstring/anotasi tipe di file ini masih
# mereferensikan nama lama (SupervisedLearnableEmbeddingModel dari versi
# nn.Embedding, dan DAEEmbeddingModel dari versi DAE).
SupervisedLearnableEmbeddingModel = PTVAEEmbeddingModel
DAEEmbeddingModel = PTVAEEmbeddingModel
VAEEmbeddingModel = PTVAEEmbeddingModel


# ===========================================================================
#  Training PT-VAE Embedding — MENGGANTIKAN Training DAE
#  Berdasarkan train_vae_embedding_model pada dataset_mrmdwith_ptvae.py,
#  DITAMBAH mekanisme fix-leakage (token missing + loss ter-mask) — PERSIS
#  pola yang dipakai versi nn.Embedding sebelumnya (SupervisedLearnableEmbeddingModel),
#  karena PT-VAE juga berbasis nn.Embedding lookup (bukan one-hot seperti DAE).
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
                                     latent_dim: int = None,
                                     encoder_ratio: float = 1.5,
                                     patience: int = 30,
                                     mask_array: np.ndarray = None,
                                     tau: float = 1.0) -> PTVAEEmbeddingModel:
    """
    Latih PTVAEEmbeddingModel dengan loss PT-VAE sesuai Liu et al. (2025)
    (lihat train_vae_embedding_model pada dataset_mrmdwith_ptvae.py untuk
    rincian formula loss L_ELBO + L_recon + L_KL + L_class, Eq. 14).

    cat_idx_array berisi SEMUA kolom (numerik bin hasil MRmD + kategorikal).

    mask_array : [N, n_cols] bool, opsional — True = nilai HILANG (missing).
        [FIX-LEAKAGE] Jika diberikan, posisi missing:
          1) Di-substitusi dengan TOKEN KHUSUS 'missing' (+1 baris tabel
             nn.Embedding per kolom) SEBELUM masuk ke Main Encoder MAUPUN
             Prior Concept Encoder (keduanya berbagi embedding & input yang
             sama — Section 3.1 paper). Encoder tidak pernah melihat nilai
             asli di posisi itu. Decoder tetap memprediksi ke ruang kelas
             ASLI (tanpa token missing) lewat cat_dims_decode.
          2) DIKECUALIKAN dari komponen reconstruction CE pada L_ELBO
             (elbo_recon), per kolom. Komponen loss lain (KL(z), KL(c),
             L_recon, L_KL, L_class) TIDAK di-mask — semuanya regularizer
             representasi laten / label per-baris, bukan perbandingan
             langsung ke nilai fitur x tertentu.

    Return : PTVAEEmbeddingModel (parameter di-freeze, eval mode)
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
    # representasi laten z_fused. Decoder tetap memprediksi ke ruang kelas
    # ASLI (tanpa token missing) lewat cat_dims_decode.
    use_missing_token = mask_array is not None
    if use_missing_token:
        cat_dims_embed = [d + 1 for d in cat_dims]
        missing_idx = torch.tensor(cat_dims, dtype=torch.long, device=device)
    else:
        cat_dims_embed = cat_dims
        missing_idx = None

    model = PTVAEEmbeddingModel(
        cat_dims        = cat_dims_embed,
        emb_sizes       = emb_sizes,
        n_classes       = n_classes,
        dropout         = dropout,
        hidden_dim      = hidden_dim,
        latent_dim      = latent_dim,
        encoder_ratio   = encoder_ratio,
        tau             = tau,   # temperature τ — tau=1.0 mencegah c_concept saturated
        cat_dims_decode = cat_dims,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    ce_loss          = nn.CrossEntropyLoss()
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

    cpu_gen = torch.Generator(device='cpu')
    loader  = torch.utils.data.DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = 0,
        pin_memory  = False,
        generator   = cpu_gen,
    )

    # K = latent_dim — dipakai untuk upper bound KL_c = log K (Eq. 11)
    K = model.latent_dim

    best_loss        = float('inf')
    patience_counter = 0
    best_model_state = None

    print(f'[PT-VAE] fix_leakage={"ON (mask_array diberikan, token missing aktif)" if use_missing_token else "OFF"}')

    model.train()
    for epoch in range(n_epochs):
        total_loss        = 0.0
        total_elbo_recon  = 0.0
        total_kl_z        = 0.0
        total_kl_c        = 0.0
        total_l_elbo      = 0.0
        total_recon_loss  = 0.0
        total_kl_loss     = 0.0
        total_class_loss  = 0.0
        n_batches         = 0

        for batch in loader:
            if mask_array is not None:
                batch_cat, batch_labels, batch_observed = batch
            else:
                batch_cat, batch_labels = batch
                batch_observed = None

            optimizer.zero_grad()

            if batch_observed is not None:
                # [FIX-LEAKAGE] Ganti index pada posisi MISSING dengan token
                # khusus 'missing' SEBELUM masuk ke encoder (main & prior,
                # karena keduanya berbagi embedding & input yang sama).
                batch_cat_in = batch_cat.clone()
                miss_pos     = ~batch_observed
                batch_cat_in[miss_pos] = missing_idx.unsqueeze(0).expand_as(batch_cat)[miss_pos]
            else:
                batch_cat_in = batch_cat

            # ── Algorithm 1, Lines 3-7 ────────────────────────────────────
            (z_fused, mu, log_var, c_concept, q_c_prior,
             mu_prior, log_var_prior, class_logits,
             recon_logits, recon_prior_logits) = model(batch_cat_in)

            # ── L_ELBO (Eq. 10) ───────────────────────────────────────────
            # Term 1: -E_q[log p(x|z,c)] ≈ CE loss.
            # [FIX-LEAKAGE] Target CE SELALU batch_cat (index ASLI, BUKAN
            # batch_cat_in), dan HANYA dihitung pada posisi OBSERVED.
            if batch_observed is not None:
                col_losses = []
                for i in range(model.n_cols):
                    obs_i = batch_observed[:, i]
                    if obs_i.any():
                        per_elem = ce_loss_noreduce(recon_logits[i], batch_cat[:, i])
                        col_losses.append(per_elem[obs_i].mean())
                if len(col_losses) > 0:
                    elbo_recon = sum(col_losses) / len(col_losses)
                else:
                    elbo_recon = torch.tensor(0.0, device=device)
            else:
                elbo_recon = sum(
                    ce_loss(recon_logits[i], batch_cat[:, i])
                    for i in range(model.n_cols)
                ) / model.n_cols

            # Term 2: KL(q(z|x) || p(z)) — Eq. 9, closed-form, ALWAYS POSITIVE
            kl_z = PTVAEEmbeddingModel.kl_divergence(mu, log_var)

            # Term 3: KL(q(c|x) || p(c)) — Eq. 11, ALWAYS POSITIVE
            kl_c = PTVAEEmbeddingModel.kl_divergence_c(c_concept, K)

            # L_ELBO (bentuk minimisasi): elbo_recon + KL_z + KL_c
            l_elbo = elbo_recon + kl_z + kl_c

            # ── L_recon (Eq. 12) — TIDAK di-mask, bukan perbandingan ke x ──
            l_recon = PTVAEEmbeddingModel.reconstruction_loss_concept(
                recon_prior_logits, recon_logits, model.n_cols
            )

            # ── L_KL (Eq. 13) — TIDAK di-mask ──────────────────────────────
            l_kl = PTVAEEmbeddingModel.kl_divergence_concept_prior(
                q_c_prior, c_concept
            )

            # ── L_class (auxiliary) — TIDAK di-mask, label per-baris ───────
            class_loss = ce_loss(class_logits, batch_labels)

            # ── Total Loss (Eq. 14) ───────────────────────────────────────
            loss = l_elbo + l_recon + l_kl + class_loss

            # Guard NaN: skip batch jika ada komponen NaN
            if not torch.isfinite(loss):
                nan_info = {
                    'elbo_recon': elbo_recon.item(),
                    'kl_z':       kl_z.item(),
                    'kl_c':       kl_c.item(),
                    'l_recon':    l_recon.item(),
                    'l_kl':       l_kl.item(),
                    'class_loss': class_loss.item(),
                }
                print(f'[WARN] PT-VAE loss NaN/Inf di-skip: {nan_info}')
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss       += loss.item()
            total_elbo_recon += elbo_recon.item()
            total_kl_z       += kl_z.item()
            total_kl_c       += kl_c.item()
            total_l_elbo     += l_elbo.item()
            total_recon_loss += l_recon.item()
            total_kl_loss    += l_kl.item()
            total_class_loss += class_loss.item()
            n_batches        += 1

        avg_loss        = total_loss        / n_batches
        avg_elbo_recon  = total_elbo_recon  / n_batches
        avg_kl_z        = total_kl_z        / n_batches
        avg_kl_c        = total_kl_c        / n_batches
        avg_l_elbo      = total_l_elbo      / n_batches
        avg_recon_loss  = total_recon_loss  / n_batches
        avg_kl_loss     = total_kl_loss     / n_batches
        avg_class_loss  = total_class_loss  / n_batches

        if (epoch + 1) % 10 == 0:
            print(
                f'[PT-VAE] Epoch {epoch+1:>4}/{n_epochs} | '
                f'Loss={avg_loss:.4f} | '
                f'L_ELBO={avg_l_elbo:.4f} '
                f'[CE(obs-only)={avg_elbo_recon:.4f}, KL(z)={avg_kl_z:.4f}, KL(c)={avg_kl_c:.4f}] | '
                f'L_recon={avg_recon_loss:.4f} | '
                f'L_KL={avg_kl_loss:.4f} | '
                f'L_class={avg_class_loss:.4f}'
            )
            losses = {
                'KL(z)':   avg_kl_z,
                'KL(c)':   avg_kl_c,
                'L_recon': avg_recon_loss,
                'L_KL':    avg_kl_loss,
            }
            for name, val in losses.items():
                if val < 1e-8:
                    print(f'  [WARN] {name} mendekati nol ({val:.2e}) — '
                          f'komponen ini mungkin tidak aktif!')
            vals = list(losses.values())
            if len(vals) > 0 and max(vals) > 10 * (sum(vals) / len(vals)):
                dominant = max(losses, key=losses.get)
                print(f'  [WARN] {dominant}={losses[dominant]:.4f} mendominasi loss '
                      f'(target masing-masing komponen ≈ 0.1–1.0)')

        if avg_loss < best_loss:
            best_loss        = avg_loss
            patience_counter = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f'[PT-VAE Embedding] Early stopping triggered at epoch {epoch+1}')
            print(f'[PT-VAE Embedding] Best total loss: {best_loss:.4f}')
            break

    if best_model_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
        print(f'[PT-VAE Embedding] Loaded best model state.')

    model.eval()

    with torch.no_grad():
        sample_cat = cat_tensor[:min(2048, len(cat_tensor))]
        # encode() ground-truth: TANPA substitusi token missing, karena
        # dipakai hanya utk memantau distribusi laten, bukan training/imputasi.
        z_sample = model.encode(sample_cat)
        print(f'[PT-VAE Embedding] Distribusi laten z_fused (N={z_sample.shape[0]}):')
        print(f'  mean={z_sample.mean().item():.4f}  '
              f'std={z_sample.std().item():.4f}  '
              f'norm_mean={z_sample.norm(dim=1).mean().item():.4f}')

    # ── Evaluasi reconstruction accuracy (diagnostik, observed-only) ──────
    if mask_array is not None:
        with torch.no_grad():
            eval_n         = min(4096, len(cat_tensor))
            eval_cat       = cat_tensor[:eval_n]
            eval_observed  = torch.tensor(
                ~np.array(mask_array, dtype=bool)[:eval_n], dtype=torch.bool, device=device
            )
            z_eval    = model.encode(eval_cat)   # deterministik (eval mode)
            recon_eval = model.decode(z_eval)
            correct_total = 0
            total_count   = 0
            for i in range(model.n_cols):
                pred_i = recon_eval[i].argmax(dim=1)
                true_i = eval_cat[:, i]
                obs_i  = eval_observed[:, i]
                correct_total += (pred_i[obs_i] == true_i[obs_i]).sum().item()
                total_count   += int(obs_i.sum().item())
            acc = correct_total / max(total_count, 1)
            print(f'[PT-VAE Embedding] Reconstruction accuracy (observed-only, '
                  f'N={eval_n}): {acc:.4f}')

    for param in model.parameters():
        param.requires_grad_(False)
    print('[PT-VAE Embedding] Seluruh parameter PT-VAE embedding di-freeze untuk training diffusion.')

    return model

# ===========================================================================
#  Encode / Decode helpers (disesuaikan ke PTVAEEmbeddingModel)
#  [DIKEMBALIKAN ke semantik softmax kategoris — BUKAN sigmoid Bernoulli
#  seperti DAE] karena decoder PT-VAE (Linear per kolom, dilatih dengan
#  CrossEntropyLoss) menghasilkan LOGITS kategoris biasa, persis seperti
#  dataset_mrmdwith_ptvae.py.
# ===========================================================================

def encode_with_embedding(model: PTVAEEmbeddingModel,
                          cat_idx_array: np.ndarray,
                          device: str,
                          batch_size: int = 4096) -> np.ndarray:
    """
    Encode integer index → embedding numpy array menggunakan PT-VAE encoder.

    Saat inference (eval mode), model.encode() mengembalikan
    z_fused = mu + c_concept secara deterministik — tanpa sampling dan
    tanpa Gumbel noise.

    [PENTING] Dipanggil TANPA substitusi token missing — dipakai untuk
    membangun train_X/test_X ground-truth (X_true) di load_dataset(), yang
    MEMANG harus memakai nilai asli (bukan token missing), karena X_true
    dipakai sebagai target evaluasi imputasi, bukan sebagai sinyal training
    embedding. Fix-leakage hanya relevan saat TRAINING model embedding
    (lihat train_supervised_embedding_model), bukan di sini.

    Output shape: [N, latent_dim] = [N, total_emb_dim].
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
            # encode() saat eval mode → z_fused = mu + c_concept (deterministik)
            z = model.encode(batch)
            all_z.append(z.cpu().numpy())

    return np.concatenate(all_z, axis=0).astype(np.float32)


def decode_cat_from_embedding(model: PTVAEEmbeddingModel,
                              emb_array: np.ndarray,
                              device: str,
                              batch_size: int = 4096) -> np.ndarray:
    """
    Decode z_fused → prediksi kelas tiap kolom (argmax logits).

    PT-VAE decode: split z_fused per emb_sizes → Linear Decoder per kolom
    (output ke ruang kelas ASLI, cat_dims_decode) → argmax.

    emb_array : [N, total_emb_dim]  — z_fused dari encoder atau pred_X dari diffusion
    Return    : [N, n_cols]         — predicted integer index
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


def decode_num_from_embedding(model: PTVAEEmbeddingModel,
                              emb_array: np.ndarray,
                              bin_midpoints: list,
                              n_num_cols: int,
                              device: str,
                              batch_size: int = 4096) -> np.ndarray:
    """
    Decode z_fused → nilai numerik kontinu (dalam skala normalisasi).

    Alur (Weighted Sum / Soft-Max Decode via PT-VAE Decoder):
      pred_X → split per emb_sizes → Linear Decoder kolom numerik
             → softmax (prob per bin) → weighted sum midpoints

    Untuk kolom ke-i (numerik):
        p_i  = softmax(decoders[i](pred_X_i))   # [N, n_bins_i]
        pred = p_i @ mids_i                      # [N] — weighted sum

    Kolom numerik diasumsikan berada di AWAL emb_model (indeks 0..n_num_cols-1),
    diikuti kolom kategorikal.

    Parameter
    ---------
    model         : PTVAEEmbeddingModel
    emb_array     : [N, total_emb_dim]  — pred_X dari diffusion (denormalisasi)
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

    [PENTING - pemisahan skenario FIT MRmD vs TRAINING Embedding]
    Kedua komponen ini punya kebutuhan validasi yang BERBEDA:

    1) MRmD (fit cut point numerik):
       WAJIB memakai split validasi internal ('validation/train.csv' 56%
       sebagai training, 'validation/val.csv' 14% sebagai validasi
       eksternal) karena Algorithm 1 MRmD butuh D_JS(P_t‖P_v) yang
       mensyaratkan validasi genuinely held-out dari data fitting.
       Cut points hasil fit ini kemudian di-TRANSFORM (fixed, tidak
       dilatih ulang) ke train_full & test.

    2) Supervised Embedding Model (kategorikal + bin numerik):
       TIDAK punya kebutuhan validasi eksternal seperti MRmD -- kriteria
       trainingnya cuma classification+reconstruction loss dengan early
       stopping berbasis loss training itu sendiri. Karena itu, embedding
       model di-LATIH ULANG (bukan sekadar dipakai lagi / di-encode saja)
       LANGSUNG di atas dataset FULL:
           train_full : 'datasets/{dataname}/train.csv'
       memakai bin numerik hasil TRANSFORM MRmD (lihat poin 1) + kolom
       kategorikal train_full. Setelah dilatih & di-freeze, model ini
       dipakai untuk meng-encode train_full & test menjadi embedding.

    Ringkasnya:
        MRmD      : FIT di validation-split -> TRANSFORM ke train_full & test
        Embedding : FIT/TRAIN LANGSUNG di train_full -> ENCODE train_full & test

    Dataset FULL (dipakai utk training embedding & pipeline diffusion):
        train_full : 'datasets/{dataname}/train.csv'
        test       : 'datasets/{dataname}/test.csv'
        mask       : 'datasets/{dataname}/masks/rate{ratio}/{mask_type}/train_mask_{idx}.npy'
                     'datasets/{dataname}/masks/rate{ratio}/{mask_type}/test_mask_{idx}.npy'

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
    train_X           : [N_train_full, total_emb_dim]           float32
    test_X            : [N_test,       total_emb_dim]           float32
    ori_train_mask    : mask asli train_full [N_train_full, total_cols]
    ori_test_mask     : mask asli test       [N_test,       total_cols]
    train_num         : [N_train_full, n_num_cols]  — float asli (ternormalisasi)
    test_num          : [N_test,       n_num_cols]
    train_all_idx     : [N_train_full, n_num_cols + n_cat_cols]  — semua bin/label idx
    test_all_idx      : [N_test,       n_num_cols + n_cat_cols]
    extend_train_mask : [N_train_full, total_emb_dim]
    extend_test_mask  : [N_test,       total_emb_dim]
    cat_bin_num       : None  (legacy)
    emb_model         : SupervisedLearnableEmbeddingModel (dilatih di train_full)
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

    # ── [FIT MRmD SAJA] Split validasi internal ────────────────────────────
    fit_train_path      = f'{data_dir}/validation/train.csv'
    fit_val_path         = f'{data_dir}/validation/val.csv'
    fit_train_mask_path  = f'{data_dir}/masks/validation/rate{ratio}/{mask_type}/train_mask_{idx}.npy'
    fit_val_mask_path    = f'{data_dir}/masks/validation/rate{ratio}/{mask_type}/val_mask_{idx}.npy'

    fit_train_df   = pd.read_csv(fit_train_path)
    fit_val_df     = pd.read_csv(fit_val_path)
    fit_train_mask = np.load(fit_train_mask_path)
    fit_val_mask   = np.load(fit_val_mask_path)

    # ── [TRAIN Embedding + Pipeline Diffusion] Dataset FULL ─────────────────
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

    # ── Labels: LabelEncoder di-fit pada TRAIN_FULL (konsumen utamanya
    #    adalah training embedding model di train_full). fit_train/fit_val
    #    (subset dari train_full) di-transform pakai encoder yang sama,
    #    HANYA dipakai sebagai kriteria relevansi (I(A;C)) & validasi
    #    eksternal MRmD. ──────────────────────────────────────────────────
    full_train_y_str = full_train_df[cols[target_col_idx]].values.ravel().astype(str)
    fit_train_y_str   = fit_train_df[cols[target_col_idx]].values.ravel().astype(str)
    fit_val_y_str     = fit_val_df[cols[target_col_idx]].values.ravel().astype(str)

    label_encoder = LabelEncoder()
    label_encoder.fit(full_train_y_str)
    n_classes = len(label_encoder.classes_)

    full_train_labels = label_encoder.transform(full_train_y_str)

    def _safe_transform_labels(y_str, split_name):
        """Transform label dgn fallback ke kelas pertama utk nilai unseen
        (seharusnya jarang terjadi karena fit_train/fit_val adalah subset
        dari train_full, tapi dijaga untuk robustness)."""
        unseen = ~np.isin(y_str, label_encoder.classes_)
        if unseen.any():
            n_unseen = int(unseen.sum())
            unseen_vals = np.unique(y_str[unseen])
            print(f'[Dataset][WARNING] {n_unseen} label pada {split_name} tidak '
                  f'dikenal saat fit LabelEncoder (train_full): {unseen_vals.tolist()}. '
                  f'Dipetakan sementara ke kelas pertama HANYA untuk kriteria '
                  f'relevansi/validasi MRmD.')
            y_str_safe = y_str.copy()
            y_str_safe[unseen] = label_encoder.classes_[0]
        else:
            y_str_safe = y_str
        return label_encoder.transform(y_str_safe)

    fit_train_labels = _safe_transform_labels(fit_train_y_str, 'fit_train (validation/train)')
    fit_val_labels   = _safe_transform_labels(fit_val_y_str,   'fit_val (validation/val)')

    print(f'[Dataset] Detected {n_classes} classes for supervised learning (fit: train_full)')
    print(f'[Dataset] Classes: {label_encoder.classes_}')

    # ── Normalisasi numerik — DIHITUNG DARI TRAIN_FULL ─────────────────────
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
        # [TIDAK BERUBAH] FIT tetap memakai fit_train(56%)+fit_val(14%).
        mrmd_cache_path = f'cache/{dataname}/rate{ratio}/{mask_type}/mrmd_{idx}.pkl'
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

        # Bin midpoints dalam skala NORMALISASI, DIHITUNG DARI TRAIN_FULL
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

    # ── Encoding kolom kategorikal — LabelEncoder di-FIT pada TRAIN_FULL ───
    # (bukan lagi pada fit_train/validation) karena embedding model dilatih
    # LANGSUNG di train_full -> vocabulary kategorikal harus konsisten
    # dengan data yang sesungguhnya dipakai untuk melatihnya.
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

    # ── Gabungkan: [num_bin | cat_idx] → satu array idx untuk embedding ────
    # (langsung dari TRAIN_FULL & TEST, bukan dari fit_train lagi)
    if n_num_cols > 0 and len(cat_col_idx) > 0:
        train_all_idx = np.concatenate([full_train_num_bin, full_train_cat_idx], axis=1)
        test_all_idx  = np.concatenate([full_test_num_bin,  full_test_cat_idx],  axis=1)
    elif n_num_cols > 0:
        train_all_idx = full_train_num_bin
        test_all_idx  = full_test_num_bin
    else:
        train_all_idx = full_train_cat_idx
        test_all_idx  = full_test_cat_idx

    # ── Dimensi embedding ─────────────────────────────────────────────────
    all_dims  = (mrmd.n_bins_ if mrmd is not None else []) + cat_dims_cat
    emb_sizes = [compute_embedding_size(n) for n in all_dims]

    print(f'[Embedding] all_dims (num_bin+cat)={all_dims}')
    print(f'[Embedding] emb_sizes={emb_sizes}, total_emb_dim={sum(emb_sizes)}')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ── Mask untuk TRAIN_FULL & TEST ────────────────────────────────────────
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

    # ── Latih PTVAEEmbeddingModel LANGSUNG di TRAIN_FULL ───────────────────
    # [PERBAIKAN] Sebelumnya model ini sempat "diwariskan" dari fit di split
    # validasi lalu cuma di-encode ke train_full/test. Sekarang model
    # BENAR-BENAR dilatih ulang (training loop penuh) di atas TRAIN_FULL,
    # karena skenarionya beda dengan MRmD -- tidak ada kebutuhan validasi
    # eksternal untuk kriteria trainingnya.
    #
    # [PT-VAE — menggantikan DAE] Berbeda dengan DAE (one-hot + hard-mask
    # internal, output = hidden_dim global), PT-VAE memakai nn.Embedding
    # lookup per kolom dan output z_fused berdimensi sum(emb_sizes) (simetris
    # per kolom, seperti versi nn.Embedding awal). mask_array (train_all_mask)
    # dipakai untuk fix-leakage via TOKEN MISSING (+1 vocab per kolom) +
    # exclude posisi missing dari komponen CE pada L_ELBO.
    print('[PT-VAE] Melatih PTVAEEmbeddingModel '
          '(L_ELBO + L_recon + L_KL + L_class) LANGSUNG pada TRAIN_FULL ...')
    print('[PT-VAE] Referensi: Liu et al. (2025) PT-VAE: Variational Autoencoder with Prior Concept Transformation')
    t_emb_start = time.time()
    print(noise_std)
    emb_model = train_supervised_embedding_model(
        cat_idx_array = train_all_idx,     # bin numerik (dari transform MRmD) + cat idx, TRAIN_FULL
        labels        = full_train_labels, # label TRAIN_FULL (dipakai utk L_class auxiliary)
        cat_dims      = all_dims,
        emb_sizes     = emb_sizes,
        n_classes     = n_classes,
        device        = device,
        n_epochs      = 1000,
        batch_size    = 1024,
        lr            = 1e-3,
        dropout       = 0.1,
        hidden_dim    = 256,               # hidden dim MLP classifier
        latent_dim    = None,              # default = total_emb_dim = sum(emb_sizes)
        encoder_ratio = 1.5,
        patience      = 40,
        mask_array    = train_all_mask,    # [FIX-LEAKAGE] mask TRAIN_FULL — posisi missing → token missing + dikecualikan dari CE pada L_ELBO
        tau           = 1.0,               # temperature Gumbel-Softmax (paper: mencegah c_concept saturated)
    )
    t_emb_end = time.time()
    t_emb = t_emb_end - t_emb_start
    print('[PT-VAE] Training selesai. Parameter di-freeze untuk diffusion.')
    print(f'[PT-VAE] Waktu komputasi embedding: {t_emb:.4f}s')

    # ── Encode TRAIN_FULL & TEST → embedding vector, memakai embedding
    #    model yang BARU SAJA dilatih & di-freeze di atas ───────────────────
    # [PENTING] encode_with_embedding TIDAK menerapkan substitusi token
    # missing — train_X/test_X di sini dibangun dari nilai ASLI (x_clean),
    # karena keduanya dipakai sebagai ground-truth (X_true) utk evaluasi
    # imputasi, BUKAN sebagai sinyal training embedding. Fix-leakage sudah
    # selesai dilakukan di TAHAP TRAINING embedding model (di atas).
    train_all_emb = encode_with_embedding(emb_model, train_all_idx, device)
    test_all_emb  = encode_with_embedding(emb_model, test_all_idx,  device)
    # shape: [N, latent_dim] = [N, total_emb_dim] = [N, sum(emb_sizes)]

    train_X = train_all_emb
    test_X  = test_all_emb

    # ── [PT-VAE] Perluas mask [N, n_cols] → [N, total_emb_dim] ─────────────
    # Berbeda dengan DAE (satu vektor laten global), z_fused PT-VAE tetap
    # berstruktur PER KOLOM (concat sesuai emb_sizes, simetris dengan
    # encoder), sehingga mask bisa diperluas PER KOLOM seperti versi
    # nn.Embedding awal (SupervisedLearnableEmbeddingModel).
    emb_sizes_arr = np.array(emb_sizes, dtype=int)

    def extend_mask_emb(mask: np.ndarray, sizes: np.ndarray) -> np.ndarray:
        """
        Perluas mask [N, n_cols] → [N, total_emb_dim].
        Kolom ke-j diperluas ke sizes[j] dimensi.
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

    return (train_X, test_X,
            full_train_mask, full_test_mask,
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
        Path ke csv asli (mis. 'datasets/{dataname}/train.csv'
        atau 'datasets/{dataname}/test.csv') yang jadi acuan
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