import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
import os
import json
import time

DATA_DIR = 'datasets'

# ===========================================================================
#  VARIAN 2 : "EMBEDDING tanpa DISKRIT"
#  ---------------------------------------------------------------------
#  - Fitur numerik      : TIDAK didiskritisasi dengan MRmD. Tetap dalam
#                          bentuk KONTINU, dinormalisasi (X-mean)/std, lalu
#                          langsung digabung (concat) ke vektor fitur --
#                          TIDAK melalui nn.Embedding sama sekali (karena
#                          nn.Embedding membutuhkan indeks diskrit).
#  - Fitur kategorikal   : tetap di-label-encode lalu di-embed memakai
#                          SupervisedLearnableEmbeddingModel (arsitektur
#                          SAMA PERSIS dengan dataset_mrmd.py: embedding
#                          per kolom + MLP + classifier + linear decoder,
#                          dilatih dengan classification + reconstruction
#                          loss), HANYA SAJA sekarang model ini cuma
#                          menerima kolom KATEGORIKAL (tanpa bin numerik).
#  - Struktur vektor fitur akhir per baris:
#        [ numerik_kontinu (n_num_cols dim) | embedding_kategorikal (sum(emb_sizes) dim) ]
#  - Evaluasi numerik (MAE/RMSE) dilakukan LANGSUNG pada segmen numerik
#    hasil rekonstruksi diffusion (karena memang representasinya kontinu,
#    tidak perlu decode bin/argmax).
#  - Evaluasi kategorikal (Accuracy) tetap lewat linear decoder embedding
#    model (decode_cat_from_embedding), sama seperti dataset_mrmd.py.
# ===========================================================================


# ===========================================================================
#  Supervised Learnable Embedding Model (TIDAK BERUBAH secara arsitektur
#  dari dataset_mrmd.py) — HANYA dipakai untuk kolom KATEGORIKAL di sini.
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
    Model Supervised Learnable Embedding untuk fitur KATEGORIKAL tabular.

    [TIDAK BERUBAH secara arsitektur dari dataset_mrmd.py] — bedanya di
    varian ini model HANYA menerima kolom kategorikal (cat_dims hanya berisi
    jumlah kategori tiap kolom kategorikal, TANPA bin numerik).

    Alur:
      cat_idx [batch, n_cat_cols]
        → nn.Embedding per kolom → concat → [batch, total_emb_dim]
        → (opsional) Linear → SiLU → Linear   (1 hidden layer, jika use_mlp=True)
        → LayerNorm                            (stabilisasi skala sebelum diffusion)
        → z [batch, total_emb_dim]
        → MLP Classifier → [batch, n_classes]  (supervised signal)
        → (+ noise σ=noise_std saat training)
        → Linear Decoder per kolom → logits rekonstruksi
    """

    def __init__(self, cat_dims: list, emb_sizes: list, n_classes: int,
                 dropout: float = 0.1, hidden_dim: int = 256,
                 use_mlp: bool = True, mlp_ratio: float = 1.5,
                 noise_std: float = 0.1,
                 cat_dims_decode: list = None):
        """
        cat_dims        : jumlah baris tabel embedding (nn.Embedding) per kolom
                           KATEGORIKAL. [FIX-LEAKAGE] Bisa berisi +1 token
                           khusus 'missing' per kolom (lihat
                           train_supervised_embedding_model).
        cat_dims_decode : jumlah kelas asli (TANPA token 'missing') untuk
                           output decoder rekonstruksi. Jika None, sama
                           dengan cat_dims.
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
            hidden_dim_mlp = max(self.total_emb_dim, int(self.total_emb_dim * mlp_ratio))
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
            nn.Linear(self.total_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_classes)
        )

        self.decoders = nn.ModuleList([
            nn.Linear(emb_size, n_cat)
            for n_cat, emb_size in zip(cat_dims_decode, emb_sizes)
        ])

    def encode(self, x_cat: torch.Tensor) -> torch.Tensor:
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
#  Training Supervised Embedding (kategorikal saja)
#  [TIDAK BERUBAH secara logika dari dataset_mrmd.py, hanya input berupa
#   cat_idx_array yang HANYA berisi kolom kategorikal]
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
    Latih SupervisedLearnableEmbeddingModel HANYA pada kolom kategorikal.

    mask_array : [N, n_cat_cols] bool, opsional — True = nilai HILANG.
        [FIX-LEAKAGE] Jika diberikan, posisi missing di-encode memakai TOKEN
        KHUSUS 'missing' dan DIKECUALIKAN dari reconstruction loss.
    """
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

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
                batch_cat_in = batch_cat.clone()
                miss_pos     = ~batch_observed
                batch_cat_in[miss_pos] = missing_idx.unsqueeze(0).expand_as(batch_cat)[miss_pos]
            else:
                batch_cat_in = batch_cat

            z, class_logits, recon_logits = model(batch_cat_in, add_noise=True)

            class_loss = ce_loss(class_logits, batch_labels)

            if batch_observed is not None:
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
#  Encode / Decode helpers (kategorikal saja)
#  [TIDAK BERUBAH dari dataset_mrmd.py]
# ===========================================================================

def encode_with_embedding(model: SupervisedLearnableEmbeddingModel,
                          cat_idx_array: np.ndarray,
                          device: str,
                          batch_size: int = 4096) -> np.ndarray:
    """Encode integer index kategorikal → embedding numpy array."""
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
    Decode embedding kategorikal → prediksi kelas tiap kolom (argmax logits).
    emb_array : [N, total_emb_dim] — HANYA segmen embedding kategorikal.
    Return    : [N, n_cat_cols]  — predicted integer index
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


# ===========================================================================
#  Load Dataset
# ===========================================================================

def load_dataset(dataname, idx=0, mask_type='MCAR', ratio='30', noise_std=0.01):
    """
    Load dataset TANPA diskritisasi MRmD untuk numerik. Numerik tetap
    kontinu (dinormalisasi). Fitur kategorikal tetap di-embed memakai
    SupervisedLearnableEmbeddingModel (arsitektur sama seperti
    dataset_mrmd.py, hanya menerima kolom kategorikal saja).

    [PENTING - TIDAK memakai split validasi sama sekali]
    Karena varian ini TIDAK melakukan diskritisasi MRmD (tidak butuh
    kriteria JS-divergence yang mensyaratkan validasi eksternal), seluruh
    proses (fitting LabelEncoder, training embedding model KATEGORIKAL,
    normalisasi numerik) LANGSUNG memakai dataset FULL:
        train_full : 'datasets/{dataname}/train.csv'
        test       : 'datasets/{dataname}/test.csv'
    beserta mask masing-masing di
        'datasets/{dataname}/masks/rate{ratio}/{mask_type}/train_mask_{idx}.npy'
        'datasets/{dataname}/masks/rate{ratio}/{mask_type}/test_mask_{idx}.npy'
    (folder 'validation' TIDAK dipakai pada varian ini).

    Struktur train_X / test_X:
        [ numerik_kontinu (n_num_cols dim) | embedding_kategorikal (sum(emb_sizes) dim) ]

    Return
    ------
    train_X            : [N_train_full, n_num_cols + total_cat_emb_dim]  float32
    test_X             : [N_test,       n_num_cols + total_cat_emb_dim]
    ori_train_mask      : mask asli train_full [N_train_full, total_cols]
    ori_test_mask       : mask asli test       [N_test,       total_cols]
    train_num           : [N_train_full, n_num_cols]  — float asli (ternormalisasi),
                           SAMA PERSIS dengan segmen numerik di train_X.
    test_num            : [N_test,       n_num_cols]
    train_all_idx       : [N_train_full, n_cat_cols]  — HANYA index kategorikal
    test_all_idx        : [N_test,       n_cat_cols]
    extend_train_mask   : [N_train_full, n_num_cols + total_cat_emb_dim]
    extend_test_mask    : [N_test,       n_num_cols + total_cat_emb_dim]
    cat_bin_num         : None  (legacy)
    emb_model           : SupervisedLearnableEmbeddingModel (kategorikal saja),
                           atau None jika tidak ada kolom kategorikal
    emb_sizes           : list[int] ukuran embedding tiap kolom kategorikal
    mrmd                : SELALU None (tidak ada diskritisasi pada varian ini)
    bin_midpoints       : SELALU None
    n_num_cols          : int
    t_mrmd              : SELALU 0.0 (tidak ada diskritisasi)
    t_emb               : float — waktu komputasi training embedding kategorikal
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

    # ── Dataset FULL (TANPA folder 'validation') ────────────────────────────
    train_path      = f'{data_dir}/train.csv'
    test_path       = f'{data_dir}/test.csv'
    train_mask_path = f'{data_dir}/masks/rate{ratio}/{mask_type}/train_mask_{idx}.npy'
    test_mask_path  = f'{data_dir}/masks/rate{ratio}/{mask_type}/test_mask_{idx}.npy'

    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)

    train_mask = np.load(train_mask_path)
    test_mask  = np.load(test_mask_path)

    cols = train_df.columns

    # ── Fitur numerik (nilai float asli) ─────────────────────────────────
    train_num_raw = train_df[cols[num_col_idx]].values.astype(np.float32)
    test_num_raw  = test_df[cols[num_col_idx]].values.astype(np.float32)

    # ── Labels untuk supervised learning embedding kategorikal ────────────
    train_y = train_df[cols[target_col_idx]]
    test_y  = test_df[cols[target_col_idx]]

    train_y_str = train_y.values.ravel().astype(str)
    test_y_str  = test_y.values.ravel().astype(str)

    label_encoder = LabelEncoder()
    label_encoder.fit(train_y_str)
    n_classes    = len(label_encoder.classes_)

    train_labels = label_encoder.transform(train_y_str)

    unseen_label_mask = ~np.isin(test_y_str, label_encoder.classes_)
    if unseen_label_mask.any():
        n_unseen = int(unseen_label_mask.sum())
        unseen_vals = np.unique(test_y_str[unseen_label_mask])
        print(f'[Dataset][WARNING] {n_unseen} label pada test tidak '
              f'dikenal saat fit LabelEncoder (train): {unseen_vals.tolist()}. '
              f'Label tsb dipetakan sementara ke kelas pertama (tidak '
              f'memengaruhi training classifier).')
        test_y_str_safe = test_y_str.copy()
        test_y_str_safe[unseen_label_mask] = label_encoder.classes_[0]
    else:
        test_y_str_safe = test_y_str

    test_labels  = label_encoder.transform(test_y_str_safe)

    print(f'[Dataset] Detected {n_classes} classes for supervised learning (fit: train only)')
    print(f'[Dataset] Classes: {label_encoder.classes_}')

    # ── Normalisasi numerik — TETAP KONTINU, TIDAK didiskritisasi ─────────
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

        train_num_norm = (train_num_raw - num_mean) / num_std
        test_num_norm  = (test_num_raw  - num_mean) / num_std

        train_num = train_num_norm.astype(np.float32)
        test_num  = test_num_norm.astype(np.float32)

        print(f'[Numerik] TIDAK didiskritisasi (MRmD dilewati). '
              f'{n_num_cols} kolom numerik tetap kontinu (dinormalisasi).')
    else:
        train_num = np.zeros((len(train_df), 0), dtype=np.float32)
        test_num  = np.zeros((len(test_df),  0), dtype=np.float32)
        num_mean = None
        num_std  = None

    mrmd          = None
    bin_midpoints = None
    t_mrmd        = 0.0

    # ── Encoding kolom kategorikal (LabelEncoder, fit HANYA pada train_full) ─
    cat_dims_cat       = []
    train_cat_idx_list = []
    test_cat_idx_list  = []

    if len(cat_col_idx) > 0:
        cat_columns = cols[cat_col_idx]
        train_cat   = train_df[cat_columns].astype(str)
        test_cat    = test_df[cat_columns].astype(str)

        UNKNOWN_TOKEN = '__unknown__'

        encoders = {}
        for col in cat_columns:
            le = LabelEncoder()
            le.fit(train_cat[col])

            train_vals = train_cat[col].values
            test_vals  = test_cat[col].values

            unseen_mask = ~np.isin(test_vals, le.classes_)

            if unseen_mask.any():
                n_unseen    = int(unseen_mask.sum())
                unseen_vals = np.unique(test_vals[unseen_mask])
                print(f"[Dataset][WARNING] Kolom '{col}': {n_unseen} nilai pada "
                      f"test tidak dikenal saat fit (train): "
                      f"{unseen_vals.tolist()}. Menambahkan 1 token khusus "
                      f"'{UNKNOWN_TOKEN}' ke vocabulary kolom ini.")
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
        encoders = {}

    # ── train_all_idx / test_all_idx = HANYA index kategorikal ─────────────
    train_all_idx = train_cat_idx
    test_all_idx  = test_cat_idx

    # ── Mask per grup kolom ──────────────────────────────────────────────
    train_num_mask = train_mask[:, num_col_idx].astype(bool) if n_num_cols > 0 else np.zeros((len(train_df), 0), dtype=bool)
    train_cat_mask = train_mask[:, cat_col_idx].astype(bool) if len(cat_col_idx) > 0 else np.zeros((len(train_df), 0), dtype=bool)
    test_num_mask  = test_mask[:, num_col_idx].astype(bool)  if n_num_cols > 0 else np.zeros((len(test_df),  0), dtype=bool)
    test_cat_mask  = test_mask[:, cat_col_idx].astype(bool)  if len(cat_col_idx) > 0 else np.zeros((len(test_df),  0), dtype=bool)

    if len(cat_col_idx) > 0:
        # ── Dimensi embedding kategorikal ────────────────────────────────
        emb_sizes = [compute_embedding_size(n) for n in cat_dims_cat]

        print(f'[Embedding] cat_dims={cat_dims_cat}')
        print(f'[Embedding] emb_sizes={emb_sizes}, total_cat_emb_dim={sum(emb_sizes)}')

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        print('[Embedding] Melatih SupervisedLearnableEmbeddingModel pada '
              'train_full (HANYA kolom kategorikal; numerik TIDAK ikut, tetap kontinu) ...')
        t_emb_start = time.time()
        emb_model = train_supervised_embedding_model(
            cat_idx_array = train_cat_idx,
            labels        = train_labels,
            cat_dims      = cat_dims_cat,
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
            mask_array    = train_cat_mask,   # [FIX-LEAKAGE] posisi missing dikecualikan dari recon loss
        )
        t_emb_end = time.time()
        t_emb = t_emb_end - t_emb_start
        print('[Embedding] Training selesai. Parameter di-freeze untuk diffusion.')
        print(f'[Embedding] Waktu komputasi embedding: {t_emb:.4f}s')

        train_cat_emb = encode_with_embedding(emb_model, train_cat_idx, device)
        test_cat_emb  = encode_with_embedding(emb_model, test_cat_idx,  device)

        emb_sizes_arr = np.array(emb_sizes, dtype=int)

        def extend_mask_emb(mask: np.ndarray, sizes: np.ndarray) -> np.ndarray:
            N      = mask.shape[0]
            cum    = np.concatenate(([0], sizes.cumsum()))
            result = np.zeros((N, sizes.sum()), dtype=bool)
            for j in range(len(sizes)):
                col_mask = mask[:, j][:, np.newaxis]
                result[:, cum[j]:cum[j + 1]] = np.tile(col_mask, sizes[j])
            return result

        extend_train_cat_mask = extend_mask_emb(train_cat_mask, emb_sizes_arr)
        extend_test_cat_mask  = extend_mask_emb(test_cat_mask,  emb_sizes_arr)
    else:
        emb_model     = None
        emb_sizes     = []
        t_emb         = 0.0
        train_cat_emb = np.zeros((len(train_df), 0), dtype=np.float32)
        test_cat_emb  = np.zeros((len(test_df),  0), dtype=np.float32)
        extend_train_cat_mask = np.zeros((len(train_df), 0), dtype=bool)
        extend_test_cat_mask  = np.zeros((len(test_df),  0), dtype=bool)

    # ── Gabungkan: [ numerik_kontinu | embedding_kategorikal ] ─────────────
    if n_num_cols > 0 and len(cat_col_idx) > 0:
        train_X = np.concatenate([train_num, train_cat_emb], axis=1)
        test_X  = np.concatenate([test_num,  test_cat_emb],  axis=1)
        extend_train_mask = np.concatenate([train_num_mask, extend_train_cat_mask], axis=1)
        extend_test_mask  = np.concatenate([test_num_mask,  extend_test_cat_mask],  axis=1)
    elif n_num_cols > 0:
        train_X = train_num
        test_X  = test_num
        extend_train_mask = train_num_mask
        extend_test_mask  = test_num_mask
    else:
        train_X = train_cat_emb
        test_X  = test_cat_emb
        extend_train_mask = extend_train_cat_mask
        extend_test_mask  = extend_test_cat_mask

    return (train_X, test_X,
            train_mask, test_mask,
            train_num, test_num,
            train_all_idx, test_all_idx,   # HANYA index kategorikal
            extend_train_mask, extend_test_mask,
            None,          # cat_bin_num (legacy)
            emb_model,     # SupervisedLearnableEmbeddingModel (kategorikal saja) atau None
            emb_sizes,     # ukuran embedding kategorikal saja
            mrmd,          # SELALU None (tidak ada diskritisasi)
            bin_midpoints, # SELALU None
            n_num_cols,    # jumlah kolom numerik (tetap dipakai utk offset)
            t_mrmd,        # SELALU 0.0
            t_emb,         # waktu training embedding kategorikal
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
             num_num, emb_model, emb_sizes, mask,
             device='cpu', oos=False,
             bin_midpoints=None, n_num_cols=0,
             num_true_norm=None):
    """
    Hitung MAE, RMSE (numerik) dan Accuracy (kategorikal).

    [VARIAN 2 - embedding tanpa diskrit]
    Numerik (MAE/RMSE):
        Numerik TIDAK melalui diskritisasi/embedding -- nilai kontinu
        prediksi diambil LANGSUNG dari n_num_cols kolom PERTAMA pada
        X_recon (tidak perlu decode bin/argmax apapun).
        Ground truth: num_true_norm (nilai float asli ternormalisasi).

    Kategorikal (Accuracy):
        decode_cat_from_embedding pada segmen embedding SETELAH n_num_cols
        kolom pertama, dibandingkan truth_all_idx (HANYA berisi index
        kategorikal pada varian ini).

    Parameter
    ---------
    bin_midpoints  : DIABAIKAN (legacy, selalu None pada varian ini)
    n_num_cols     : int — jumlah kolom numerik kontinu di awal X_recon
    num_num        : int — DIABAIKAN (legacy)
    truth_all_idx  : [N, n_cat_cols]  integer index kategorikal SAJA
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

    # ── Numerik: MAE & RMSE di skala normalisasi (LANGSUNG, tanpa decode) ──
    mae  = np.nan
    rmse = np.nan

    if n_num_cols > 0 and num_mask is not None:
        num_pred_norm = X_recon[:, :n_num_cols]

        if num_true_norm is not None:
            gt_norm = num_true_norm
        else:
            gt_norm = X_true[:, :n_num_cols]

        diff = num_pred_norm[num_mask] - gt_norm[num_mask]
        mae  = float(np.abs(diff).mean())
        rmse = float(np.sqrt((diff ** 2).mean()))

    # ── Kategorikal: Akurasi via Linear Decoder embedding ─────────────────
    acc = np.nan
    if (truth_all_idx is not None
            and len(cat_col_idx) > 0
            and emb_model is not None
            and emb_sizes is not None
            and cat_mask is not None):

        cat_emb_segment = X_recon[:, n_num_cols:]
        pred_cat_idx = decode_cat_from_embedding(
            emb_model, cat_emb_segment, device
        )  # [N, n_cat_cols]

        n_cat_cols    = len(cat_col_idx)
        correct_total = 0
        total_missing = 0

        for j in range(n_cat_cols):
            rows_miss = cat_mask[:, j]
            if rows_miss.sum() == 0:
                continue

            pred_j = pred_cat_idx[:, j]
            true_j = truth_all_idx[:, j].astype(int)

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


def save_imputed_csv_emb_nodiscrete(dataname, pred_X, mask, split_df_path, save_path,
                                    emb_model, emb_sizes, n_num_cols,
                                    num_mean, num_std, cat_encoders, device, oos=False):
    """
    Simpan hasil imputasi (pipeline embedding kategorikal TANPA diskritisasi
    numerik) ke file CSV dengan struktur kolom asli dataset.

    Aturan penyusunan nilai:
      - Posisi yang OBSERVED (mask == False) -> diambil dari nilai ASLI
        (train.csv / val.csv).
      - Posisi yang MISSING  (mask == True)  -> diambil dari hasil
        rekonstruksi:
          * numerik    -> segmen n_num_cols PERTAMA pada pred_X (kontinu),
                          langsung didenormalisasi ke skala asli.
          * kategorik  -> decode_cat_from_embedding pada segmen setelah
                          n_num_cols kolom, lalu inverse_transform via
                          LabelEncoder (cat_encoders).

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

    pred_X_full = np.array(pred_X, copy=True)

    result_df = orig_df.copy()

    if dataname == 'news' and oos is True:
        drop = 6265
        if drop < len(result_df):
            result_df = result_df.drop(index=drop).reset_index(drop=True)
        if num_mask is not None:
            num_mask = np.delete(num_mask, drop, axis=0)
        if cat_mask is not None:
            cat_mask = np.delete(cat_mask, drop, axis=0)
        pred_X_full = np.delete(pred_X_full, drop, axis=0)

    # ===== Kolom numerik: langsung ambil segmen kontinu, denormalisasi =====
    if len(num_col_idx) > 0:
        num_pred_norm = pred_X_full[:, :n_num_cols]
        num_mean_arr  = np.asarray(num_mean)
        num_std_arr   = np.asarray(num_std)
        num_pred_orig = (num_pred_norm * num_std_arr + num_mean_arr).astype(np.float32)

        num_cols = cols[num_col_idx]
        for i, col in enumerate(num_cols):
            col_values = result_df[col].values.astype(np.float32).copy()
            miss_rows = num_mask[:, i]
            col_values[miss_rows] = num_pred_orig[miss_rows, i]
            result_df[col] = col_values

    # ===== Kolom kategorik: decode via emb_model =====
    if len(cat_col_idx) > 0 and emb_model is not None:
        cat_cols = cols[cat_col_idx]
        cat_emb_segment = pred_X_full[:, n_num_cols:]

        pred_cat_idx = decode_cat_from_embedding(
            emb_model, cat_emb_segment, device
        )  # [N, n_cat_cols]

        for j, col in enumerate(cat_cols):
            miss_rows = cat_mask[:, j]
            if miss_rows.sum() == 0:
                continue

            pred_idx_col = pred_cat_idx[:, j]

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