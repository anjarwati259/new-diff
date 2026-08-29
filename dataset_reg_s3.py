import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import os
import json

DATA_DIR = 'datasets'

# ===========================================================================
#  SKENARIO 3 [FIX-LEAKAGE]: min-max murni + MSELoss (tanpa MRmD)
#                            + penanganan missing value saat training embedding
#
#  Normalisasi label: min-max langsung ke [0, 1], TANPA sigmoid tambahan.
#  (Sama persis dengan dataset_reg_s3.py sebelumnya)
#
#  [BARU - dibanding dataset_reg_s3.py] Penanganan missing value:
#  Sebelumnya, seluruh baris train_cat_idx (termasuk posisi yang nantinya
#  di-mask sebagai "missing" untuk keperluan evaluasi imputasi) ikut
#  dipakai APA ADANYA untuk melatih embedding model (classification +
#  reconstruction loss). Ini artinya nilai ASLI di posisi yang seharusnya
#  "tidak diketahui" tetap dipelajari oleh model -> data leakage.
#
#  Perbaikan (mengikuti pola dataset_mrmd_test.py):
#    1) Tambahkan 1 token khusus 'missing' per kolom kategorikal pada
#       tabel embedding (num_embeddings = n_kategori + 1).
#    2) Sebelum encode(), posisi yang di-mask missing diganti dengan
#       token khusus ini (BUKAN nilai aslinya).
#    3) Reconstruction loss HANYA dihitung pada posisi yang OBSERVED
#       (bukan missing) per kolom.
#    4) Regression loss (MSE terhadap label target) TETAP dihitung
#       penuh atas semua baris -- karena label target adalah kolom
#       terpisah dari num_col_idx/cat_col_idx dan tidak ikut di-mask
#       oleh skenario missing-value fitur.
#    5) Setelah training selesai & model di-freeze, encoding FINAL untuk
#       pipeline diffusion (train_X, test_X) tetap memakai index ASLI
#       (bukan token missing) -- karena posisi yang missing pada
#       pipeline DiffPuter nantinya akan digantikan/diestimasi ulang
#       lewat proses E-step, bukan dipakai langsung sebagai nilai fixed.
#
#  Arsitektur: nn.Embedding (cat only, +1 token missing per kolom) + MLP
#             + LayerNorm → Regressor (Sigmoid output) + MSELoss
#             + Reconstruction CE loss (masked, hanya posisi observed)
# ===========================================================================

def compute_embedding_size(n_categories: int) -> int:
    """
    Hitung ukuran embedding optimal berdasarkan jumlah kategori.
    Rumus: min(600, round(1.6 * n_categories^0.56))
    Referensi: Guo & Berkhahn (2016)
    [TIDAK BERUBAH]
    """
    return min(600, round(1.6 * n_categories ** 0.56))


def minmax_normalize_labels(labels_raw: np.ndarray,
                             label_min: float = None,
                             label_max: float = None):
    """
    Normalisasi label ke range [0, 1] menggunakan min-max murni.

    [TIDAK BERUBAH dari dataset_reg_s3.py]
    Alur:
        label_raw → min-max → [0, 1]   (TANPA sigmoid)

    Parameter
    ---------
    labels_raw  : np.ndarray [N]  — nilai label float asli
    label_min   : float — min dari train labels (untuk transform test)
    label_max   : float — max dari train labels (untuk transform test)

    Return
    ------
    labels_norm : np.ndarray [N]  float32  — label ternormalisasi [0, 1]
    label_min   : float
    label_max   : float
    """
    labels_raw = labels_raw.astype(np.float32)

    if label_min is None:
        label_min = float(labels_raw.min())
    if label_max is None:
        label_max = float(labels_raw.max())

    label_range = label_max - label_min + 1e-8

    # Min-max → [0, 1]  (tanpa sigmoid)
    labels_norm = ((labels_raw - label_min) / label_range).astype(np.float32)

    return labels_norm, label_min, label_max


def inverse_minmax_labels(labels_norm: np.ndarray,
                           label_min: float,
                           label_max: float) -> np.ndarray:
    """
    Inverse normalisasi label dari [0, 1] kembali ke skala asli.
    [TIDAK BERUBAH dari dataset_reg_s3.py]

    Parameter
    ---------
    labels_norm : np.ndarray [N]  float32  — output sigmoid regressor [0, 1]
    label_min   : float
    label_max   : float

    Return
    ------
    labels_asli : np.ndarray [N]  float32  — nilai label di skala asli
    """
    label_range = label_max - label_min + 1e-8

    # Inverse min-max (linear, tanpa logit)
    labels_asli = (labels_norm * label_range + label_min).astype(np.float32)

    return labels_asli


class SupervisedLearnableEmbeddingModel(nn.Module):
    """
    Model Supervised Learnable Embedding dengan Regression Head.

    [BERUBAH dari dataset_reg_s3.py]
    Tambah parameter `cat_dims_decode` untuk mendukung token 'missing':
      - cat_dims        : jumlah baris tabel embedding (nn.Embedding) per
                          kolom. Bisa berisi +1 token khusus 'missing' per
                          kolom (lihat train_supervised_embedding_model),
                          supaya posisi yang di-mask missing bisa di-encode
                          dengan token netral, bukan nilai aslinya.
      - cat_dims_decode : jumlah kelas ASLI (TANPA token 'missing') untuk
                          output decoder rekonstruksi. Jika None, sama
                          dengan cat_dims (perilaku lama / tidak ada token
                          missing) -- default ini menjaga backward
                          compatibility dengan dataset_reg_s3.py.

    Alur (tidak berubah kecuali dimensi tabel embedding & decoder):
      cat_idx [batch, n_cat_cols]
        → nn.Embedding per kolom → concat → [batch, total_emb_dim]
        → (opsional) Linear → SiLU → Linear   (use_mlp=True)
        → LayerNorm
        → z [batch, total_emb_dim]
        → Regressor → Sigmoid → [batch]
        → (+ noise σ=noise_std saat training)
        → Linear Decoder per kolom → logits rekonstruksi (dimensi = kelas ASLI)

    Parameter
    ---------
    cat_dims        : list[int]   jumlah baris embedding per kolom (bisa +1 token missing)
    emb_sizes       : list[int]   dimensi embedding per kolom
    dropout         : float       dropout pada regressor head
    hidden_dim      : int         dimensi hidden layer untuk regressor
    use_mlp         : bool        aktifkan 1 hidden layer setelah concat
    mlp_ratio       : float       hidden_dim_mlp = int(total_emb_dim * mlp_ratio)
    noise_std       : float       std Gaussian noise sebelum decoding saat training
    cat_dims_decode : list[int] atau None — jumlah kelas asli per kolom untuk decoder
    """

    def __init__(self, cat_dims: list, emb_sizes: list,
                 dropout: float = 0.1, hidden_dim: int = 256,
                 use_mlp: bool = True, mlp_ratio: float = 1.5,
                 noise_std: float = 0.1,
                 cat_dims_decode: list = None):
        super().__init__()

        # [BERUBAH] Default cat_dims_decode = cat_dims (backward compatible)
        cat_dims_decode = cat_dims_decode if cat_dims_decode is not None else cat_dims

        # Satu nn.Embedding per kolom kategorikal
        # [BERUBAH] num_embeddings bisa n_kategori+1 jika ada token missing
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=n_cat, embedding_dim=emb_dim)
            for n_cat, emb_dim in zip(cat_dims, emb_sizes)
        ])

        self.total_emb_dim = sum(emb_sizes)
        self.n_cols        = len(cat_dims)
        self.cat_dims      = cat_dims
        self.emb_sizes     = emb_sizes
        self.noise_std     = noise_std
        self.use_mlp       = use_mlp

        # Optional MLP setelah concat [TIDAK BERUBAH]
        if use_mlp:
            hidden_dim_mlp = max(self.total_emb_dim, int(self.total_emb_dim * mlp_ratio))
            self.mlp = nn.Sequential(
                nn.Linear(self.total_emb_dim, hidden_dim_mlp),
                nn.SiLU(),
                nn.Linear(hidden_dim_mlp, self.total_emb_dim),
            )
        else:
            self.mlp = None

        # LayerNorm [TIDAK BERUBAH]
        self.layer_norm = nn.LayerNorm(self.total_emb_dim)
        self.out_dim    = self.total_emb_dim

        # ── Regression Head [TIDAK BERUBAH dari dataset_reg_s3.py] ─────────
        # Output: 1 nilai kontinu + Sigmoid → range (0, 1)
        # Cocok dengan label yang dinormalisasi via min-max murni
        # Loss: MSELoss (bukan CrossEntropyLoss)
        self.dropout   = nn.Dropout(dropout)
        self.regressor = nn.Sequential(
            nn.Linear(self.total_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()          # output → (0, 1), konsisten dengan label min-max
        )

        # Linear Decoder per kolom
        # [BERUBAH] output dimension pakai cat_dims_decode (kelas ASLI,
        # tanpa token missing), bukan cat_dims (yang mungkin +1)
        self.decoders = nn.ModuleList([
            nn.Linear(emb_size, n_cat)
            for n_cat, emb_size in zip(cat_dims_decode, emb_sizes)
        ])

    def encode(self, x_cat: torch.Tensor) -> torch.Tensor:
        """
        Encode integer index kategorikal → vektor embedding dense + LayerNorm.
        [TIDAK BERUBAH]
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

    def regress(self, z: torch.Tensor) -> torch.Tensor:
        """
        Regression head: embedding → prediksi label (0, 1).
        [TIDAK BERUBAH]

        z      : [batch, total_emb_dim]
        return : [batch]  — nilai float (0, 1) setelah sigmoid
        """
        return self.regressor(z).squeeze(1)   # [batch, 1] → [batch]

    def decode(self, z: torch.Tensor) -> list:
        """
        Linear Decoder: embedding → logit tiap kolom kategorikal.
        [TIDAK BERUBAH]
        """
        per_col = torch.split(z, self.emb_sizes, dim=1)
        return [self.decoders[i](per_col[i]) for i in range(self.n_cols)]

    def forward(self, x_cat: torch.Tensor, add_noise: bool = False):
        """
        Full forward: encode → regress + (noise) decode.
        [TIDAK BERUBAH]

        return : (z, reg_output, recon_logits)
            z            [batch, total_emb_dim]
            reg_output   [batch]               — prediksi label (0, 1)
            recon_logits list[n_cat_cols] of [batch, vocab_i]
        """
        z          = self.encode(x_cat)
        reg_output = self.regress(z)           # [batch] float (0, 1)

        # Noise sebelum decoding saat training [TIDAK BERUBAH]
        if add_noise and self.training and self.noise_std > 0:
            z_noisy = z + torch.randn_like(z) * self.noise_std
        else:
            z_noisy = z

        recon_logits = self.decode(z_noisy)
        return z, reg_output, recon_logits


def train_supervised_embedding_model(cat_idx_array: np.ndarray,
                                     labels: np.ndarray,
                                     cat_dims: list,
                                     emb_sizes: list,
                                     device: str,
                                     n_epochs: int = 1000,
                                     batch_size: int = 1024,
                                     lr: float = 1e-3,
                                     dropout: float = 0.1,
                                     hidden_dim: int = 256,
                                     use_mlp: bool = True,
                                     mlp_ratio: float = 1.5,
                                     noise_std: float = 0.01,
                                     patience: int = 40,
                                     alpha: float = 1.0,
                                     beta: float = 0.25,
                                     mask_array: np.ndarray = None,
                                     ) -> SupervisedLearnableEmbeddingModel:
    """
    Latih SupervisedLearnableEmbeddingModel dengan Regression Head.

    [BERUBAH dari dataset_reg_s3.py]
    Tambah parameter `mask_array` untuk penanganan missing value saat
    training embedding, mengikuti pola dataset_mrmd_test.py:

      - mask_array : [N, n_cat_cols] bool, opsional — True = nilai HILANG
        (missing). Jika diberikan:
          1) Setiap kolom kategorikal ditambah 1 token khusus 'missing'
             pada tabel embedding (num_embeddings = n_kategori + 1).
          2) Posisi yang di-mask missing di-encode memakai token ini
             (bukan nilai aslinya) SEBELUM masuk ke encode().
          3) Reconstruction loss (CE) HANYA dihitung pada posisi OBSERVED
             per kolom -- posisi missing dikecualikan supaya model tidak
             "menghafal" nilai asli yang seharusnya diimputasi.
          4) Regression loss (MSE terhadap label target) TETAP dihitung
             penuh atas semua baris pada batch -- karena target regresi
             adalah kolom terpisah yang tidak ikut di-mask oleh skenario
             missing-value fitur ini.
      - Jika mask_array=None (default), perilaku identik dengan
        dataset_reg_s3.py -- seluruh baris/posisi dipakai apa adanya.

    Loss total:
      loss = alpha * MSE(reg_output, label_minmax)
           + beta  * CE(recon_logits, cat_idx)   [masked jika mask_array diberikan]

    Parameter
    ---------
    labels     : np.ndarray [N] float32 — label yang sudah dinormalisasi min-max [0,1]
    alpha      : float — bobot regression loss (MSE). Default 1.0 (sesuai paper).
    beta       : float — bobot reconstruction loss (CE). Default 0.25 (sesuai paper).
    mask_array : np.ndarray [N, n_cat_cols] bool, opsional — True = missing.
    """
    # Fix random seed agar hasil embedding reproducible setiap run
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # ── [BARU - FIX-LEAKAGE] Siapkan token 'missing' jika mask_array ada ───
    # Tabel embedding diperbesar +1 baris per kolom untuk token 'missing'.
    # Decoder tetap memprediksi ke ruang kelas ASLI (cat_dims, tanpa +1)
    # lewat parameter cat_dims_decode.
    use_missing_token = mask_array is not None
    if use_missing_token:
        cat_dims_embed = [d + 1 for d in cat_dims]
        missing_idx = torch.tensor(cat_dims, dtype=torch.long, device=device)
    else:
        cat_dims_embed = cat_dims
        missing_idx = None

    model = SupervisedLearnableEmbeddingModel(
        cat_dims_embed,
        emb_sizes,
        dropout         = dropout,
        hidden_dim      = hidden_dim,
        use_mlp         = use_mlp,
        mlp_ratio       = mlp_ratio,
        noise_std       = noise_std,
        cat_dims_decode = cat_dims,   # decoder tetap output kelas ASLI
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Loss functions
    mse_loss         = nn.MSELoss()               # untuk regression (supervised signal)
    ce_loss          = nn.CrossEntropyLoss()       # reconstruction (tanpa mask)
    ce_loss_noreduce = nn.CrossEntropyLoss(reduction='none')  # reconstruction (masked)

    # Label dtype: float32
    cat_tensor   = torch.tensor(cat_idx_array, dtype=torch.long,    device=device)
    label_tensor = torch.tensor(labels,        dtype=torch.float32, device=device)

    if use_missing_token:
        # observed_tensor: True = observed (bukan missing)
        observed_tensor = torch.tensor(~np.array(mask_array, dtype=bool),
                                       dtype=torch.bool, device=device)
        dataset = torch.utils.data.TensorDataset(cat_tensor, label_tensor, observed_tensor)
    else:
        dataset = torch.utils.data.TensorDataset(cat_tensor, label_tensor)

    cpu_gen  = torch.Generator(device='cpu')
    loader   = torch.utils.data.DataLoader(
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

    print(f'[Embedding] alpha={alpha} (regression weight), '
          f'beta={beta} (reconstruction weight), '
          f'missing_token={"ON" if use_missing_token else "OFF"}')

    model.train()
    for epoch in range(n_epochs):
        total_loss      = 0.0
        total_reg_loss  = 0.0
        total_recon_loss = 0.0
        n_batches       = 0

        for batch in loader:
            if use_missing_token:
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

            z, reg_output, recon_logits = model(batch_cat_in, add_noise=True)

            # Regression loss (MSE) — TIDAK terpengaruh mask fitur, karena
            # label target adalah kolom terpisah yang selalu observed.
            reg_loss = mse_loss(reg_output, batch_labels)

            if batch_observed is not None:
                # [FIX-LEAKAGE] recon_loss HANYA dihitung pada posisi OBSERVED,
                # dibandingkan dengan nilai ASLI (batch_cat, bukan batch_cat_in)
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

            # Combined loss
            loss = alpha * reg_loss + beta * recon_loss

            loss.backward()
            optimizer.step()

            total_loss       += loss.item()
            total_reg_loss   += reg_loss.item()
            total_recon_loss += (recon_loss.item() if torch.is_tensor(recon_loss) else recon_loss)
            n_batches        += 1

        avg_loss       = total_loss       / n_batches
        avg_reg_loss   = total_reg_loss   / n_batches
        avg_recon_loss = total_recon_loss / n_batches

        if (epoch + 1) % 10 == 0:
            print(f'[Embedding] Epoch {epoch+1}/{n_epochs} - '
                  f'Loss: {avg_loss:.4f} (Reg: {avg_reg_loss:.4f}, '
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

    # Monitor distribusi embedding [TIDAK BERUBAH]
    with torch.no_grad():
        sample_cat = cat_tensor[:min(2048, len(cat_tensor))]
        z_sample   = model.encode(sample_cat)
        print(f'[Embedding] Distribusi embedding (N={z_sample.shape[0]}):')
        print(f'  mean={z_sample.mean().item():.4f}  '
              f'std={z_sample.std().item():.4f}  '
              f'norm_mean={z_sample.norm(dim=1).mean().item():.4f}')

        # Monitor prediksi label regression (skala min-max)
        reg_sample = model.regress(z_sample)
        print(f'[Embedding] Prediksi label regression (skala min-max):')
        print(f'  min={reg_sample.min().item():.4f}  '
              f'max={reg_sample.max().item():.4f}  '
              f'mean={reg_sample.mean().item():.4f}')

    # Freeze seluruh parameter embedding [TIDAK BERUBAH]
    for param in model.parameters():
        param.requires_grad_(False)
    print('[Embedding] Seluruh parameter embedding di-freeze untuk training diffusion.')

    return model


def encode_with_embedding(model: SupervisedLearnableEmbeddingModel,
                          cat_idx_array: np.ndarray,
                          device: str,
                          batch_size: int = 4096) -> np.ndarray:
    """
    Encode seluruh data kategorikal → embedding numpy array.
    [TIDAK BERUBAH]

    [CATATAN] Dipanggil dengan index ASLI (bukan token missing) baik untuk
    train maupun test -- karena pada tahap ini kita membentuk representasi
    penuh (train_X/test_X) yang akan diberikan ke pipeline DiffPuter.
    Posisi yang missing pada pipeline tersebut akan digantikan/diestimasi
    ulang lewat proses E-step berdasarkan mask (extend_train_mask /
    extend_test_mask), bukan dipakai langsung sebagai nilai fixed di sini.
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
    Decode embedding → prediksi kelas kategorikal (argmax logits).
    [TIDAK BERUBAH]
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

def load_dataset(dataname, idx=0, mask_type='MCAR', ratio='30'):
    """
    Load dataset dengan regression embedding learning.

    [BERUBAH dari dataset_reg_s3.py]
    - Sekarang menghitung `train_cat_mask` (dari train_mask, kolom
      kategorikal) dan meneruskannya sebagai `mask_array` ke
      train_supervised_embedding_model -- supaya training embedding TIDAK
      "menghafal" nilai asli di posisi yang di-mask sebagai missing
      (lihat penjelasan FIX-LEAKAGE di atas).
    - Numerik TETAP tidak diubah (tidak di-embed, langsung dipakai raw
      concat seperti dataset_reg_s3.py) -- konsisten dengan paper yang
      tidak mendiskritisasi/meng-embed numerik pada dataset regresi.
      Penanganan missing untuk numerik tetap ditangani terpisah oleh
      pipeline DiffPuter (lewat extend_train_mask/extend_test_mask),
      bukan di tahap embedding ini.

    Return sama persis dengan dataset_reg_s3.py untuk kompatibilitas
    main_class.py:
      train_X, test_X, ori_train_mask, ori_test_mask,
      train_num, test_num, train_cat_idx, test_cat_idx,
      extend_train_mask, extend_test_mask,
      cat_bin_num (None), emb_model, emb_sizes
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

    # Fitur numerik [TIDAK BERUBAH]
    data_num  = data_df[cols[num_col_idx]].values.astype(np.float32)
    train_num = train_df[cols[num_col_idx]].values.astype(np.float32)
    test_num  = test_df[cols[num_col_idx]].values.astype(np.float32)

    # Normalisasi label pakai min-max murni (tanpa sigmoid) [TIDAK BERUBAH]
    train_y = train_df[cols[target_col_idx]]
    test_y  = test_df[cols[target_col_idx]]

    train_labels_raw = train_y.values.ravel().astype(np.float32)
    test_labels_raw  = test_y.values.ravel().astype(np.float32)

    label_min   = float(train_labels_raw.min())
    label_max   = float(train_labels_raw.max())
    label_range = label_max - label_min + 1e-8

    train_labels = ((train_labels_raw - label_min) / label_range).astype(np.float32)
    test_labels  = ((test_labels_raw  - label_min) / label_range).astype(np.float32)

    print(f'[Dataset][Skenario 3 + FIX-LEAKAGE] min-max murni (tanpa sigmoid):')
    print(f'  label_min={label_min:.2f}, label_max={label_max:.2f}')
    print(f'  Label range train : [{train_labels.min():.4f}, {train_labels.max():.4f}]')
    print(f'  Label range test  : [{test_labels.min():.4f}, {test_labels.max():.4f}]')
    print(f'  Range efektif     : {train_labels.max()-train_labels.min():.4f} (ideal: 1.0)')

    # Kasus: hanya fitur numerik [TIDAK BERUBAH]
    if len(cat_col_idx) == 0:
        train_X = train_num
        test_X  = test_num

        extend_train_mask = train_mask[:, num_col_idx]
        extend_test_mask  = test_mask[:, num_col_idx]

        return (train_X, test_X,
                train_mask, test_mask,
                train_num, test_num,
                None, None,
                extend_train_mask, extend_test_mask,
                None, None, None)

    # Kasus: ada fitur kategorikal
    cat_columns = cols[cat_col_idx]

    data_cat  = data_df[cat_columns].astype(str)
    train_cat = train_df[cat_columns].astype(str)
    test_cat  = test_df[cat_columns].astype(str)

    encoders           = {}
    cat_dims           = []
    train_cat_idx_list = []
    test_cat_idx_list  = []

    for col in cat_columns:
        le = LabelEncoder()
        le.fit(data_cat[col])
        encoders[col] = le
        cat_dims.append(len(le.classes_))

        train_cat_idx_list.append(
            le.transform(train_cat[col]).astype(np.int64)
        )
        test_cat_idx_list.append(
            le.transform(test_cat[col]).astype(np.int64)
        )

    train_cat_idx = np.stack(train_cat_idx_list, axis=1)
    test_cat_idx  = np.stack(test_cat_idx_list,  axis=1)

    emb_sizes = [compute_embedding_size(n) for n in cat_dims]

    print(f'[Embedding] cat_dims={cat_dims}, emb_sizes={emb_sizes}, '
          f'total_emb_dim={sum(emb_sizes)}')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ── [BARU - FIX-LEAKAGE] Mask kategorikal untuk TRAIN ───────────────────
    # True = missing (nilai ini akan diganti token khusus saat training
    # embedding, dan dikecualikan dari reconstruction loss).
    train_cat_mask = train_mask[:, cat_col_idx].astype(bool)

    print('[Embedding][Skenario 3 + FIX-LEAKAGE] Melatih '
          'SupervisedLearnableEmbeddingModel (regression MSE + '
          'reconstruction CE loss, missing value dikecualikan) ...')
    emb_model = train_supervised_embedding_model(
        cat_idx_array = train_cat_idx,
        labels        = train_labels,   # float32 min-max murni [0, 1]
        cat_dims      = cat_dims,
        emb_sizes     = emb_sizes,
        device        = device,
        n_epochs      = 1000,
        batch_size    = 1024,
        lr            = 1e-3,
        dropout       = 0.1,
        hidden_dim    = 256,
        use_mlp       = True,
        mlp_ratio     = 1.5,
        noise_std     = 0.01,
        patience      = 40,
        alpha         = 1.0,    # sesuai nilai optimal hasil tuning paper
        beta          = 0.25,   # sesuai nilai optimal hasil tuning paper
        mask_array    = train_cat_mask,   # [BARU] posisi missing dikecualikan
    )
    print('[Embedding] Training selesai. Parameter di-freeze untuk diffusion.')

    # Encode FINAL menggunakan index ASLI (bukan token missing) — lihat
    # catatan di docstring encode_with_embedding(). [TIDAK BERUBAH]
    train_cat_emb = encode_with_embedding(emb_model, train_cat_idx, device)
    test_cat_emb  = encode_with_embedding(emb_model, test_cat_idx,  device)

    # Gabungkan numerik + embedding [TIDAK BERUBAH]
    train_X = np.concatenate([train_num, train_cat_emb], axis=1)
    test_X  = np.concatenate([test_num,  test_cat_emb],  axis=1)

    # Extended mask [TIDAK BERUBAH]
    train_num_mask = train_mask[:, num_col_idx]
    train_cat_mask_orig = train_mask[:, cat_col_idx]
    test_num_mask  = test_mask[:, num_col_idx]
    test_cat_mask  = test_mask[:, cat_col_idx]

    emb_sizes_arr = np.array(emb_sizes, dtype=int)

    def extend_mask_emb(mask: np.ndarray, sizes: np.ndarray) -> np.ndarray:
        N      = mask.shape[0]
        cum    = np.concatenate(([0], sizes.cumsum()))
        result = np.zeros((N, sizes.sum()), dtype=bool)
        for j in range(len(sizes)):
            col_mask = mask[:, j][:, np.newaxis]
            result[:, cum[j]:cum[j + 1]] = np.tile(col_mask, sizes[j])
        return result

    ext_train_cat_mask = extend_mask_emb(train_cat_mask_orig, emb_sizes_arr)
    ext_test_cat_mask  = extend_mask_emb(test_cat_mask,       emb_sizes_arr)

    extend_train_mask = np.concatenate([train_num_mask, ext_train_cat_mask], axis=1)
    extend_test_mask  = np.concatenate([test_num_mask,  ext_test_cat_mask],  axis=1)

    return (train_X, test_X,
            train_mask, test_mask,
            train_num, test_num,
            train_cat_idx, test_cat_idx,
            extend_train_mask, extend_test_mask,
            None,       # cat_bin_num (legacy)
            emb_model,
            emb_sizes)


# ===========================================================================
#  Utilities
# ===========================================================================

def mean_std(data, mask):
    """[TIDAK BERUBAH]"""
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

def get_eval(dataname, X_recon, X_true, truth_cat_idx,
             num_num, emb_model, emb_sizes, mask,
             device='cpu', oos=False):
    """
    Hitung MAE, RMSE (numerik) dan Accuracy (kategorikal).
    [TIDAK BERUBAH] — evaluasi imputasi tidak menyentuh label regression sama sekali.

    Label regression (target_col_idx) tidak dievaluasi di sini.
    Jika ingin evaluasi prediksi label, gunakan inverse_minmax_labels()
    secara terpisah di luar fungsi ini.
    """
    info_path = f'datasets/Info/{dataname}.json'
    with open(info_path, 'r') as f:
        info = json.load(f)

    num_col_idx = info['num_col_idx']
    cat_col_idx = info['cat_col_idx']

    num_mask = mask[:, num_col_idx].astype(bool)
    cat_mask = mask[:, cat_col_idx].astype(bool) if len(cat_col_idx) > 0 else None

    num_pred = X_recon[:, :num_num]
    num_true = X_true[:, :num_num]

    cat_emb_pred = X_recon[:, num_num:]

    # Special case: news dataset [TIDAK BERUBAH]
    if dataname == 'news' and oos:
        drop = 6265
        num_mask     = np.delete(num_mask,     drop, axis=0)
        num_pred     = np.delete(num_pred,     drop, axis=0)
        num_true     = np.delete(num_true,     drop, axis=0)
        if cat_mask is not None:
            cat_mask = np.delete(cat_mask,     drop, axis=0)
        if truth_cat_idx is not None:
            truth_cat_idx = np.delete(truth_cat_idx, drop, axis=0)
        cat_emb_pred = np.delete(cat_emb_pred, drop, axis=0)

    # Numerik: MAE & RMSE [TIDAK BERUBAH]
    div  = num_pred[num_mask] - num_true[num_mask]
    mae  = np.abs(div).mean()
    rmse = np.sqrt((div ** 2).mean())

    # Kategorikal: Akurasi via Linear Decoder [TIDAK BERUBAH]
    acc = np.nan
    if (truth_cat_idx is not None
            and len(cat_col_idx) > 0
            and emb_model is not None
            and emb_sizes is not None):

        pred_cat_idx = decode_cat_from_embedding(emb_model, cat_emb_pred, device)

        correct_total = 0
        total_missing = 0

        for j in range(len(cat_col_idx)):
            rows_miss = cat_mask[:, j]
            if rows_miss.sum() == 0:
                continue

            pred_j = pred_cat_idx[:, j]
            true_j = truth_cat_idx[:, j].astype(int)

            correct = (pred_j[rows_miss] == true_j[rows_miss]).sum()
            correct_total += int(correct)
            total_missing += int(rows_miss.sum())

        if total_missing > 0:
            acc = correct_total / total_missing

    return mae, rmse, acc