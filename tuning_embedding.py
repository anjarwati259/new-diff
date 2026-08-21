import os
import json
import time
import argparse
import warnings
import itertools

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder

from dataset_mrmd import (
    load_dataset_mrmd_only,
    compute_embedding_size,
    SupervisedLearnableEmbeddingModel,
    decode_cat_from_embedding,
)

warnings.filterwarnings('ignore')

# ===========================================================================
#  tuning_embedding.py — Stage 1 & 2 Grid Search untuk Embedding Model
#
#  SCOPE DATA: TRAIN_NEW (56%) untuk fit, VALIDATION (14%) untuk evaluasi.
#  TEST (30%) TIDAK PERNAH DI-LOAD SAMA SEKALI di file ini — firewall fisik
#  supaya tidak ada risiko tersentuh, bukan cuma "tidak dipakai".
#
#  Alur:
#    STAGE 1 — Grid search hidden_dim x dropout (alpha:beta FIX = 1.0:0.25)
#              -> pilih 1 kombinasi terbaik (proxy accuracy tertinggi)
#    STAGE 2 — Grid search alpha x beta (hidden_dim/dropout FIX dari Stage 1)
#              -> ambil TOP 3 kombinasi terbaik
#
#  Metrik ranking: Accuracy rekonstruksi GABUNGAN (numerik-bin + kategorikal,
#  argmax vs bin/index asli) di posisi missing VALIDATION.
#
#  Early stopping saat tuning: val loss (alpha*class_loss + beta*recon_loss),
#  dihitung di VALIDATION dengan model dalam mode eval (tidak ada gradient
#  update dari VALIDATION sama sekali).
# ===========================================================================


# ---------------------------------------------------------------------------
#  Encoding kolom kategorikal — fit HANYA dari TRAIN_NEW (scope tahap tuning)
# ---------------------------------------------------------------------------

def encode_categorical_train_new_val(train_new_df, val_df, cols, cat_col_idx):
    """
    Encode kolom kategorikal. Encoder di-fit HANYA dari TRAIN_NEW (56%),
    VALIDATION ditransform pakai encoder itu (kategori tak dikenal -> bucket
    'unknown', pola yang sama dengan test set di pipeline lama).

    Return: train_new_cat_idx, val_cat_idx [N, n_cat_cols] int64, cat_dims (list, +1 utk unknown)
    """
    if len(cat_col_idx) == 0:
        n_tr, n_vl = len(train_new_df), len(val_df)
        return (np.zeros((n_tr, 0), dtype=np.int64),
                np.zeros((n_vl, 0), dtype=np.int64), [])

    cat_columns = cols[cat_col_idx]
    train_new_cat = train_new_df[cat_columns].astype(str)
    val_cat       = val_df[cat_columns].astype(str)

    cat_dims = []
    train_new_idx_list, val_idx_list = [], []

    for col in cat_columns:
        le = LabelEncoder()
        le.fit(train_new_cat[col])
        known_classes = set(le.classes_)
        unknown_idx   = len(le.classes_)
        cat_dims.append(len(le.classes_) + 1)  # +1 slot 'unknown'

        train_new_idx_list.append(le.transform(train_new_cat[col]).astype(np.int64))

        val_vals = val_cat[col].values
        val_idx_col = np.empty(len(val_vals), dtype=np.int64)
        n_unknown = 0
        for i, v in enumerate(val_vals):
            if v in known_classes:
                val_idx_col[i] = le.transform([v])[0]
            else:
                val_idx_col[i] = unknown_idx
                n_unknown += 1
        if n_unknown > 0:
            print(f'[Tuning] PERINGATAN: kolom "{col}" — {n_unknown} nilai '
                  f'di VALIDATION tidak muncul di TRAIN_NEW, dipetakan ke '
                  f'bucket "unknown".')
        val_idx_list.append(val_idx_col)

    train_new_cat_idx = np.stack(train_new_idx_list, axis=1)
    val_cat_idx       = np.stack(val_idx_list, axis=1)
    return train_new_cat_idx, val_cat_idx, cat_dims


# ---------------------------------------------------------------------------
#  Training embedding dengan early stopping berdasar VAL LOSS
#  [BEDA dari train_supervised_embedding_model di dataset_mrmd.py, yang
#  early stopping-nya berdasar TRAIN loss — dipakai khusus untuk FINAL RUN]
# ---------------------------------------------------------------------------

def train_embedding_val_early_stopping(
        train_idx_array, train_labels, train_mask_array,
        val_idx_array, val_labels, val_mask_array,
        cat_dims, emb_sizes, n_classes, device,
        alpha=1.0, beta=0.25,
        hidden_dim=256, dropout=0.1, noise_std=0.01,
        use_mlp=True, mlp_ratio=1.5,
        n_epochs=1000, batch_size=1024, lr=1e-3, patience=40,
        eval_every=5, verbose=False):
    """
    Latih SupervisedLearnableEmbeddingModel di TRAIN_NEW, early stopping
    berdasarkan VAL LOSS (alpha*class_loss + beta*recon_loss) yang dihitung
    di VALIDATION setiap `eval_every` epoch, model dalam mode eval (no_grad
    — VALIDATION TIDAK PERNAH ikut backward pass).

    Return: model (state terbaik ter-load), best_val_loss, history (list of dict)
    """
    torch.manual_seed(42)
    np.random.seed(42)

    cat_dims_embed = [d + 1 for d in cat_dims]   # +1 token 'missing'
    missing_idx = torch.tensor(cat_dims, dtype=torch.long, device=device)

    model = SupervisedLearnableEmbeddingModel(
        cat_dims_embed, emb_sizes, n_classes,
        dropout=dropout, hidden_dim=hidden_dim,
        use_mlp=use_mlp, mlp_ratio=mlp_ratio, noise_std=noise_std,
        cat_dims_decode=cat_dims,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    ce_loss_noreduce = nn.CrossEntropyLoss(reduction='none')

    train_cat_t = torch.tensor(train_idx_array, dtype=torch.long, device=device)
    train_lbl_t = torch.tensor(train_labels, dtype=torch.long, device=device)
    train_obs_t = torch.tensor(~np.array(train_mask_array, dtype=bool),
                               dtype=torch.bool, device=device)

    val_cat_t = torch.tensor(val_idx_array, dtype=torch.long, device=device)
    val_lbl_t = torch.tensor(val_labels, dtype=torch.long, device=device)
    val_obs_t = torch.tensor(~np.array(val_mask_array, dtype=bool),
                             dtype=torch.bool, device=device)

    dataset = torch.utils.data.TensorDataset(train_cat_t, train_lbl_t, train_obs_t)
    loader  = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=False,
        generator=torch.Generator(device='cpu'),
    )

    def _forward_loss(batch_cat, batch_labels, batch_observed, train_mode):
        batch_cat_in = batch_cat.clone()
        miss_pos = ~batch_observed
        batch_cat_in[miss_pos] = missing_idx.unsqueeze(0).expand_as(batch_cat)[miss_pos]

        z, class_logits, recon_logits = model(batch_cat_in, add_noise=train_mode)
        class_loss = nn.functional.cross_entropy(class_logits, batch_labels)

        col_losses = []
        for i in range(model.n_cols):
            obs_i = batch_observed[:, i]
            if obs_i.any():
                per_elem = ce_loss_noreduce(recon_logits[i], batch_cat[:, i])
                col_losses.append(per_elem[obs_i].mean())
        recon_loss = (sum(col_losses) / len(col_losses)) if col_losses else torch.tensor(0.0, device=device)

        loss = alpha * class_loss + beta * recon_loss
        return loss, class_loss, recon_loss

    best_val_loss    = float('inf')
    patience_counter = 0
    best_model_state = None
    history = []

    for epoch in range(n_epochs):
        model.train()
        for batch_cat, batch_labels, batch_observed in loader:
            optimizer.zero_grad()
            loss, _, _ = _forward_loss(batch_cat, batch_labels, batch_observed, train_mode=True)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % eval_every == 0 or epoch == n_epochs - 1:
            model.eval()
            with torch.no_grad():
                val_loss, val_class_loss, val_recon_loss = _forward_loss(
                    val_cat_t, val_lbl_t, val_obs_t, train_mode=False
                )
            val_loss_val = val_loss.item()
            history.append({
                'epoch': epoch + 1, 'val_loss': val_loss_val,
                'val_class_loss': val_class_loss.item(),
                'val_recon_loss': val_recon_loss.item(),
            })

            if verbose:
                print(f'  [Epoch {epoch+1}] val_loss={val_loss_val:.4f} '
                      f'(class={val_class_loss.item():.4f}, recon={val_recon_loss.item():.4f})')

            if val_loss_val < best_val_loss:
                best_val_loss    = val_loss_val
                patience_counter = 0
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1

            if patience_counter >= patience:
                if verbose:
                    print(f'  [Early stop] epoch {epoch+1}, best_val_loss={best_val_loss:.4f}')
                break

    if best_model_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    return model, best_val_loss, history


# ---------------------------------------------------------------------------
#  Proxy metric: Accuracy rekonstruksi GABUNGAN (numerik-bin + kategorikal)
#  di posisi missing VALIDATION — TANPA diffusion.
# ---------------------------------------------------------------------------

def proxy_reconstruction_accuracy(model, idx_array, mask_array, device):
    """
    Encode VALIDATION (posisi missing -> token 'missing'), decode balik,
    lalu hitung accuracy SEMUA kolom (numerik-bin + kategorikal) di posisi
    yang di-mask missing, dibandingkan ke bin/index index aslinya.

    Return: accuracy (float, 0-1)
    """
    model.eval()
    mask_bool = np.array(mask_array, dtype=bool)
    n_cols    = idx_array.shape[1]

    cat_dims  = model.cat_dims  # sudah termasuk +1 token missing di embedding, tapi decode ke cat_dims_decode asli
    missing_idx_np = np.array([d for d in [c - 1 for c in cat_dims]])  # cat_dims di sini = cat_dims_embed = asli+1

    idx_in = idx_array.copy()
    idx_in[mask_bool] = np.take(missing_idx_np, np.where(mask_bool)[1])

    idx_tensor = torch.tensor(idx_in, dtype=torch.long, device=device)
    with torch.no_grad():
        z, _, _ = model(idx_tensor, add_noise=False)

    emb_np = z.cpu().numpy()
    pred_idx = decode_cat_from_embedding(model, emb_np, device)  # [N, n_cols]

    correct_total = 0
    total_missing = 0
    for j in range(n_cols):
        rows_miss = mask_bool[:, j]
        if rows_miss.sum() == 0:
            continue
        correct = (pred_idx[rows_miss, j] == idx_array[rows_miss, j]).sum()
        correct_total += int(correct)
        total_missing += int(rows_miss.sum())

    return (correct_total / total_missing) if total_missing > 0 else float('nan')


# ---------------------------------------------------------------------------
#  Main — Stage 1 & Stage 2 Grid Search
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Tuning Embedding (Stage 1 & 2, MRmD + Grid Search)')
    parser.add_argument('--dataname',  type=str, default='shoppers')
    parser.add_argument('--split_idx', type=int, default=0)
    parser.add_argument('--ratio',     type=str, default='30')
    parser.add_argument('--mask',      type=str, default='MCAR')
    parser.add_argument('--n_epochs',  type=int, default=300)
    parser.add_argument('--patience',  type=int, default=10)
    args = parser.parse_args()

    dataname, split_idx, ratio, mask_type = (
        args.dataname, args.split_idx, args.ratio, args.mask
    )
    if mask_type == 'MNAR':
        mask_type = 'MNAR_logistic_T2'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'[Tuning] Device: {device}')

    # =====================================================================
    #  Load MRmD result (TRAIN_NEW & VALIDATION num_bin) — TEST TIDAK di-load
    # =====================================================================
    mrmd_result = load_dataset_mrmd_only(dataname, split_idx, mask_type, ratio)

    cols          = mrmd_result['cols']
    cat_col_idx   = mrmd_result['cat_col_idx']
    target_col_idx = mrmd_result['target_col_idx']
    train_new_df  = mrmd_result['train_new_df']
    val_df        = mrmd_result['val_df']
    train_new_num_bin = mrmd_result['train_new_num_bin']
    val_num_bin       = mrmd_result['val_num_bin']
    train_new_mask    = mrmd_result['train_new_mask']
    val_mask          = mrmd_result['val_mask']

    # =====================================================================
    #  Encode kategorikal — fit HANYA dari TRAIN_NEW
    # =====================================================================
    train_new_cat_idx, val_cat_idx, cat_dims_cat = encode_categorical_train_new_val(
        train_new_df, val_df, cols, cat_col_idx
    )

    # =====================================================================
    #  Target label encoder — fit HANYA dari TRAIN_NEW (scope tahap tuning,
    #  TERPISAH dari label encoder yang dipakai MRmD/TRAIN_FULL)
    # =====================================================================
    target_col = cols[target_col_idx]
    train_new_y = train_new_df[target_col].values.ravel().astype(str)
    val_y       = val_df[target_col].values.ravel().astype(str)

    tgt_encoder = LabelEncoder()
    tgt_encoder.fit(train_new_y)
    n_classes = len(tgt_encoder.classes_)
    known_tgt = set(tgt_encoder.classes_)

    train_new_labels = tgt_encoder.transform(train_new_y)
    val_labels = np.array([
        tgt_encoder.transform([v])[0] if v in known_tgt else n_classes
        for v in val_y
    ])
    n_unknown_val_labels = int((val_labels == n_classes).sum())
    if n_unknown_val_labels > 0:
        print(f'[Tuning] PERINGATAN: {n_unknown_val_labels} label target di '
              f'VALIDATION tidak muncul di TRAIN_NEW.')
        n_classes_eff = n_classes + 1
    else:
        n_classes_eff = n_classes

    # =====================================================================
    #  Gabungkan numerik-bin + kategorikal -> all_idx, all_dims, all_mask
    # =====================================================================
    train_new_all_idx = np.concatenate([train_new_num_bin, train_new_cat_idx], axis=1)
    val_all_idx       = np.concatenate([val_num_bin, val_cat_idx], axis=1)

    n_num_cols = mrmd_result['n_num_cols']
    mrmd = mrmd_result['mrmd']
    all_dims = (mrmd.n_bins_ if mrmd is not None else []) + cat_dims_cat
    emb_sizes = [compute_embedding_size(n) for n in all_dims]

    print(f'[Tuning] all_dims={all_dims}')
    print(f'[Tuning] emb_sizes={emb_sizes}, total_emb_dim={sum(emb_sizes)}')

    num_col_idx = mrmd_result['num_col_idx']
    train_new_num_mask_only = train_new_mask[:, num_col_idx].astype(bool) if n_num_cols > 0 else np.zeros((len(train_new_df), 0), dtype=bool)
    val_num_mask_only       = val_mask[:, num_col_idx].astype(bool)       if n_num_cols > 0 else np.zeros((len(val_df), 0), dtype=bool)
    train_new_cat_mask = train_new_mask[:, cat_col_idx].astype(bool) if len(cat_col_idx) > 0 else np.zeros((len(train_new_df), 0), dtype=bool)
    val_cat_mask       = val_mask[:, cat_col_idx].astype(bool)       if len(cat_col_idx) > 0 else np.zeros((len(val_df), 0), dtype=bool)

    train_new_all_mask = np.concatenate([train_new_num_mask_only, train_new_cat_mask], axis=1)
    val_all_mask       = np.concatenate([val_num_mask_only, val_cat_mask], axis=1)

    # =====================================================================
    #  STAGE 1 — Grid search hidden_dim x dropout x batch_size x lr x mlp_ratio
    #  (alpha:beta FIX 1.0:0.25). Total = 3x3x3x3x3 = 243 kombinasi.
    # =====================================================================
    HIDDEN_DIMS = [128, 256, 512]
    DROPOUTS    = [0.1, 0.2, 0.3]
    BATCH_SIZES = [512, 1024, 2048]
    LRS         = [1e-4, 1e-3, 1e-2]
    MLP_RATIOS  = [1.5, 2, 4]
    ALPHA_FIX, BETA_FIX = 1.0, 0.25

    stage1_grid = list(itertools.product(
        HIDDEN_DIMS, DROPOUTS, BATCH_SIZES, LRS, MLP_RATIOS
    ))
    print(f'\n{"="*60}\n[STAGE 1] Grid Search hidden_dim x dropout x batch_size '
          f'x lr x mlp_ratio\n(alpha:beta FIX = {ALPHA_FIX}:{BETA_FIX})\n'
          f'Total kombinasi: {len(stage1_grid)}\n{"="*60}')

    stage1_results = []
    for i, (hidden_dim, dropout, batch_size, lr, mlp_ratio) in enumerate(stage1_grid, 1):
        print(f'\n[Stage 1 {i}/{len(stage1_grid)}] hidden_dim={hidden_dim}, '
              f'dropout={dropout}, batch_size={batch_size}, lr={lr}, '
              f'mlp_ratio={mlp_ratio} ...')
        t0 = time.time()
        model, best_val_loss, _ = train_embedding_val_early_stopping(
            train_new_all_idx, train_new_labels, train_new_all_mask,
            val_all_idx, val_labels, val_all_mask,
            all_dims, emb_sizes, n_classes_eff, device,
            alpha=ALPHA_FIX, beta=BETA_FIX,
            hidden_dim=hidden_dim, dropout=dropout,
            batch_size=batch_size, lr=lr, mlp_ratio=mlp_ratio,
            n_epochs=args.n_epochs, patience=args.patience,
        )
        acc = proxy_reconstruction_accuracy(model, val_all_idx, val_all_mask, device)
        elapsed = time.time() - t0

        print(f'[Stage 1 {i}/{len(stage1_grid)}] -> '
              f'val_loss={best_val_loss:.4f}, proxy_acc={acc:.4f} ({elapsed:.1f}s)')

        stage1_results.append({
            'hidden_dim': hidden_dim, 'dropout': dropout,
            'batch_size': batch_size, 'lr': lr, 'mlp_ratio': mlp_ratio,
            'val_loss': best_val_loss, 'proxy_accuracy': acc,
            'elapsed_sec': elapsed,
        })

    stage1_results.sort(key=lambda r: r['proxy_accuracy'], reverse=True)
    best_stage1 = stage1_results[0]
    print(f'\n[STAGE 1] Pemenang: hidden_dim={best_stage1["hidden_dim"]}, '
          f'dropout={best_stage1["dropout"]}, batch_size={best_stage1["batch_size"]}, '
          f'lr={best_stage1["lr"]}, mlp_ratio={best_stage1["mlp_ratio"]} '
          f'(proxy_acc={best_stage1["proxy_accuracy"]:.4f})')

    # =====================================================================
    #  STAGE 2 — Grid search alpha x beta
    #  (hidden_dim, dropout, batch_size, lr, mlp_ratio FIX dari Stage 1)
    # =====================================================================
    ALPHA_BETA_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0]

    print(f'\n{"="*60}\n[STAGE 2] Grid Search alpha x beta\n'
          f'(hidden_dim={best_stage1["hidden_dim"]}, dropout={best_stage1["dropout"]}, '
          f'batch_size={best_stage1["batch_size"]}, lr={best_stage1["lr"]}, '
          f'mlp_ratio={best_stage1["mlp_ratio"]} FIX)\n{"="*60}')

    stage2_results = []
    for alpha, beta in itertools.product(ALPHA_BETA_VALUES, ALPHA_BETA_VALUES):
        if alpha == 0.0 and beta == 0.0:
            print(f'[Stage 2] alpha=0, beta=0 -> SKIP (degenerate, loss selalu 0)')
            continue

        print(f'\n[Stage 2] alpha={alpha}, beta={beta} ...')
        t0 = time.time()
        model, best_val_loss, _ = train_embedding_val_early_stopping(
            train_new_all_idx, train_new_labels, train_new_all_mask,
            val_all_idx, val_labels, val_all_mask,
            all_dims, emb_sizes, n_classes_eff, device,
            alpha=alpha, beta=beta,
            hidden_dim=best_stage1['hidden_dim'], dropout=best_stage1['dropout'],
            batch_size=best_stage1['batch_size'], lr=best_stage1['lr'],
            mlp_ratio=best_stage1['mlp_ratio'],
            n_epochs=args.n_epochs, patience=args.patience,
        )
        acc = proxy_reconstruction_accuracy(model, val_all_idx, val_all_mask, device)
        elapsed = time.time() - t0

        print(f'[Stage 2] alpha={alpha}, beta={beta} -> '
              f'val_loss={best_val_loss:.4f}, proxy_acc={acc:.4f} ({elapsed:.1f}s)')

        stage2_results.append({
            'alpha': alpha, 'beta': beta,
            'val_loss': best_val_loss, 'proxy_accuracy': acc,
            'elapsed_sec': elapsed,
        })

    stage2_results.sort(key=lambda r: r['proxy_accuracy'], reverse=True)
    top3_stage2 = stage2_results[:3]

    print(f'\n[STAGE 2] TOP 3 kombinasi alpha:beta:')
    for i, r in enumerate(top3_stage2, 1):
        print(f'  #{i}: alpha={r["alpha"]}, beta={r["beta"]}, '
              f'proxy_acc={r["proxy_accuracy"]:.4f}')

    # =====================================================================
    #  Simpan hasil
    # =====================================================================
    result_dir = f'tuning_results/{dataname}/rate{ratio}/{mask_type}/{split_idx}'
    os.makedirs(result_dir, exist_ok=True)

    with open(f'{result_dir}/stage1_results.json', 'w') as f:
        json.dump(stage1_results, f, indent=2)
    with open(f'{result_dir}/stage2_results.json', 'w') as f:
        json.dump(stage2_results, f, indent=2)

    winning_config = {
        'hidden_dim': best_stage1['hidden_dim'],
        'dropout':    best_stage1['dropout'],
        'batch_size': best_stage1['batch_size'],
        'lr':         best_stage1['lr'],
        'mlp_ratio':  best_stage1['mlp_ratio'],
        'noise_std':  0.01,
        'use_mlp':    True,
        'top3_alpha_beta': [{'alpha': r['alpha'], 'beta': r['beta'],
                             'proxy_accuracy': r['proxy_accuracy']} for r in top3_stage2],
    }
    with open(f'{result_dir}/winning_config.json', 'w') as f:
        json.dump(winning_config, f, indent=2)

    print(f'\n[Tuning] Hasil disimpan ke {result_dir}/')
    print(f'  - stage1_results.json   (semua {len(stage1_grid)} kombinasi '
          f'hidden_dim x dropout x batch_size x lr x mlp_ratio)')
    print(f'  - stage2_results.json   (semua kombinasi alpha x beta)')
    print(f'  - winning_config.json   (config pemenang + top-3 alpha:beta)')
    print(f'\n[NEXT] Lanjut ke tuning_verify_diffusion.py dengan top-3 kandidat '
          f'alpha:beta di atas.')


if __name__ == '__main__':
    main()