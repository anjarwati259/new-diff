import os
import torch

import numpy as np
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
import warnings
import time
from tqdm import tqdm

from model import MLPDiffusion, Model
from dataset import load_dataset, get_eval, mean_std, load_meta, save_imputed_csv
from diffusion_utils import sample_step, impute_mask

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='Missing Value Imputation')

parser.add_argument('--dataname', type=str, default='california', help='Name of dataset.')
parser.add_argument('--gpu', type=int, default=0, help='GPU index.')
parser.add_argument('--split_idx', type=int, default=0, help='Split idx.')
parser.add_argument('--max_iter', type=int, default=5, help='Maximum iteration.')
parser.add_argument('--ratio', type=str, default=30, help='Masking ratio.')
parser.add_argument('--hid_dim', type=int, default=1024, help='Hidden dimension.')
parser.add_argument('--mask', type=str, default='MCAR', help='Masking machenisms.')
parser.add_argument('--num_trials', type=int, default=2, help='Number of sampling times.')
parser.add_argument('--num_steps', type=int, default=20, help='Number of diffusion steps.')
# [BARU] Opsi untuk menyimpan hasil imputasi (train/val/test) ke CSV.
# True  -> hasil imputasi tiap iterasi disimpan sebagai file .csv
# False -> hasil imputasi TIDAK disimpan (default, tidak ada perubahan perilaku lama)
parser.add_argument('--save_imputation', type=lambda x: str(x).lower() in ('true', '1', 'yes'),
                     default=True, help='Simpan hasil imputasi train/val/test ke CSV (True/False).')

args = parser.parse_args()

# Force GPU usage - akan error jika GPU tidak tersedia
if not torch.cuda.is_available():
    raise RuntimeError("GPU tidak tersedia! Script ini membutuhkan GPU untuk berjalan.")

args.device = f'cuda:{args.gpu}'
torch.cuda.set_device(args.gpu)

# Set default tensor type ke CUDA
torch.set_default_device(args.device)


if __name__ == '__main__':

    dataname = args.dataname
    split_idx = args.split_idx
    device = args.device
    hid_dim = args.hid_dim
    mask_type = args.mask
    ratio = args.ratio
    num_trials = args.num_trials
    num_steps = args.num_steps

    if mask_type == 'MNAR':
        mask_type = 'MNAR_logistic_T2'

    # =========================================================================
    # [MODIFIKASI] load_dataset sekarang mengembalikan 16 nilai (3-way split:
    # train/val/test), bukan 11 nilai (train/test) seperti sebelumnya.
    # =========================================================================
    (train_X, val_X, test_X,
     ori_train_mask, ori_val_mask, ori_test_mask,
     train_num, val_num, test_num,
     train_cat_idx, val_cat_idx, test_cat_idx,
     train_mask, val_mask, test_mask,   # ini extend_*_mask (bit-level), sama seperti pola nama asli
     cat_bin_num) = load_dataset(dataname, split_idx, mask_type, ratio)

    # [BARU] Metadata (nama kolom asli + target asli) hanya dibutuhkan kalau
    # kita mau menyimpan hasil imputasi ke CSV.
    meta = load_meta(dataname) if args.save_imputation else None

    # mean & std HANYA dihitung dari train (observed entries) -> tidak ada
    # kebocoran informasi dari val atau test ke proses normalisasi.
    mean_X, std_X = mean_std(train_X, train_mask)
    in_dim = train_X.shape[1]

    # Langsung convert ke GPU tensor
    X     = torch.tensor((train_X - mean_X) / std_X / 2, device=device, dtype=torch.float32)
    X_val = torch.tensor((val_X   - mean_X) / std_X / 2, device=device, dtype=torch.float32)   # [BARU]
    X_test = torch.tensor((test_X - mean_X) / std_X / 2, device=device, dtype=torch.float32)

    mask_train = torch.tensor(train_mask, device=device, dtype=torch.float32)
    mask_val   = torch.tensor(val_mask,   device=device, dtype=torch.float32)   # [BARU]
    mask_test  = torch.tensor(test_mask,  device=device, dtype=torch.float32)

    # Convert mean dan std ke GPU tensor untuk operasi selanjutnya
    mean_X_gpu = torch.tensor(mean_X, device=device, dtype=torch.float32)
    std_X_gpu = torch.tensor(std_X, device=device, dtype=torch.float32)

    MAEs = []
    RMSEs = []
    ACCs = []

    # [BARU] Nampung hasil imputasi (pred_X sudah didenormalisasi) tiap
    # iterasi di memori dulu. TIDAK langsung ditulis ke CSV per-iterasi --
    # nanti di akhir cuma iterasi TERBAIK yang ditulis ke folder 'best/'.
    imputed_per_iter = {}   # { iteration: {'train': pred_X, 'val': pred_X, 'test': pred_X} }

    MAEs_val = []     # [BARU]
    RMSEs_val = []    # [BARU]
    ACCs_val = []     # [BARU]

    MAEs_out = []
    RMSEs_out = []
    ACCs_out = []

    batch_size = 4096

    # Custom Dataset untuk GPU tensor
    # [MODIFIKASI] dipindah ke luar loop iterasi (dulunya didefinisikan ulang
    # tiap iterasi) karena definisinya statis, tidak tergantung iterasi.
    class GPUTensorDataset(torch.utils.data.Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    # =========================================================================
    # [BARU] val_loader dibangun SEKALI di luar loop iterasi EM, karena isi
    # val (X_val, mask_val) tidak berubah antar iterasi -- beda dengan
    # train_data yang di-update tiap iterasi pakai hasil rekonstruksi
    # sebelumnya (lihat blok "if iteration == 0 ... else ..." di bawah).
    # Dipakai untuk menghitung val_loss tiap epoch -> dasar checkpoint
    # selection & early stopping (menggantikan train_loss).
    # =========================================================================
    val_data_fixed = (1. - mask_val) * X_val
    val_loader = DataLoader(
        GPUTensorDataset(val_data_fixed),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    start_time = time.time()
    for iteration in range(args.max_iter):

        ## M-Step: Density Estimation
     
        ckpt_dir = f'ckpt/{dataname}/rate{ratio}/{mask_type}/{split_idx}/{num_trials}_{num_steps}'
        os.makedirs(f'{ckpt_dir}/{iteration}', exist_ok=True)

        # [MODIFIKASI] result_save_path dipindah ke awal loop (sebelumnya
        # dibuat di akhir loop) supaya sudah tersedia saat kita mau
        # menyimpan hasil imputasi (train/val/test) ke CSV di tengah proses,
        # bukan hanya dipakai untuk result.txt di akhir iterasi.
        result_save_path = f'results/{dataname}/rate{ratio}/{mask_type}/{split_idx}/{num_trials}_{num_steps}'
        os.makedirs(result_save_path, exist_ok=True)

        print(f'iteration: {iteration}')
        print(ckpt_dir)

        if iteration == 0:
            X_miss = (1. - mask_train) * X
            train_data = X_miss
        else:
            print(f'Loading X_miss from {ckpt_dir}/iter_{iteration}.npy')
            # Load langsung ke GPU
            X_miss = torch.tensor(np.load(f'{ckpt_dir}/iter_{iteration}.npy') / 2, device=device, dtype=torch.float32)
            train_data = X_miss

        print(f'[INFO] Loaded X_miss shape: {train_data.shape}, range: [{train_data.min():.4f}, {train_data.max():.4f}]')

        # Buat generator untuk GPU
        generator = torch.Generator(device=device)

        train_loader = DataLoader(
            GPUTensorDataset(train_data),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # Set 0 karena data sudah di GPU
            pin_memory=False,  # Tidak perlu pin_memory karena sudah di GPU
            generator=generator  # Gunakan GPU generator
        )

        num_epochs = 100 + 1

        denoise_fn = MLPDiffusion(in_dim, hid_dim).to(device)

        if iteration == 0:
            print(denoise_fn)

        model = Model(denoise_fn=denoise_fn, hid_dim=in_dim).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=50)

        model.train()

        # [MODIFIKASI] checkpoint & early stopping sekarang berbasis val_loss,
        # bukan train_loss (best_loss) seperti sebelumnya -- supaya model
        # yang disimpan adalah model yang paling general (tidak overfit ke
        # data training), bukan cuma yang paling hafal data training.
        best_val_loss = float('inf')
        patience = 0

        # progress bar
        pbar = tqdm(range(num_epochs), desc='Training')
        for epoch in pbar:

            batch_loss = 0.0
            len_input = 0
 
            for batch in train_loader:
                inputs = batch.float()  # Sudah di GPU, tidak perlu .to(device)
                loss = model(inputs)

                loss = loss.mean()
                batch_loss += loss.item() * len(inputs)
                len_input += len(inputs)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            curr_loss = batch_loss / len_input   # train loss, dipakai buat logging saja sekarang

            # =================================================================
            # [BARU] Hitung val_loss -- TIDAK backprop, cuma forward pass buat
            # memantau seberapa baik model generalisasi ke data yang tidak
            # ikut update parameter.
            # =================================================================
            model.eval()
            with torch.no_grad():
                val_batch_loss = 0.0
                val_len_input = 0
                for vbatch in val_loader:
                    vinputs = vbatch.float()
                    vloss = model(vinputs)
                    vloss = vloss.mean()
                    val_batch_loss += vloss.item() * len(vinputs)
                    val_len_input += len(vinputs)
            model.train()

            curr_val_loss = val_batch_loss / val_len_input

            # [MODIFIKASI] scheduler & checkpoint sekarang pakai val_loss
            scheduler.step(curr_val_loss)

            if curr_val_loss < best_val_loss:
                best_val_loss = curr_val_loss
                patience = 0
                torch.save(model.state_dict(), f'{ckpt_dir}/{iteration}/model.pt')
            else:
                patience += 1
                if patience == 500:
                    print('Early stopping')
                    break
            
            pbar.set_postfix(train_loss=curr_loss, val_loss=curr_val_loss)

            if epoch % 1000 == 0:
                torch.save(model.state_dict(), f'{ckpt_dir}/{iteration}/model_{epoch}.pt')

        end_time = time.time()
        
        print(f'Iteration {iteration} training time: {end_time - start_time:.2f} seconds')
        print(f'Best val_loss iterasi {iteration}: {best_val_loss:.6f}')

        ## E-Step: Missing Value Imputation

        # In-sample imputation
        
        impute_start_time = time.time()

        rec_Xs = []

        for trial in tqdm(range(num_trials), desc='In-sample imputation'):
        
            X_miss = (1. - mask_train) * X
            impute_X = X_miss  # Sudah di GPU
  
            in_dim = X.shape[1]

            denoise_fn = MLPDiffusion(in_dim, hid_dim).to(device)

            model = Model(denoise_fn=denoise_fn, hid_dim=in_dim).to(device)
            model.load_state_dict(torch.load(f'{ckpt_dir}/{iteration}/model.pt'))

            # ==========================================================

            net = model.denoise_fn_D

            num_samples, dim = X.shape[0], X.shape[1]
            rec_X = impute_mask(net, impute_X, mask_train, num_samples, dim, num_steps, device)
            
            mask_int = mask_train.float()  # Sudah di GPU
            rec_X = rec_X * mask_int + impute_X * (1 - mask_int)
            rec_Xs.append(rec_X)
            
            

        rec_X = torch.stack(rec_Xs, dim=0).mean(0) 

        # Simpan hasil (hanya saat save ke disk yang perlu CPU)
        rec_X_save = (rec_X * 2).cpu().numpy()
        X_true_save = (X * 2).cpu().numpy()

        np.save(f'{ckpt_dir}/iter_{iteration+1}.npy', rec_X_save)

        # Lakukan komputasi di GPU
        pred_X_gpu = rec_X * 2
        X_true_gpu = X * 2

        # Denormalisasi di GPU
        len_num = train_num.shape[1]
        pred_X_gpu[:, len_num:] = pred_X_gpu[:, len_num:] * std_X_gpu[len_num:] + mean_X_gpu[len_num:]

        # Convert ke CPU hanya untuk evaluasi
        pred_X = pred_X_gpu.cpu().numpy()
        X_true = X_true_gpu.cpu().numpy()

        mae, rmse, acc = get_eval(dataname, pred_X, X_true, train_cat_idx, train_num.shape[1], cat_bin_num, ori_train_mask)
        MAEs.append(mae)
        RMSEs.append(rmse)
        ACCs.append(acc)

        # [BARU] Simpan pred_X train ke memori dulu (belum ditulis ke CSV)
        if args.save_imputation:
            imputed_per_iter.setdefault(iteration, {})['train'] = pred_X

        impute_end_time = time.time()
        print(f'In-sample imputation time: {impute_end_time - impute_start_time:.2f} seconds')

        print('in-sample', mae, rmse, acc)

        # =====================================================================
        # [BARU] Validation imputation -- diagnostic tambahan. TIDAK dipakai
        # untuk pemilihan model (itu sudah dilakukan lewat val_loss di atas),
        # cuma dilaporkan berdampingan dengan in-sample & out-of-sample supaya
        # kelihatan posisi val di antara keduanya (seharusnya val lebih dekat
        # ke out-of-sample daripada ke in-sample, karena val tidak ikut
        # backprop -- kalau val ternyata sama bagusnya dengan in-sample,
        # itu tanda model masih overfit).
        # =====================================================================
        val_impute_start_time = time.time()

        rec_Xs = []

        for trial in tqdm(range(num_trials), desc='Validation imputation'):

            X_miss = (1. - mask_val) * X_val
            impute_X = X_miss

            in_dim = X_val.shape[1]

            denoise_fn = MLPDiffusion(in_dim, hid_dim).to(device)

            model = Model(denoise_fn=denoise_fn, hid_dim=in_dim).to(device)
            model.load_state_dict(torch.load(f'{ckpt_dir}/{iteration}/model.pt'))

            net = model.denoise_fn_D

            num_samples, dim = X_val.shape[0], X_val.shape[1]
            rec_X = impute_mask(net, impute_X, mask_val, num_samples, dim, num_steps, device)

            mask_int = mask_val.float()
            rec_X = rec_X * mask_int + impute_X * (1 - mask_int)
            rec_Xs.append(rec_X)

        rec_X = torch.stack(rec_Xs, dim=0).mean(0)

        pred_X_gpu = rec_X * 2
        X_true_gpu = X_val * 2

        len_num = val_num.shape[1]
        pred_X_gpu[:, len_num:] = pred_X_gpu[:, len_num:] * std_X_gpu[len_num:] + mean_X_gpu[len_num:]

        pred_X = pred_X_gpu.cpu().numpy()
        X_true = X_true_gpu.cpu().numpy()

        mae_val, rmse_val, acc_val = get_eval(dataname, pred_X, X_true, val_cat_idx, val_num.shape[1], cat_bin_num, ori_val_mask, oos=False)
        MAEs_val.append(mae_val)
        RMSEs_val.append(rmse_val)
        ACCs_val.append(acc_val)

        # [BARU] Simpan pred_X val ke memori dulu (belum ditulis ke CSV)
        if args.save_imputation:
            imputed_per_iter.setdefault(iteration, {})['val'] = pred_X

        val_impute_end_time = time.time()
        print(f'Validation imputation time: {val_impute_end_time - val_impute_start_time:.2f} seconds')

        print('validation', mae_val, rmse_val, acc_val)

        # out-of-sample imputation
        
        oos_impute_start_time = time.time()

        rec_Xs = []

        for trial in tqdm(range(num_trials), desc='Out-of-sample imputation'):
            
            # For out-of-sample imputation, no results from previous iterations are used

            X_miss = (1. - mask_test) * X_test
            impute_X = X_miss  # Sudah di GPU

            in_dim = X_test.shape[1]

            denoise_fn = MLPDiffusion(in_dim, hid_dim).to(device)

            model = Model(denoise_fn=denoise_fn, hid_dim=in_dim).to(device)
            model.load_state_dict(torch.load(f'{ckpt_dir}/{iteration}/model.pt'))

            # ==========================================================
            net = model.denoise_fn_D

            num_samples, dim = X_test.shape[0], X_test.shape[1]
            rec_X = impute_mask(net, impute_X, mask_test, num_samples, dim, num_steps, device)
            
            mask_int = mask_test.float()  # Sudah di GPU
            rec_X = rec_X * mask_int + impute_X * (1 - mask_int)
            rec_Xs.append(rec_X)
            
    
        rec_X = torch.stack(rec_Xs, dim=0).mean(0) 

        # Lakukan komputasi di GPU
        pred_X_gpu = rec_X * 2
        X_true_gpu = X_test * 2

        # Denormalisasi di GPU
        len_num = train_num.shape[1]
        pred_X_gpu[:, len_num:] = pred_X_gpu[:, len_num:] * std_X_gpu[len_num:] + mean_X_gpu[len_num:]

        # Convert ke CPU hanya untuk evaluasi
        pred_X = pred_X_gpu.cpu().numpy()
        X_true = X_true_gpu.cpu().numpy()

        mae_out, rmse_out, acc_out = get_eval(dataname, pred_X, X_true, test_cat_idx, test_num.shape[1], cat_bin_num, ori_test_mask, oos=True)
        MAEs_out.append(mae_out)
        RMSEs_out.append(rmse_out)
        ACCs_out.append(acc_out)

        # [BARU] Simpan pred_X test ke memori dulu (belum ditulis ke CSV)
        if args.save_imputation:
            imputed_per_iter.setdefault(iteration, {})['test'] = pred_X

        oos_impute_end_time = time.time()
        print(f'Out-of-sample imputation time: {oos_impute_end_time - oos_impute_start_time:.2f} seconds')

        with open(f'{result_save_path}/result.txt', 'a+') as f:
            f.write(f'iteration {iteration}, MAE: in-sample: {mae}, validation: {mae_val}, out-of-sample: {mae_out} \n')
            f.write(f'iteration {iteration}: RMSE: in-sample: {rmse}, validation: {rmse_val}, out-of-sample: {rmse_out} \n')
            f.write(f'iteration {iteration}: ACC: in-sample: {acc}, validation: {acc_val}, out-of-sample: {acc_out} \n')
            f.write(f'iteration {iteration}: best_val_loss (checkpoint selection): {best_val_loss:.6f} \n')
            f.write(f'iteration {iteration}: Training time: {end_time - start_time:.2f}s, In-sample imputation time: {impute_end_time - impute_start_time:.2f}s, Validation imputation time: {val_impute_end_time - val_impute_start_time:.2f}s, Out-of-sample imputation time: {oos_impute_end_time - oos_impute_start_time:.2f}s \n\n')

        print('out-of-sample', mae_out, rmse_out, acc_out)

        print(f'saving results to {result_save_path}')
        
        # Reset start_time untuk iterasi berikutnya
        start_time = time.time()

    # =========================================================================
    # [BARU] Setelah SEMUA iterasi (max_iter) selesai: pilih 1 iterasi TERBAIK
    # berdasarkan metrik VALIDATION (ACCs_val, MAEs_val, RMSEs_val), lalu
    # simpan cuma CSV imputasi (train/val/test) dari iterasi itu ke folder
    # 'best/'. Validation dipakai sebagai kriteria karena prinsipnya sama
    # dengan best_val_loss yang sudah dipakai buat checkpoint selection di
    # atas -- val tidak ikut backprop, jadi tidak bias/overfit.
    #
    # "Terbaik" = ACC paling TINGGI, MAE & RMSE paling RENDAH. Karena bisa
    # saja iterasi terbaik untuk ACC beda dengan iterasi terbaik untuk
    # MAE/RMSE, dipakai RANKING GABUNGAN (rank tiap metrik dijumlah, total
    # rank terkecil = terbaik secara keseluruhan).
    # =========================================================================
    if args.save_imputation and len(imputed_per_iter) > 0:

        import numpy as _np

        acc_arr  = _np.array(ACCs_val,  dtype=float)
        mae_arr  = _np.array(MAEs_val,  dtype=float)
        rmse_arr = _np.array(RMSEs_val, dtype=float)

        # rank: index 0 = terbaik. NaN (acc) diletakkan paling akhir (terburuk).
        acc_rank  = _np.argsort(_np.argsort(-_np.nan_to_num(acc_arr, nan=-_np.inf)))
        mae_rank  = _np.argsort(_np.argsort(mae_arr))
        rmse_rank = _np.argsort(_np.argsort(rmse_arr))

        total_rank = acc_rank + mae_rank + rmse_rank
        best_iter = int(_np.argmin(total_rank))

        print(f'[INFO] Iterasi terbaik (val ACC tertinggi, MAE & RMSE terendah): {best_iter}')
        print(f'[INFO] val -> ACC: {acc_arr[best_iter]}, MAE: {mae_arr[best_iter]}, RMSE: {rmse_arr[best_iter]}')

        best_dir = f'{result_save_path}/best'
        os.makedirs(best_dir, exist_ok=True)

        best_pred = imputed_per_iter.get(best_iter, {})

        if 'train' in best_pred:
            save_imputed_csv(
                save_path=f'{best_dir}/imputed_train.csv',
                dataname=dataname,
                X_pred=best_pred['train'],
                num_col_idx=meta['num_col_idx'],
                cat_col_idx=meta['cat_col_idx'],
                target_col_idx=meta['target_col_idx'],
                cols=meta['cols'],
                cat_bin_num=cat_bin_num,
                y=meta['train_y'],
            )

        if 'val' in best_pred:
            save_imputed_csv(
                save_path=f'{best_dir}/imputed_val.csv',
                dataname=dataname,
                X_pred=best_pred['val'],
                num_col_idx=meta['num_col_idx'],
                cat_col_idx=meta['cat_col_idx'],
                target_col_idx=meta['target_col_idx'],
                cols=meta['cols'],
                cat_bin_num=cat_bin_num,
                y=meta['val_y'],
            )

        if 'test' in best_pred:
            save_imputed_csv(
                save_path=f'{best_dir}/imputed_test.csv',
                dataname=dataname,
                X_pred=best_pred['test'],
                num_col_idx=meta['num_col_idx'],
                cat_col_idx=meta['cat_col_idx'],
                target_col_idx=meta['target_col_idx'],
                cols=meta['cols'],
                cat_bin_num=cat_bin_num,
                y=meta['test_y'],
            )

        # Ringkasan metrik iterasi terbaik yang dipilih
        with open(f'{best_dir}/best_iteration_summary.txt', 'w') as f:
            f.write(f'best_iteration: {best_iter}\n')
            f.write(f'val   -> ACC: {acc_arr[best_iter]}, MAE: {mae_arr[best_iter]}, RMSE: {rmse_arr[best_iter]}\n')
            f.write(f'train -> ACC: {ACCs[best_iter]}, MAE: {MAEs[best_iter]}, RMSE: {RMSEs[best_iter]}\n')
            f.write(f'test  -> ACC: {ACCs_out[best_iter]}, MAE: {MAEs_out[best_iter]}, RMSE: {RMSEs_out[best_iter]}\n')
            f.write(f'\nSeluruh metrik val per iterasi (dipakai untuk pemilihan):\n')
            for i in range(len(ACCs_val)):
                f.write(f'  iter {i}: ACC={ACCs_val[i]}, MAE={MAEs_val[i]}, RMSE={RMSEs_val[i]}\n')

        print(f'[INFO] Hasil imputasi terbaik disimpan di folder: {best_dir}')