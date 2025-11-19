# Metopsi-Final-Project

Repository untuk preprocessing data MCU dan menyiapkan fitur untuk clustering.


## Anggota

- Tegar Prasetyo (23/520277/PA/22364)
- Benedictus Erwin Widianto (23/520176/PA/22350)

## Ringkasan
Notebook utama: `convert_excel.ipynb`  
Dokumentasi pipeline: [preprocessing_docs.md](preprocessing_docs.md)  <!-- link to detailed steps -->
Input data (harus disimpan di root repo): `data_2_MCU.xlsx`  
Output yang dihasilkan oleh notebook:
- `data_2_MCU.csv` (snapshot awal)
- `Agg_pasien_MCU_2.csv` (hasil agregasi per pasien)
- `Cleaned_agg_pasien_MCU_2.csv` (hasil preprocessing final)
...existing code...
## File penting
- `convert_excel.ipynb` — kode preprocessing
- `preprocessing_docs.md` — dokumentasi alur (ringkasan) ← lihat detail langkah: [preprocessing_docs.md](preprocessing_docs.md)
- `requirements.txt` — paket yang dibutuhkan

## Tujuan
Membersihkan dan meng-encode data MCU:
- drop kolom ber-Missing tinggi
- agregasi per `BADGE` (median numeric, mode kategori)
- normalisasi string kategori
- pemetaan manual (binary / ordinal)
- binary encoding untuk sisanya
- agregasi fitur binary hasil encoding → fitur ringkas
- imputasi missing & ekspor CSV final

## Quick start (Windows, tanpa virtualenv)
1. Install dependency (global / user)
   - pip install --user -r requirements.txt
   - atau (jika ingin instal system-wide): pip install -r requirements.txt

2. Jalankan notebook
   - jupyter lab
   - atau: jupyter notebook lalu buka `convert_excel.ipynb`

## Catatan penting
- Backup file `Agg_pasien_MCU_2.csv` sebelum operasi destruktif (drop kolom).
- Periksa unique values tiap kolom kategori (`value_counts()`) sebelum mapping manual untuk menghindari NaN tak terduga.
- Mapping yang dipakai ada di dalam notebook — simpan salinan mapping jika ingin direvisi.
- Jika dataset besar, pertimbangkan menjalankan notebook pada subset saat debugging.

## File penting
- `convert_excel.ipynb` — kode preprocessing
- `preprocessing_docs.md` — dokumentasi alur (ringkasan)
- `requirements.txt` — paket yang dibutuhkan
