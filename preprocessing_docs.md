# Preprocessing pipeline — convert_excel.ipynb

Ringkasan alur preprocessing yang dijalankan di notebook:

## Input / output utama
- Input: `data_2_MCU.xlsx`
- Intermediate CSV: `data_2_MCU.csv`
- Aggregated: `Agg_pasien_MCU_2.csv`
- Final cleaned: `Cleaned_agg_pasien_MCU_2.csv`

---

## Langkah-langkah (cell-by-cell)

1. Load & initial column drop
   - Baca Excel dengan `pd.read_excel(..., engine='openpyxl')`.
   - Hitung missing per kolom; drop kolom yang memiliki > 3000 missing.
   - Simpan snapshot ke `data_2_MCU.csv`.
   - Set index ke kolom `TANGGAL`.

2. Identifikasi kolom kategorial & pembersihan awal
   - Buat daftar kolom object/category, hitung unique values.
   - Tandai kolom dengan unique > 100 untuk di-drop (terlalu banyak kategori).
   - Buang kolom kategorial yang hanya punya 1 nilai.

3. Aggregation per pasien (`BADGE`)
   - Numeric: ambil median.
   - Kategorial: ambil mode (jika ada).
   - Hasil disimpan sebagai `agg_df` dan diekspor ke `Agg_pasien_MCU_2.csv`.

4. Standardisasi string kategorial
   - Trim/strip dan normalisasi teks (perbaiki variasi `'tidak diperiksa'`).
   - Konversi ke lower/strip di langkah selanjutnya.

5. Identifikasi kolom binary vs multi-cardinality
   - Tampilkan kolom kandidat binary (nunique == 2) dan kolom ber-more-than-2 unique (multi-card).
   - Simpan unique values untuk keputusan pemetaan manual.

6. Manual mapping (binary & ordinal)
   - Terapkan mapping informed untuk beberapa kolom binary (contoh: `OLAHRAGA`, `ALERGI`, `CONJUNGTIVA`, dsb).
   - Terapkan mapping ordinal untuk beberapa kolom urine sedimen (`ERITROSIT_RBC`, `LEKOSIT_WBC`, `SEL_EPITEL`, dsb).
   - Mapping diimplementasikan dengan `.map()`; nilai yang tidak tercakup tetap menjadi NaN atau dibiarkan.

   Contoh mapping ringkas (ada di notebook, sudah lebih lengkap):
   - Binary: `'+' -> 1`, `'-' -> 0`, `'normal' -> 0`, `'ada' -> 1` ...
   - Ordinal (contoh): `'Negatif' -> 0`, `'+' -> 1`, `'++' -> 2`, `'+++' -> 3`

7. Encoding sisanya
   - Hapus kolom kategorial yang seluruhnya NaN.
   - Untuk kolom kategorial yang masih tersisa, gunakan `category_encoders.BinaryEncoder` (binary encoding) untuk mengubah menjadi kolom numerik.

8. Ekstraksi & parsing spesifik kolom
   - TENSI → pecah menjadi `SISTOLIK` dan `DIASTOLIK` dengan fungsi parsing.
   - KEHAMILAN → parse menjadi `TRIMESTER` (0 = tidak hamil, 1/2/3) berdasarkan teks/perkiraan minggu/bulan.
   - Urine microscopic (`ERITROSIT_RBC`, `LEKOSIT_WBC`, `SEL_EPITEL`) → parse:
     - Jika ada angka atau range → ambil rata-rata angka.
     - Jika teks seperti `nihil/negatif/sedikit/+/++/+++` → konversi ke skor numerik via mapping.
   - Semua kolom object/category → lower + strip + normalisasi bentuk `'tidak diperiksa'`.

9. Post-encoding cleaning & column aggregation
   - Hapus kolom hasil encoding yang sangat jarang (rare) jika mean < threshold (contoh threshold 0.01).
   - Gabungkan beberapa kolom binary-encoded logically menjadi fitur ringkas (contoh groupings: `ANY_EAR_ABNORMAL`, `ANY_NOSE_ABNORMAL`, `ANY_URINE_SEDIMENT_POS`, dst). Implementasi: `new_col = old_cols.max(axis=1)` lalu drop kolom lama.
   - Pilih daftar kolom yang ingin di-keep langsung (list `keep_direct`) bila ada.

10. Final selection & feature engineering
    - Pilih kolom numeric & lab penting (`numeric_lab_cols`) yang tersedia.
    - Gabungkan numeric_lab_cols + keep_direct + grouped features menjadi `final_clustering_df`.
    - Hitung tambahan fitur opsional:
      - BMI = BERAT / (TINGGI/100)^2
      - MAP = DIASTOLIK + (SISTOLIK - DIASTOLIK) / 3

11. Final cleanup & missing handling
    - Drop kolom yang hanya memiliki satu nilai (`nunique() == 1`).
    - Isi missing:
      - Numeric → fill dengan median kolom.
      - Categorical/binary → fill dengan mode kolom.
    - Simpan hasil akhir ke `Cleaned_agg_pasien_MCU_2.csv`.

---

## Files & variabel penting dalam notebook
- df: dataframe awal dari Excel
- agg_df: hasil aggregasi per BADGE (median/mode)
- encoded_df: hasil setelah parsing + mapping + binary encoding
- final_df: hasil setelah menghapus kolom rare & menggabung fitur
- final_clustering_df: features final untuk clustering
- output file: `Cleaned_agg_pasien_MCU_2.csv`

---

## Catatan & rekomendasi singkat
- Periksa coverage mapping: jalankan `value_counts()` sebelum `.map()` untuk memastikan semua variasi string tertangani.
- Simpan mapping dictionary terpisah (file json) supaya dapat direview dan direvisi tanpa mengubah kode.
- Jangan lupa backup `Agg_pasien_MCU_2.csv` sebelum operasi destruktif (drop kolom).
- Untuk debugging, jalankan langkah mapping pada subset kecil (print unique values setelah replace) agar tidak kehilangan data karena `.map()` menghasilkan NaN.
- Setelah preprocessing, cek distribusi fitur (missing, skew, outlier) sebelum clustering.
