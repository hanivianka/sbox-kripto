# S-Box Analyzer

S-Box Analyzer merupakan alat untuk menyediakan analisis properti kriptografi untuk S-Box menggunakan berbagai metrik, termasuk Nonlinearity (NL), Strict Avalanche Criterion (SAC), Bit Independence Criterion—Nonlinearity (BIC-NL), Bit Independence Criterion—Nonlinearity (BIC-NL), Linear Approximation Probability (LAP), dan Differential Approximation Probability (DAP).

# Fitur

- **Nonlinearity (NL):** Mengukur ketahanan S-Box terhadap analisis kriptografi linear.
- **Strict Avalanche Criterion (SAC):** Menganalisis bagaimana bit output berubah ketika satu bit input diubah.
- **Bit Independence Criterion - Nonlinearity (BIC-NL):** Mengukur independensi pasangan bit output dalam hal nonlinearity.
- **Bit Independence Criterion - Strict Avalanche Criterion (BIC-SAC):** Mengevaluasi independensi pasangan bit output dalam hal avalanche.
- **Linear Approximation Probability (LAP):** Menghitung bias maksimum dalam tabel aproksimasi linear.
- **Differential Approximation Probability (DAP):** Menganalisis tabel distribusi diferensial.
- **Streamlit GUI:** Antarmuka interaktif untuk mengunggah, menganalisis, dan mengekspor hasil.

# Instalasi

1. Clone repositori:
    ```bash
    git clone https://github.com/hanivianka/sbox-kripto
    ```
    
2. Masuk ke direktori tempat meng-clone repositori:
    ```bash
    cd [nama-repositori]
    ```

3. Instal dependensi yang diperlukan:
    ```bash
    pip install -r requirements.txt
    ```

4. Jalankan aplikasi:
    ```bash
    streamlit run kripto-gui.py
    ```

5. Buka aplikasi di browser `http://localhost:8501`.

# Penggunaan

1. Unggah file S-Box dalam format Excel (`.xlsx` atau `.xls`).
2. Pilih properti kriptografi yang ingin dianalisis dari sidebar.
3. Lihat hasil analisis di aplikasi.
4. Unduh hasil analisis dalam format file Excel.

# Struktur File

- `kripto-gui.py`: Kode utama aplikasi.
- `requirements.txt`: Daftar pustaka Python yang diperlukan.
- `README.md`: Dokumentasi proyek.

# Persyaratan

Pastikan memiliki:

- Python 3.8 atau lebih tinggi
- Pustaka Python yang diperlukan:
  ```
  numpy==2.2.0
  openpyxl==3.1.5
  pandas==2.2.3
  streamlit==1.38.0
  ```

# Program Hasil Deploy

https://sbox-testing-kripto.streamlit.app/

# Paper Referensi

https://link.springer.com/article/10.1007/s11071-024-10414-3
