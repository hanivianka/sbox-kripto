import streamlit as st
import pandas as pd
import numpy as np
import openpyxl
from io import BytesIO

# Fungsi konversi output S-box ke bentuk biner
def sbox_to_binary(sbox, m):
    return np.array([[int(bit) for bit in format(val, f'0{m}b')] for val in sbox])

# Fungsi Walsh-Hadamard Transform (WHT)
def walsh_hadamard_transform(func):
    size = len(func)
    wht = np.copy(func) * 2 - 1
    step = 1
    while step < size:
        for i in range(0, size, step * 2):
            for j in range(step):
                u = wht[i + j]
                v = wht[i + j + step]
                wht[i + j] = u + v
                wht[i + j + step] = u - v
        step *= 2
    return wht

# Fungsi Nonlinearity (NL)
def compute_nl(binary_sbox, n, m):
    nl_values = []
    for i in range(m):
        func = binary_sbox[:, i]
        wht = walsh_hadamard_transform(func)
        max_correlation = np.max(np.abs(wht))
        nl = 2**(n-1) - (max_correlation // 2)
        nl_values.append(nl)
    return min(nl_values), nl_values

# Fungsi Strict Avalanche Criterion (SAC)
def compute_sac(binary_sbox, n, m):
    num_inputs = 2**n
    sac_matrix = np.zeros((n, m))
    for i in range(num_inputs):
        original_output = binary_sbox[i]
        for bit in range(n):
            flipped_input = i ^ (1 << bit)
            flipped_output = binary_sbox[flipped_input]
            bit_changes = original_output ^ flipped_output
            sac_matrix[bit] += bit_changes
    sac_matrix /= num_inputs
    return np.mean(sac_matrix), sac_matrix

# Fungsi Bit Independence Criterion—Nonlinearity (BIC-NL)
def compute_bic_nl(binary_sbox, n, m):
    bic_nl_values = []
    for i in range(m):
        for j in range(i + 1, m):
            combined_func = binary_sbox[:, i] ^ binary_sbox[:, j]
            wht = walsh_hadamard_transform(combined_func)
            max_correlation = np.max(np.abs(wht))
            nl = 2**(n-1) - (max_correlation // 2)
            bic_nl_values.append(nl)
    return min(bic_nl_values), bic_nl_values

# Fungsi Bit Independence Criterion—Strict Avalanche Criterion (BIC-SAC)
def compute_bic_sac(binary_sbox, n, m):
    bic_sac_values = []
    for i in range(m):
        for j in range(i + 1, m):
            sac_matrix = np.zeros(n)
            for k in range(n):
                count = 0
                for x in range(2**n):
                    flipped_x = x ^ (1 << k)
                    original = binary_sbox[x, i] ^ binary_sbox[x, j]
                    flipped = binary_sbox[flipped_x, i] ^ binary_sbox[flipped_x, j]
                    if original != flipped:
                        count += 1
                sac_matrix[k] = count / (2**n)
            bic_sac_values.append(np.mean(sac_matrix))
    return np.mean(bic_sac_values), bic_sac_values

# Fungsi Linear Approximation Probability (LAP)
def compute_lap(sbox, n):
    bias_table = np.zeros((2**n, 2**n))
    for a in range(1, 2**n):
        for b in range(2**n):
            count = 0
            for x in range(2**n):
                input_dot = bin(x & a).count('1') % 2
                output_dot = bin(sbox[x] & b).count('1') % 2
                if input_dot == output_dot:
                    count += 1
            bias_table[a, b] = abs(count - 2**(n-1))
    lap_max = np.max(bias_table) / (2**n)
    return lap_max, bias_table

# Fungsi Differential Approximation Probability (DAP)
def compute_dap(sbox, n):
    diff_table = np.zeros((2**n, 2**n), dtype=int)
    for delta_x in range(1, 2**n):
        for x in range(2**n):
            x_prime = x ^ delta_x
            delta_y = sbox[x] ^ sbox[x_prime]
            diff_table[delta_x, delta_y] += 1
    dap_max = np.max(diff_table) / (2**n)
    return dap_max, diff_table

# Fungsi utama untuk Streamlit GUI
def main():
    # Konfigurasi halaman 
    st.set_page_config(page_title="S-Box Analyzer", page_icon="🧮", layout="wide")

    # Sidebar
    with st.sidebar:
        st.markdown("<h1 font-size: 26px; color: black;'>⚙️ S-Box Analyzer</h1>", unsafe_allow_html=True)
        st.image("https://www.freeiconspng.com/uploads/3d-cube-transparent-png-4.png", width=200)
        st.markdown("<p style='font-size: 18px;'>Upload file, pilih analisis, dan unduh hasil.</p>", unsafe_allow_html=True)
        st.markdown("### Contoh File S-Box")
        st.markdown("Anda dapat mengunduh contoh file S-Box di bawah ini:")
        
        # Membuat DataFrame contoh S-Box dengan data yang diberikan
        example_sbox = pd.DataFrame([
            [99, 77, 116, 121, 232, 43, 110, 44, 199, 231, 38, 172, 229, 223, 165, 164],
            [205, 177, 143, 220, 50, 237, 176, 66, 101, 53, 56, 74, 195, 42, 2, 69],
            [85, 201, 10, 131, 21, 119, 221, 151, 203, 1, 36, 181, 138, 115, 146, 41],
            [71, 78, 14, 210, 84, 183, 134, 32, 52, 124, 108, 118, 18, 238, 204, 230],
            [25, 5, 54, 83, 215, 7, 19, 137, 57, 0, 8, 39, 93, 139, 120, 104],
            [127, 48, 169, 135, 242, 12, 6, 31, 4, 28, 126, 227, 207, 243, 241, 47],
            [113, 29, 148, 250, 180, 125, 187, 35, 153, 142, 9, 130, 240, 23, 163, 144],
            [219, 30, 68, 149, 198, 55, 59, 147, 197, 170, 189, 89, 95, 98, 128, 251],
            [63, 87, 49, 253, 168, 252, 26, 236, 88, 158, 81, 245, 58, 100, 22, 20],
            [244, 96, 75, 72, 167, 206, 129, 247, 76, 51, 234, 166, 255, 178, 90, 17],
            [109, 193, 171, 182, 103, 235, 112, 117, 202, 226, 212, 16, 209, 37, 60, 150],
            [159, 80, 157, 33, 122, 160, 73, 132, 174, 65, 224, 61, 217, 97, 185, 225],
            [11, 156, 92, 190, 152, 140, 175, 79, 233, 123, 13, 106, 15, 191, 67, 254],
            [228, 200, 114, 141, 162, 133, 45, 136, 62, 186, 3, 196, 111, 213, 34, 161],
            [94, 216, 188, 249, 145, 179, 24, 154, 208, 239, 40, 211, 46, 222, 27, 102],
            [214, 86, 194, 82, 184, 192, 218, 105, 91, 246, 64, 107, 173, 155, 248, 70]
        ])
        
        # Mengonversi DataFrame ke format Excel
        excel_buffer = BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            example_sbox.to_excel(writer, index=False, sheet_name='S-Box', header=False)
            writer.close()
        
        # Tombol unduh untuk file Excel
        st.download_button(
            label="Unduh Contoh S-Box",
            data=excel_buffer,
            file_name="sbox_table.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
        uploaded_file = st.file_uploader("Upload S-Box File (Excel)", type=["xlsx", "xls"])

    # Halaman utama
    st.title("S-Box Analyzer GUI 🔍")

    st.markdown("""
        <div style="font-size: 28px; font-weight: bold;">
            Selamat datang di <span style="color: #4CAF50;">S-Box Analyzer</span>!  
        </div>
        <p style="font-size: 20px;">
            Gunakan alat ini untuk menganalisis properti kriptografi dari S-Box, seperti:
        </p>
        <ul style="font-size: 18px; line-height: 1.6;">
            <li><b>Nonlinearity (NL)</b></li>
            <li><b>Strict Avalanche Criterion (SAC)</b></li>
            <li><b>Bit Independence Criterion—Nonlinearity (BIC-NL)</b></li>
            <li><b>Bit Independence Criterion—Strict Avalanche Criterion (BIC-SAC)</b></li>
            <li><b>Linear Approximation Probability (LAP)</b></li>
            <li><b>Differential Approximation Probability (DAP)</b></li>
        </ul>
    """, unsafe_allow_html=True)


    # Jika file diunggah
    if uploaded_file:
        col1, col2 = st.columns([2, 1])  # Dua kolom untuk tata letak
        with col1:
            st.subheader("📋 Imported S-Box")
            df = pd.read_excel(uploaded_file, header=None)
            st.dataframe(df)

        sbox = df.values.flatten()
        n, m = 8, 8  # Default bit input/output
        binary_sbox = sbox_to_binary(sbox, m)

        # Pilihan tes analisis di sidebar
        with st.sidebar:
            st.subheader("Pilih Tes Analisis yang Ingin Dilakukan:")
            test_options = ["NL", "SAC", "BIC-NL", "BIC-SAC", "LAP", "DAP"]
            selected_tests = st.multiselect("Pilih:", test_options)

        # Progress bar
        progress_bar = st.progress(0)
        results = {}

        # Eksekusi tes analisis
        total_tests = len(selected_tests)
        for i, test in enumerate(selected_tests):
            if test == "NL":
                nl_min, nl_values = compute_nl(binary_sbox, n, m)
                results["NL"] = {"NL Minimum": float(nl_min), "NL Per Fungsi Komponen": [float(val) for val in nl_values]}

            elif test == "SAC":
                sac_avg, sac_matrix = compute_sac(binary_sbox, n, m)
                results["SAC"] = {"SAC Rerata": float(sac_avg), "SAC Matriks": sac_matrix.astype(float)}

            elif test == "BIC-NL":
                bic_nl_min, bic_nl_values = compute_bic_nl(binary_sbox, n, m)
                results["BIC-NL"] = {
                    "BIC-NL Minimum": float(bic_nl_min), 
                    "BIC-NL Semua Pasangan": [float(val) for val in bic_nl_values]
                }

            elif test == "BIC-SAC":
                bic_sac_avg, bic_sac_values = compute_bic_sac(binary_sbox, n, m)
                results["BIC-SAC"] = {
                    "BIC-SAC Rerata": float(bic_sac_avg), 
                    "BIC-SAC Semua Pasangan": [float(val) for val in bic_sac_values]
                }

            elif test == "LAP":
                lap_max, bias_table = compute_lap(sbox, n)
                results["LAP"] = {"LAP Maksimum": float(lap_max), "Tabel Bias": bias_table.astype(float)}

            elif test == "DAP":
                dap_max, diff_table = compute_dap(sbox, n)
                results["DAP"] = {"DAP Maksimum": float(dap_max), "Tabel Diferensial": diff_table.astype(float)}

            progress_bar.progress((i + 1) / total_tests)

        # Menampilkan hasil
        st.subheader("📊 Hasil Analisis")
        for test, result in results.items():
            st.markdown(f"#### **{test} Results**", unsafe_allow_html=True)
            for key, value in result.items():
                if isinstance(value, np.ndarray):
                    st.write(f"**{key}**")
                    st.dataframe(value)
                else:
                    st.write(f"**{key}**: {value}")

        # Ekspor hasil ke excel
        if results:
            st.subheader("📥 Unduh Hasil Analisis")
            output = BytesIO()
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                for test, result in results.items():
                    for key, value in result.items():
                        if isinstance(value, np.ndarray) or isinstance(value, list):
                            if isinstance(value, list):
                                value = np.array(value)
                            if value.ndim == 1:  # 1D Array
                                pd.DataFrame(value, columns=[key]).to_excel(writer, sheet_name=f"{test}_{key}", index=False)
                            elif value.ndim == 2:  # 2D Array
                                pd.DataFrame(value).to_excel(writer, sheet_name=f"{test}_{key}", index=False, header=False)
                        else:
                            pd.DataFrame({key: [value]}).to_excel(writer, sheet_name=f"{test}_{key}", index=False)

            st.download_button("📥 Download Hasil", data=output.getvalue(), file_name="hasil_analisis_sbox.xlsx", mime="application/vnd.ms-excel")

    else:
        st.info("📂 Silakan unggah file Excel S-Box untuk memulai analisis.")


if __name__ == "__main__":
    main()