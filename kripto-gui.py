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
                results["NL"] = {"NL Minimum": nl_min, "NL Per Fungsi Komponen": nl_values}

            elif test == "SAC":
                sac_avg, sac_matrix = compute_sac(binary_sbox, n, m)
                results["SAC"] = {"SAC Rerata": sac_avg, "SAC Matriks": sac_matrix}

            elif test == "BIC-NL":
                bic_nl_min, bic_nl_values = compute_bic_nl(binary_sbox, n, m)
                results["BIC-NL"] = {"BIC-NL Minimum": bic_nl_min, "BIC-NL Semua Pasangan": bic_nl_values}

            elif test == "BIC-SAC":
                bic_sac_avg, bic_sac_values = compute_bic_sac(binary_sbox, n, m)
                results["BIC-SAC"] = {"BIC-SAC Rerata": bic_sac_avg, "BIC-SAC Semua Pasangan": bic_sac_values}

            elif test == "LAP":
                lap_max, bias_table = compute_lap(sbox, n)
                results["LAP"] = {"LAP Maksimum": lap_max, "Tabel Bias": bias_table}

            elif test == "DAP":
                dap_max, diff_table = compute_dap(sbox, n)
                results["DAP"] = {"DAP Maksimum": dap_max, "Tabel Diferensial": diff_table}

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