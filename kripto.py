import numpy as np

# Definisi S-box
sbox = np.array([
    99, 205, 85, 71, 25, 127, 113, 219, 63, 244, 109, 159, 11, 228, 94, 214,
    77, 177, 201, 78, 5, 48, 29, 30, 87, 96, 193, 80, 156, 200, 216, 86,
    116, 143, 10, 14, 54, 169, 148, 68, 49, 75, 171, 157, 92, 114, 188, 194,
    121, 220, 131, 210, 83, 135, 250, 149, 253, 72, 182, 33, 190, 141, 249, 82,
    232, 50, 21, 84, 215, 242, 180, 198, 168, 167, 103, 122, 152, 162, 145, 184,
    43, 237, 119, 183, 7, 12, 125, 55, 252, 206, 235, 160, 140, 133, 179, 192,
    110, 176, 221, 134, 19, 6, 187, 59, 26, 129, 112, 73, 175, 45, 24, 218,
    44, 66, 151, 32, 137, 31, 35, 147, 236, 247, 117, 132, 79, 136, 154, 105,
    199, 101, 203, 52, 57, 4, 153, 197, 88, 76, 202, 174, 233, 62, 208, 91,
    231, 53, 1, 124, 0, 28, 142, 170, 158, 51, 226, 65, 123, 186, 239, 246,
    38, 56, 36, 108, 8, 126, 9, 189, 81, 234, 212, 224, 13, 3, 40, 64,
    172, 74, 181, 118, 39, 227, 130, 89, 245, 166, 16, 61, 106, 196, 211, 107,
    229, 195, 138, 18, 93, 207, 240, 95, 58, 255, 209, 217, 15, 111, 46, 173,
    223, 42, 115, 238, 139, 243, 23, 98, 100, 178, 37, 97, 191, 213, 222, 155,
    165, 2, 146, 204, 120, 241, 163, 128, 22, 90, 60, 185, 67, 34, 27, 248,
    164, 69, 41, 230, 104, 47, 144, 251, 20, 17, 150, 225, 254, 161, 102, 70
])

# Parameter S-box
n = 8  # Panjang input dalam bit
m = 8  # Panjang output dalam bit

# Fungsi konversi output S-box ke bentuk biner
def sbox_to_binary(sbox, n, m):
    return np.array([[int(bit) for bit in format(val, f'0{m}b')] for val in sbox])

binary_sbox = sbox_to_binary(sbox, n, m)

# Fungsi Walsh-Hadamard Transform (WHT)
def walsh_hadamard_transform(func):
    size = len(func)
    wht = np.copy(func) * 2 - 1  # Konversi 0/1 ke -1/+1
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

# Fungsi untuk menghitung NL
def compute_nl(sbox, n, m):
    nl_values = []
    for i in range(m):  # Tiap bit output dari S-box
        func = binary_sbox[:, i]  # Fungsi komponen
        wht = walsh_hadamard_transform(func)
        max_correlation = np.max(np.abs(wht))
        nl = 2**(n-1) - (max_correlation // 2)
        nl_values.append(nl)
    return min(nl_values), nl_values

# Fungsi untuk menghitung SAC
def compute_sac(sbox, n, m):
    num_inputs = 2**n
    binary_sbox = np.array([[int(bit) for bit in format(val, f'0{m}b')] for val in sbox])
    sac_matrix = np.zeros((n, m))  # Matriks untuk menyimpan SAC untuk setiap bit input dan bit output
    
    for i in range(num_inputs):  # Untuk setiap input
        original_output = binary_sbox[i]
        for bit in range(n):  # Balikkan setiap bit input 
            flipped_input = i ^ (1 << bit)  # Balikkan bit pada posisi `bit`
            flipped_output = binary_sbox[flipped_input]
            # Hitung perubahan bit output
            bit_changes = original_output ^ flipped_output
            sac_matrix[bit] += bit_changes

    # Normalisasi nilai SAC menjadi probabilitas
    sac_matrix /= num_inputs
    sac_average = np.mean(sac_matrix)  # Nilai SAC rata-rata dari semua bit
    return sac_average, sac_matrix

# Fungsi untuk menghitung BIC-NL
def compute_bic_nl(binary_sbox, n, m):
    num_inputs = 2**n
    bic_nl_values = []
    for i in range(m):  # Bit output pertama
        for j in range(i + 1, m):  # Bit output kedua
            combined_func = binary_sbox[:, i] ^ binary_sbox[:, j]  # XOR dari dua bit
            wht = walsh_hadamard_transform(combined_func)
            max_correlation = np.max(np.abs(wht))
            nl = 2**(n-1) - (max_correlation // 2)
            bic_nl_values.append(nl)
    return min(bic_nl_values), bic_nl_values

# Fungsi untuk menghitung BIC-SAC
def compute_bic_sac(binary_sbox, n, m):
    num_inputs = 2**n
    bic_sac_values = []
    
    for i in range(m):  # Bit output pertama
        for j in range(i + 1, m):  # Bit output kedua
            sac_matrix = np.zeros(n)  # Nilai SAC untuk pasangan bit saat ini
            
            for k in range(n):  # Untuk setiap bit input
                count = 0
                for x in range(num_inputs):
                    # Balikkan bit ke-k dari input
                    flipped_x = x ^ (1 << k)
                    # XOR dua bit output sebelum dan sesudah pembalikan
                    original = binary_sbox[x, i] ^ binary_sbox[x, j]
                    flipped = binary_sbox[flipped_x, i] ^ binary_sbox[flipped_x, j]
                    if original != flipped:  # Periksa apakah hasil XOR berubah
                        count += 1
                sac_matrix[k] = count / num_inputs  # Normalisasi jumlah
            
            bic_sac_values.append(np.mean(sac_matrix))  # SAC rata-rata untuk pasangan bit ini
    
    return np.mean(bic_sac_values), bic_sac_values

# Fungsi untuk menghitung LAP
def compute_lap(sbox, n):
    num_outputs = len(sbox)
    bias_table = np.zeros((2**n, 2**n))  # Ukuran tabel: 256x256 untuk 8-bit S-box

    # Menghitung tabel bias
    for a in range(1, 2**n):  # Mask input (tidak termasuk 0)
        for b in range(2**n):  # Mask output
            count = 0
            for x in range(num_outputs):  # Input x
                # Hitung dot product input
                input_dot = bin(x & a).count('1') % 2
                # Hitung dot product output
                output_dot = bin(sbox[x] & b).count('1') % 2
                if input_dot == output_dot:
                    count += 1
            # Hitung bias
            bias_table[a, b] = abs(count - num_outputs / 2)

    # Menghitung LAP
    lap_max = np.max(bias_table) / (2 ** n)

    return lap_max, bias_table

# Fungsi untuk menghitung DAP
def compute_dap(sbox, n):
    num_outputs = len(sbox)
    diff_table = np.zeros((num_outputs, num_outputs), dtype=int)

    # Menghitung tabel Diferensial
    for delta_x in range(1, num_outputs):  # Perbedaan input (tidak termasuk 0)
        for x in range(num_outputs):  # Semua nilai input
            x_prime = x ^ delta_x  # Input setelah flipping
            delta_y = sbox[x] ^ sbox[x_prime]  # Perbedaan output
            diff_table[delta_x, delta_y] += 1

    # Menghitung DAP
    dap_max = np.max(diff_table) / (2 ** n)

    return dap_max, diff_table

# Memanggil dan menampilkan hasil fungsi
# Hitung NL
nl, nl_values = compute_nl(sbox, n, m)
print(f"--- Nonlinearity (NL) ---")
print(f"Nilai NL minimum: {nl}")
print(f"Nilai NL untuk tiap fungsi komponen:\n{nl_values}")

# Hitung SAC
sac_average, sac_matrix = compute_sac(sbox, n, m)
print(f"\n--- Strict Avalanche Criterion (SAC) ---")
print(f"Nilai SAC rata-rata: {sac_average}")
print(f"Nilai SAC untuk tiap bit input-output:\n{sac_matrix}")

# Hitung BIC-NL
bic_nl_min, bic_nl_values = compute_bic_nl(binary_sbox, n, m)
print(f"\n--- Bit Independence Criterion—Nonlinearity (BIC-NL) ---")
print(f"Nilai BIC-NL minimum: {bic_nl_min}")
print(f"Nilai BIC-NL untuk semua pasangan:\n{bic_nl_values}")

# Hitung BIC-SAC
bic_sac_avg, bic_sac_values = compute_bic_sac(binary_sbox, n, m)
print(f"\n--- Bit Independence Criterion—Strict Avalanche Criterion (BIC-SAC) ---")
print(f"Nilai BIC-SAC rata-rata: {bic_sac_avg}")
print(f"Nilai BIC-SAC untuk semua pasangan bit:\n{bic_sac_values}")

# Hitung LAP
lap_max, bias_table = compute_lap(sbox, n)
print(f"\n--- Linear Approximation Probability (LAP) ---")
print(f"Tabel Bias:\n{bias_table}")
print(f"Nilai LAP maksimum: {lap_max:.5f}")

# Hitung DAP
dap_max, diff_table = compute_dap(sbox, n)
print(f"\n--- Differential Approximation Probability (DAP) ---")
print(f"Tabel Diferensial:\n{diff_table}")
print(f"Nilai DAP maksimum: {dap_max:.5f}")