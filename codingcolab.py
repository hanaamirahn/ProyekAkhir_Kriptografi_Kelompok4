# -*- coding: utf-8 -*-
"""Kriptografi_Kelompok 4_Rombel 4

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/19CIP6CDgkcKHqM4H4aY-ZJBWoXKkudMn

## **Nonlinearity (NL)**
"""

def calculate_nonlinearity(sbox):
    # Fungsi untuk menghitung transformasi Walsh-Hadamard
    def walsh_hadamard_transform(function):
        num_elements = 256  # Panjang input (8-bit)
        transform_result = np.zeros(num_elements, dtype=int)  # Array untuk hasil transformasi

        for coeff in range(num_elements):  # Iterasi untuk setiap koefisien Walsh-Hadamard
            correlation_sum = 0  # Nilai akumulasi korelasi
            for input_value in range(num_elements):  # Iterasi untuk semua input
                # Korelasi antara input dan fungsi linier/afin
                parity_input = (-1) ** (bin(coeff & input_value).count('1') % 2)
                parity_output = (-1) ** (bin(coeff & function[input_value]).count('1') % 2)

                # Akumulasi hasil korelasi
                correlation_sum += parity_input * parity_output

            # Simpan hasil korelasi untuk koefisien ini
            transform_result[coeff] = correlation_sum

        return transform_result

    # Hitung transformasi Walsh-Hadamard untuk S-box
    wht_result = walsh_hadamard_transform(sbox)

    # Hitung nilai nonlinearitas berdasarkan hasil Walsh-Hadamard
    max_correlation = max(abs(value) for value in wht_result[1:])  # Abaikan elemen pertama
    nonlinearity = (256 // 2) - (max_correlation // 2)

    return nonlinearity

"""## **Strict Avalanche Criterion (SAC)**"""

def calculate_sac(sbox):
    # Fungsi untuk menghitung Strict Avalanche Criterion (SAC)
    def calculate_bit_changes(sbox):
        num_elements = 256  # Panjang input (8-bit)
        changes_per_bit = []

        for bit_position in range(8):  # Iterasi untuk setiap posisi bit
            total_changes = 0  # Akumulasi perubahan bit
            for idx in range(num_elements):  # Iterasi untuk semua elemen
                # Hitung perbedaan bit antara dua nilai S-box yang hanya berbeda pada posisi bit ini
                bit_diff = bin(sbox[idx] ^ sbox[idx ^ (1 << bit_position)]).count('1')
                total_changes += bit_diff

            # Simpan hasil perubahan bit untuk posisi bit ini
            changes_per_bit.append(total_changes / (num_elements * 8))

        return changes_per_bit

    # Hitung perubahan bit untuk S-box
    bit_changes_result = calculate_bit_changes(sbox)

    # Hitung nilai SAC berdasarkan hasil perubahan bit
    sac_value = np.mean(bit_changes_result)

    return sac_value

"""## **Bit Independence Criterion-NonLinearity (BIC-NL)**"""

def calculate_bic_nl(sbox):
    """
    Menghitung Bit Independence Criterion-Nonlinearity (BIC-NL).
    """
    num_bits = 8  # Jumlah bit pada keluaran S-box (8-bit)
    bic_nl_values = []  # Menyimpan nilai non-linearitas untuk setiap pasangan bit

    # Fungsi untuk menghitung transformasi Walsh-Hadamard
    def walsh_hadamard_transform(function):
        num_elements = 256  # Panjang input (8-bit)
        transform_result = np.zeros(num_elements, dtype=int)

        for coeff in range(num_elements):
            correlation_sum = 0
            for input_value in range(num_elements):
                parity_input = (-1) ** (bin(coeff & input_value).count('1') % 2)
                parity_output = (-1) ** (function[input_value])
                correlation_sum += parity_input * parity_output
            transform_result[coeff] = correlation_sum

        return transform_result

    # Iterasi untuk setiap pasangan bit
    for bit_i in range(num_bits):
        for bit_j in range(bit_i + 1, num_bits):
            # Ekstrak fungsi Boolean untuk bit ke-i dan ke-j
            boolean_function = [
                ((sbox[input_value] >> bit_i) & 1) ^ ((sbox[input_value] >> bit_j) & 1)
                for input_value in range(256)
            ]

            # Hitung transformasi Walsh-Hadamard untuk fungsi Boolean
            wht_result = walsh_hadamard_transform(boolean_function)

            # Hitung non-linearitas sebagai setengah jarak Hamming maksimum
            max_correlation = max(abs(value) for value in wht_result)
            nonlinearity = (256 // 2) - (max_correlation // 2)

            # Simpan nilai non-linearitas untuk pasangan ini
            bic_nl_values.append(nonlinearity)

    # Hitung rata-rata non-linearitas untuk semua pasangan bit
    bic_nl_score = np.mean(bic_nl_values)

    return bic_nl_score

"""## **Bit Independence Criterion-Strict Avalanche Criterion (BIC-SAC)**"""

def calculate_bic_sac_v2(substitution_box):
    # Menyimpan panjang S-box dan bit length
    box_size = len(substitution_box)
    bit_count = 8  # Panjang bit dalam S-box (AES menggunakan 8 bit)

    total_bit_pairs = 0  # Variabel untuk jumlah pasangan bit
    total_independence_score = 0  # Variabel untuk jumlah total independensi

    # Iterasi untuk setiap pasangan bit output (i, j)
    for bit_i in range(bit_count):
        for bit_j in range(bit_i + 1, bit_count):  # Hanya pasangan bit unik (i, j)
            bit_pair_independence_sum = 0

            # Iterasi untuk setiap input x dan untuk setiap pembalikan bit pada input
            for input_val in range(box_size):
                for flip_bit in range(bit_count):
                    # Membalikkan satu bit pada input x
                    flipped_input = input_val ^ (1 << flip_bit)

                    # Output asli dan hasil output dari input yang dibalik
                    original_output = substitution_box[input_val]
                    flipped_output = substitution_box[flipped_input]

                    # Ambil nilai bit ke-i dan ke-j dari output
                    bit_original_i = (original_output >> bit_i) & 1
                    bit_original_j = (original_output >> bit_j) & 1
                    bit_flipped_i = (flipped_output >> bit_i) & 1
                    bit_flipped_j = (flipped_output >> bit_j) & 1

                    # Menambahkan hasil independensi perubahan bit pada pasangan bit (i, j)
                    bit_pair_independence_sum += ((bit_original_i ^ bit_flipped_i) ^ (bit_original_j ^ bit_flipped_j))

            # Normalisasi hasil independensi untuk pasangan bit (i, j)
            normalized_independence = bit_pair_independence_sum / (box_size * bit_count)
            total_independence_score += normalized_independence
            total_bit_pairs += 1

    # Menghitung rata-rata dari semua pasangan bit output
    bic_sac_value = total_independence_score / total_bit_pairs
    return round(bic_sac_value, 5)

"""## **Linear Approximation Probability (LAP)**"""

def calculate_lap(sbox):
    """
    Menghitung Linear Approximation Probability (LAP).
    """
    max_bias_value = 0  # Nilai bias maksimum yang akan dihitung
    for approximation_input in range(1, 256):  # Iterasi untuk semua kemungkinan nilai input 'a'
        for approximation_output in range(1, 256):  # Iterasi untuk semua kemungkinan nilai output 'b'
            correlation_total = sum(
                (bin(approximation_input & idx).count('1') % 2) == (bin(approximation_output & sbox44[idx]).count('1') % 2)
                for idx in range(256)  # Iterasi untuk semua elemen dalam S-box
            )
            # Hitung bias untuk pasangan input-output (a, b)
            bias_value = abs(correlation_total / 256.0 - 0.5)
            max_bias_value = max(max_bias_value, bias_value)  # Simpan bias maksimum yang ditemukan

    return max_bias_value  # Kembalikan bias maksimum

"""## **Differential Approximation Probability (DAP)**"""

def calculate_dap(sbox):
    """
    Menghitung Differential Approximation Probability (DAP).
    """
    highest_probability = 0  # Probabilitas maksimum yang akan dihitung
    for input_difference in range(1, 256):  # Iterasi untuk setiap perbedaan input
        output_difference_count = {}  # Dictionary untuk menghitung frekuensi perbedaan output
        for input_value in range(256):  # Iterasi untuk semua nilai input
            # Hitung perbedaan output berdasarkan perbedaan input
            output_difference = sbox[input_value ^ input_difference] ^ sbox[input_value]
            output_difference_count[output_difference] = output_difference_count.get(output_difference, 0) + 1

        # Hitung probabilitas maksimum untuk perbedaan input saat ini
        max_probability_for_input = max(count / 256.0 for count in output_difference_count.values())
        highest_probability = max(highest_probability, max_probability_for_input)  # Simpan probabilitas tertinggi

    return highest_probability  # Kembalikan probabilitas maksimum