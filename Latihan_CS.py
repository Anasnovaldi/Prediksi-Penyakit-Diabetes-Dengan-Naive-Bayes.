import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

if __name__ == "__main__":
    while True:
        os.system("cls" if os.name == "nt" else "clear")
        print("="*100)
        print(f"{"================================== Prediksi Diabetes Naive Bayess ==================================":^100}")
        print(f"{"="*100}\n")

        try:
            # Data Loading
            df = pd.read_csv("Latihan-Naive-Bayes-Klasifikasi-Diabetes/data_diabetes.csv")

            # Encoding Kategori
            le_riwayat = LabelEncoder()
            le_aktivitas = LabelEncoder()
            df["riwayat_keluarga"] = le_riwayat.fit_transform(df["riwayat_keluarga"])
            df["aktivitas_fisik"] = le_aktivitas.fit_transform(df["aktivitas_fisik"])

            # Data test
            usia         =   int(input("Masukkan Usia Disini (th)               : "))
            berat        = float(input("Masukkan Berat Badan Disini (kg)        : "))
            tinggi_cm    = float(input("Masukkan Tinggi Badan Disini (cm)       : "))
            tinggi_m     = tinggi_cm / 100
            bmi          = berat / (tinggi_m ** 2)
            kadar_gula_d =   int(input("Masukkan Kadar Gula Darah Disini        : "))
            Tekanan_d    =   int(input("Masukkan Tekanan Darah Disini           : "))
            Riwayat_k    = int(input('''Apakah Kamu Memiliki Riwayat Keluarga   : 
    1. Memiliki
    2. Tidak Memiliki
Pilih Antara (1/2) : '''))
            if Riwayat_k == 1:
                Riwayat_k = le_riwayat.transform(["Ya"])[0]
            elif Riwayat_k == 2:
                Riwayat_k = le_riwayat.transform(["Tidak"])[0]
            else:
                print("Pilihan Hanya 1/2...")

            aktivitas_f  =   int(input('''Aktivitas Belakangan Ini  : 
    1. Aktivitas Tinggi
    2. Aktivitas Sedang
    3. Aktivitas Rendah
Pilih Antara (1/2/3) : '''))

            if aktivitas_f == 1:
                aktivitas_f = le_aktivitas.transform(["Tinggi"])[0]
            elif aktivitas_f == 2:
                aktivitas_f = le_aktivitas.transform(["Sedang"])[0]
            elif aktivitas_f == 3:
                aktivitas_f = le_aktivitas.transform(["Rendah"])[0]
            else:
                print("Pilihan Hanya 1/2/3...")

            # Deklarasi Fiture Dan Target
            X = df[["usia","bmi","kadar_gula_darah","tekanan_darah","riwayat_keluarga","aktivitas_fisik"]]
            y = df["diabetes_target"]

            # Train Test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Buat Model Random Forest
            model = GaussianNB()
            model.fit(X_train,y_train)

            # Akurasi Model
            y_pred = model.predict(X_test)
            akurasi = accuracy_score(y_test, y_pred)
            print("="*100)
            print(f"Akurasi model di test set: {akurasi*100:.2f}%")
            print("="*100)

            # Data Uji
            user = pd.DataFrame([[usia,bmi,kadar_gula_d,Tekanan_d,Riwayat_k,aktivitas_f]], columns=["usia","bmi","kadar_gula_darah","tekanan_darah","riwayat_keluarga","aktivitas_fisik"])

            # Prediksi Model
            hasil_prediksi = model.predict(user)

            # Hasil Prediksi Model
            print(f"\n{"="*100}")
            print(f"{"===================================== Hasil Prediksi Diabetes ======================================":^100}")
            print(f"{"="*100}\n")
            print(f"{str(hasil_prediksi):^100}")
            print(f"\n{"="*100}")
            
        except ValueError:
            print("Input Yang Dimasukkan Salah...")

        option_close = input('Apakah Kamu Ingin Menjalankan Ulang (yes/no) : ').lower()
        if option_close == 'n' or option_close == 'no':
            break