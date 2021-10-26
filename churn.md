# Laporan Proyek Machine Learning Customer Churn Analysist - Ersan Fernando Samjaya

## Domain Proyek
Domain proyek yang dipilih dalam proyek machine learning ini adalah mengenai Telekomunikasi dengan judul proyek "Menganalisa Pelanggan Churn Dalam Sebulan Terakhir di Perusahaan Telekomunikasi".
<p align="center">
  <img width="460" height="300" src="https://s16353.pcdn.co/wp-content/uploads/2018/06/Churn.png">
</p>

### Latar belakang :

Retensi pelanggan adalah salah satu KPI (Key Performance Indicators) utama untuk perusahaan dengan model bisnis berbasis langganan. Persaingan ketat terutama di pasar SaaS (Softwere as a service, perangkat lunak yang bisa digunakan dan diakses melalui internet tanpa harus melakukan instalasi) di mana pelanggan bebas memilih dari banyak penyedia. Satu pengalaman buruk dan pelanggan mungkin saja pindah ke pesaing yang mengakibatkan churn pelanggan.

Memprediksi churn pelanggan adalah masalah bisnis yang menantang tetapi sangat penting terutama di industri di mana biaya akuisisi pelanggan tinggi seperti teknologi, telekomunikasi, keuangan, dll. Kemampuan untuk memprediksi bahwa pelanggan tertentu berisiko tinggi untuk churn , selagi masih ada waktu untuk melakukan sesuatu, ini merupakan sumber pendapatan potensial tambahan yang sangat besar bagi perusahaan.

[referensi : Towards Data Science](https://towardsdatascience.com/predict-customer-churn-the-right-way-using-pycaret-8ba6541608ac)

## Bussiness Understanding

Customer Churn adalah persentase pelanggan yang berhenti menggunakan produk atau layanan perusahaan selama jangka waktu tertentu. Salah satu cara untuk menghitung churn rate adalah dengan membagi jumlah pelanggan yang hilang selama interval waktu tertentu dengan jumlah pelanggan aktif pada awal periode. Misalnya, jika Anda mendapatkan 1000 pelanggan dan kehilangan 50 orang dibulan lalu, maka tingkat churn bulanan Anda adalah 5 persen.

### Problem Statements
 * Bagaimana memprediksi Customer Churn dengan pendekatan model Machine Learning?
 * Bagaimana kita dapat melatih dan memilih model yang memaksimalkan nilai bisnis?
 

### Goals
* Tujuan utama dari model prediktif churn pelanggan adalah untuk mempertahankan pelanggan pada risiko churn tertinggi dengan terlibat secara proaktif dengan mereka. Misalnya: Tawarkan voucher hadiah atau harga promosi apa pun dan pertahankan selama satu atau dua tahun tambahan untuk memperpanjang nilai masa pakainya bagi perusahaan.
* Memaksimalkan keuntungan, memangkas biaya untuk promosi.

### Solution Statements
* Mennggunakan 2 model ,yaitu: Model Random Forest Classifier dan Extra Tree Classifier
#penjelasasn algoritma

### Data Understanding
Sumber = [Kaggle : Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn)

Dataset mencakup informasi tentang:

* Pelanggan yang pergi dalam sebulan terakhir – kolomnya disebut Churn
* Layanan yang telah didaftarkan oleh setiap pelanggan – telepon, banyak saluran, internet, keamanan online, pencadangan online, perlindungan perangkat, dukungan teknis, serta streaming TV dan film
* Informasi akun pelanggan – sudah berapa lama mereka menjadi pelanggan, kontrak, metode pembayaran, tagihan tanpa kertas, tagihan bulanan, dan total tagihan
* Info demografis tentang pelanggan – jenis kelamin, rentang usia, dan jika mereka memiliki pasangan dan tanggungan

Berikut ini merupakan deskripsi untuk setiap variabel:

-   `CustomerID`: Customer ID
-   `Gender`: Gender pelanggan yaitu Female dan Male
-   `SeniorCitizen`: Apakah pelanggan merupakan senio citizen (0: No, 1: Yes)
-   `Partner`: Apakah pelanggan memiliki partner atau tidak (Yes, No)
-   `Dependents`: Apakah pelanggan memiliki tanggungan atau tidak (Yes, No)
-   `Tenure`: Jumlah bulan dalam menggunakan produk perusahaan
-   `MultipleLines`: Apakah pelanggan memiliki banyak saluran atau tidak (Yes, No, No phone service)
-   `OnlineSecurity`: Apakah pelanggan memiliki keamanan online atau tidak
-   `OnlineBackup`: Apakah pelanggan memiliki cadangan online atau tidak
-   `DeviceProtection`: Apakah pelanggan memiliki perlindungan perangkat atau tidak
-   `TechSupport`: Apakah pelanggan memiliki dukungan teknis atau tidak
-   `StreamingTV`: Apakah pelanggan berlangganan TV streaming atau tidak
-   `StreamingMovies`: Apakah pelanggan berlangganan movies streaming atau tidak
-   `Contract`: Ketentuan kontrak berlangganan (Month-to-month, One year, Two year)
-   `PaperlessBilling`: Apakah pelanggan memiliki tagihan tanpa kertas atau tidak (Yes, No)
-   `PaymentMethod`: Metode pembayaran (Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic))
-   `MonthlyCharges`: Jumlah pembayaran yang dilakukan setiap bulan
-   `TotalCharges`: Jumlah total yang dibebankan oleh pelanggan
-   `Churn`: Apakah pelanggan Churn atau tidak (Yes or No)
### Data Preparation
* mengimport library yang dibutuhkan
* Menggunakan library pandas-profiling untuk efisiensi proses EDA
* Mengganti nilai yang kosong dengan np.nan
* Cek outliers dengan visualisasi
* Menghapus nilai 0 di kolom tenor dan menghapus kolom customer ID
* Visualisasi Data bertipe numerik untuk mengetahui seberapa besar korelasi antar variable numerik
* Konversi semua kolom bertipe categorical ke numerik dan menerapkan label encoding untuk kolom bertipe categorical
* Membagi datset menjadi data latih dan data uji
* Treatment imbalance Dataset menggunakan teknik upsampling
* Normalisasi data dengan MinMaxScaler dengan skala range 0 dan 1

### Modeling
 Untuk pemilihan model, saya menggunakan algoritma Random Forest Classifier dan Extra Tree Classifier
 Random Forest : 
 * Random Forest adalah suatu algoritma yang digunakan pada klasifikasi data dalam jumlah yang besar. 
 * Klasifikasi random forest dilakukan melalui penggabungan pohon (tree) dengan melakukan training pada sampel data yang dimiliki. 
 * Penggunaan pohon (tree) yang semakin banyak akan mempengaruhi akurasi yang akan didapatkan menjadi lebih baik. 
 * Penentuan klasifikasi dengan random forest diambil berdasarkan hasil voting dari tree yang terbentuk. 
 * Pemenang dari tree yang terbentuk ditentukan dengan vote terbanyak. 
 * Pembangunan pohon (tree) pada random forest sampai dengan mencapai ukuran maksimum dari pohon data. Akan tetapi,pembangunan pohon random forest tidak dilakukan pemangkasan (pruning) yang merupakan sebuah metode untuk mengurangi kompleksitas ruang. 
 * Pembangunan dilakukan dengan penerapan metode random feature selection untuk meminimalisir kesalahan. 
 * Pembentukan pohon (tree) dengan sample data menggunakan variable yang diambil secara acak dan menjalankan klasifikasi pada semua tree yang terbentuk.
 * Random forest menggunakan Decision Tree untuk melakukan proses seleksi. 
 * Pohon yang dibangun dibagi secara rekursif dari data pada kelas yang sama. 
 * Pemecahan (split) digunakan untuk membagi data berdasarkan jenis atribut yang digunakan. 
 * Pembuatan decision tree pada saat penentuan klasifikasi,pohon yang buruk akan membuat prediksi acak yang saling bertentangan. 
 * Sehingga,beberapa decision tree akan menghasilkan jawaban yang baik. 
 * Random forest merupakan salah satu cara penerapan dari pendekatan diskriminasi stokastik pada klasifikasi. 
 * Proses Klasifikasi akan berjalan jika semua tree telah terbentuk.
 * Pada saat proses klasifikasi selesai dilakukan, inisialisasi dilakukan dengan sebanyak data berdasarkan nilai akurasinya. 
 * Keuntungan penggunaan random forest yaitu mampu mengklasifiksi data yang memiliki atribut yang tidak lengkap,dapat digunakan untuk klasifikasi dan regresi akan tetapi tidak terlalu bagus untuk regresi, lebih cocok untuk pengklasifikasian data serta dapat digunakan untuk menangani data sampel yang banyak. 
 * Proses klasifikasi pada random forest berawal dari memecah data sampel yang ada kedalam decision tree secara acak. 
 * Setelah pohon terbentuk,maka akan dilakukan voting pada setiap kelas dari data sampel. Kemudian, mengkombinasikan vote dari setiap kelas kemudian diambil vote yang paling banyak.Dengan menggunakan random forest pada klasifikasi data maka, akan menghasilkan vote yang paling baik 
 Model Extra Tree Classifier :
 * Random Forest memilih pemisahan optimal sementara Extra Tree memilihnya secara acak. Namun, setelah titik split dipilih, kedua algoritme memilih yang terbaik di antara semua subset fitur. Oleh karena itu, Pohon Ekstra menambahkan pengacakan tetapi masih memiliki pengoptimalan

* Menguji data latih dengan kedua model, lalu bandingkan kinerjanya, pilih model yang paling baik kinerjanya.
* Mengembangkan model dengan melakukan tuning hyperparameter 
* Menggunakan Matriks bisnis yaitu laba yang diperoleh serta cost yang dikeluarkan untuk evaluasi kinerja kedua model
* Bandingkan kinerja model sebelum dan sesudah di tunning
* Memberikan estimasi biaya promosi dan estimasi laba yang akan diperoleh setelah melakukan promosi
 
 ### Menguji kedua model dan membandingkan kinerjanya
 * Menguji kedua model dengan data latih, lalu memilih model dengan kinerja terbaik
 * menampilkan klasifikasi report dari kedua model
 ### Mengembangkan model dasar
 * Melakukan hyperparameter tuning dan melatihnya dengan data training
 * Bandingkan kinerja model yang telah di tuning dengan yang standard 
 * Menampilkan confussion matrix kedua model
 ### Evaluasi
 * Asumsikan bahwa pelanggan yang berlangganan akan memberikan keuntungan sebesar Rp.200.000,00 per bulan.
 * Promosi yang didapat oleh pelanggan yang diprediksi churn adalah sebesar Rp.50.000,00
 * Menghitung Laba yang diperoleh serta Cost yang dikeluarkan oleh perusahaan berdasarkan kedua model
 * Melilih model yang ekonomis namun dapat menghasilkan profit yang besar
 * Memberikan estimasi biaya yang dibutuhkan kepada tim Marketing untuk promosi dan estimasi laba yang akan diperoleh / harus dicapai setelah promosi diberikan  


# Conclusion

Dengan adanya model untuk memprediksi *customer churn*, pihak perusahaan telekomunikasi dengan mudah mengetahui pelanggan mana yang memiliki kecenderungan untuk *churn*. 
Dari sini, pihak *marketing* dapat melakukan promosi produk dengan sifat kontrak yang jangkanya lebih panjang sehingga para pelanggan  dapat bertahan lebih lama.
Berdasarkan metrik dari bisnis tersebut, perusahaan akan memberikan estimasi biaya untuk promotion dan mengetahui estimasi laba yang akan diperoleh setelah melakukan promosi kepada tim marketing.

# External Resources

- Reference: [Algoritma Book: ML Application in Industry](https://algoml-industry.netlify.app/)
- Dataset: [Kaggle: Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn)
