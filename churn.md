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
* Menggunakan 2 model ,yaitu: Model Random Forest Classifier dan Extra Tree Classifier
##### Penjelasan Random Forest : 
Dalam machine learning sering kita mendengar  tentang metode Random Forest yang digunakan untuk menyelesaikan permasalahan. Metode Random Forest  merupakan salah satu metode dalam Decision Tree. Decision Tree atau pohon pengambil keputusan adalah sebuah diagram alir yang berbentuk seperti pohon yang memiliki sebuah root node yang digunakan untuk mengumpulkan data, Sebuah inner node yang berada pada root node yang berisi tentang pertanyaan tentang data dan  sebuah leaf node yang digunakan untuk memecahkan masalah serta membuat keputusan. Decision tree mengklasifikasikan suatu sampel data yang belum diketahui kelasnya kedalam kelas – kelas yang ada. Penggunaan decision tree agar dapat menghindari overfitting pada sebuah set data saat mencapai akurasi yang maksimum.

Random forest  adalah kombinasi dari  masing – masing tree yang baik kemudian dikombinasikan  ke dalam satu model. Random Forest bergantung pada sebuah nilai vector random dengan distribusi yang sama pada semua pohon yang masing masing decision tree memiliki kedalaman yang maksimal. Random forest adalah classifier yang terdiri dari classifier yang berbentuk pohon {h(x, θ k ), k = 1, . . .} dimana θk adalah random vector yang diditribusikan secara independen dan masing masing tree pada sebuah unit kan memilih class yang paling popular pada input x. 
 
 ###### Kelebihan Random Forest : 
 * Keuntungan penggunaan random forest yaitu mampu mengklasifiksi data yang memiliki atribut yang tidak lengkap,dapat digunakan untuk klasifikasi dan regresi akan tetapi tidak terlalu bagus untuk regresi, lebih cocok untuk pengklasifikasian data serta dapat digunakan untuk menangani data sampel yang banyak. 
 * Random Forest bagus untuk klasifikasi. Dapat digunakan untuk membuat prediksi kategori dengan beberapa nilai yang mungkin dan dapat dikalibrasi untuk probabilitas output.
 * Menghasilkan eror yang lebih rendah.
 * Memberikan hasil yang bagus dalam klasifikasi.
 * Dapat mengatasi data training dalam jumlah sangat besar secara efisien.
 * Metode yang efektif untuk mengestimasi hilangnya data.
 * Dapat memperkiraan variabel apa yang penting dalam klasifikasi.
 * Menyediakan metode eksperimental untuk mendeteksi interaksi variabel.
 
 ###### Kekurangan Random Forest :
 * Pembuatan decision tree pada saat penentuan klasifikasi,pohon yang buruk akan membuat prediksi acak yang saling bertentangan. Sehingga,beberapa decision tree akan menghasilkan jawaban yang baik. 
 * Random Forest rawan terjadi overfitting, terutama ketika bekerja dengan dataset yang relatif kecil. Perlu di curigai jika model data dapat membuat prediksi yang "terlalu bagus" pada set uji menggunakan Random Forest. Salah satu cara overfitting adalah menggunakan fitur yang benar-benar relevan dalam model data yang digunakan.
 * Waktu pemrosesan yang lama karena menggunakan data yang banyak dan membangun model tree yang banyak pula untuk membentuk random trees karena menggunakan single processor.
 * Interpretasi yang sulit dan membutuhkan mode penyetelan yang tepat untuk data.
 * Ketika digunakan untuk regresi, mereka tidak dapat memprediksi di luar kisaran dalam data percobaan, hal ini di mungkinkan data terlalu cocok dengan kumpulan data pengganggu (noisy).


 ##### Penjelasan Extra Tree Classifier :
  Extra Trees Classifier adalah jenis teknik pembelajaran ensemble yang menggabungkan hasil dari beberapa pohon keputusan yang tidak berkorelasi yang dikumpulkan di "forest" untuk menghasilkan hasil klasifikasinya. Secara konsep, sangat mirip dengan Random Forest Classifier dan hanya berbeda dalam cara konstruksi pohon keputusan di hutan. Random Forest memilih pemisahan optimal sementara Extra Tree memilihnya secara acak. Namun, setelah titik split dipilih, kedua algoritme memilih yang terbaik di antara semua subset fitur.

##### Kelebihan Extra Tree Classifier: 
Pohon Ekstra menambahkan pengacakan tetapi masih memiliki pengoptimalan. Setiap Pohon Keputusan di Hutan Pohon Ekstra dibangun dari sampel pelatihan asli. Kemudian, pada setiap node pengujian, Setiap pohon disediakan dengan sampel acak k-fitur dari set fitur , yang mana setiap pohon keputusan harus memilih fitur terbaik untuk membagi data berdasarkan beberapa kriteria matematis (biasanya Indeks Gini). 

##### Kekurangan Extra Tree Classifier: 
Sampel fitur acak ini mengarah pada pembuatan beberapa pohon keputusan yang tidak berkorelasi.

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
### Data Preparation/Prepocessing
 Mengapa data perlu di-preprocessing? Karena data masih belum siap untuk diproses di Model machine Learning. Jika tanpa Data preprocessing, maka hasil analisa akan terjadi bias, kurang akurat, waktu training yang lama, dll.
* Cek outliers dengan visualisasi. Outliers adalah data yang memiliki nilai sangat jauh dari nilai umumnya, atau dengan kata lain memiliki nilai yang ekstrem. Adanya outliers ini dapat berpengaruh pada hasil uji asumsi, seperti uji normalitas, lineraritas, maupun homogenitas varians. Lebih parah lagi, outliers ini dapat berpengaruh pada pegambilan kesimpulan penelitian dari hasil uji statistik
* Mengganti nilai yang kosong dengan np.nan. kolom TotalCharges adalah tipe objek, bukan float64. Setelah diselidiki, saya menemukan ada beberapa ruang kosong di kolom ini yang menyebabkan Python memaksa tipe data sebagai object . Untuk memperbaikinya, kita harus trimming ruang kosong itu  sebelum mengubah tipe data. Kemudian mengisinya dengan rata-rata (mean) dikarenakan tidak terdapat outliers pada kolom tersebut.
* Menghapus nilai 0 di kolom tenor dan menghapus kolom customer ID. Kita bisa menghapus kolom tenor yang bernilai 0 dikarenakan tidak bisa mengisinya dengan nilai mean ataupun median, dan karena jumlahnya hanya 11, tidak akan berdampak besar terhadap hasil analisa. Kemudian menghapus kolom customer ID, karena kita tidak akan menggunakan data tersebut.
* Visualisasi Data pada kolom bertipe category untuk mengetahui korelasi dengan kolom target
* Visualisasi Data pada kolom bertipe numerik untuk mengetahui seberapa besar korelasi antar variable numerik
* Konversi semua kolom bertipe categorical ke numerik dan menerapkan label encoding untuk kolom bertipe categorical. Model Machine Learning hanya bisa memproses data angka, bukan data teks, sehingga perlu dilakukan konversi.
* Membagi dataset menjadi data latih dan data uji, bertujuan untuk menghindari terjadinya overfitting, yaitu suatu kondisi pelatihan yang hasil uji terhadap data yang dilatih sangat bagus tetapi diuji oleh data lain yang tidak digunakan dalam pelatihan sangat buruk
* Treatment imbalance Dataset menggunakan teknik upsampling, bertujuan agar model Machine Learning meminimalisir kekeliruan dalam memprediksi target 
* Normalisasi data dengan MinMaxScaler dengan skala range 0 dan 1, bertujuan untuk membuat nilai data menjadi lebih kecil tanpa merubah informasi yang dikandungnya

### Modeling
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
 Kita dapat menggunakan 2 buah metrik. Yaitu Metrik F1-Score dan metrik laba/profit/untung.
 F1- Score:
 * F1-Score merupakan perbandingan rata-rata presisi dan recall yang dibobotkan. F1 Score = 2 * (Recall*Precission) / (Recall + Precission).
 
 Bisnis:
 * Asumsikan bahwa pelanggan yang berlangganan akan memberikan keuntungan sebesar Rp.200.000,00 per bulan.
 * Promosi yang didapat oleh pelanggan yang diprediksi churn adalah sebesar Rp.50.000,00
 * Menghitung Laba yang diperoleh serta Cost yang dikeluarkan oleh perusahaan berdasarkan kedua model
 * Melilih model yang ekonomis namun dapat menghasilkan profit yang besar
 * Memberikan estimasi biaya yang dibutuhkan kepada tim Marketing untuk promosi dan estimasi laba yang akan diperoleh / harus dicapai setelah promosi diberikan  


### Conclusion
1. Apabila keputusan diambil berdasarkan besarnya nilai F1-Score, maka tuned model lah yang akan dipilih.

2. Sedangkan apabila keputusan diambil berdasarkan metrik bisnis, maka akan memilih model dengan laba yang besar serta cost yang seminimum mungkin.
Berdasarkan metrik dari bisnis tersebut, meskipun base model memiliki nilai F1-Score yang lebih kecil, namun laba yang dihasilkan lebih besar,yaitu sebesar Rp.191.450.000,00;  jika dibandingkan dengan model yang telah dituning menggunakan hyper parameters, yaitu hanya sebesar Rp.183.000.000,00. Selain itu cost yang digunakan oleh base model lebih sedikit, yaitu dengan estimasi sebesar Rp. 17.150.000,00, sedangkan cost yang dibutuhkan oleh tuning model adalah dengan estimasi sebesar Rp.28.800.000  Sehingga dapat disimpulkan:
Perusahaan akan memberikan estimasi biaya untuk promotion sebesar  Rp.17.000.000 kepada team marketing. Dengan estimasi keuntungan sebesar Rp. 191.450.000,00.

# External Resources

- Reference: [Algoritma Book: ML Application in Industry](https://algoml-industry.netlify.app/)
- Dataset: [Kaggle: Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn)
