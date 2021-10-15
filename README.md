# Predictive-Analysist-Customer-Churn-Analysist
Repository ini sebagai bagian dalam submission untuk lanjut di kelas Machine Learning Terapan di Dicoding

# Bussiness Understanding

Kita pasti pernah merasa kurang puas dengan sebuah perusahaan telekomunikasi dan akhirnya memutuskan pindah ke perusahaan lain, bukan? Entah karena harganya terlalu mahal, sinyalnya yang kurang bagus, atau karena pelayanannya yang kurang baik. Nah hal itu disebut dengan *Customer Churn*.

***Customer churn*** didefinisikan sebagai kecenderungan pelanggan untuk berhenti melakukan interaksi dengan sebuah perusahaan. Perusahaan telekomunikasi memiliki kebutuhan untuk mengetahui apakah pelanggan akan berhenti berlangganan atau tidak, karena biaya untuk mempertahankan pelanggan yang sudah ada jauh lebih sedikit dibandingkan memperoleh pelanggan baru.

Perusahaan biasanya mendefinisikan 2 tipe *customer churn*, yaitu *voluntary* dan *involuntary*. ***Voluntary churn*** merupakan pelanggan yang dengan sengaja berhenti dan beralih ke perusahaan lain, sedangkan ***involuntary churn*** merupakan pelanggan yang berhenti karena sebab eksternal seperti berpindah lokasi, kematian, atau alasan lainnya.

Diantara kedua tipe tersebut, *voluntary churn* lah yang tidak sulit untuk dilakukan karena kita dapat mempelajari karakteristik pelanggan yang dapat dilihat dari profil pelanggan. Permasalahan ini dapat dijawab dengan membuat sebuah model *Machine Learning* yang dapat memprediksi apakah seorang pelanggan akan *churn* atau tidak. Harapannya, dengan adanya model ini, pihak perusahaan telekomunikasi dapat melakukan tindak preventif bagi pelanggan yang berpeluang besar untuk *churn*.

# Workflow

## Understanding Data

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

## Import Data

Data yang digunakan merupakan data profil pelanggan dari sebuah perusahaan telekomunikasi yang diperoleh dari [Kaggle](https://www.kaggle.com/blastchar/telco-customer-churn). Dataset tersebut berisikan data untuk 7043 pelanggan yang meliputi demografis pelanggan, informasi pembayaran akun, serta produk layanan yang didaftarkan oleh tiap pelanggan. Dari informasi tersebut, kita ingin memprediksi apakah seorang pelanggan akan `Churn` atau tidak.

## menginstall library pandas profiling untuk Exploratory Data Analysis
```
!pip install pandas-profiling==2.7.1
```
## import Library yang akan digunakan
```
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pandas_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
```
## Membaca Dataset
```
dataset = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
dataset
```
## EDA dengan ProfileReport
```
ProfileReport(dataset)
```

```
dataset.info()
```
## Data Cleansing

Sebelum masuk ke tahap modeling, kita bersihkan datanya terlebih dahulu.

Mengganti yang kolom kosong dengan np.nan dan mengganti tipe kolom Senior Citizen menjadi tipe kolom categorical 

```
dataset['TotalCharges'] = dataset['TotalCharges'].replace(' ', np.nan)
dataset['TotalCharges'] = dataset['TotalCharges'].astype('float64')
# konversi ke float64
dataset['SeniorCitizen'] = dataset['SeniorCitizen'].astype('object')
dataset = dataset.drop(dataset.columns[0], axis=1)
```

```
dataset.describe()
```
Cek kelengkapan data, dari tahap ini kita akan memperoleh informasi apakah data kita sudah lengkap.
```
tenure = (dataset.tenure == 0).sum()
print("Nilai 0 di kolom tenure ada: ", tenure)
```
cek kolom tenure yg terdapat nilai 0
```
dataset.loc[(dataset['tenure']==0)]
```
Mengambil dataset dengan kolom tenure bukan 0
```
dataset = dataset.loc[(dataset['tenure']!=0)]
dataset.shape
```
## Analisis Univariat
Visualisasikan kolom apakah terdapat outlier
```
sns.boxplot(x=dataset['tenure'])
```
Visualisasikan kolom apakah terdapat outlier
```
sns.boxplot(x=dataset['TotalCharges'])
```

```
numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
categorical_features = ['gender', 'SeniorCitizen', 'Partner', 'Dependents','PhoneService', 'MultipleLines', 'InternetService',
       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
       'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
       'PaymentMethod','Churn']
```
## Analisis Multivariat
Mengamati hubungan antar fitur numerik dengan fungsi pairplot()
```
sns.pairplot(dataset, diag_kind = 'kde')
```

```
plt.figure(figsize=(10, 8))
correlation_matrix = dataset.corr().round(2)
# annot = True to print the values inside the square
sns.heatmap(data=correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, )
plt.title("Correlation Matrix untuk Fitur Numerik ", size=20)
```
Konversi semua kolom bertipe categorical ke numerik
```
for column in dataset.columns:
	if dataset[column].dtype == np.number: continue
	# menerapkan label encoding untuk kolom bertipe categorical
	dataset[column] = LabelEncoder().fit_transform(dataset[column])
print(dataset.describe())
```

```
dataset.columns
```

## Train-Test Splitting

Setelah kita melakukan *data cleansing* dan eksplorasi data, tahap berikutnya adalah *train-test splitting* yaitu membagi data menjadi data *train* dan *test* dengan proporsi 80:20. Data *train* digunakan untuk membuat model sedangkan data *test* digunakan untuk mengevaluasi performa model.
```
X = dataset.drop(["Churn"],axis =1)
y = dataset["Churn"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 15102021)
```

Mengecek isi kolom
```
print(f'Total # of sample in whole dataset: {len(X)}')
print(f'Total # of sample in train dataset: {len(X_train)}')
print(f'Total # of sample in test dataset: {len(X_test)}')
```

Mengecek distribusi data 
```
dataset[numerical_features].hist(bins=50, figsize=(20,15))
plt.show()
```

Mengecek persentase data churn
```
print('Jumlah baris dan kolom dari x_train:', X_train.shape,'\nJumlah baris dan kolom dari y_train:', y_train.shape)
print('Prosentase Churn di data Training adalah:')
print(pd.Series(y_train).value_counts(normalize=True))
```

Menangani imbalance data, melakukan *upsampling*, yang artinya kita akan menyetarakan proporsi target variabel menjadi sama besar.
```
over_sampler = RandomOverSampler(random_state=42)
X_res, y_res = over_sampler.fit_resample(X_train, y_train)
print(f"Training target statistics: {Counter(y_res)}")
print(f"Testing target statistics: {Counter(y_test)}")
```

Cek proporsi isi churn setelah di treatment 
```
print('Jumlah baris dan kolom dari x_train:', X_res.shape,'\nJumlah baris dan kolom dari y_train:', y_res.shape)
print('Prosentase Churn di data Training adalah:')
print(pd.Series(y_res).value_counts(normalize=True))
```

Normalisasi data dengan standar scaller
```
scaler = StandardScaler()
scaler.fit(X_res)
X_res = scaler.transform(X_res) 
X_test = scaler.transform(X_test)
print(X_res)
print(X_test)
```
## Model Evaluation

Terakhir, mari kita uji model random forest yang telah kita buat ke data test. Pada kasus ini, kita ingin memperoleh nilai recall yang sebesar mungkin agar model kita dapat mendeteksi pelanggan yang sebenarnya Churn sebanyak-banyaknya.

Model selection dengan Logistic Regression standard
```
LR = LogisticRegression()
LR.fit(X_res, y_res)
```

Model selection dengan Gradient Boosting Classifier standard
```
gbc = GradientBoostingClassifier()
gbc.fit(X_res, y_res)
```

Model selection dengan Random Forest Classifier standard
```
rfc = RandomForestClassifier()
rfc.fit(X_res, y_res)
```

Cek akurasi train dan test
```
mse = pd.DataFrame(columns=['train', 'test'], index=['Gradient Boosting Classifie', 'Logistic Regression', 'Random Forest Classifier'])
model_dict = {'Gradient Boosting Classifie': gbc, 'Logistic Regression': LR, 'Random Forest Classifier': rfc}
  
for name, model in model_dict.items():
    mse.loc[name, 'train'] = model.score(X_res, y_res)*100
    mse.loc[name, 'test'] = model.score(X_test, y_test)*100
 
mse
```

Visualisasikan akurasi model 
```
fig, ax = plt.subplots()
mse.sort_values(by='test', ascending=False).plot(kind='barh', ax=ax, zorder=3)
ax.grid(zorder=0)
```

Memilih model RFC, kemudian cek precision,recall, f1-score dan support nya
```
y_test_pred = rfc.predict(X_test)
print('Classification Report Testing Model (Random Forest Classifier):')
print(classification_report(y_test, y_test_pred))
```

Tuning model RFC
```
rfc_tuned = RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight={},
                       criterion='entropy', max_depth=11, max_features='log2',
                       min_impurity_decrease=0.001,
                       min_samples_leaf=4, min_samples_split=10,
                       min_weight_fraction_leaf=0.0, n_estimators=180,
                       verbose=0)
rfc_tuned.fit(X_res,y_res)
```

Predict
```
y_train_pred = rfc_tuned.predict(X_test)
# Print classification report
print('Classification Report Training Model (Gradient Boosting):')
print(classification_report(y_test, y_train_pred))
```

Membuat confusion Matrix
```
confusion_matrix_tuned = pd.DataFrame((confusion_matrix(y_test, rfc_tuned.predict(X_test))), ('No churn', 'Churn'), ('No churn', 'Churn'))
confusion_matrix_tuned
```

Plot confusion matrix
```
plt.figure()
heatmap = sns.heatmap(confusion_matrix_tuned, annot=True, annot_kws={'size': 14}, fmt='d', cmap='YlGnBu')
heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)
heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)

plt.title('Confusion Matrix for Training Model\n(Random Forest Classifier)', fontsize=18, color='darkblue')
plt.ylabel('True label', fontsize=14)
plt.xlabel('Predicted label', fontsize=14)
plt.show()
```

# Conclusion

Dengan adanya model untuk memprediksi *customer churn*, pihak perusahaan telekomunikasi dengan mudah mengetahui pelanggan mana yang memiliki kecenderungan untuk *churn*. 

Dari sini, pihak *marketing* dapat melakukan promosi produk dengan sifat kontrak yang jangkanya lebih panjang sehingga para pelanggan  dapat bertahan lebih lama.

# External Resources

- Reference: [Algoritma Book: ML Application in Industry](https://algoml-industry.netlify.app/)
- Dataset: [Kaggle: Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn)
