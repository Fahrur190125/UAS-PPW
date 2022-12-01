import streamlit as st
st.set_page_config(
   page_title="Klasifikasi Komentar Video Youtube",
   page_icon="🧊",
   layout="wide",
   initial_sidebar_state="expanded",
)
import pandas as pd
from googleapiclient.discovery import build
import nltk
import string 
import re
from nltk.tokenize import word_tokenize 
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import swifter
import csv
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

st.title('Klasifikasi Komentar Video Youtube', anchor=None)
""" Pada pembahasan kali ini akan membahas klasifikasi komentar publik di youtube dengan menggunakan  Youtube Data API dengan Algoritma atau Metode klasifikasi yang telah dipelajari pada Matkul Pencarian dan Penambangan Web yang ada pada library Scikit-Learn. 
Metode-metode yang akan digunakan yaitu :"""
"""## A. KNN (K-Nearest Neighbor)
K-Nearest Neighbor (KNN) merupakan salah satu metode yang digunakan
dalam menyelesaikan masalah pengklasifikasian. Prinsip KNN yaitu
mengelompokkan atau mengklasifikasikan suatu data baru yang belum diketahui
kelasnya berdasarkan jarak data baru itu ke beberapa tetangga (neighbor) terdekat.
Tetangga terdekat adalah objek latih yang memiliki nilai kemiripan terbesar atau
ketidakmiripan terkecil dari data lama. Jumlah tetangga terdekat dinyatakan
dengan k. Nilai k yang terbaik tergantung pada data. 
Nilai k umumnya ditentukan dalam jumlah ganjil (3, 5, 7) untuk
menghindari munculnya jumlah jarak yang sama dalam proses pengklasifikasian.
Apabila terjadi dua atau lebih jumlah kelas yang muncul sama maka nilai k
menjadi k – 1 (satu tetangga kurang), jika masih ada yang sama lagi maka nilai k
menjadi k – 2 , begitu seterusnya sampai tidak ditemukan lagi kelas yang sama
banyak. Banyaknya kelas yang paling banyak dengan jarak terdekat akan menjadi
kelas dimana data yang dievaluasi berada. Dekat atau jauhnya tetangga (neighbor)
biasanya dihitung berdasarkan jarak Euclidean (Euclidean Distance). Berikut
rumus pencarian jarak menggunakan rumus Euclidian :

$$d_i = \sqrt{\sum_{i=1}^{p}(x_2i-x_1i)^{2}}$$

dengan:

$x_1$ = sampel data

$x_2$ = data uji

i = variabel data


$d_i$ = jarak

p = dimensi data
""" 
st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/5/52/Map1NN.png/183px-Map1NN.png","Deskripsi KNN")
"""## B. Naive Bayes
Naive Bayes adalah teknik sederhana untuk membangun pengklasifikasi, model yang menetapkan label kelas ke instance masalah, direpresentasikan sebagai vektor nilai fitur , di mana label kelas diambil dari beberapa himpunan terbatas. Tidak ada satu algoritme untuk melatih pengklasifikasi semacam itu, tetapi keluarga algoritme berdasarkan prinsip umum: semua pengklasifikasi naif Bayes berasumsi bahwa nilai fitur tertentu tidak bergantung pada nilai fitur lainnya, mengingat variabel kelas. Misalnya, buah dapat dianggap sebagai apel jika berwarna merah, bulat, dan berdiameter sekitar 10 cm. Pengklasifikasi naif Bayes menganggap masing-masing fitur ini berkontribusi secara independen terhadap probabilitas bahwa buah ini adalah apel, terlepas dari kemungkinan apa pun.korelasi antara fitur warna, kebulatan, dan diameter.

Dalam banyak aplikasi praktis, estimasi parameter untuk model naive bayes menggunakan metode maximum likelihood ; dengan kata lain, seseorang dapat bekerja dengan model naif Bayes tanpa menerima probabilitas Bayesian atau menggunakan metode Bayesian apa pun."""
st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/4f/ROC_curves.svg/220px-ROC_curves.svg.png","Deskripsi Naive Bayes")
"""Terlepas dari desainnya yang naif dan asumsi yang tampaknya terlalu disederhanakan, pengklasifikasi naif Bayes telah bekerja cukup baik dalam banyak situasi dunia nyata yang kompleks. Pada tahun 2004, analisis masalah klasifikasi Bayesian menunjukkan bahwa ada alasan teoretis yang masuk akal untuk kemanjuran pengklasifikasi naif Bayes yang tampaknya tidak masuk akal. Namun, perbandingan komprehensif dengan algoritme klasifikasi lain pada tahun 2006 menunjukkan bahwa klasifikasi Bayes mengungguli pendekatan lain, seperti pohon yang diperkuat atau hutan acak .

Keuntungan dari naive bayes adalah hanya membutuhkan sejumlah kecil data pelatihan untuk mengestimasi parameter yang diperlukan untuk klasifikasi.
"""
"""## C. SVM(Support Vector Machine)
SVM adalah salah satu metode prediksi yang paling kuat, yang didasarkan pada kerangka pembelajaran statistik atau teori VC yang diusulkan oleh Vapnik (1982, 1995) dan Chervonenkis (1974). Diberikan satu set contoh pelatihan, masing-masing ditandai sebagai milik salah satu dari dua kategori, algoritma pelatihan SVM membangun sebuah model yang memberikan contoh baru untuk satu kategori atau yang lain, menjadikannya sebagai pengklasifikasi linier biner non- probabilistik (walaupun metode seperti Platt penskalaan ada untuk menggunakan SVM dalam pengaturan klasifikasi probabilistik). SVM memetakan contoh-contoh pelatihan ke titik-titik dalam ruang untuk memaksimalkan lebar celah antara kedua kategori tersebut. Contoh-contoh baru kemudian dipetakan ke dalam ruang yang sama dan diprediksi termasuk dalam kategori berdasarkan di sisi celah mana mereka jatuh.

Selain melakukan klasifikasi linier , SVM dapat secara efisien melakukan klasifikasi non-linier menggunakan apa yang disebut trik kernel , yang secara implisit memetakan masukannya ke dalam ruang fitur berdimensi tinggi."""
st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/7/72/SVM_margin.png/300px-SVM_margin.png","Deskripsi SVM")
"""Ketika data tidak diberi label, pembelajaran yang diawasi tidak dimungkinkan, dan diperlukan pendekatan pembelajaran yang tidak diawasi , yang berupaya menemukan pengelompokan alami data ke dalam kelompok, dan kemudian memetakan data baru ke kelompok yang terbentuk ini. 
"""
"""## D. Bagging Classification
Bagging merupakan metode yang dapat memperbaiki hasil dari algoritma klasifikasi machine learning dengan menggabungkan klasifikasi prediksi dari beberapa model. Hal ini digunakan untuk mengatasi ketidakstabilan pada model yang kompleks dengan kumpulan data yang relatif kecil. Bagging adalah salah satu algoritma berbasis ensemble yang paling awal dan paling sederhana, namun efektif. Bagging paling cocok untuk masalah dengan dataset pelatihan yang relatif kecil. Bagging mempunyai variasi yang disebut Pasting Small Votes. cara ini dirancang untuk masalah dengan dataset pelatihan yang besar, mengikuti pendekatan yang serupa, tetapi membagi dataset besar menjadi segmen yang lebih kecil. Penggolong individu dilatih dengan segmen ini, yang disebut bites, sebelum menggabungkannya melalui cara voting mayoritas."""
st.image("https://upload.wikimedia.org/wikipedia/commons/6/6b/Bagging.png","Deskripsi Bagging")
"""Bagging mengadopsi distribusi bootstrap supaya menghasilkan base learner yang berbeda, untuk memperoleh data subset. sehingga melatih base learners. dan bagging juga mengadopsi strategi aggregasi output base leaner, yaitu metode voting untuk kasus klasifikasi dan averaging untuk kasus regresi. Untuk melakukan bagging pada data yang sudah di precocessing dengan menngunakan libary skikit learn seperti berikut.
"""
"""## E. Stacking Classification
Stacking merupakan cara untuk mengkombinasi beberapa model, dengan konsep meta learner. dipakai setelah bagging dan boosting. tidak seperti bagging dan boosting, stacking memungkinkan mengkombinasikan model dari tipe yang berbeda. Ide dasarnya adalah untuk train learner tingkat pertama menggunakan kumpulan data training asli, dan kemudian menghasilkan kumpulan data baru untuk melatih learner tingkat kedua, di mana output dari learner tingkat pertama dianggap sebagai fitur masukan sementara yang asli label masih dianggap sebagai label data training baru. Pembelajar tingkat pertama sering dihasilkan dengan menerapkan algoritma learning yang berbeda.
Dalam fase training pada stacking, satu set data baru perlu dihasilkan dari classifier tingkat pertama. Jika data yang tepat yang digunakan untuk melatih classifier tingkat pertama juga digunakan untuk menghasilkan kumpulan data baru untuk melatih classifier tingkat kedua. proses tersebut memiliki risiko yang tinggi yang akan mengakibatkan overfitting. sehingga disarankan bahwa contoh yang digunakan untuk menghasilkan kumpulan data baru dikeluarkan dari contoh data training untuk learner tingkat pertama, dan prosedur crossvalidasi."""
st.image("https://upload.wikimedia.org/wikipedia/commons/d/de/Stacking.png","Deskripsi Stacking")
"""## F Random Forest Classification
Random forest (RF) adalah suatu algoritma yang digunakan pada klasifikasi data dalam jumlah yang besar. Klasifikasi random forest dilakukan melalui penggabungan pohon (tree) dengan melakukan training pada sampel data yang dimiliki. Penggunaan pohon (tree) yang semakin banyak akan mempengaruhi akurasi yang akan didapatkan menjadi lebih baik. Penentuan klasifikasi dengan random forest diambil berdasarkan hasil voting dari tree yang terbentuk. Pemenang dari tree yang terbentuk ditentukan dengan vote terbanyak. Pembangunan pohon (tree) pada random forest sampai dengan mencapai ukuran maksimum dari pohon data. Akan tetapi,pembangunan pohon random forest tidak dilakukan pemangkasan (pruning) yang merupakan sebuah metode untuk mengurangi kompleksitas ruang. Pembangunan dilakukan dengan penerapan metode random feature selection untuk meminimalisir kesalahan. Pembentukan pohon (tree) dengan sample data menggunakan variable yang diambil secara acak dan menjalankan klasifikasi pada semua tree yang terbentuk. Random forest menggunakan Decision Tree untuk melakukan proses seleksi. Pohon yang dibangun dibagi secara rekursif dari data pada kelas yang sama. Pemecahan (split) digunakan untuk membagi data berdasarkan jenis atribut yang digunakan. Pembuatan decision tree pada saat penentuan klasifikasi,pohon yang buruk akan membuat prediksi acak yang saling bertentangan. Sehingga,beberapa decision tree akan menghasilkan jawaban yang baik. Random forest merupakan salah satu cara penerapan dari pendekatan diskriminasi stokastik pada klasifikasi. Proses Klasifikasi akan berjalan jika semua tree telah terbentuk.Pada saat proses klasifikasi selesai dilakukan, inisialisasi dilakukan dengan sebanyak data berdasarkan nilai akurasinya. Keuntungan penggunaan random forest yaitu mampu mengklasifiksi data yang memiliki atribut yang tidak lengkap,dapat digunakan untuk klasifikasi dan regresi akan tetapi tidak terlalu bagus untuk regresi, lebih cocok untuk pengklasifikasian data serta dapat digunakan untuk menangani data sampel yang banyak. Proses klasifikasi pada random forest berawal dari memecah data sampel yang ada kedalam decision tree secara acak. Setelah pohon terbentuk,maka akan dilakukan voting pada setiap kelas dari data sampel. Kemudian, mengkombinasikan vote dari setiap kelas kemudian diambil vote yang paling banyak.Dengan menggunakan random forest pada klasifikasi data maka, akan menghasilkan vote yang paling baik."""
st.image("https://upload.wikimedia.org/wikipedia/commons/7/76/Random_forest_diagram_complete.png","Deskripsi Random Forest Classication")

"""Untuk dapat mengklasifikasi komentar youtube tersebut dengan menggunakan metode yang dipaparkan diatas dapat dilakukan dengan melakukan tahapan-tahapan berikut.
1. Crawling
2. Prepocessing
3. Pemberian Label
4. TF(Term Frequncy)
5. Modelling
Setelah melalui tahapan-tahapan diatas maka kommentar publik di youtube dapat diketahui akurasi dari klasifikasi tersebut. Tahapan-tahapan tersebut akan dipaparkan lebih jelas seperti berikut."""
""" ## 1. Crawling
Proses crawling ini dilakukan untuk dapat mengambil dataset yang didapat dari komentar publik di youtube. Untuk dapat melakukan crawling pada youtube, terlebih dahulu import library yang akan digunakan, dan library yang akan dipakai yaitu google api client dari google dan httplib.
Setelah itu lakukan proses crawling, namun sebelum melakukan crawling terlebih dahulu kita harus nemilki data api key youtube yang dapat diperoleh dengan langkah-langkah berikut.
1. Login ke Google Developer Console (https://console.developers.google.com/) dengan akun Google Anda
2. Buat project baru dan lengkapi isian yang diminta.
3. Aktifkan Layanan API pada halaman project, dan cari **Youtube Data API v3**.
4. Dari halaman dashboard, buat kredential agar API tersebut dapat digunakan. Klik tombol **Buat Kredensial** (**Create Credential**). Lengkapi isian formnya.
5. Anda dapat mengakses / melihat API KEY pada tab **Credentials**.
Setelah melakukan langkah-langkah tersebut kita akan memiliki data api key youtube. Setelah didapat kita copy agar dapat digunakan untuk mengcrawling komentar dari video youtube. Sebelumnya kita inputkan Video ID dari url youtube yang akan di crawling (contoh url video = https://youtu.be/nhBpK69g8pM, maka Video ID = nhBpK69g8pM). Disini saya mengambil contoh dari Video ID tersebut, jika teman teman ingin mengambil dari video lain maka inputkan Video ID nya pada inputan berikut."""
def video_comments(video_id):
	# empty list for storing reply
	replies = []

	# creating youtube resource object
	youtube = build('youtube', 'v3', developerKey=api_key)

	# retrieve youtube video results
	video_response = youtube.commentThreads().list(part='snippet,replies', videoId=video_id).execute()

	# iterate video response
	while video_response:
		
		# extracting required info
		# from each result object
		for item in video_response['items']:

			# Extracting comments
			comment = item['snippet']['topLevelComment']['snippet']['textDisplay']


			replies.append([comment])
			
			# counting number of reply of comment
			replycount = item['snippet']['totalReplyCount']

			# if reply is there
			if replycount>0:
				# iterate through all reply
				for reply in item['replies']['comments']:
					
					# Extract reply
					repl = reply['snippet']['textDisplay']
					
					# Store reply is list
					#replies.append(reply)
					replies.append([ repl])

			# print comment with list of reply
			#print(comment, replies, end = '\n\n')

			# empty reply list
			#replies = []

		# Again repeat
		if 'nextPageToken' in video_response:
			video_response = youtube.commentThreads().list(
					part = 'snippet,replies',
					pageToken = video_response['nextPageToken'], 
					videoId = video_id
				).execute()
		else:
			break
	#endwhile
	return replies

	# isikan dengan api key Anda
api_key = 'AIzaSyBaM_0Q-FXvN2nfsWVqOLeO0ztdT2ovP3Q'

# Enter video id
# contoh url video = https://youtu.be/nhBpK69g8pM
video_id = st.text_input("Masukkan Video ID","tCrGtfDOQhA") #isikan dengan kode / ID video
"""Maka akan diperoleh hasil crawling komentar youtube seperti berikut."""

# Call function
if video_id != "" :
	comments = video_comments(video_id)
	df = pd.DataFrame(comments, columns=['Komentar'])
	df.to_csv('youtube-comments.csv', index=False)
	comments = pd.read_csv("youtube-comments.csv",index_col=False)
	comments

"""## 2. Prepocessing
Setelah proses crawling, selanjutnya lakukan prepocessing text, yaitu sebuah proses mesin yang digunakan untuk menyeleksi data teks agar lebih terstruktur dengan melalui beberapa tahapan-tahapan yang meliputi tahapan case folding, tokenizing, filtering dan stemming."""
""" ### a. Case Folding
Setelah berhasil mengambil dataset, selanjutnya proses prepocessing, tahapan case folding yaitu tahapan pertama untuk melakukan prepocessing text dengan mengubah text menjadi huruf kecil semua dengan menghilangkan juga karakter spesial, angka, tanda baca, spasi serta huruf yang tidak penting.
#### 1. Merubah Huruf Kecil Semua
Tahapan case folding yang pertama yaitu merubah semua huruf menjadi huruf kecil semua menggunakan fungsi lower().
#### 2. Menghapus Karakter Spesial
Tahapan case folding selanjutnya ialah menghapus karakter spesial dengan menggunakan library nltk.
#### 3. Menghapus Angka
Selanjutnya melakukan penghapusan angka, penghapusan angka disini fleksibel, jika angka ingin dijadikan fitur maka penghapusan angka tidak perlu dilakukan.
#### 4. Menghapus Tanda Baca
Selanjutnya penghapusan tanda baca yang tidak perlu yang dilakukan dengan function punctuation.
#### 5. Menghapus Spasi
Selanjutnya melakukan penghapusan spasi yang tidak dibutuhkan.
#### 6. Menghapus Huruf
Selanjutnya melakukan penghapusan huruf yang tidak bermakna.
"""
""" Setelah melakukan case folding dengan mengikuti langkah-langkah diatas maka dataset yang diperoleh setelah dilakukan case folding sebagai berikut. """
#lower
comments['Komentar'] = comments['Komentar'].str.lower()

#char spesial
#regex library
# import word_tokenize & FreqDist from NLTK
def remove_special(text):
    # remove tab, new line, ans back slice
    text = text.replace('\\t'," ").replace('\\n'," ").replace('\\u'," ").replace('\\'," ").replace('\\f'," ").replace('\\r'," ")
    # remove non ASCII (emoticon, chinese word, .etc)
    text = text.encode('ascii', 'replace').decode('ascii')
    # remove mention, link, hashtag
    text = ' '.join(re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)"," ", text).split())
    # remove incomplete URL
    return text.replace("http://", " ").replace("https://", " ")
                
comments['Komentar'] = comments['Komentar'].apply(remove_special)

#remove number
def remove_number(text):
    return  re.sub(r"\d+", "", text)

comments['Komentar'] = comments['Komentar'].apply(remove_number)

#remove punctuation
def remove_punctuation(text):
    return text.translate(str.maketrans("","",string.punctuation))

comments['Komentar'] = comments['Komentar'].apply(remove_punctuation)
#remove whitespace leading & trailing
def remove_whitespace_LT(text):
    return text.strip()

comments['Komentar'] = comments['Komentar'].apply(remove_whitespace_LT)


#remove multiple whitespace into single whitespace
def remove_whitespace_multiple(text):
    return re.sub('\s+',' ',text)
comments['Komentar'] = comments['Komentar'].apply(remove_whitespace_multiple)

# remove single char
def remove_singl_char(text):
    return re.sub(r"\b[a-zA-Z]\b", " ", text)

comments['Komentar'] = comments['Komentar'].apply(remove_singl_char)
comments['Komentar']
"""### b. Tokenizing
Setelah tahapan case folding selesai, selanjutnya masuk ke tahapan tokenizing yang merupakan tahapan prepocessing yang memecah kalimat dari text menjadi kata agar membedakan antara kata pemisah atau bukan. Untuk melakukan tokenizing dapat menggunakan dengan library nltk dan function untuk melakukan Tokenizing. Dan Hasil yang diperoleh dari tahapan Tokenizing sebagai berikut"""
nltk.download('punkt')
# NLTK word Tokenize

# NLTK word Tokenize 
def word_tokenize_wrapper(text):
    return word_tokenize(text)

comments['Komentar'] = comments['Komentar'].apply(word_tokenize_wrapper)
comments['Komentar']
"""### c. Filtering(Stopword)
Tahapan prepocessing selanjutnya ialah filtering atau disebut juga stopword yang merupakan lanjutan dari tahapan tokenizing yang digunakan untuk mengambil kata-kata penting dari hasil tokenizing tersebut dengan menghapus kata hubung yang tidak memiliki makna.
Proses stopword dapat dilakukan dengan mengimport library stopword dan sebuah function untuk dapat melakukan stopword. Hasil yang didapat setelah dilakukan stopword ialah sebagai berikut.
"""
nltk.download('stopwords')

list_stopwords = stopwords.words('indonesian')

# append additional stopword
list_stopwords.extend(["yg", "dg", "rt", "dgn", "ny", "d", 'klo', 
                       'kalo', 'amp', 'biar', 'bikin', 'bilang', 
                       'gak', 'ga', 'krn', 'nya', 'nih', 'sih', 
                       'si', 'tau', 'tdk', 'tuh', 'utk', 'ya', 
                       'jd', 'jgn', 'sdh', 'aja', 'n', 't', 
                       'nyg', 'hehe', 'pen', 'u', 'nan', 'loh', 'rt',
                       '&amp', 'yah'])

# convert list to dictionary
list_stopwords = set(list_stopwords)

#Menghapus Stopword dari list token
def stopwords_removal(words):
    return [word for word in words if word not in list_stopwords]

comments['Komentar'] = comments['Komentar'].apply(stopwords_removal)

comments['Komentar']
"""### d. Stemming
Tahapan terakhir dari proses prepocessing ialah stemming yang merupakan penghapusan suffix maupun prefix pada text sehingga menjadi kata dasar. Proses ini dapat dilakukan dengan menggunakan library sastrawi dan swifter. Dan hasil dari tahapan stemming yang merupakan tahapan akhir dari prepocessing text ialah sebagai berikut."""

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import swifter


# create stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# stemmed
def stemmed_wrapper(term):
    return stemmer.stem(term)

term_dict = {}

for document in comments['Komentar']:
    for term in document:
        if term not in term_dict:
            term_dict[term] = ' '
            
print(len(term_dict))
print("------------------------")

for term in term_dict:
    term_dict[term] = stemmed_wrapper(term)
    print(term,":" ,term_dict[term])
    
print(term_dict)
print("------------------------")


# apply stemmed term to dataframe
def get_stemmed_term(document):
    return [term_dict[term] for term in document]

comments['Komentar'] = comments['Komentar'].swifter.apply(get_stemmed_term)
comments['Komentar']
"""## 3. Pemberian Label

Setelah proses prepocesing selesai didapat sebuah dataset yang masih belum memiliki label, untuk itu pada tahapan ini dataset akan diberikan kelas atau label yang sesuai. Akan tetapi tahap pelabelan ini akan memerlukan waktu yang lama jika dilakukan secara manual. Untuk itu pada tahapan ini saya memberikan kelas atau label pada masing-masing data secara otomatis dengan menggunakan nilai polarity.

### a. Nilai Polarity
Nilai polarity merupakan nilai yang menunjukkan apakah kata tersebut bernilai negatif atau positif ataupun netral. Nilai polarity didapatkan dengan menjumlahkan nilai dari setiap kata dataset yang menunjukkan bahwa kata tersebut bernilai positif atau negatif ataupun netral.

Didalam satu kalimat atau data,nilai dari kata-kata didalam satu kalimat tersebut akan dijumlah sehingga akan didapatkan nilai atau skor polarity. Nilai atau skor tersebutlah yang akan menentukan kalimat atau data tersebut berkelas positif(pro) atau negatif(kontra) ataupun netral.

Jika nilai polarity yang didapat lebih dari 0 maka kalimat atau data tersebut diberi label atau kelas pro. Jika nilai polarity yang didapat kurang dari 0 maka kalimat atau data tersebut diberi label atau kelas kontra. Sedangkan jika nilai polarity sama dengan 0 maka kalimat atau data tersebut diberi label netral.

### b. Ambil Nilai Polarity
Sebelum melakukan pemberian label atau kelas dengan menggunakan nilai polarity, kita ambil nilai polarity dari setiap kata apakah positif atau negatif. Untuk itu saya mengambil nilai polarity dari github yang di dapat dari link github berikut https://github.com/fajri91/InSet
Nilai lexicon positif dan negatif yang didapat dari github tersebut saya download kemudian saya upload ke github saya dan kemudian saya ambil data lexicon positif dan negatif tersebut untuk digunakan sebagai menentukan sentimen positif, negatif, maupun netral.
"""

positive = pd.read_csv("https://raw.githubusercontent.com/Fahrur190125/Data/main/positive.csv")
positive.to_csv('lexpos.csv',index=False)
negative = pd.read_csv("https://raw.githubusercontent.com/Fahrur190125/Data/main/negative.csv")
negative.to_csv('lexneg.csv',index=False)

"""### c. Menentukan Kelas/Label dengan Nilai Polarity
Setelah berhasil mengambil nilai polarity lexicon positif dan negatif selanjutnya kita tentukan kelas dari masing masing data dengan menjumlahkan nilai polarity yang didapat dengan ketentuan jika lebih dari 0 maka memiliki kelas positif, jika kurang dari 0 maka diberi kelas negatif, dan jika sama dengan 0 maka memiliki kelas netral, dan hasilnya sebagai berikut.
"""
# Determine sentiment polarity of tweets using indonesia sentiment lexicon (source : https://github.com/fajri91/InSet)
# Loads lexicon positive and negative data
lexicon_positive = dict()
import csv
with open('lexpos.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        lexicon_positive[row[0]] = int(row[1])

lexicon_negative = dict()
import csv
with open('lexneg.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        lexicon_negative[row[0]] = int(row[1])
        
# Function to determine sentiment polarity of tweets        
def sentiment_analysis_lexicon_indonesia(text):
    #for word in text:
    score = 0
    for word in text:
        if (word in lexicon_positive):
            score = score + lexicon_positive[word]
    for word in text:
        if (word in lexicon_negative):
            score = score + lexicon_negative[word]
    polarity=''
    if (score > 0):
        polarity = 'positif'
    elif (score < 0):
        polarity = 'negatif'
    else:
        polarity = 'netral'
    return score, polarity

# Results from determine sentiment polarity of comments

results = comments['Komentar'].apply(sentiment_analysis_lexicon_indonesia)
results = list(zip(*results))
comments['polarity_score'] = results[0]
comments['label'] = results[1]
n = comments['label'].value_counts()
""" Jumlah masing masing sentimen dengan menggunakan polarity score"""
n
# Export to csv file
comments.to_csv('Prepocessing.csv',index=False)
""" Hasil pelabelan otomatis menggunakan polarity score """
comments

"""## 4. TF(Term Frequency)

Term Frequency(TF) merupakan banyaknya jumlah kemunculan term pada suatu dokumen. Untuk menghitung nilai TF terdapat beberapa cara, cara yang paling sederhana ialah dengan menghitung banyaknya jumlah kemunculan kata dalam 1 dokumen.
Sedangkan untuk menghitung nilai TF dengan menggunakan mesin dapat menggunakan library sklearn. Hasilnya seperti berikut.
"""

#Membuat Dataframe
dataTextPre = pd.read_csv('Prepocessing.csv',index_col=False)
dataTextPre.drop("polarity_score", axis=1, inplace=True)
vectorizer = CountVectorizer(min_df=1)
bag = vectorizer.fit_transform(dataTextPre['Komentar'])
dataTextPre

"""### Matrik VSM(Visual Space Model)
Sebelum menghitung nilai TF, terlebih dahulu buat matrik vsm untuk menentukan bobot nilai term pada dokumen, hasilnya sebagaii berikut.
"""

matrik_vsm = bag.toarray()
#print(matrik_vsm)
matrik_vsm.shape

matrik_vsm[0]


a=vectorizer.get_feature_names()

print(len(matrik_vsm[:,1]))
#dfb =pd.DataFrame(data=matrik_vsm,index=df,columns=[a])
dataTF =pd.DataFrame(data=matrik_vsm,index=list(range(1, len(matrik_vsm[:,1])+1, )),columns=[a])
dataTF.to_csv('TF.csv',index=False)

"""### Nilai Term Dokumen
Setelah didapat nilai matrik vsm, maka nilai term frequency yang didapat pada masing masing dokumen ialah seperti berikut.
"""

datalabel = pd.read_csv('Prepocessing.csv',index_col=False)
TF = pd.read_csv('TF.csv',index_col=False)
dataJurnal = pd.concat([TF, datalabel["label"]], axis=1)
dataJurnal

"""### Split Data
Selanjutnya split dataset menjadi data training dan testing. Atur size atau ukuran yang akan menjadi data testing berapa persen, untuk defaultnya saya atur 0,10 atau 10%, jika teman teman ingin mengubah sizenya, maka ubah pada inputan berikut.
"""

size = st.number_input("Masukkan size : ",0.10)
### Train test split to avoid overfitting
X_train,X_test,y_train,y_test=train_test_split(dataJurnal.drop(labels=['label'], axis=1),
    dataJurnal['label'],
    test_size=size,
    random_state=0)
"""Data Training"""
X_train
"""Data Testing""" 
X_test 

#KNN
KNN_range = []
KNN_score = []
for i in range (2,len(y_test)):
	classifier = KNeighborsClassifier(n_neighbors=i) 
	classifier.fit(X_train, y_train)
	y_pred = classifier.predict(X_test)
	score = classifier.score(X_test, y_test)
	KNN_range.append(i)
	KNN_score.append(score)

# Naive Bayes
# Mengaktifkan/memanggil/membuat fungsi klasifikasi Naive Bayes
modelnb = GaussianNB()

# Memasukkan data training pada fungsi klasifikasi Naive Bayes
nbtrain = modelnb.fit(X_train, y_train)

# Menentukan hasil prediksi dari x_test
#y_pred = nbtrain.predict(X_test)

nb_score = nbtrain.score(X_test, y_test)

#SVM
#Create a svm Classifier
svm = svm.SVC() # Linear Kernel

#Train the model using the training sets
svm.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = svm.predict(X_test)

# Model Accuracy: how often is the classifier correct?
svm_score = svm.score(X_test, y_test)


#Bagging
# load the data
X = X_train
Y = y_train
  
seed = 20
#kfold = model_selection.KFold(n_splits = 3,random_state = seed)
  
# initialize the base classifier
base_cls = DecisionTreeClassifier()

#Menyimpan Hasil Nilai Base Classifier
base_classifier=[]
bg_score=[]

for i in range (1, 50):
  # no. of base classifier
  num_trees = i
    
  # bagging classifier
  model = BaggingClassifier(base_estimator = base_cls,
                            n_estimators = num_trees,
                            random_state = seed)
    
  results = model_selection.cross_val_score(model, X, Y)

  #Nilai base classifier dan hasil nilai classifier disimpan dan akan ditampilkan di grafik
  base_classifier.append(i)
  bg_score.append(results.mean())

#Stacking
estimators = [
    ('rf', RandomForestClassifier(n_estimators=20, random_state=42),'rf1', RandomForestClassifier(n_estimators=20, random_state=42)),
    ('svr', make_pipeline(StandardScaler(), LinearSVC(random_state=42)))
]
clf = StackingClassifier(
    estimators=estimators, final_estimator=RandomForestClassifier()
)

st_score = clf.fit(X_train, y_train).score(X_test, y_test)

#Random Forest
#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)

# prediction on test set
y_pred=clf.predict(X_test)

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
rf_score = metrics.accuracy_score(y_test, y_pred)

"""## 5. Modelling
Selanjutnya setelah didapat data training dan testing, lakukan tahapan modelling dengan menggunakan metode atau algoritma yang paling optimal guna mencari nilai akurasi yang terbaik. Untuk menghasilkan akurasi yang terbaik, maka lakukan eksperimen atau percobaan dengan beberapa metode atau algoritma.
Pada pembahasan kali ini kita akan menggunakan metode yang dijelaskan di atas. Untuk melakukan eksperimen tersebut lakukan percobaan dengan metode yang disediakan seperti berikut."""

option = st.selectbox(
    'Pilih Metode atau Algoritma yang akan digunakan?',
    ('KNN (K-Nearest Neighbor)', 'Naive Bayes Classification', 'SVM (Support Vektor Macchine)', 'Bagging Classification', 'Stacking Classification', 'Random Forest Classification'))

st.write('Metode yang dipilih :', option)
if option == 'KNN (K-Nearest Neighbor)':
	st.write('Akurasi yang diperoleh :', max(KNN_score))
elif option == 'Naive Bayes Classification':
	st.write('Akurasi yang diperoleh :', nb_score)
elif option == 'SVM (Support Vektor Macchine)':
	st.write('Akurasi yang diperoleh :', svm_score)
elif option == 'Bagging Classification':
	st.write('Akurasi yang diperoleh :', max(bg_score))
elif option == 'Stacking Classification':
	st.write('Akurasi yang diperoleh :', st_score)
elif option == 'Random Forest Classification':
	st.write('Akurasi yang diperoleh :', rf_score)