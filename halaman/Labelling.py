from pyparsing import col
import streamlit as st
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize 
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import swifter
import string 
import time
import os
import re


#lower
comment = pd.read_csv("data/data.csv",index_col=False)
#lower
comment['Komentar'] = comment['Komentar'].str.lower()

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
                
comment['Komentar'] = comment['Komentar'].apply(remove_special)

#remove number
def remove_number(text):
    return  re.sub(r"\d+", "", text)

comment['Komentar'] = comment['Komentar'].apply(remove_number)

#remove punctuation
def remove_punctuation(text):
    return text.translate(str.maketrans("","",string.punctuation))

comment['Komentar'] = comment['Komentar'].apply(remove_punctuation)
#remove whitespace leading & trailing
def remove_whitespace_LT(text):
    return text.strip()

comment['Komentar'] = comment['Komentar'].apply(remove_whitespace_LT)


#remove multiple whitespace into single whitespace
def remove_whitespace_multiple(text):
    return re.sub('\s+',' ',text)
comment['Komentar'] = comment['Komentar'].apply(remove_whitespace_multiple)

# remove single char
def remove_singl_char(text):
    return re.sub(r"\b[a-zA-Z]\b", " ", text)

comment['Komentar'] = comment['Komentar'].apply(remove_singl_char)

nltk.download('punkt')
# NLTK word Tokenize

# NLTK word Tokenize 
def word_tokenize_wrapper(text):
    return word_tokenize(text)

comment['Komentar'] = comment['Komentar'].apply(word_tokenize_wrapper)

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

comment['Komentar'] = comment['Komentar'].apply(stopwords_removal)


from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import swifter


# create stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# stemmed
def stemmed_wrapper(term):
    return stemmer.stem(term)

term_dict = {}

for document in comment['Komentar']:
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

comment['Komentar'] = comment['Komentar'].swifter.apply(get_stemmed_term)

positive = pd.read_csv("https://raw.githubusercontent.com/Fahrur190125/Data/main/positive.csv")
positive.to_csv('data/lexpos.csv',index=False)
negative = pd.read_csv("https://raw.githubusercontent.com/Fahrur190125/Data/main/negative.csv")
negative.to_csv('data/lexneg.csv',index=False)

lexicon_positive = dict()
import csv
with open('data/lexpos.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        lexicon_positive[row[0]] = int(row[1])

lexicon_negative = dict()
import csv
with open('data/lexneg.csv', 'r') as csvfile:
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


def app():
    st.title('KLASIFIKASI KOMENTAR YOUTUBE')
    st.header('Memberi Label Pada Data')
    st.markdown('Setelah didapat dataset hasil dari proses crawling yang berupa komentar publik terhadap video youtube, selanjutnya memberikan label terhadap data yang telah diperoleh. Pemberian label pada dataset dapat dilkukan secara manual maupun otomatis, untuk lebih jelasnya seperti berikut.')
    st.subheader('Memberi Label Secara Manual')
    st.markdown('Pemberian label secara manual dapat dilakukan dengan mendownload dataset hasil crawling dan buka dataset tersebut di excel dan berikan label pada setiap data yang terdapat pada dataset. Kemudian dataset yang sudah diberikan label tersebut di upload kembali ke website ini pada inputan file berformat csv berikut.')
    data = st.file_uploader("upload data berlabel berformat csv", type=['csv'])
    if data is not None:
        try :
            dataframe = pd.read_csv(data)
            st.write(dataframe)

            col1, col2 = st.columns(2)
            with col1 :
                column = st.selectbox("Pilih Kolom yang akan di proses :",
                list(dataframe.columns))
            with col2 :
                label = st.selectbox("Pilih Kolom yang akan dijadikan label atau class :",
                list(dataframe.columns))
  
            column_data = pd.DataFrame(data={'column': [column], 'label': [label]})
            if st.button('simpan data') :
                column_data.to_csv('data/meta/column_data.csv',index=False)
                if os.path.exists("data/data_branch.csv"):
                    os.remove("data/data_branch.csv")
                if os.path.exists("data/tf_idf.csv"):
                    os.remove("data/tf_idf.csv")
                dataframe = dataframe[[column_data['column'][0],column_data['label'][0]]]
                dataframe.to_csv('data/main_data.csv',index=False)
                with st.spinner('tunggu sebentar ...'):
                    time.sleep(1)
                st.success('data berhasil disimpan')
                st.info('column ' + column_data['column'][0] + ' akan diproses')
                st.info('column ' + column_data['label'][0] + ' akan dijadikan label')
        except :
            st.error('error : periksa lagi inputan anda')
    st.subheader('Memberi Label Secara Otomatis')
    st.markdown('Pemberian label secara otomatis dapat dilakukan dengan menggunakan nilai polarity. Nilai polarity merupakan nilai yang menunjukkan apakah kata tersebut bernilai negatif atau positif ataupun netral. Nilai polarity didapatkan dengan menjumlahkan nilai dari setiap kata dataset yang menunjukkan bahwa kata tersebut bernilai positif atau negatif ataupun netral. Didalam satu kalimat atau data,nilai dari kata-kata didalam satu kalimat tersebut akan dijumlah sehingga akan didapatkan nilai atau skor polarity. Nilai atau skor tersebutlah yang akan menentukan kalimat atau data tersebut berkelas positif(pro) atau negatif(kontra) ataupun netral. Jika nilai polarity yang didapat lebih dari 0 maka kalimat atau data tersebut diberi label atau kelas pro. Jika nilai polarity yang didapat kurang dari 0 maka kalimat atau data tersebut diberi label atau kelas kontra. Sedangkan jika nilai polarity sama dengan 0 maka kalimat atau data tersebut diberi label netral. Untuk nilai polarity yang saya gunakan, saya mengambil nilai polarity dari github yang di dapat dari link github berikut https://github.com/fajri91/InSet. Pelabelalan dataset secara otomatis dapat dilihat dengan mengklik tombol berikut.')
    with st.expander("Hasil Pelabelan Otomatis"):
        colI, colII = st.columns(2)
        with colI:
            comments = pd.read_csv("data/data.csv",index_col=False)
            st.write('Data sebelum dilakukan pelabelalan otomatis')
            st.write(comments)
            column = st.selectbox("Pilih Kolom yang akan di proses :", list(comments.columns))
        with colII:
            results = comment['Komentar'].apply(sentiment_analysis_lexicon_indonesia)
            results = list(zip(*results))
            comment['label'] = results[1]
            datalabel = pd.concat([comments, comment['label']], axis=1)
            st.write('Data setelah dilakukan pelabelalan otomatis')
            st.write(datalabel)
            label = st.selectbox("Pilih Kolom yang akan dijadikan label atau class :", list(comment.columns))

        column_data = pd.DataFrame(data={'column': [column], 'label': [label]})
        if st.button('simpan data') :
            column_data.to_csv('data/meta/column_data.csv',index=False)
            if os.path.exists("data/data_branch.csv"):
                os.remove("data/data_branch.csv")
            if os.path.exists("data/tf_idf.csv"):
                os.remove("data/tf_idf.csv")
            dataframe = datalabel[[column_data['column'][0],column_data['label'][0]]]
            dataframe.to_csv('data/main_data.csv',index=False)
            with st.spinner('tunggu sebentar ...'):
                time.sleep(1)
            st.success('data berhasil disimpan')
            st.info('column ' + column_data['column'][0] + ' akan diproses')
            st.info('column ' + column_data['label'][0] + ' akan dijadikan label')