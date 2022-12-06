import streamlit as st
import pandas as pd
import time
import timeit

import swifter
def text_preprocessing(data,step,column='column'):
    col1, col2 = st.columns(2)
    with col1 :
        st.subheader('Data sebelum di proses')
        st.write(data)
        if step == "filtering" :
            options = st.multiselect('Pilih proses',
            ['hapus angka', 'hapus single karakter', 'hapus tanda baca'],
            ['hapus tanda baca'])    
            start = timeit.default_timer()
            for option in options:
                if option == 'hapus angka':
                    data[column] = data[column].str.replace('\d+', '')
                if option == 'hapus single karakter':
                    data[column] = data[column].str.replace(r"\b[a-zA-Z]\b", "")
                if option == "hapus tanda baca":
                    data[column] = data[column].str.replace(r'[^\w\s]+', '')
            stop = timeit.default_timer()
        if step == "remove stopwords":
            language = st.selectbox("Pilih bahasa yang anda gunakan :",
            ('indonesian','english'))
        if step == 'stemming' :
            language = st.selectbox("Pilih bahasa yang anda gunakan :",
            ('indonesian','english'))
    with col2 :
        with st.spinner('tunggu sebentar ...'):
            time.sleep(2)
            if step == 'case fold':
                start = timeit.default_timer()
                data[column] = data[column].str.lower()
                stop = timeit.default_timer()
            if step == 'remove stopwords' :
                import nltk
                nltk.download('punkt')
                from nltk.tokenize import word_tokenize
                from nltk.corpus import stopwords
                nltk.download('stopwords')
                start = timeit.default_timer()
                data[column] = data[column].apply(word_tokenize)
                data[column] =data[column].apply(lambda x: [token for token in x if token not in stopwords.words(language)])
                stop = timeit.default_timer()
                data[column] = data[column].str.join(" ")
            if step == 'stemming' :
                if language == 'indonesian' :
                    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
                    factory = StemmerFactory()
                    stemmer = factory.create_stemmer()
                    data[column] = data[column].str.split()
                    start = timeit.default_timer()
                    data[column] = data[column].swifter.apply(lambda x: [stemmer.stem(y) for y in x])
                    stop = timeit.default_timer()
                    data[column] = data[column].str.join(" ")
                else :
                    from nltk.stem.snowball import SnowballStemmer
                    stemmer = SnowballStemmer(language)
                    data[column] = data[column].str.split()
                    start = timeit.default_timer()
                    data[column] = data[column].apply(lambda x: [stemmer.stem(y) for y in x])
                    stop = timeit.default_timer()
                    data[column] = data[column].str.join(" ")
            data.to_csv('data/data_branch.csv',index=False)
            data_branch = pd.read_csv('data/data_branch.csv')
        
            st.subheader('Data setelah di proses')
            st.write(data_branch)
            st.write('lama proses : ', stop-start,' detik')

            if st.button('simpan data'):
                data_branch.to_csv('data/main_data.csv',index=False)
                st.success('Berhasil disimpan')
def app():
    #Your statements here
    data = pd.read_csv('data/main_data.csv')
    column_data = pd.read_csv('data/meta/column_data.csv')
    column = column_data['column'][0]
    st.title('KLASIFIKASI KOMENTAR YOUTUBE')
    st.sidebar.markdown("lakukan preprosesing data")
    steps = st.sidebar.radio('Langkah langkah pre-prosessing : ',('case fold','filtering','remove stopwords','stemming'))
    st.header(f'Preprosesing - {steps}')
    if steps == 'case fold':
        text_preprocessing(data,'case fold',column)
    if steps == 'filtering':
        text_preprocessing(data,'filtering',column)
    if steps == "remove stopwords":
        text_preprocessing(data,'remove stopwords',column)
    if steps == 'stemming' :
        text_preprocessing(data, 'stemming',column)
