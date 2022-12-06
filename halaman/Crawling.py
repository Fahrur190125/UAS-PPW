import streamlit as st
import pandas as pd
import os
from googleapiclient.discovery import build
# isikan dengan api key Anda
api_key = 'AIzaSyBaM_0Q-FXvN2nfsWVqOLeO0ztdT2ovP3Q'
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

def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv(index=False).encode('utf-8')

def app():
    st.header('Crawling Komentar Youtube')
    st.markdown('Proses crawling ini dilakukan untuk dapat mengambil dataset yang didapat dari komentar publik di youtube. Untuk dapat melakukan crawling pada youtube, terlebih dahulu import library yang akan digunakan, dan library yang akan dipakai yaitu google api client dari google dan httplib. Setelah itu lakukan proses crawling, namun sebelum melakukan crawling terlebih dahulu kita harus nemilki data api key youtube yang dapat diperoleh dengan langkah-langkah berikut.  \n1. Login ke Google Developer Console (https://console.developers.google.com/) dengan akun Google Anda  \n2. Buat project baru dan lengkapi isian yang diminta.  \n3. Aktifkan Layanan API pada halaman project, dan cari **Youtube Data API v3**.  \n4. Dari halaman dashboard, buat kredential agar API tersebut dapat digunakan. Klik tombol **Buat Kredensial** (**Create Credential**). Lengkapi isian formnya.  \n5. Anda dapat mengakses / melihat API KEY pada tab **Credentials**.  \nSetelah melakukan langkah-langkah tersebut kita akan memiliki data api key youtube. Setelah didapat kita copy agar dapat digunakan untuk mengcrawling komentar dari video youtube. Sebelumnya kita inputkan Video ID dari url youtube yang akan di crawling (contoh url video = https://youtu.be/tCrGtfDOQhA, maka ID Videonya = tCrGtfDOQhA). Disini saya mengambil contoh dari Video ID tersebut, jika teman teman ingin mengambil dari video lain maka inputkan Video ID nya pada inputan berikut.')

    # Enter video id
    # contoh url video = https://youtu.be/tCrGtfDOQhA
    video_id = st.text_input("Masukkan ID Video","tCrGtfDOQhA",placeholder='Masukkan ID Video Youtube') #isikan dengan kode / ID video
    # Call function
    if video_id != "" :
        comments = video_comments(video_id)
        df = pd.DataFrame(comments, columns=['Komentar'])
        if os.path.exists("data/data.csv"):
            os.remove("data/data.csv")
            df.to_csv('data/data.csv',index=False)
        comments = pd.read_csv("data/data.csv",index_col=False)
        st.success('Maka akan diperoleh hasil crawling komentar youtube seperti berikut.')
        st.write(comments)
    else:
        st.error('Inputkan ID Video Terlebih Dahulu.')

    csv = convert_df(df)
    st.download_button(
        label="Download data",
        data=csv,
        file_name='main_data.csv',
        mime='text/csv',
        )