import streamlit as st
import pandas as pd
import joblib 
from google_play_scraper import Sort, reviews
import os
from datetime import datetime

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Analisis Ojol Pro v4.0", layout="wide")

# --- FUNGSI LOAD MODEL ---
@st.cache_resource
def load_all_models():
    path = 'models/'
    try:
        m_s = joblib.load(path + 'model_sentimen.pkl')
        t_s = joblib.load(path + 'tfidf_sentimen.pkl')
        m_k = joblib.load(path + 'model_kategori.pkl')
        t_k = joblib.load(path + 'tfidf_kategori.pkl')
        return m_s, t_s, m_k, t_k
    except:
        return None, None, None, None

m_s, t_s, m_k, t_k = load_all_models()

# --- UI UTAMA ---
st.title("🚖 Dashboard Analisis Ojol (Mode Full Year)")
st.info("Sistem akan menarik seluruh ulasan pada tahun yang dipilih tanpa batasan kedalaman data.")

col_app, col_year = st.columns([3, 1])
with col_app:
    dict_apps = {
        "Grab": "com.grabtaxi.passenger",
        "Gojek": "com.gojek.app",
        "Maxim": "com.taxsee.taxsee",
        "inDrive": "sinet.startup.inDriver"
    }
    selected_app_name = st.selectbox("Pilih Aplikasi:", list(dict_apps.keys()))
    app_id = dict_apps[selected_app_name]

with col_year:
    selected_year = st.selectbox("Pilih Tahun Target:", list(range(2026, 2019, -1)), index=2)

if st.button("🚀 Mulai Scraping Seluruh Tahun"):
    all_year_data = []
    continuation_token = None
    
    status_msg = st.empty()
    progress_bar = st.progress(0)
    
    # Flag untuk mengontrol loop
    stop_scraping = False
    batch_count = 0

    with st.spinner(f"Sedang mengunduh seluruh ulasan {selected_app_name} tahun {selected_year}..."):
        while not stop_scraping:
            batch_count += 1
            try:
                # Ambil data per 1000 (maksimal per tarikan)
                res, continuation_token = reviews(
                    app_id, lang='id', country='id', sort=Sort.NEWEST, 
                    count=1000, continuation_token=continuation_token
                )
                
                if not res: break
                
                df_batch = pd.DataFrame(res)
                df_batch['at'] = pd.to_datetime(df_batch['at'])
                
                # Filter ulasan yang masuk tahun target
                mask_target = df_batch['at'].dt.year == selected_year
                target_reviews = df_batch[mask_target]
                
                if not target_reviews.empty:
                    all_year_data.extend(target_reviews.to_dict('records'))
                
                # Cek ulasan tertua dalam batch ini
                min_date_in_batch = df_batch['at'].min()
                
                # Update status ke user
                status_msg.info(f"Telah memeriksa ulasan hingga tanggal: {min_date_in_batch.strftime('%Y-%m-%d')} | Terkumpul: {len(all_year_data)} data")

                # LOGIKA STOP:
                # Jika tanggal terkecil di batch sudah lebih lama dari tahun target, maka BERHENTI.
                if min_date_in_batch.year < selected_year:
                    stop_scraping = True
                
                # Jika token habis
                if not continuation_token:
                    stop_scraping = True
                    
            except Exception as e:
                st.error(f"Terjadi kendala teknis: {e}")
                break

    if all_year_data:
        df_final = pd.DataFrame(all_year_data).drop_duplicates(subset=['content'])
        
        # PROSES PREDIKSI
        if m_s:
            with st.spinner("Mengkategorikan data..."):
                texts = df_final['content'].fillna("")
                df_final['sentimen'] = m_s.predict(t_s.transform(texts))
                df_final['kategori'] = m_k.predict(t_k.transform(texts))
            
            # --- TAMPILAN STATISTIK ---
            st.divider()
            st.subheader(f"📊 Hasil Akhir Tahun {selected_year} ({len(df_final)} Ulasan)")
            
            total = len(df_final)
            
            # Persentase Kategori
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### 🏷️ Kategori")
                kat_counts = df_final['kategori'].value_counts()
                for kat, count in kat_counts.items():
                    st.write(f"**{kat.upper()}**: {count} ulasan ({ (count/total)*100 :.1f}%)")
            
            with c2:
                st.markdown("#### 😊 Sentimen")
                sent_counts = df_final['sentimen'].value_counts()
                for sent, count in sent_counts.items():
                    st.write(f"**{sent.upper()}**: {count} ulasan ({ (count/total)*100 :.1f}%)")
            
            st.divider()
            st.dataframe(df_final[['at', 'userName', 'content', 'sentimen', 'kategori']])
            st.download_button("Download Data Lengkap", df_final.to_csv(index=False).encode('utf-8'), f"{selected_app_name}_{selected_year}.csv")
    else:
        st.warning(f"Tidak ada data ditemukan untuk tahun {selected_year}.")