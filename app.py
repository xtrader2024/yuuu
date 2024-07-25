import streamlit as st
import pandas as pd

# Streamlit uygulaması başlık
st.title('TradingView Pine Script Sinyal Analizi')

# Webhook verilerini tutacak bir DataFrame oluşturalım
df_webhooks = pd.DataFrame(columns=['Date', 'Signal', 'Message'])

# Streamlit arayüzü
st.header('Webhook Verileri')
st.dataframe(df_webhooks)
