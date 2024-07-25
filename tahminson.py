from flask import Flask, request, jsonify
import requests
import threading
import pandas as pd
import time

app = Flask(__name__)

# Webhook verilerini tutacak DataFrame
df_webhooks = pd.DataFrame(columns=['Date', 'Signal', 'Message'])

# Streamlit uygulamasının URL'si
STREAMLIT_URL = 'http://localhost:8501'  # Streamlit uygulamasının adresi

# Webhook endpoint'i
WEBHOOK_ENDPOINT = '/webhook'

# Streamlit'e webhook verilerini göndermek için HTTP POST isteği
def send_to_streamlit(data):
    requests.post(STREAMLIT_URL + WEBHOOK_ENDPOINT, json=data)

# Webhook endpoint'i
@app.route(WEBHOOK_ENDPOINT, methods=['POST'])
def webhook_receiver():
    data = request.json
    print('Received webhook:', data)
    update_dataframe(data)
    send_to_streamlit(data)
    return jsonify({'message': 'Webhook received'}), 200

# Webhook verilerini DataFrame'e eklemek için fonksiyon
def update_dataframe(data):
    global df_webhooks
    df_webhooks = df_webhooks.append({
        'Date': pd.Timestamp.now(),
        'Signal': data.get('title', ''),
        'Message': data.get('message', '')
    }, ignore_index=True)

# Webhook dinleme işlemi başlatılıyor
def start_webhook_listener():
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

# Streamlit verilerini gönderme işlemi
def send_streamlit_data():
    while True:
        # DataFrame'i Streamlit'e gönder
        try:
            requests.post(STREAMLIT_URL + '/postdata', json=df_webhooks.to_dict(orient='records'))
        except requests.exceptions.ConnectionError:
            print('Connection error. Trying again in 5 seconds...')
            time.sleep(5)
        time.sleep(1)  # Her saniye kontrol et

# Streamlit verilerini gönderme işlemi başlatılıyor
thread_streamlit = threading.Thread(target=send_streamlit_data)
thread_streamlit.start()

# Webhook dinleme işlemi başlatılıyor
if __name__ == '__main__':
    thread_listener = threading.Thread(target=start_webhook_listener)
    thread_listener.start()
