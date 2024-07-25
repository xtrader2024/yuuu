from flask import Flask, render_template
import numpy as np
import ccxt
import talib
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

app = Flask(__name__)

# API anahtarları
API_KEY = 'TAAgJilKcF9LHg977hGa3fVXdd9TUv6EmaZu7YgkCa4f8aAcxT5lvRI1gkh8mvw2'
API_SECRET = 'Yw48JHkJu3dz0YpJrPJz9ektNHUvYZtNePTeQLzDAe0CRk33wyKbebyRV0q4xwJk'

# Binance API istemcisi
binance = ccxt.binance({
    'apiKey': API_KEY,
    'secret': API_SECRET
})

# Tüm sembolleri al
markets = binance.load_markets()
symbols = list(markets.keys())

# USDT ile eşleştirilmiş sembolleri al
usdt_pairs = [pair for pair in symbols if pair.endswith('/USDT')]


# Ayarlar
sma_time_period = 50
ema_time_period = 50
rsi_time_period = 14
macd_fast_period = 12
macd_slow_period = 26
macd_signal_period = 9
stoch_fastk_period = 14
stoch_slowk_period = 3
stoch_slowd_period = 3
bollinger_window = 20
sar_acceleration = 0.02
sar_maximum = 0.2
chakin_period = 14
kama_period = 10
atr_period = 14


def chakin_money_flow(highs, lows, closes, volumes, period):
    # Chakin Money Flow hesaplama
    high_low_diff = highs - lows
    high_low_diff[high_low_diff == 0] = 0.000001  # sıfıra bölme hatası önleme
    money_flow_multiplier = ((closes - lows) - (highs - closes)) / high_low_diff
    money_flow_volume = money_flow_multiplier * volumes
    chakin_money_flow = np.zeros_like(closes)
    for i in range(1, len(closes)):
        chakin_money_flow[i] = np.sum(money_flow_volume[i - period:i])
    return chakin_money_flow

def kama(close_prices, period):
    # KAMA (Kaufman's Adaptive Moving Average) hesaplama
    change = np.abs(close_prices - np.roll(close_prices, period))
    volatility = np.sum(change)
    efficiency_ratio = change / volatility
    smoothing_constant = np.square(efficiency_ratio * (2 / (period + 1) - 1) + 1)
    kama = np.zeros_like(close_prices)
    kama[period - 1] = close_prices[period - 1]
    for i in range(period, len(close_prices)):
        kama[i] = kama[i - 1] + smoothing_constant[i] * (close_prices[i] - kama[i - 1])
    return kama

def atr(highs, lows, closes, period):
    # ATR (Average True Range) hesaplama
    tr1 = np.abs(highs - lows)
    tr2 = np.abs(highs - np.roll(closes, 1))
    tr3 = np.abs(np.roll(closes, 1) - np.roll(lows, 1))
    true_range = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = np.zeros_like(closes)
    atr[period - 1] = np.mean(true_range[:period])
    for i in range(period, len(closes)):
        atr[i] = (atr[i - 1] * (period - 1) + true_range[i]) / period
    return atr

def elliott_wave_oscillator(closes, highs, lows, periods):
    # Elliott Wave Oscillator'ı hesaplama
    ewo = np.zeros_like(closes)
    for i in range(periods, len(closes)):
        wave5 = closes[i] - closes[i - periods]
        wave3 = highs[i] - lows[i]
        ewo[i] = wave5 - wave3
    return ewo, wave5


@app.route('/')
def home():
    # Kripto para listesi ve fiyatları
    cryptos = {}

   

    # Tüm USDT çiftlerinin fiyatlarını çekme ve analiz etme
    for symbol in usdt_pairs:
        base, quote = symbol.split('/')

        try:
            # Kripto para fiyatını al
            ticker = binance.fetch_ticker(symbol)
            price = ticker.get('close')

            # Fiyat None ise, bu kripto para geç
            if price is None:
                print(f"Uyarı: {symbol} kripto para fiyatı alınamadı.")
                continue

            # Hareketli Ortalama (SMA ve EMA), RSI ve MACD hesaplamaları
            history = binance.fetch_ohlcv(symbol, timeframe='4h', limit=51)  # 51 veri alıyoruz, en son veriyi bugünkü fiyat hesaplamaları için kullanacağız
            if history is None or len(history) < 51:
                print(f"Uyarı: {symbol} kripto para verisi alınamadı veya yetersiz.")
                continue

            closes = np.array([ohlcv[4] for ohlcv in history[:-1]])  # Son veriyi hariç tutuyoruz
            highs = np.array([ohlcv[2] for ohlcv in history[:-1]])
            lows = np.array([ohlcv[3] for ohlcv in history[:-1]])
            volumes = np.array([ohlcv[5] for ohlcv in history[:-1]])

            sma_50 = talib.SMA(closes, timeperiod=sma_time_period)[-1]
            ema_50 = talib.EMA(closes, timeperiod=ema_time_period)[-1]

            # SMA ve EMA'nın None olup olmadığını kontrol et
            if np.isnan(sma_50) or np.isnan(ema_50):
                print(f"SMA veya EMA hesaplanamadı: {symbol}")
                continue

            rsi_14 = talib.RSI(closes, timeperiod=rsi_time_period)[-1]
            macd, signal, _ = talib.MACD(closes, fastperiod=macd_fast_period, slowperiod=macd_slow_period, signalperiod=macd_signal_period)
            stoch_k, stoch_d = talib.STOCH(high=highs, low=lows, close=closes, fastk_period=stoch_fastk_period, slowk_period=stoch_slowk_period, slowd_period=stoch_slowd_period)
            sar = talib.SAR(high=highs, low=lows, acceleration=sar_acceleration, maximum=sar_maximum)[-1]
            bollinger_upper, bollinger_middle, bollinger_lower = talib.BBANDS(closes, timeperiod=bollinger_window)
            chakin = chakin_money_flow(highs, lows, closes, volumes, chakin_period)
            kama_val = kama(closes, kama_period)
            atr_val = atr(highs, lows, closes, atr_period)
            
            # Elliott Wave Oscillator'ı hesapla
            ewo, wave5 = elliott_wave_oscillator(closes, highs, lows, periods=10)
            
            # Elliott Wave Oscillator'dan elde edilen verileri analiz sonuçlarına ekle
            wave5_starts = np.zeros_like(closes)
            wave5_ends = np.zeros_like(closes)
            for i in range(5, len(closes)):
                wave5_starts[i] = closes[i - 5]
                wave5_ends[i] = closes[i]

            # Pivot Points hesaplama
            pivot_point = (highs[-2] + closes[-2] + lows[-2]) / 3
            r1 = 2 * pivot_point - lows[-2]
            s1 = 2 * pivot_point - highs[-2]

            cryptos[base] = {'symbol': base, 'price': price, 'SMA_50': sma_50, 'EMA_50': ema_50, 'RSI_14': rsi_14,
                             'MACD': macd[-1], 'Signal': signal[-1], 'Stoch_K': stoch_k[-1], 'Stoch_D': stoch_d[-1],
                             'SAR': sar, 'Bollinger_Upper': bollinger_upper[-1], 'Bollinger_Middle': bollinger_middle[-1], 'Bollinger_Lower': bollinger_lower[-1],
                             'Pivot_Point': pivot_point, 'R1': r1, 'S1': s1,
                             'Chakin': chakin[-1], 'KAMA': kama_val[-1], 'ATR': atr_val[-1], 'EWO': ewo[-1],
                             'Wave5_Starts': wave5_starts[-1], 'Wave5_Ends': wave5_ends[-1]}
        except (ccxt.NetworkError, ccxt.ExchangeError, ccxt.BaseError) as e:
            print(f"Hata: {e}")
            continue

    # Veri çerçevesi oluşturma
    df = pd.DataFrame.from_dict(cryptos, orient='index')

    # Al sinyali veren coinleri saklamak için bir liste
    selected_coins = []

    # Tüm coinler için al sinyali testi
    for index, row in df.iterrows():
        # SMA ve EMA'nın None olup olmadığını kontrol et
        if np.isnan(row['SMA_50']) :
            print(f"Yetersiz veri: {index}")
            continue

        # Tüm sembollerin veri çerçevesinde bulunup bulunmadığını kontrol et
        if not all(df.loc[index].notnull()):
            print(f"Uyarı: {index} sembolü eksik verilere sahip.")
            continue

        if row['price'] > row['SMA_50'] and row['price'] > row['EMA_50']:
           
                # MACD al sinyali testi
                if row['MACD'] > row['Signal']:
                    # Stokastik Osilatör al sinyali testi
                    if row['Stoch_K'] > row['Stoch_D'] and row['Stoch_K'] > 20:
                        # Parabolic SAR al sinyali testi
                        if row['price'] > row['SAR']:
                            selected_coins.append((index, row['price'], row['SMA_50'], row['EMA_50'], row['RSI_14'], row['MACD'], row['Signal'], row['Stoch_K'], row['Stoch_D'], row['SAR'], row['Bollinger_Upper'], row['Bollinger_Middle'], row['Bollinger_Lower'], row['Pivot_Point'], row['R1'], row['S1'], row['Chakin'], row['KAMA'], row['ATR'], row['EWO'], row['Wave5_Starts'], row['Wave5_Ends']))

    # Beklenen artış yüzdesine göre seçilen coinleri sıralama
    selected_coins.sort(key=lambda x: float(x[3]), reverse=True)
    top_20_coins = selected_coins[:20]

    # Seçilen coinlerin analizini yapma ve yazdırma
    selected_coins_data = []
    for coin_data in top_20_coins:
        coin_name, price, sma_50, ema_50, rsi_14, macd, signal, stoch_k, stoch_d, sar, bollinger_upper, bollinger_middle, bollinger_lower, pivot_point, r1, s1, chakin, kama_val, atr_val, ewo, wave5_starts, wave5_ends = coin_data

        # Beklenen fiyatı hesapla
        expected_price = price * (1 + (price - sma_50) / sma_50)

        # Beklenen artış yüzdesini hesapla
        expected_increase_percentage = ((expected_price - price) / price) * 100 - 2  # Burada %1 eksik yazdırmak için -1 eklenmiştir


        # Dalga tahmini
        if ewo > 0:
            wave = "Yukarı"
        else:
            wave = "Aşağı"

        # Coin'in analiz sonuçlarını içeren bir sözlük oluştur
        coin_analysis = {
            'coin_name': coin_name,
            'price': f"{price:.8f}",
            'expected_price': f"{expected_price:.8f}",
            'expected_increase_percentage': f"{expected_increase_percentage:.2f}",
            'sma_50': f"{sma_50:.8f}",
            'ema_50': f"{ema_50:.8f}",
            'rsi_14': f"{rsi_14:.2f}",
            'macd': f"{macd:.8f}",
            'signal': f"{signal:.8f}",
            'stoch_k': f"{stoch_k:.2f}",
            'stoch_d': f"{stoch_d:.2f}",
            'sar': f"{sar:.8f}",
            'bollinger_upper': f"{bollinger_upper:.8f}",
            'bollinger_middle': f"{bollinger_middle:.8f}",
            'bollinger_lower': f"{bollinger_lower:.8f}",
            'pivot_point': f"{pivot_point:.8f}",
            'r1': f"{r1:.8f}",
            's1': f"{s1:.8f}",
            'chakin': f"{chakin:.8f}",
            'kama': f"{kama_val:.8f}",
            'atr': f"{atr_val:.8f}",
            'ewo': f"{ewo:.8f}",
            'wave5_starts': f"{wave5_starts:.8f}",
            'wave5_ends': f"{wave5_ends:.8f}",
            'wave': wave  # Yeni eklenen dalga bilgisi
        }
        selected_coins_data.append(coin_analysis)

    
    

    # BTC ve diğer belirlenen coinler için analiz sonuçlarının yazdırılması
    other_coins_data = []
    for coin in ['BTC', 'ETH', 'TAO', 'GALA', 'PENDLE', 'ADA', 'SOL', 'XRP', 'DOT', 'DOGE', 'BUSD', 'UNI', 'LINK', 'LTC', 'BCH', 'ALGO', 'TRX', 'ETC', 'XTZ', 'VET', 'EGLD', 'FIL', 'EOS', 'MATIC', 'IOTA', 'SUSHI', 'XLM', 'CAKE', 'AAVE', 'LUNA', 'ATOM', 'FTM', 'TRTL', 'COTI', 'DGB', 'VGX', 'HOT', 'RSR', 'BTT', 'WAVES', 'XDC', 'RVN', 'SAND', 'XEM', 'MANA', 'BAT', 'ZRX', 'KNC', 'IOST', 'WIN', 'ONE', 'ENA', 'WIF', 'JTO', 'UMA', 'RUNE', 'ALT', 'HIGH', 'INJ', 'PEPE', 'NOT', 'GRT', 'QTUM', 'OCEAN', 'LRC', 'SXP', 'TOMO', 'STMX', 'AEVO','ANKR', 'CHR', 'DODO', 'CELR', 'TWT', 'BLZ', 'BAND', 'MIR', 'STORJ', 'CVC', 'AKRO', '1INCH', 'SC', 'GTO', 'TFUEL', 'DATA', 'ONG', 'RLC', 'OXT', 'WRX', 'HARD', 'DIA', 'DUSK', 'LIT', 'RSR', 'AUDIO', 'BTS', 'YFI', 'CRV', 'PHA', 'SFP', 'API3', 'CTSI', 'MTL', 'OGN', 'FOR', 'LPT', 'STPT', 'BEL', 'NPXS', 'LINA', 'MDX', 'FLM', 'UNFI', 'NKN', 'YFII', 'UNI']:
        if coin in df.index:
            coin_analysis = df.loc[coin]
            expected_price_24h = coin_analysis['price'] * (1 + (coin_analysis['price'] - coin_analysis['SMA_50']) / coin_analysis['SMA_50'])
            coin_expected_increase = (coin_analysis['price'] - coin_analysis['SMA_50']) / coin_analysis['SMA_50'] * 100
            coin_data = {
                'coin_name': coin,
                'price': f"{coin_analysis['price']:.8f}",
                'expected_price_24h': f"{expected_price_24h:.8f}",
                'expected_increase_percentage': f"{coin_expected_increase:.2f}",
                'sma_50': f"{coin_analysis['SMA_50']:.8f}",
                'ema_50': f"{coin_analysis['EMA_50']:.8f}",
                'rsi_14': f"{coin_analysis['RSI_14']:.2f}",
                'macd': f"{coin_analysis['MACD']:.8f}",
                'signal': f"{coin_analysis['Signal']:.8f}",
                'stoch_k': f"{coin_analysis['Stoch_K']:.2f}",
                'stoch_d': f"{coin_analysis['Stoch_D']:.2f}",
                'sar': f"{coin_analysis['SAR']:.8f}",
                'bollinger_upper': f"{coin_analysis['Bollinger_Upper']:.8f}",
                'bollinger_middle': f"{coin_analysis['Bollinger_Middle']:.8f}",
                'bollinger_lower': f"{coin_analysis['Bollinger_Lower']:.8f}",
                'pivot_point': f"{coin_analysis['Pivot_Point']:.8f}",
                'r1': f"{coin_analysis['R1']:.8f}",
                's1': f"{coin_analysis['S1']:.8f}",
                'chakin': f"{coin_analysis['Chakin']:.8f}",
                'kama': f"{coin_analysis['KAMA']:.8f}",
                'atr': f"{coin_analysis['ATR']:.8f}",
                'ewo': f"{coin_analysis['EWO']:.8f}",
                'wave5_starts': f"{coin_analysis['Wave5_Starts']:.8f}",
                'wave5_ends': f"{coin_analysis['Wave5_Ends']:.8f}"
            }
            other_coins_data.append(coin_data)

    return render_template('index2.html', selected_coins_data=selected_coins_data, other_coins_data=other_coins_data, )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)


