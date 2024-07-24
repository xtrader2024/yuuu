import numpy as np
import ccxt
import talib
import pandas as pd
import streamlit as st

# Binance API keys (replace with your own keys)
API_KEY = 'TAAgJilKcF9LHg977hGa3fVXdd9TUv6EmaZu7YgkCa4f8aAcxT5lvRI1gkh8mvw2'
API_SECRET = 'Yw48JHkJu3dz0YpJrPJz9ektNHUvYZtNePTeQLzDAe0CRk33wyKbebyRV0q4xwJk'

# Binance API client
binance = ccxt.binance({
    'apiKey': API_KEY,
    'secret': API_SECRET
})

# Load all markets and filter pairs ending with USDT
markets = binance.load_markets()
symbols = list(markets.keys())
usdt_pairs = [pair for pair in symbols if pair.endswith('/USDT')]

# Parameters
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


def calculate_rsi(close_prices, period):
    # Relative Strength Index (RSI) calculation
    delta = np.diff(close_prices)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close_prices)
    avg_loss = np.zeros_like(close_prices)
    
    # Calculate initial average gain and loss
    avg_gain[period] = np.mean(gain[:period])
    avg_loss[period] = np.mean(loss[:period])
    
    # Calculate average gain and loss for the rest of the data
    for i in range(period + 1, len(close_prices)):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i - 1]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i - 1]) / period
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_sar(highs, lows, close_prices, sar_acceleration, sar_maximum):
    # SAR (Stop and Reverse) calculation
    sar = np.zeros_like(close_prices)
    af = sar_acceleration
    ep = highs[0]
    sar[0] = lows[0]
    
    for i in range(1, len(close_prices)):
        if sar[i - 1] < close_prices[i - 1]:
            sar[i] = sar[i - 1] + af * (ep - sar[i - 1])
            sar[i] = min(sar[i], lows[i])
            if highs[i] > ep:
                ep = highs[i]
                af = min(af + sar_acceleration, sar_maximum)
        else:
            sar[i] = sar[i - 1] - af * (sar[i - 1] - ep)
            sar[i] = max(sar[i], highs[i])
            if lows[i] < ep:
                ep = lows[i]
                af = min(af + sar_acceleration, sar_maximum)
    
    return sar


def chakin_money_flow(highs, lows, closes, volumes, period):
    # Chakin Money Flow calculation
    high_low_diff = highs - lows
    high_low_diff[high_low_diff == 0] = 0.000001  # Prevent division by zero error
    money_flow_multiplier = ((closes - lows) - (highs - closes)) / high_low_diff
    money_flow_volume = money_flow_multiplier * volumes
    chakin_money_flow = np.zeros_like(closes)
    for i in range(1, len(closes)):
        chakin_money_flow[i] = np.sum(money_flow_volume[i - period:i])
    return chakin_money_flow


def kama(close_prices, period):
    # KAMA (Kaufman's Adaptive Moving Average) calculation
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
    # ATR (Average True Range) calculation
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
    # Elliott Wave Oscillator calculation
    ewo = np.zeros_like(closes)
    for i in range(periods, len(closes)):
        wave5 = closes[i] - closes[i - periods]
        wave3 = highs[i] - lows[i]
        ewo[i] = wave5 - wave3
    return ewo


def fetch_and_analyze_data(symbol, placeholder):
    base, quote = symbol.split('/')
    
    try:
        # Get cryptocurrency price
        ticker = binance.fetch_ticker(symbol)
        price = ticker.get('close')

        # If price is None, skip this cryptocurrency
        if price is None:
            placeholder.warning(f"Warning: Price data for {symbol} could not be retrieved.")
            return None

        # Calculate Moving Averages (SMA and EMA), RSI, and MACD
        history = binance.fetch_ohlcv(symbol, timeframe='4h', limit=51)  # Fetch 51 data points, excluding the latest one for today's price calculations
        if history is None or len(history) < 51:
            placeholder.warning(f"Warning: Data for {symbol} could not be retrieved or is insufficient.")
            return None

        closes = np.array([ohlcv[4] for ohlcv in history[:-1]])  # Exclude the latest data point
        highs = np.array([ohlcv[2] for ohlcv in history[:-1]])
        lows = np.array([ohlcv[3] for ohlcv in history[:-1]])
        volumes = np.array([ohlcv[5] for ohlcv in history[:-1]])

        sma_50 = talib.SMA(closes, timeperiod=sma_time_period)[-1]
        ema_50 = talib.EMA(closes, timeperiod=ema_time_period)[-1]

        # Check if SMA or EMA are NaN
        if np.isnan(sma_50) or np.isnan(ema_50):
            placeholder.warning(f"SMA or EMA calculation failed: {symbol}")
            return None

        rsi_14 = calculate_rsi(closes, rsi_time_period)[-1]
        macd, signal, _ = talib.MACD(closes, fastperiod=macd_fast_period, slowperiod=macd_slow_period, signalperiod=macd_signal_period)
        stoch_k, stoch_d = talib.STOCH(high=highs, low=lows, close=closes, fastk_period=stoch_fastk_period, slowk_period=stoch_slowk_period, slowd_period=stoch_slowd_period)
        sar = calculate_sar(highs, lows, closes, sar_acceleration, sar_maximum)
        bollinger_upper, bollinger_middle, bollinger_lower = talib.BBANDS(closes, timeperiod=bollinger_window)
        chakin = chakin_money_flow(highs, lows, closes, volumes, chakin_period)
        kama_val = kama(closes, kama_period)
        atr_val = atr(highs, lows, closes, atr_period)

        # Calculate Elliott Wave Oscillator
        ewo = elliott_wave_oscillator(closes, highs, lows, periods=10)
        
        # Calculate expected price
        expected_price = price * (1 + (price - sma_50) / sma_50)

        # Calculate expected increase percentage
        expected_increase_percentage = ((expected_price - price) / price) * 100 - 1  # Subtracting 1

        # Check MACD signal and expected increase percentage
        if macd[-1] > signal[-1] and expected_increase_percentage >= 5:
            return {
                'symbol': base,
                'price': price,
                'expected_price': expected_price,
                'expected_increase_percentage': expected_increase_percentage,
                'SMA_50': sma_50,
                'EMA_50': ema_50,
                'RSI_14': rsi_14,
                'MACD': macd[-1],
                'STOCH_K': stoch_k[-1],
                'STOCH_D': stoch_d[-1],
                'SAR': sar[-1],
                'BBANDS_upper': bollinger_upper[-1],
                'BBANDS_middle': bollinger_middle[-1],
                'BBANDS_lower': bollinger_lower[-1],
                'Chakin_Money_Flow': chakin[-1],
                'KAMA': kama_val[-1],
                'ATR': atr_val[-1],
                'Elliott_Wave_Oscillator': ewo[-1]
            }
        else:
            return None

    except Exception as e:
        st.error(f"Error processing {symbol}: {str(e)}")
        return None


def main():
    # Streamlit setup
    st.title('Cryptocurrency Analysis')
    placeholder = st.empty()

    # Iterate through all USDT pairs and fetch data
    results = []
    for symbol in usdt_pairs:
        data = fetch_and_analyze_data(symbol, placeholder)
        if data:
            results.append(data)

    # Display results
    if results:
        df = pd.DataFrame(results)
        st.dataframe(df)
    else:
        st.warning("No cryptocurrencies meet the criteria.")

if __name__ == "__main__":
    main()
