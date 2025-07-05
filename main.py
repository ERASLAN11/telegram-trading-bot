# GELİŞMİŞ TELEGRAM TRADING BOT - TEMIZ VERSİYON v3.0
import os
import sys
import json
import time
import asyncio
import statistics
from datetime import datetime, timedelta
import logging
import sqlite3
import pytz
import numpy as np
from pathlib import Path

# Temel kütüphaneler
try:
    import requests
    import numpy as np
except ImportError:
    print("📦 Temel kütüphaneler yükleniyor...")
    os.system("pip install requests numpy")
    import requests
    import numpy as np

# CCXT
try:
    import ccxt
except ImportError:
    print("📦 CCXT yükleniyor...")
    os.system("pip install ccxt")
    import ccxt

# Telegram
try:
    from telegram import Bot, Update
    from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
    from telegram.error import TelegramError
except ImportError:
    print("📦 Telegram bot yükleniyor...")
    os.system("pip install python-telegram-bot==20.7")
    from telegram import Bot, Update
    from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
    from telegram.error import TelegramError

# Scipy for signal processing
try:
    from scipy.signal import find_peaks
except ImportError:
    print("📦 Scipy yükleniyor...")
    os.system("pip install scipy")
    from scipy.signal import find_peaks

# === LOGGING ===
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('advanced_trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

print("🚀 GELİŞMİŞ TELEGRAM TRADING BOT - CLEAN v3.0")

# === AYARLAR ===
BOT_TOKEN = "7904118290:AAGFFLeGjG4s3VhnvuT8YcPL0_x1AvdGXBo"
CHAT_ID = "1075504620"

# Takip edilecek coinler
COINS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "DOGE/USDT",
    "SHIB/USDT", "XRP/USDT", "AVAX/USDT", "NEIRO/USDT"
]

# Bot ayarları
CHECK_INTERVAL = 180  # 3 dakika
DAILY_SIGNAL_LIMIT = 15  # Günde 15 sinyal

# Formation ağırlıkları
FORMATION_WEIGHTS = {
    'wyckoff': 2.0,      # En önemli - Kurumsal hareket
    'smc': 1.8,          # Çok önemli - Smart money
    'elliott': 1.5,      # Önemli - Market psychology
    'divergence': 1.3,   # Önemli - Early warning
    'volume_profile': 1.0, # Normal - Price acceptance
    'harmonic': 0.8      # Destekleyici - Mathematical
}

# Dinamik eşik ayarları
THRESHOLD_SETTINGS = {
    'STRONG_TREND': {'min_formations': 2, 'min_score': 4},
    'WEAK_TREND': {'min_formations': 3, 'min_score': 6},
    'SIDEWAYS': {'min_formations': 4, 'min_score': 8}
}

def detect_market_condition(ohlcv_1d, ohlcv_4h):
    # Market koşulunu tespit et
    try:
        if len(ohlcv_1d) < 10 or len(ohlcv_4h) < 20:
            return "WEAK_TREND"

        # Daily trend gücü
        daily_closes = [x[4] for x in ohlcv_1d[-10:]]
        daily_range = (max(daily_closes) - min(daily_closes)) / min(daily_closes)

        # 4H volatilite
        h4_closes = [x[4] for x in ohlcv_4h[-20:]]
        h4_volatility = np.std(h4_closes) / np.mean(h4_closes)

        # Market koşulu belirleme
        if daily_range > 0.15 and h4_volatility > 0.03:
            return "STRONG_TREND"
        elif daily_range < 0.05 and h4_volatility < 0.015:
            return "SIDEWAYS"
        else:
            return "WEAK_TREND"

    except Exception as e:
        return "WEAK_TREND"

def calculate_weighted_score(formations, market_condition):
    # Ağırlıklı skor hesapla
    try:
        weighted_total = 0
        for formation_name, formation_data in formations.items():
            weight = FORMATION_WEIGHTS.get(formation_name, 1.0)
            score = formation_data.get('score', 0)
            weighted_total += score * weight

        # Market koşuluna göre bonus/penalty
        if market_condition == "STRONG_TREND":
            weighted_total *= 1.1  # %10 bonus
        elif market_condition == "SIDEWAYS":
            weighted_total *= 0.9  # %10 azalt

        return weighted_total

    except Exception as e:
        return 0

def get_dynamic_threshold(market_condition):
    # Market koşuluna göre dinamik eşik
    return THRESHOLD_SETTINGS.get(market_condition, THRESHOLD_SETTINGS['WEAK_TREND'])

class SessionAnalyzer:
    def __init__(self):
        self.utc_tz = pytz.UTC

    def get_current_session(self):
        utc_now = datetime.now(self.utc_tz)
        hour = utc_now.hour

        if 0 <= hour < 9:
            return "ASIAN_SESSION"
        elif 9 <= hour < 17:
            return "LONDON_SESSION"
        elif 17 <= hour < 24:
            return "NEW_YORK_SESSION"
        else:
            return "TRANSITION"

    def get_kill_zone(self):
        utc_now = datetime.now(self.utc_tz)
        hour = utc_now.hour

        if 2 <= hour < 5:
            return "ASIAN_KILLZONE"
        elif 7 <= hour < 10:
            return "LONDON_KILLZONE"
        elif 12 <= hour < 15:
            return "NY_KILLZONE"
        elif 15 <= hour < 17:
            return "LONDON_CLOSE_KILLZONE"
        else:
            return "NO_KILLZONE"

    def get_session_multiplier(self):
        session = self.get_current_session()
        kill_zone = self.get_kill_zone()

        kill_zone_bonus = 1.3 if kill_zone != "NO_KILLZONE" else 1.0

        session_multipliers = {
            "ASIAN_SESSION": 1.0,
            "LONDON_SESSION": 1.2,
            "NEW_YORK_SESSION": 1.3,
            "TRANSITION": 0.8
        }

        return session_multipliers.get(session, 1.0) * kill_zone_bonus

class TechnicalAnalyzer:
    def __init__(self):
        pass

    def calculate_rsi(self, prices, period=14):
        if len(prices) < period + 1:
            return 50

        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains[:period]) if len(gains) > 0 else 0
        avg_loss = np.mean(losses[:period]) if len(losses) > 0 else 0

        if avg_loss == 0:
            return 100

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def analyze_wyckoff_phase(self, ohlcv_data, volumes):
        if len(ohlcv_data) < 50:
            return {'phase': 'UNKNOWN', 'score': 0}

        recent_data = ohlcv_data[-50:]
        recent_volumes = volumes[-50:]

        highs = [x[2] for x in recent_data]
        lows = [x[3] for x in recent_data]
        closes = [x[4] for x in recent_data]

        price_range = max(highs) - min(lows)
        avg_volume = statistics.mean(recent_volumes)
        recent_avg_volume = statistics.mean(recent_volumes[-10:])

        volume_increasing = recent_avg_volume > avg_volume * 1.2
        price_consolidating = price_range < (max(highs) * 0.08)
        price_trending_up = closes[-1] > closes[-20]
        price_trending_down = closes[-1] < closes[-20]

        if price_consolidating and volume_increasing:
            if closes[-1] < statistics.mean(closes):
                return {'phase': 'ACCUMULATION', 'score': 3}
            else:
                return {'phase': 'DISTRIBUTION', 'score': -3}
        elif price_trending_up and volume_increasing:
            return {'phase': 'MARKUP', 'score': 4}
        elif price_trending_down and volume_increasing:
            return {'phase': 'MARKDOWN', 'score': -4}
        else:
            return {'phase': 'CONSOLIDATION', 'score': 0}

    def analyze_smc_structure(self, ohlcv_data):
        if len(ohlcv_data) < 20:
            return {'structure': 'UNKNOWN', 'score': 0}

        highs = [x[2] for x in ohlcv_data[-20:]]
        lows = [x[3] for x in ohlcv_data[-20:]]

        recent_high = max(highs[-5:])
        previous_high = max(highs[-15:-5])
        recent_low = min(lows[-5:])
        previous_low = min(lows[-15:-5])

        if recent_high > previous_high and recent_low > previous_low:
            return {'structure': 'BULLISH_BOS', 'score': 3}
        elif recent_high < previous_high and recent_low < previous_low:
            return {'structure': 'BEARISH_BOS', 'score': -3}
        else:
            return {'structure': 'CONSOLIDATION', 'score': 0}

    def analyze_elliott_waves(self, closes):
        if len(closes) < 30:
            return {'current_wave': 'UNKNOWN', 'score': 0}

        # Basit trend analizi
        short_ma = np.mean(closes[-10:])
        long_ma = np.mean(closes[-30:])

        if short_ma > long_ma * 1.02:
            return {'current_wave': 'WAVE_3', 'score': 4}
        elif short_ma > long_ma:
            return {'current_wave': 'WAVE_1', 'score': 2}
        elif short_ma < long_ma * 0.98:
            return {'current_wave': 'WAVE_C', 'score': -3}
        else:
            return {'current_wave': 'WAVE_4', 'score': 1}

    def analyze_volume_profile(self, ohlcv_data):
        if len(ohlcv_data) < 50:
            return {'position': 'UNKNOWN', 'score': 0}

        prices = []
        volumes = []

        for candle in ohlcv_data[-50:]:
            high, low, volume = candle[2], candle[3], candle[5]
            mid_price = (high + low) / 2
            prices.append(mid_price)
            volumes.append(volume)

        # Basit volume analizi
        high_volume_price = prices[np.argmax(volumes)]
        current_price = ohlcv_data[-1][4]

        if current_price < high_volume_price * 0.95:
            return {'position': 'BELOW_VA', 'score': 2}
        elif current_price > high_volume_price * 1.05:
            return {'position': 'ABOVE_VA', 'score': -2}
        else:
            return {'position': 'AT_POC', 'score': 1}

    def analyze_momentum_divergence(self, ohlcv_data):
        if len(ohlcv_data) < 20:
            return {'divergences': [], 'score': 0}

        closes = [x[4] for x in ohlcv_data[-20:]]

        rsi_values = []
        for i in range(14, len(closes)):
            rsi = self.calculate_rsi(closes[:i+1])
            rsi_values.append(rsi)

        if len(rsi_values) < 6:
            return {'divergences': [], 'score': 0}

        # Basit divergence kontrolü
        price_trend = closes[-1] - closes[-6]
        rsi_trend = rsi_values[-1] - rsi_values[-6]

        if price_trend < 0 and rsi_trend > 0:
            return {'divergences': ['BULLISH_DIVERGENCE'], 'score': 3}
        elif price_trend > 0 and rsi_trend < 0:
            return {'divergences': ['BEARISH_DIVERGENCE'], 'score': -3}
        else:
            return {'divergences': [], 'score': 0}

    def analyze_harmonic_patterns(self, ohlcv_data):
        if len(ohlcv_data) < 20:
            return {'patterns': [], 'score': 0}

        closes = [x[4] for x in ohlcv_data[-20:]]

        # Basit ABCD pattern
        if len(closes) >= 4:
            a, b, c, d = closes[-20], closes[-15], closes[-10], closes[-1]

            # ABCD oranları kontrol et
            ab = abs(b - a)
            bc = abs(c - b) 
            cd = abs(d - c)

            if ab > 0 and bc > 0:
                bc_ab_ratio = bc / ab
                cd_bc_ratio = cd / bc if bc > 0 else 0

                # Fibonacci oranları (0.618, 0.786)
                if 0.6 <= bc_ab_ratio <= 0.8 and 1.2 <= cd_bc_ratio <= 1.6:
                    if d > a:  # Bullish pattern
                        return {'patterns': ['ABCD_BULLISH'], 'score': 2}
                    else:  # Bearish pattern
                        return {'patterns': ['ABCD_BEARISH'], 'score': -2}

        return {'patterns': [], 'score': 0}

class KeyLevelsCalculator:
    def __init__(self):
        pass

    def calculate_fibonacci_levels(self, high, low):
        if high <= low:
            return {}

        range_val = high - low
        return {
            'fib_23_6': high - (range_val * 0.236),
            'fib_38_2': high - (range_val * 0.382),
            'fib_50': high - (range_val * 0.5),
            'fib_61_8': high - (range_val * 0.618),
            'fib_78_6': high - (range_val * 0.786)
        }

    def calculate_pivot_points(self, high, low, close):
        pp = (high + low + close) / 3

        return {
            'pp': pp,
            'r1': 2 * pp - low,
            's1': 2 * pp - high,
            'r2': pp + (high - low),
            's2': pp - (high - low),
            'r3': high + 2 * (pp - low),
            's3': low - 2 * (high - pp)
        }

    def calculate_session_levels(self, ohlcv_data):
        if len(ohlcv_data) < 24:
            logger.warning("Yeterli veri yok")
            return {'daily': {}, 'weekly': {}, 'monthly': {}}

        # Günlük seviyeleri
        daily_data = ohlcv_data[-24:]
        daily_open = daily_data[0][1]
        daily_high = max([x[2] for x in daily_data])
        daily_low = min([x[3] for x in daily_data])
        daily_close = daily_data[-1][4]

        # Haftalık seviyeleri
        weekly_data = ohlcv_data[-168:] if len(ohlcv_data) >= 168 else ohlcv_data
        weekly_open = weekly_data[0][1]
        weekly_high = max([x[2] for x in weekly_data])
        weekly_low = min([x[3] for x in weekly_data])
        weekly_close = weekly_data[-1][4]

        # Aylık seviyeleri
        monthly_data = ohlcv_data[-720:] if len(ohlcv_data) >= 720 else ohlcv_data
        monthly_open = monthly_data[0][1]
        monthly_high = max([x[2] for x in monthly_data])
        monthly_low = min([x[3] for x in monthly_data])
        monthly_close = monthly_data[-1][4]

        return {
            'daily': {
                'open': daily_open, 'high': daily_high, 
                'low': daily_low, 'close': daily_close
            },
            'weekly': {
                'open': weekly_open, 'high': weekly_high, 
                'low': weekly_low, 'close': weekly_close
            },
            'monthly': {
                'open': monthly_open, 'high': monthly_high, 
                'low': monthly_low, 'close': monthly_close
            }
        }

class TradingSetupCalculator:
    def __init__(self):
        pass

    def calculate_entry_point(self, current_price, signal_direction):
        if signal_direction == "LONG":
            return current_price * 0.999  # %0.1 alt
        else:
            return current_price * 1.001  # %0.1 üst

    def calculate_stop_loss(self, entry_price, signal_direction, signal_type):
        if signal_type == "COUNTER_TREND":
            stop_distance = entry_price * 0.015  # %1.5
        else:
            stop_distance = entry_price * 0.025  # %2.5

        if signal_direction == "LONG":
            return entry_price - stop_distance
        else:
            return entry_price + stop_distance

    def calculate_take_profits(self, entry_price, stop_price, signal_direction, signal_type):
        risk = abs(entry_price - stop_price)

        if signal_type == "COUNTER_TREND":
            reward1 = risk * 1.5  # R:R 1:1.5
            reward2 = risk * 2.5  # R:R 1:2.5
        else:
            reward1 = risk * 2.0  # R:R 1:2
            reward2 = risk * 4.0  # R:R 1:4

        if signal_direction == "LONG":
            tp1 = entry_price + reward1
            tp2 = entry_price + reward2
        else:
            tp1 = entry_price - reward1
            tp2 = entry_price - reward2

        return tp1, tp2

    def suggest_leverage(self, signal_type, strength, session):
        base_leverage = {
            "COUNTER_TREND": {5: 8, 4: 6, 3: 4, 2: 3, 1: 2},
            "NORMAL_TREND": {5: 15, 4: 12, 3: 8, 2: 5, 1: 3}
        }

        suggested = base_leverage[signal_type][strength]

        session_multipliers = {
            "ASIAN_SESSION": 0.8,
            "LONDON_SESSION": 1.0,
            "NEW_YORK_SESSION": 1.1,
            "TRANSITION": 0.6
        }

        suggested = int(suggested * session_multipliers.get(session, 1.0))
        return max(2, min(20, suggested))

class DatabaseManager:
    def __init__(self, db_path="advanced_trading_bot.db"):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT,
                    signal_type TEXT,
                    signal_direction TEXT,
                    total_score REAL,
                    strength INTEGER,
                    entry_price REAL,
                    stop_loss REAL,
                    tp1 REAL,
                    tp2 REAL,
                    leverage INTEGER,
                    session TEXT,
                    kill_zone TEXT,
                    formations TEXT,
                    market_condition TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    username TEXT,
                    message TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            conn.commit()
            conn.close()
            logger.info("✅ Veritabanı başlatıldı")

        except Exception as e:
            logger.error(f"❌ Veritabanı hatası: {e}")

    def save_signal(self, signal_data):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO signals (
                    symbol, signal_type, signal_direction, total_score, strength,
                    entry_price, stop_loss, tp1, tp2, leverage, session, kill_zone, 
                    formations, market_condition
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                signal_data['symbol'], signal_data['signal_type'], 
                signal_data['signal_direction'], signal_data['total_score'], 
                signal_data['strength'], signal_data['entry_price'], 
                signal_data['stop_loss'], signal_data['tp1'], signal_data['tp2'],
                signal_data['leverage'], signal_data['session'], 
                signal_data['kill_zone'], signal_data['formations'],
                signal_data['market_condition']
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"❌ Sinyal kaydetme hatası: {e}")

    def save_conversation(self, user_id, username, message):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO conversations (user_id, username, message)
                VALUES (?, ?, ?)
            ''', (str(user_id), username or "Unknown", message))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"❌ Konuşma kaydetme hatası: {e}")

    def get_daily_stats(self):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            today = datetime.now().date()
            cursor.execute('''
                SELECT COUNT(*) FROM signals 
                WHERE DATE(timestamp) = ?
            ''', (today,))

            daily_count = cursor.fetchone()[0]
            conn.close()

            return {'daily_signals': daily_count}
        except Exception as e:
            logger.error(f"❌ İstatistik hatası: {e}")
            return {'daily_signals': 0}

class AdvancedTradingBot:
    def __init__(self):
        self.last_signals = {}
        self.daily_count = 0
        self.last_reset = datetime.now().date()
        self.db = DatabaseManager()
        self.session_analyzer = SessionAnalyzer()
        self.technical_analyzer = TechnicalAnalyzer()
        self.key_levels_calc = KeyLevelsCalculator()
        self.setup_calc = TradingSetupCalculator()
        self.application = None
        self.running = False

    def can_send_signal(self, symbol):
        if datetime.now().date() != self.last_reset:
            self.daily_count = 0
            self.last_reset = datetime.now().date()

        stats = self.db.get_daily_stats()
        daily_count = stats['daily_signals']

        if daily_count >= DAILY_SIGNAL_LIMIT:
            return False

        last_time = self.last_signals.get(symbol)
        if last_time:
            if (datetime.now() - last_time).seconds < 600:  # 10 dakika cooldown
                return False

        return True

    def signal_sent(self, symbol):
        self.last_signals[symbol] = datetime.now()
        self.daily_count += 1

def comprehensive_analysis(symbol, session_analyzer, technical_analyzer, key_levels_calc):
    # Kapsamlı teknik analiz
    try:
        exchange = ccxt.binance()

        # Multi-timeframe veri çek
        ohlcv_1h = exchange.fetch_ohlcv(symbol, '1h', limit=720)  # 30 gün
        ohlcv_4h = exchange.fetch_ohlcv(symbol, '4h', limit=180)  # 30 gün
        ohlcv_1d = exchange.fetch_ohlcv(symbol, '1d', limit=30)   # 30 gün
        ticker = exchange.fetch_ticker(symbol)

        if not all([ohlcv_1h, ohlcv_4h, ohlcv_1d]) or len(ohlcv_1h) < 50:
            return None

        # Temel veriler
        current_price = ticker['last']
        price_change_24h = ticker['percentage'] if ticker['percentage'] else 0

        # Volume analizi
        volumes_1h = [x[5] for x in ohlcv_1h]
        current_volume = volumes_1h[-1]
        avg_volume = statistics.mean(volumes_1h[-48:-1]) if len(volumes_1h) > 1 else volumes_1h[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1

        # RSI hesaplama (1h)
        closes_1h = [x[4] for x in ohlcv_1h]
        rsi = technical_analyzer.calculate_rsi(closes_1h)

        # Session analizi
        current_session = session_analyzer.get_current_session()
        kill_zone = session_analyzer.get_kill_zone()
        session_multiplier = session_analyzer.get_session_multiplier()

        # Multi-timeframe trend analizi
        closes_4h = [x[4] for x in ohlcv_4h]
        closes_1d = [x[4] for x in ohlcv_1d]

        # Daily trend
        daily_trend = "BULLISH" if closes_1d[-1] > closes_1d[-5] else "BEARISH"

        # Market koşulunu tespit et
        market_condition = detect_market_condition(ohlcv_1d, ohlcv_4h)

        # 6 Formasyon analizi
        wyckoff = technical_analyzer.analyze_wyckoff_phase(ohlcv_1h, volumes_1h)
        smc = technical_analyzer.analyze_smc_structure(ohlcv_4h)
        elliott = technical_analyzer.analyze_elliott_waves(closes_4h)
        harmonic = technical_analyzer.analyze_harmonic_patterns(ohlcv_4h)
        volume_profile = technical_analyzer.analyze_volume_profile(ohlcv_1h)
        divergence = technical_analyzer.analyze_momentum_divergence(ohlcv_1h)

        # Ağırlıklı skor hesapla
        formations_dict = {
            'wyckoff': wyckoff,
            'smc': smc,
            'elliott': elliott,
            'harmonic': harmonic,
            'volume_profile': volume_profile,
            'divergence': divergence
        }

        weighted_technical_score = calculate_weighted_score(formations_dict, market_condition)

        # Volume score
        if volume_ratio >= 4:
            volume_score = 3
        elif volume_ratio >= 3:
            volume_score = 2
        elif volume_ratio >= 2:
            volume_score = 1
        else:
            volume_score = 0

        # RSI score (LONG için)
        if rsi < 30:
            rsi_score = 4
        elif rsi < 35:
            rsi_score = 3
        elif rsi < 40:
            rsi_score = 2
        elif rsi < 50:
            rsi_score = 1
        else:
            rsi_score = 0

        # Total technical score
        total_technical_score = weighted_technical_score + volume_score

        # Dinamik eşik al
        threshold = get_dynamic_threshold(market_condition)

        # Pozitif/Negatif formation sayısı
        positive_formations = sum(1 for f in formations_dict.values() if f.get('score', 0) > 0)
        negative_formations = sum(1 for f in formations_dict.values() if f.get('score', 0) < 0)

# RSI oversold/overbought kontrolü (FILTER olarak)
        rsi_filter = ""
        if rsi < 30:
            rsi_filter = "OVERSOLD"  # LONG timing iyi
            rsi_score_boost = 2  # LONG için bonus
        elif rsi > 70:
            rsi_filter = "OVERBOUGHT"  # SHORT timing iyi  
            rsi_score_boost = -2  # SHORT için bonus (negatif)
        else:
            rsi_filter = "NEUTRAL"  # 30-70 arası nötr
            rsi_score_boost = 0

        # Total technical score hesapla
        total_technical_score = weighted_technical_score + volume_score + rsi_score_boost

        # Gelişmiş sinyal yönü belirleme (RSI filter ile)
        if (total_technical_score >= threshold['min_score'] and 
            positive_formations >= threshold['min_formations']):
            if rsi_filter == "OVERSOLD" or rsi_filter == "NEUTRAL":
                signal_direction = "LONG"
            else:
                return None  # OVERBOUGHT'ta LONG verme
        elif (total_technical_score <= -threshold['min_score'] and 
              negative_formations >= threshold['min_formations']):
            if rsi_filter == "OVERBOUGHT" or rsi_filter == "NEUTRAL":
                signal_direction = "SHORT"
            else:
                return None  # OVERSOLD'da SHORT verme
        else:
            return None  # Yetersiz sinyal

        # Signal type belirleme (counter-trend vs normal)
        if (daily_trend == "BEARISH" and signal_direction == "LONG") or \
           (daily_trend == "BULLISH" and signal_direction == "SHORT"):
            signal_type = "COUNTER_TREND"
        else:
            signal_type = "NORMAL_TREND"

        # Session multiplier uygula
        final_score = total_technical_score * session_multiplier
        strength = min(int(abs(final_score) / 4), 5)  # 1-5 arası

        if strength < 3:  # Minimum 3 yıldız
            return None

        # Key levels hesapla
        session_levels = key_levels_calc.calculate_session_levels(ohlcv_1h)

        # Trading setup hesapla
        setup_calc = TradingSetupCalculator()
        entry_price = setup_calc.calculate_entry_point(current_price, signal_direction)
        stop_loss = setup_calc.calculate_stop_loss(entry_price, signal_direction, signal_type)
        tp1, tp2 = setup_calc.calculate_take_profits(entry_price, stop_loss, signal_direction, signal_type)
        leverage = setup_calc.suggest_leverage(signal_type, strength, current_session)

        # Ana tetikleyicileri belirle
        main_triggers = []
        if abs(wyckoff['score']) >= 3:
            main_triggers.append(f"Wyckoff: {wyckoff['phase']} ({wyckoff['score']:+d})")
        if abs(smc['score']) >= 3:
            main_triggers.append(f"SMC: {smc['structure']} ({smc['score']:+d})")
        if abs(elliott['score']) >= 3:
            main_triggers.append(f"Elliott: {elliott['current_wave']} ({elliott['score']:+d})")
        if abs(divergence['score']) >= 3:
            div_name = divergence['divergences'][0] if divergence['divergences'] else 'MOMENTUM'
            main_triggers.append(f"Divergence: {div_name} ({divergence['score']:+d})")

        # Destekleyici faktörler
        supporting_factors = []
        if 1 <= abs(wyckoff['score']) < 3:
            supporting_factors.append(f"Wyckoff: {wyckoff['phase']}")
        if 1 <= abs(smc['score']) < 3:
            supporting_factors.append(f"SMC: {smc['structure']}")
        if volume_ratio >= 2:
            supporting_factors.append(f"Volume: {volume_ratio:.1f}x")
        if abs(harmonic['score']) >= 1:
            pattern_name = harmonic['patterns'][0] if harmonic['patterns'] else 'Pattern'
            supporting_factors.append(f"Harmonic: {pattern_name}")
        if abs(volume_profile['score']) >= 1:
            supporting_factors.append(f"Volume Profile: {volume_profile['position']}")

        return {
            'symbol': symbol,
            'signal_direction': signal_direction,
            'signal_type': signal_type,
            'strength': strength,
            'total_score': final_score,
            'current_price': current_price,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'tp1': tp1,
            'tp2': tp2,
            'leverage': leverage,
            'rsi': rsi,
            'volume_ratio': volume_ratio,
            'price_change_24h': price_change_24h,
            'session': current_session,
            'kill_zone': kill_zone,
            'daily_trend': daily_trend,
            'market_condition': market_condition,
            'main_triggers': main_triggers,
            'supporting_factors': supporting_factors,
            'formations': {
                'wyckoff': wyckoff,
                'smc': smc,
                'elliott': elliott,
                'harmonic': harmonic,
                'volume_profile': volume_profile,
                'divergence': divergence
            },
            'positive_formations': positive_formations,
            'negative_formations': negative_formations
        }

    except Exception as e:
        logger.error(f"❌ {symbol} analiz hatası: {e}")
        return None

def format_advanced_signal_message(data):
    # Gelişmiş sinyal mesajını formatla
    symbol = data['symbol']
    signal_direction = data['signal_direction']
    signal_type = data['signal_type']
    strength = data['strength']
    total_score = data['total_score']

    # Header belirleme
    if signal_type == "COUNTER_TREND":
        if signal_direction == "LONG":
            header = "🔄 KISA VADELİ LONG SİNYALİ ⚡"
            trend_warning = f"⚠️ ANA TREND: DÜŞÜŞ YÖNLÜ 📉"
        else:
            header = "🔄 KISA VADELİ SHORT SİNYALİ ⚡"
            trend_warning = f"⚠️ ANA TREND: YÜKSELIŞ YÖNLÜ 📈"
    else:
        if signal_direction == "LONG":
            header = "🚀 GÜÇLÜ LONG SİNYALİ 📈"
            trend_warning = "✅ ANA TREND: YÜKSELIŞ YÖNLÜ"
        else:
            header = "🚀 GÜÇLÜ SHORT SİNYALİ 📉"
            trend_warning = "✅ ANA TREND: DÜŞÜŞ YÖNLÜ"

    direction_emoji = "🟢" if signal_direction == "LONG" else "🔴"
    stars = "⭐" * strength

    # Risk-Reward hesaplama
    risk = abs(data['entry_price'] - data['stop_loss'])
    reward1 = abs(data['tp1'] - data['entry_price'])
    reward2 = abs(data['tp2'] - data['entry_price'])
    rr1 = reward1 / risk if risk > 0 else 0
    rr2 = reward2 / risk if risk > 0 else 0
    risk_percent = (risk / data['entry_price']) * 100

    message = f"""{header}
{trend_warning}

{direction_emoji} <b>{signal_direction} - {symbol}</b>
💰 Fiyat: <b>${data['current_price']:.6f}</b>
{stars} Güç: <b>{strength}/5</b> (Skor: {total_score:.1f})

📋 <b>TRADING SETUP:</b>
🎯 Giriş: <b>${data['entry_price']:.6f}</b>
🛑 Stop: <b>${data['stop_loss']:.6f}</b>
🎯 TP1: <b>${data['tp1']:.6f}</b> (R:R 1:{rr1:.1f})
🎯 TP2: <b>${data['tp2']:.6f}</b> (R:R 1:{rr2:.1f})
⚡ Kaldıraç: <b>{data['leverage']}x</b> (Risk: {risk_percent:.1f}%)

🎯 <b>ANA TETİKLEYİCİLER:</b>"""

    # Ana tetikleyicileri listele
    for trigger in data['main_triggers'][:3]:  # En fazla 3 tane
        message += f"\n• {trigger}"

    if data['supporting_factors']:
        message += f"\n\n📈 <b>DESTEKLEYEN FAKTÖRLER:</b>"
        for factor in data['supporting_factors'][:4]:  # En fazla 4 tane
            message += f"\n• {factor}"

    message += f"""

🌍 <b>SESSION & MARKET:</b>
🕐 Session: <b>{data['session'].replace('_', ' ')}</b>
🎯 Kill Zone: <b>{data['kill_zone'].replace('_', ' ')}</b>
📊 Market: <b>{data['market_condition']}</b>

✅ <b>DOĞRULAMA:</b> {data['positive_formations']}/6 formasyon pozitif"""

    # Strateji önerisi
    if signal_type == "COUNTER_TREND":
        message += f"""

💡 <b>COUNTER-TREND STRATEJİSİ:</b>
• <b>50% pozisyon TP1'de kapat</b>
• <b>50% pozisyon TP2'ye kadar tut</b>
• <b>Maksimum 6 saat pozisyon tut</b>
• <b>Daily trend yönüne dikkat et!</b>

⚠️ <b>RİSK UYARISI:</b>
Ana trend devam edebilir - Hızlı hareket gerekli!"""
    else:
        message += f"""

💡 <b>TREND FOLLOWİNG STRATEJİSİ:</b>
• <b>30% pozisyon TP1'de kapat</b>
• <b>70% pozisyon TP2'ye kadar tut</b>
• <b>Stop'u trailing yapabilirsin</b>
• <b>Trend devam ederse position artır</b>

✅ <b>GÜVEN SEVİYESİ YÜKSEK</b>
Ana trend ile uyumlu pozisyon"""

    message += f"""

🕒 {datetime.now().strftime('%H:%M:%S')} | Advanced Analysis v3.0
🎯 Weighted Scoring & Dynamic Thresholds"""

    return message

# === TELEGRAM BOT KOMUTLARI ===
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    bot.db.save_conversation(user.id, user.username, "/start")

    message = f"""🚀 <b>GELİŞMİŞ TRADING BOT'A HOŞGELDİN!</b>

Merhaba {user.first_name}! 👋

📊 <b>v3.0 YENİ ÖZELLİKLER:</b>
🔹 Ağırlıklı Formasyon Sistemi
🔹 Dinamik Eşik Ayarlama
🔹 Market Koşulu Tespiti
🔹 6 Teknik Formasyon Analizi
🔹 Multi-Timeframe Analiz
🔹 Counter-Trend Detection

📋 <b>Komutlar:</b>
/start - Botu başlat
/status - Bot durumu
/stats - Detaylı istatistikler
/formations - Formasyon açıklamaları
/help - Yardım

🎯 Bot otomatik sinyaller gönderecek!"""

    await update.message.reply_text(message, parse_mode='HTML')

async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    bot.db.save_conversation(user.id, user.username, "/status")

    stats = bot.db.get_daily_stats()
    session = bot.session_analyzer.get_current_session()
    kill_zone = bot.session_analyzer.get_kill_zone()

    message = f"""📊 <b>BOT DURUMU</b>

🎯 Bugün gönderilen: <b>{stats['daily_signals']}/{DAILY_SIGNAL_LIMIT}</b>
🌍 Aktif Session: <b>{session.replace('_', ' ')}</b>
🎯 Kill Zone: <b>{kill_zone.replace('_', ' ')}</b>

⏰ Son güncelleme: <b>{datetime.now().strftime('%H:%M:%S')}</b>
🔄 Kontrol aralığı: <b>{CHECK_INTERVAL} saniye</b>
💰 Takip edilen: <b>{len(COINS)} coin</b>

🟢 <b>Bot aktif çalışıyor...</b>
📊 Weighted Scoring System v3.0"""

    await update.message.reply_text(message, parse_mode='HTML')

async def formations_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    bot.db.save_conversation(user.id, user.username, "/formations")

    message = f"""📊 <b>AĞIRLIKLI FORMASYON SİSTEMİ</b>

🎯 <b>Formation Ağırlıkları:</b>
🏗️ Wyckoff: <b>2.0x</b> (En önemli)
🧠 SMC: <b>1.8x</b> (Çok önemli)
〰️ Elliott: <b>1.5x</b> (Önemli)
📉 Divergence: <b>1.3x</b> (Önemli)
📈 Volume Profile: <b>1.0x</b> (Normal)
🔸 Harmonic: <b>0.8x</b> (Destekleyici)

📊 <b>Dinamik Eşikler:</b>
💪 Strong Trend: 3 formasyon, 6 skor
⚖️ Weak Trend: 4 formasyon, 8 skor  
↔️ Sideways: 5 formasyon, 12 skor

🎯 <b>Market Koşulu Tespiti:</b>
• Daily range + 4H volatilite analizi
• Otomatik eşik ayarlama
• Market koşuluna göre bonus/penalty

✅ <b>SONUÇ:</b> Daha az false signal, daha kaliteli sinyaller!"""

    await update.message.reply_text(message, parse_mode='HTML')

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    bot.db.save_conversation(user.id, user.username, "/help")

    message = f"""❓ <b>GELİŞMİŞ TRADING BOT YARDIM v3.0</b>

🤖 <b>Yenilikler:</b>
Bu versiyonda ağırlıklı skorlama sistemi ve dinamik eşikler eklendi.

📊 <b>Nasıl çalışır?</b>
• Her 3 dakikada coinleri analiz eder
• 6 formasyonu ağırlığına göre skorlar
• Market koşulunu tespit eder
• Dinamik eşik uygular
• En az 3 yıldız güçte sinyal gönderir

⚙️ <b>Yeni Sinyal Kriterleri:</b>
• Ağırlıklı skor hesaplama
• Market koşuluna göre eşik ayarlama
• Session timing multiplier
• Counter-trend detection

📋 <b>Sinyal Türleri:</b>
🚀 <b>Normal Trend:</b> Ana trend yönü (güvenli)
🔄 <b>Counter-Trend:</b> Ara trend fırsatı (risk)

⚠️ <b>Risk Uyarısı:</b>
Sinyaller bilgi amaçlıdır. DYOR!

🎯 <b>v3.0 = Daha akıllı, daha az spam!</b>"""

    await update.message.reply_text(message, parse_mode='HTML')

async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    bot.db.save_conversation(user.id, user.username, "/stats")

    try:
        conn = sqlite3.connect(bot.db.db_path)
        cursor = conn.cursor()

        # Son sinyaller
        cursor.execute('''
            SELECT symbol, signal_direction, strength, market_condition, timestamp 
            FROM signals 
            ORDER BY timestamp DESC 
            LIMIT 5
        ''')
        recent_signals = cursor.fetchall()

        # Market koşulu istatistikleri
        cursor.execute('''
            SELECT market_condition, COUNT(*), AVG(strength)
            FROM signals 
            WHERE timestamp >= date('now', '-7 days')
            GROUP BY market_condition
        ''')
        market_stats = cursor.fetchall()

        conn.close()

        recent_text = "\n".join([
            f"• {sig[0]} {sig[1]} {'⭐'*sig[2]} ({sig[3]}) - {sig[4][:16]}" 
            for sig in recent_signals
        ]) if recent_signals else "Henüz sinyal yok"

        market_text = "\n".join([
            f"📊 {condition}: {count} sinyal (Ort: {avg:.1f}⭐)" 
            for condition, count, avg in market_stats
        ]) if market_stats else "Henüz veri yok"

        message = f"""📊 <b>GELİŞMİŞ İSTATİSTİKLER v3.0</b>

🚀 <b>Son 5 Sinyal:</b>
{recent_text}

📈 <b>Market Koşulu Performansı:</b>
{market_text}

⏰ Rapor zamanı: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
🎯 Weighted Scoring & Dynamic Thresholds"""

        await update.message.reply_text(message, parse_mode='HTML')

    except Exception as e:
        logger.error(f"❌ İstatistik hatası: {e}")
        await update.message.reply_text("❌ İstatistikler alınırken hata oluştu.")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    message_text = update.message.text

    bot.db.save_conversation(user.id, user.username, message_text)

    if any(word in message_text.lower() for word in ['merhaba', 'selam', 'hello', 'hi']):
        await update.message.reply_text(f"Merhaba {user.first_name}! 👋 /help komutuyla v3.0 özelliklerini öğrenebilirsin.")
    elif any(word in message_text.lower() for word in ['teşekkür', 'sağol', 'thanks']):
        await update.message.reply_text("Rica ederim! 😊 İyi tradeler!")
    elif "analiz" in message_text.lower() or "sinyal" in message_text.lower():
        await update.message.reply_text("Bot v3.0 ağırlıklı sistem ile analiz yapıyor! 📊 /stats ile son sinyalleri gör.")
    else:
        await update.message.reply_text("Mesajın için teşekkürler! 💬 /help komutuyla v3.0 yeniliklerini öğren.")

async def send_signal_to_telegram(signal_data):
    try:
        message = format_advanced_signal_message(signal_data)
        await bot.application.bot.send_message(
            chat_id=CHAT_ID, 
            text=message, 
            parse_mode='HTML'
        )

        # Veritabanına kaydet
        formations_text = f"Wyckoff:{signal_data['formations']['wyckoff']['phase']}, SMC:{signal_data['formations']['smc']['structure']}, Elliott:{signal_data['formations']['elliott']['current_wave']}"

        db_data = {
            'symbol': signal_data['symbol'],
            'signal_type': signal_data['signal_type'],
            'signal_direction': signal_data['signal_direction'],
            'total_score': signal_data['total_score'],
            'strength': signal_data['strength'],
            'entry_price': signal_data['entry_price'],
            'stop_loss': signal_data['stop_loss'],
            'tp1': signal_data['tp1'],
            'tp2': signal_data['tp2'],
            'leverage': signal_data['leverage'],
            'session': signal_data['session'],
            'kill_zone': signal_data['kill_zone'],
            'formations': formations_text,
            'market_condition': signal_data['market_condition']
        }

        bot.db.save_signal(db_data)
        logger.info("✅ Telegram sinyali gönderildi ve kaydedildi")
        return True
    except Exception as e:
        logger.error(f"❌ Telegram hatası: {e}")
        return False

async def trading_loop():
    loop_count = 0

    while bot.running:
        try:
            loop_count += 1
            logger.info(f"🔄 Döngü #{loop_count} - {len(COINS)} coin analiz ediliyor...")

            signals_found = 0

            for coin in COINS:
                try:
                    if not bot.can_send_signal(coin):
                        continue

                    # Kapsamlı analiz yap
                    analysis_result = comprehensive_analysis(
                        coin, 
                        bot.session_analyzer, 
                        bot.technical_analyzer, 
                        bot.key_levels_calc
                    )

                    if analysis_result:
                        success = await send_signal_to_telegram(analysis_result)

                        if success:
                            bot.signal_sent(coin)
                            signals_found += 1
                            logger.info(f"🎯 {coin} - {analysis_result['signal_direction']} sinyali gönderildi!")

                        await asyncio.sleep(5)

                except Exception as e:
                    logger.error(f"❌ {coin} analiz hatası: {e}")

                await asyncio.sleep(2)

            # Durum raporu (her 20 döngüde)
            if loop_count % 20 == 0:
                stats = bot.db.get_daily_stats()
                session = bot.session_analyzer.get_current_session()
                kill_zone = bot.session_analyzer.get_kill_zone()

                status = f"""📊 <b>BOT DURUM RAPORU v3.0</b>

🎯 Bugün gönderilen: <b>{stats['daily_signals']}/{DAILY_SIGNAL_LIMIT}</b>
🔄 Tamamlanan döngü: <b>{loop_count}</b>
🌍 Session: <b>{session.replace('_', ' ')}</b>
🎯 Kill Zone: <b>{kill_zone.replace('_', ' ')}</b>
⏰ Zaman: <b>{datetime.now().strftime('%H:%M:%S')}</b>

🤖 <b>Weighted Scoring System aktif!</b>
📊 Dynamic Thresholds çalışıyor"""

                try:
                    await bot.application.bot.send_message(
                        chat_id=CHAT_ID, 
                        text=status, 
                        parse_mode='HTML'
                    )
                except Exception as e:
                    logger.error(f"❌ Durum raporu hatası: {e}")

            logger.info(f"✅ Döngü #{loop_count} tamamlandı. {signals_found} sinyal gönderildi.")

            await asyncio.sleep(CHECK_INTERVAL)

        except Exception as e:
            logger.error(f"❌ Trading döngüsü hatası: {e}")
            await asyncio.sleep(60)

async def main():
    global bot
    bot = AdvancedTradingBot()

    # Telegram bot uygulamasını başlat
    bot.application = Application.builder().token(BOT_TOKEN).build()

    # Komut işleyicileri ekle
    bot.application.add_handler(CommandHandler("start", start_command))
    bot.application.add_handler(CommandHandler("status", status_command))
    bot.application.add_handler(CommandHandler("stats", stats_command))
    bot.application.add_handler(CommandHandler("formations", formations_command))
    bot.application.add_handler(CommandHandler("help", help_command))
    bot.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Başlangıç mesajı
    start_msg = f"""🚀 <b>GELİŞMİŞ TRADING BOT v3.0 BAŞLADI!</b>

📊 <b>YENİ ÖZELLİKLER:</b>
🔹 Ağırlıklı Formasyon Sistemi
🔹 Dinamik Eşik Ayarlama (STRONG/WEAK/SIDEWAYS)
🔹 Market Koşulu Tespiti
🔹 6 Teknik Formasyon Analizi
🔹 Multi-Timeframe (1h, 4h, daily)
🔹 Session & Kill Zone Timing
🔹 Counter-Trend Detection

⚙️ <b>AYARLAR:</b>
⏱️ Kontrol: Her {CHECK_INTERVAL} saniye
🎯 Günlük limit: {DAILY_SIGNAL_LIMIT} sinyal
📊 Minimum güç: 3/5 yıldız

💰 <b>TAKİP EDİLEN COİNLER ({len(COINS)}):</b>
{', '.join([c.replace('/USDT', '') for c in COINS])}

🎯 <b>WEIGHTED SCORING:</b>
Wyckoff (2.0x), SMC (1.8x), Elliott (1.5x), 
Divergence (1.3x), Volume Profile (1.0x), Harmonic (0.8x)

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
🎯 Advanced Analysis v3.0 - Daha akıllı, daha az spam!"""

    try:
        await bot.application.bot.send_message(
            chat_id=CHAT_ID, 
            text=start_msg, 
            parse_mode='HTML'
        )
    except Exception as e:
        logger.error(f"❌ Başlangıç mesajı gönderilemedi: {e}")

    # Bot'u başlat
    bot.running = True

    # Telegram bot'u arka planda başlat
    await bot.application.initialize()
    await bot.application.start()
    await bot.application.updater.start_polling()

    logger.info("🔥 Gelişmiş Trading Bot v3.0 başlatıldı!")

    # Trading döngüsünü başlat
    await trading_loop()

if __name__ == "__main__":
    try:
        print("🚀 GELİŞMİŞ TELEGRAM TRADING BOT v3.0")
        print("📊 Weighted Scoring & Dynamic Thresholds")
        print("📦 Kütüphaneler kontrol ediliyor...")

        # Event loop ayarla
        if sys.platform.startswith('win'):
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

        # Botu çalıştır
        asyncio.run(main())

    except KeyboardInterrupt:
        print("\n⛔ Bot durduruldu!")
        logger.info("Bot kullanıcı tarafından durduruldu")
    except Exception as e:
        print(f"❌ Kritik hata: {e}")
        logger.error(f"Kritik hata: {e}")
        input("Enter tuşuna basın...")
    finally:
        print("🔚 Bot kapatıldı")
        logger.info("Bot kapatıldı")          
    
        
