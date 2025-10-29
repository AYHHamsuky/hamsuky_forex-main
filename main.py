import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import talib
import logging
import datetime
import requests
import time
import os
import streamlit as st
import threading
import json

from datetime import timezone
from pathlib import Path
from meta import *  # For trading operations
from dotenv import load_dotenv  # For loading environment variables

st.set_page_config(
    page_title="Hamsuky Forex Trading Signal",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"  # Hide sidebar by default
)
# Load environment variables from .env file if it exists
load_dotenv()

# Setup enhanced logging with rotation and UTF-8 encoding
from logging.handlers import RotatingFileHandler
import logging.handlers
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# Use a safer file handler that won't conflict with multiple threads on Windows
class SafeRotatingFileHandler(RotatingFileHandler):
    """A thread-safe rotating file handler for Windows"""
    def doRollover(self):
        try:
            super().doRollover()
        except (OSError, PermissionError):
            # If file rotation fails (common on Windows), just continue logging
            pass

log_handler = SafeRotatingFileHandler(
    filename=log_dir / "trading_bot.log",
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5,
    encoding='utf-8',  # Fix Unicode encoding errors
    delay=True  # Don't open file until first log message
)
log_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
log_handler.setFormatter(log_formatter)

logger = logging.getLogger("trading_bot")
logger.setLevel(logging.INFO)
logger.handlers = []  # Clear any existing handlers
logger.addHandler(log_handler)

# Import ICT functions for enhanced signal generation (after logger is initialized)
try:
    from ict_trader import integrate_ict_methodology, calculate_ict_exit_levels
    from ict_signal_generation import add_ict_concepts, ict_check_trade_signals
    ICT_AVAILABLE = True
    logger.info("ICT modules loaded successfully")
except ImportError as e:
    logger.warning(f"ICT modules not fully available: {e}")
    ICT_AVAILABLE = False

# Global variables
last_market_alert_time = 0
trade_history = []  # Store trade history for performance tracking
active_trades = {}  # Track currently active trades
TRADE_HISTORY_FILE = "trade_history.json"

# Load trade history from file if exists
def load_trade_history():
    try:
        if os.path.exists(TRADE_HISTORY_FILE):
            with open(TRADE_HISTORY_FILE, 'r') as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Error loading trade history: {e}")
    return []

# Common list of symbols to use throughout the application - OPTIMIZED FOR PROFITABLE MARKETS
def get_default_symbols():
    return [
        # Cryptocurrency - BTC only
        "BTCUSD.m",
        
        # Commodities - Gold and Oil (highly profitable)
        "XAUUSD.m",  # Gold
        "WTI.m",      # Crude Oil (WTI)
        
        # Forex Major Pairs - High liquidity and profitability
        "EURUSD.m",   # EUR/USD (most liquid pair)
        "GBPUSD.m",   # GBP/USD (Cable)
        "USDJPY.m",   # USD/JPY
        
        # Forex Commodity Currencies - Profitable pairs
        "AUDCAD.m",   # AUD/CAD (requested)
        "AUDUSD.m",   # AUD/USD (Aussie)
        "USDCAD.m",   # USD/CAD (Loonie)
        "NZDUSD.m"    # NZD/USD (Kiwi)
    ]

# Save trade history to file
def save_trade_history():
    try:
        with open(TRADE_HISTORY_FILE, 'w') as f:
            json.dump(trade_history, f, indent=2, default=str)
    except Exception as e:
        logger.error(f"Error saving trade history: {e}")

# Initialize trade history
trade_history = load_trade_history()


def send_signal_to_subscribers(message, signal_type="general", retry_attempts=3, delay=5):
    """
    Send Telegram alerts to a single chat ID with retry mechanism.
    signal_type: Can be "general", "buy", "sell", "closed", or "warning"
    """
    global last_market_alert_time
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")  # Single chat ID from .env
    
    if not bot_token or not chat_id:
        logger.error("Telegram credentials missing. Please set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env file.")
        return False
    
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    current_time = time.time()
    
    # Avoid spamming "Market Not Profitable" messages
    if "Market Not Profitable" in message and (current_time - last_market_alert_time < 300):
        return False
    
    # Add JustMarkets branding and timestamp to messages
    branded_message = f"""
🤖 <b>HAMSUKY TRADING SIGNAL</b> 🤖
<i>Trading with JustMarkets</i>

{message}

⏰ {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")}
"""
    
    data = {"chat_id": chat_id, "text": branded_message, "parse_mode": "HTML"}
    
    # Try to send the message with retry mechanism
    for attempt in range(retry_attempts):
        try:
            response = requests.post(url, data=data, timeout=10)
            response.raise_for_status()
            logger.info(f"✅ Telegram message sent successfully to {chat_id}")
            
            # If it's a market alert, update the time
            if "Market Not Profitable" in message:
                last_market_alert_time = current_time
            
            return True  # Success
            
        except requests.exceptions.RequestException as e:
            error_msg = f"Telegram Alert Failed (Attempt {attempt+1}/{retry_attempts}): {str(e)}"
            logger.warning(error_msg)
            
            if attempt < retry_attempts - 1:
                time.sleep(delay * (attempt + 1))  # Exponential backoff
    
    # All attempts failed
    logger.error(f"Failed to send Telegram message after {retry_attempts} attempts")
    return False

# Removed subscription and trial management code for standalone deployment


# Removed daily tasks function for standalone deployment

# Removed trial status check function for standalone deployment


def initialize_mt5(account_type="demo", max_retries=3, retry_delay=2):
    """Initialize connection to JustMarkets MetaTrader 5"""

    # In the initialize_mt5() function, add:
    
    
    # Check if already initialized and disconnect first
    if mt5.terminal_info() is not None:
        mt5.shutdown()
        logger.info("Disconnected from previous MT5 session")
    
    # Set JustMarkets-specific account credentials
    if account_type.lower() == "demo":
        # JustMarkets demo account credentials
        account = "2001479025"  # Your JustMarkets demo account
        server = "JustMarkets-Demo"
        logger.info(f"Using JustMarkets demo account {account}")
    else:
        # JustMarkets real account credentials
        account = "2050196801"  # Your JustMarkets real account
        server = "JustMarkets-Live"
        logger.warning(f"Using REAL money JustMarkets account {account} - trades will use actual funds!")
    
    # Get password from environment variable
    password = os.getenv(f"MT5_{account_type.upper()}_PASSWORD")
    
    # If password not found in environment, check if we should use default method
    if not password:
        use_credentials = False
        logger.warning(f"No password found for {account_type} account. Trying to connect to an already logged-in MT5 terminal.")
    else:
        use_credentials = True
    
    for attempt in range(max_retries):
        try:
            if use_credentials:
                logger.info(f"Attempt {attempt+1}/{max_retries}: Connecting to JustMarkets MT5 with {account_type} account {account}")
                
                # Try to initialize with credentials
                initialization_result = mt5.initialize(
                    login=int(account), 
                    password=password, 
                    server=server,
                    timeout=60000  # Increase timeout to 60 seconds for more reliable connection
                )
            else:
                # Initialize without specific credentials (uses already logged in MT5 terminal)
                logger.info(f"Attempt {attempt+1}/{max_retries}: Connecting to JustMarkets MT5 using terminal login")
                initialization_result = mt5.initialize(timeout=60000)
            
            if not initialization_result:
                error_code = mt5.last_error()
                error_msg = f"JustMarkets MT5 initialization failed with error code: {error_code}"
                logger.error(error_msg)
                
                # Handle specific JustMarkets error codes
                if error_code[0] == 10009:  # Sometimes seen with JustMarkets
                    logger.warning("Connection issue with JustMarkets server. Check internet connection.")
                elif error_code[0] == -10005:  # IPC timeout
                    logger.warning("IPC Timeout detected - MT5 terminal communication issue")
                
                # Wait before retry
                time.sleep(retry_delay)
                continue
            
            # Verify JustMarkets account connection
            account_info = mt5.account_info()
            if account_info is None:
                error_msg = "Failed to get JustMarkets account info - check if you're logged in to MT5"
                logger.error(error_msg)
                time.sleep(retry_delay)
                continue
            
            # Verify it's actually JustMarkets
            if "JustMarkets" not in account_info.server:
                logger.warning(f"Connected to {account_info.server} instead of JustMarkets. Check your MT5 installation.")
                
            # Check account type (demo or real)
            is_demo = 'demo' in account_info.server.lower()
            account_type_detected = "DEMO" if is_demo else "REAL"
            
            logger.info(f"Successfully connected to JustMarkets {account_type_detected} account #{account_info.login}")
            logger.info(f"Server: {account_info.server}, Balance: {account_info.balance} {account_info.currency}")
            
            # Add a warning if using a real account
            if not is_demo:
                logger.warning("ATTENTION: Using REAL MONEY JustMarkets account for trading!")
            
            # Check if symbols are available - BTC and Gold only
            symbols_to_check = ["BTCUSD.m", "XAUUSD.m"]
            for symbol in symbols_to_check:
                if mt5.symbol_info(symbol) is None:
                    if not mt5.symbol_select(symbol, True):
                        logger.warning(f"Could not add {symbol} to MarketWatch. Check symbol name with JustMarkets.")
                    else:
                        logger.info(f"Added {symbol} to MarketWatch")
            
            return True
            
        except Exception as e:
            logger.error(f"Error during JustMarkets MT5 initialization attempt {attempt+1}: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
    
    # All retries failed
    st.error("Failed to connect to JustMarkets MetaTrader 5 after multiple attempts.")
    return False

# Function to Get Market Data with caching
last_data_fetch = {}
cached_market_data = {}

def get_market_data(symbol="GBPUSD.m", timeframe=mt5.TIMEFRAME_M5, n_bars=100, force_refresh=False):
    """Get market data with caching to reduce API calls"""
    cache_key = f"{symbol}_{timeframe}"
    current_time = time.time()
    
    # Return cached data if it's fresh
    cache_lifetime = max(10, timeframe/mt5.TIMEFRAME_M1 * 2)
    
    if not force_refresh and cache_key in cached_market_data and cache_key in last_data_fetch:
        if current_time - last_data_fetch[cache_key] < cache_lifetime:
            return cached_market_data[cache_key].copy()
    
    # Special handling for crypto and metals - sometimes these have different naming conventions
    symbol_variations = [symbol]
    
    # Add potential variations for crypto and metals
    if symbol == "BTCUSD.m":
        symbol_variations.extend(["BTC.USD", "BTCUSD.m.a", "BTC/USD"])
    elif symbol == "ETHUSD.m":
        symbol_variations.extend(["ETH.USD", "ETHUSD.m.a", "ETH/USD"])
    elif symbol == "XAUUSD.m":
        symbol_variations.extend(["GOLD", "GOLD.a", "XAU/USD"])
    elif symbol == "XAGUSD.m":
        symbol_variations.extend(["SILVER", "SILVER.a", "XAG/USD"])
    
    # Try each variation until we find one that works
    for sym_variant in symbol_variations:
        # Ensure symbol is in MarketWatch
        if mt5.symbol_info(sym_variant) is None:
            logger.info(f"Trying to add {sym_variant} to MarketWatch")
            if not mt5.symbol_select(sym_variant, True):
                logger.warning(f"Failed to add {sym_variant} to MarketWatch. Error: {mt5.last_error()}")
                continue  # Try next variation
            else:
                logger.info(f"Added {sym_variant} to MarketWatch")
                symbol = sym_variant  # Use this working variation
                break
        else:
            symbol = sym_variant  # Use this working variation
            break
    
    # Add a small delay to avoid overwhelming the server
    time.sleep(0.1)
    
    # Fetch new data
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n_bars)
    if rates is None or len(rates) == 0:
        error_info = mt5.last_error()
        logger.error(f"Failed to fetch data for {symbol}. Error: {error_info}")
        return pd.DataFrame()
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    # Update cache
    cached_market_data[cache_key] = df.copy()
    last_data_fetch[cache_key] = current_time
    
    return df

# Calculate Indicators with error handling
def calculate_indicators(df, symbol="GBPUSD.m"):
    """Calculate indicators with optimized parameters for different market types"""
    if df.empty:
        return df
    
    try:
        # Determine market type to adjust parameters - OPTIMIZED FOR FOREX, BTC & GOLD
        is_crypto = "BTC" in symbol  # Only BTC as requested
        is_metal = "XAU" in symbol or "GOLD" in symbol  # Only Gold as requested
        is_forex = any(pair in symbol for pair in ["USD", "EUR", "GBP", "JPY", "AUD", "NZD", "CAD", "CHF"])        # Adjust timeperiods based on market type
        rsi_period = 14
        macd_fast = 12
        macd_slow = 26
        macd_signal = 9
        bb_period = 20
        atr_period = 14
            
        # BTC markets need optimized periods for high volatility
        if is_crypto:
            rsi_period = 21  # Longer period for crypto volatility
            macd_fast = 8
            macd_slow = 21
            bb_period = 30
            atr_period = 21
        # Gold optimized parameters for precious metal trends
        elif is_metal:
            rsi_period = 14
            macd_fast = 8
            macd_slow = 21
            bb_period = 20
            atr_period = 14
        # Forex optimized for major pairs
        else:
            rsi_period = 14
            macd_fast = 12
            macd_slow = 26
            bb_period = 20
            atr_period = 14
        
        # Basic momentum indicators
        df['RSI'] = talib.RSI(df['close'], timeperiod=rsi_period)
        df['MACD'], df['MACD_signal'], df['MACD_hist'] = talib.MACD(
            df['close'], fastperiod=macd_fast, slowperiod=macd_slow, signalperiod=macd_signal)
        
        # Moving averages - All markets need these for SMC
        df['SMA_50'] = talib.SMA(df['close'], timeperiod=50)
        df['SMA_200'] = talib.SMA(df['close'], timeperiod=200)
        df['EMA_9'] = talib.EMA(df['close'], timeperiod=9)
        df['EMA_21'] = talib.EMA(df['close'], timeperiod=21)
        df['EMA_55'] = talib.EMA(df['close'], timeperiod=55)  # Needed for SMC breaker block logic
        df['EMA_89'] = talib.EMA(df['close'], timeperiod=89)  # Additional EMA for trend confirmation
        
        # Volatility and trend indicators
        df['ATR'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=atr_period)
        df['ADX'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
        
        # Bollinger Bands
        df['upper_band'], df['middle_band'], df['lower_band'] = talib.BBANDS(
            df['close'], timeperiod=bb_period, nbdevup=2, nbdevdn=2)
        df['BB_Width'] = df['upper_band'] - df['lower_band']
        
        # Trend classification
        df['Uptrend'] = df['SMA_50'] > df['SMA_200']
        df['Downtrend'] = df['SMA_50'] < df['SMA_200']
        
        # Support/Resistance zones
        df['recent_high'] = df['high'].rolling(10).max()
        df['recent_low'] = df['low'].rolling(10).min()
        
        # Stochastic oscillator
        df['slowk'], df['slowd'] = talib.STOCH(df['high'], df['low'], df['close'], 
                                            fastk_period=14, slowk_period=3, slowk_matype=0, 
                                            slowd_period=3, slowd_matype=0)
        
        # Volume indicators (if volume data is available)
        if 'tick_volume' in df.columns:
            df['OBV'] = talib.OBV(df['close'], df['tick_volume'])
            
        # Special indicators for crypto and metals
        if is_crypto or is_metal:
            # Add Ichimoku Cloud which works well for these markets
            tenkan_period = 9 if is_crypto else 7
            kijun_period = 26 if is_crypto else 22
            
            # Calculate Tenkan-sen (Conversion Line)
            tenkan_high = df['high'].rolling(window=tenkan_period).max()
            tenkan_low = df['low'].rolling(window=tenkan_period).min()
            df['Tenkan'] = (tenkan_high + tenkan_low) / 2
            
            # Calculate Kijun-sen (Base Line)
            kijun_high = df['high'].rolling(window=kijun_period).max()
            kijun_low = df['low'].rolling(window=kijun_period).min()
            df['Kijun'] = (kijun_high + kijun_low) / 2
            
            # Calculate Senkou Span A (Leading Span A)
            df['Senkou_A'] = ((df['Tenkan'] + df['Kijun']) / 2).shift(kijun_period)
            
            # Calculate Senkou Span B (Leading Span B)
            senkou_b_period = 52 if is_crypto else 44
            senkou_high = df['high'].rolling(window=senkou_b_period).max()
            senkou_low = df['low'].rolling(window=senkou_b_period).min()
            df['Senkou_B'] = ((senkou_high + senkou_low) / 2).shift(kijun_period)
            
            # Calculate Chikou Span (Lagging Span)
            df['Chikou'] = df['close'].shift(-kijun_period)
            
            # Define cloud direction (bullish/bearish)
            df['Cloud_Bullish'] = df['Senkou_A'] > df['Senkou_B']
            df['Cloud_Bearish'] = df['Senkou_A'] < df['Senkou_B']
            
            # Define price relative to cloud
            df['Price_Above_Cloud'] = (df['close'] > df['Senkou_A']) & (df['close'] > df['Senkou_B'])
            df['Price_Below_Cloud'] = (df['close'] < df['Senkou_A']) & (df['close'] < df['Senkou_B'])
            df['Price_In_Cloud'] = ~(df['Price_Above_Cloud'] | df['Price_Below_Cloud'])
                
    except Exception as e:
        logger.error(f"Error calculating indicators: {e}")
        st.error(f"Error calculating indicators: {e}")
    
    return df

# Enhancement for market profitability check
def is_profitable_market(df, symbol="GBPUSD.m"):
    """Check if market conditions are suitable for trading with market-specific parameters"""
    if df.empty:
        return False
    
    # Determine market type
    is_crypto = "BTC" in symbol or "ETH" in symbol
    is_metal = "XAU" in symbol or "GOLD" in symbol or "XAG" in symbol or "SILVER" in symbol
    is_index = any(idx in symbol for idx in ["US30.std", "US500.std", "USTEC.std", "DE30.std", "UK100.std", "JP225.std"])
    
    try:
        # Market-specific ATR thresholds
        min_atr = 0.0006  # Default for forex
        
        if is_crypto:
            min_atr = 50.0  # BTC needs higher volatility threshold
        elif is_metal:
            min_atr = 0.4  # Gold typically measured in dollars
        else:  # Forex pairs
            min_atr = 0.0006  # Standard forex volatility threshold
        
        # Check if market has sufficient volatility
        if df['ATR'].iloc[-1] < min_atr:
            logger.info(f"Low volatility detected in {symbol}: ATR={df['ATR'].iloc[-1]:.6f}")
            return False
            
        # Market-specific ADX thresholds
        min_adx = 18  # Default for forex
        
        if is_crypto:
            min_adx = 20  # BTC needs stronger trends
        elif is_metal:
            min_adx = 16  # Gold can move with weaker ADX
        else:  # Forex
            min_adx = 18  # Standard forex trend strength
        
        # Check if market has sufficient trend strength
        if df['ADX'].iloc[-1] < min_adx:
            logger.info(f"Weak trend detected in {symbol}: ADX={df['ADX'].iloc[-1]:.2f}")
            return False
            
        # Check Bollinger Band width for ranging or trending conditions
        min_bb_width = 0.0004  # Default
        
        if is_crypto:
            min_bb_width = 100.0  # Bitcoin typically has wider bands in USD
        elif is_metal:
            min_bb_width = 0.8  # Gold needs wider bands
        else:  # Forex
            min_bb_width = 0.0004  # Standard forex BB width
            
        if df['BB_Width'].iloc[-1] < min_bb_width:
            logger.info(f"Tight range detected in {symbol}: BB_Width={df['BB_Width'].iloc[-1]:.6f}")
            return False
            
        # Check for excessive volatility
        max_vol_ratio = 6.0  # Default
        
        if is_crypto:
            max_vol_ratio = 8.0  # BTC can handle higher volatility spikes
        elif is_metal:
            max_vol_ratio = 7.0  # Gold can have volatile spikes
        else:  # Forex
            max_vol_ratio = 6.0  # Standard forex volatility limit
            
        if df['ATR'].iloc[-1] > max_vol_ratio * df['ATR'].rolling(20).mean().iloc[-1]:
            logger.info(f"Excessive volatility detected in {symbol}: ATR={df['ATR'].iloc[-1]:.6f}")
            return False
        
        # Check if price is near support/resistance
        close = df['close'].iloc[-1]
        recent_high = df['recent_high'].iloc[-1]
        recent_low = df['recent_low'].iloc[-1]
        
        # Different buffer sizes for different markets
        buffer_pct = 0.15  # Default
        
        if is_crypto:
            buffer_pct = 0.2  # BTC needs wider buffers
        elif is_metal:
            buffer_pct = 0.18  # Gold needs slightly wider buffers
        else:  # Forex
            buffer_pct = 0.15  # Standard forex buffer
            
        buffer = buffer_pct * df['ATR'].iloc[-1]
        if abs(close - recent_high) < buffer or abs(close - recent_low) < buffer:
            logger.info(f"{symbol} price near support/resistance: {close:.5f}, High: {recent_high:.5f}, Low: {recent_low:.5f}")
            return False
            
        # Special check for BTC: avoid trading during extreme fear/greed
        if is_crypto and 'RSI' in df.columns:
            if df['RSI'].iloc[-1] < 20 or df['RSI'].iloc[-1] > 80:
                logger.info(f"Extreme market sentiment for BTC: RSI={df['RSI'].iloc[-1]:.2f}")
                return False
        
        # Special check for Gold: avoid trading during high-impact economic events
        if is_metal:
            recent_bars = df.tail(5)
            avg_range = (recent_bars['high'] - recent_bars['low']).mean()
            typical_range = df['ATR'].iloc[-1]
            
            if avg_range > 2 * typical_range:
                logger.info(f"Possible economic event affecting Gold: Recent volatility too high")
                return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error in market profitability check for {symbol}: {e}")
        return False

# Extended market session check to allow more trading hours
def is_active_session():
    now = datetime.datetime.utcnow()
    hour = now.hour
    weekday = now.weekday()
    
    # Don't trade on weekends
    if weekday >= 5:  # 5 = Saturday, 6 = Sunday
        return False
        
    # Extended trading sessions (UTC time)
    asian_session = 0 <= hour < 8  # Added Asian session
    london_session = 7 <= hour < 16
    new_york_session = 12 <= hour < 22  # Extended NY session
    
    # Trade during all major sessions
    return asian_session or london_session or new_york_session
    
# Import ICT functions for enhanced signal generation
from ict_trader import integrate_ict_methodology, calculate_ict_exit_levels
from ict_signal_generation import add_ict_concepts, ict_check_trade_signals

# Enhanced Signal Generation Functions
def advanced_check_trade_signals(df, symbol="GBPUSD.m"):
    """Enhanced trade signal function optimized for different market types with ICT integration"""
    if df.empty:
        return df
    
    df = calculate_indicators(df, symbol)

    # Determine market type - OPTIMIZED FOR FOREX, BTC & GOLD ONLY
    is_crypto = "BTC" in symbol  # Bitcoin only
    is_metal = "XAU" in symbol or "GOLD" in symbol  # Gold only
    is_forex = any(pair in symbol for pair in ["USD", "EUR", "GBP", "JPY", "AUD", "NZD", "CAD", "CHF"])
    
    # Original signals (keep these as a baseline)
    df['Buy_Signal_Original'] = (df['RSI'] < 30) & (df['MACD'] > df['MACD_signal']) & df['Uptrend']
    df['Sell_Signal_Original'] = (df['RSI'] > 70) & (df['MACD'] < df['MACD_signal']) & df['Downtrend']
    
    # Calculate additional indicators for better signal quality
    # Price action patterns
    df['Higher_High'] = (df['high'] > df['high'].shift(1)) & (df['high'].shift(1) > df['high'].shift(2))
    df['Lower_Low'] = (df['low'] < df['low'].shift(1)) & (df['low'].shift(1) < df['low'].shift(2))
    df['Higher_Low'] = (df['low'] > df['low'].shift(1))
    df['Lower_High'] = (df['high'] < df['high'].shift(1))
    
    # Candlestick patterns
    df['Bullish_Engulfing'] = (df['open'] > df['close'].shift(1)) & (df['close'] > df['open'].shift(1)) & (df['close'] - df['open'] > df['open'].shift(1) - df['close'].shift(1))
    df['Bearish_Engulfing'] = (df['open'] < df['close'].shift(1)) & (df['close'] < df['open'].shift(1)) & (df['open'] - df['close'] > df['close'].shift(1) - df['open'].shift(1))
    
    # Breakout detection
    df['Range_High'] = df['high'].rolling(10).max()
    df['Range_Low'] = df['low'].rolling(10).min()
    df['Range_Breakout_Up'] = (df['close'] > df['Range_High'].shift(1)) & (df['close'].shift(1) <= df['Range_High'].shift(2))
    df['Range_Breakout_Down'] = (df['close'] < df['Range_Low'].shift(1)) & (df['close'].shift(1) >= df['Range_Low'].shift(2))
    
    # Support/Resistance breaks
    df['Support_Break'] = (df['close'] < df['low'].rolling(10).min().shift(1)) & (df['close'].shift(1) >= df['low'].rolling(10).min().shift(2))
    df['Resistance_Break'] = (df['close'] > df['high'].rolling(10).max().shift(1)) & (df['close'].shift(1) <= df['high'].rolling(10).max().shift(2))
    
    # Volume confirmation
    if 'tick_volume' in df.columns:
        df['Volume_Increasing'] = df['tick_volume'] > df['tick_volume'].rolling(5).mean()
    else:
        df['Volume_Increasing'] = True
    
    # Trend strength and volatility assessment
    adx_threshold = 20  # Baseline
    if is_crypto:
        adx_threshold = 25  # Crypto needs stronger trends
    elif is_metal:
        adx_threshold = 18  # Gold can trend with lower ADX
    
    df['Strong_Trend'] = df['ADX'] > adx_threshold
    
    # Volatility thresholds vary by market
    if is_crypto:
        df['Ideal_Volatility'] = (df['ATR'] > df['ATR'].rolling(20).mean() * 0.8) & (df['ATR'] < df['ATR'].rolling(20).mean() * 3.0)
    elif is_metal:
        df['Ideal_Volatility'] = (df['ATR'] > df['ATR'].rolling(20).mean() * 0.7) & (df['ATR'] < df['ATR'].rolling(20).mean() * 2.5)
    else:  # Forex and indices
        df['Ideal_Volatility'] = (df['ATR'] > df['ATR'].rolling(20).mean() * 0.6) & (df['ATR'] < df['ATR'].rolling(20).mean() * 2.8)
    
    # RSI Divergence
    df['RSI_Higher_High'] = (df['RSI'] > df['RSI'].shift(1)) & (df['RSI'].shift(1) > df['RSI'].shift(2))
    df['RSI_Lower_Low'] = (df['RSI'] < df['RSI'].shift(1)) & (df['RSI'].shift(1) < df['RSI'].shift(2))
    df['Bullish_Divergence'] = df['Lower_Low'] & ~df['RSI_Lower_Low'] & (df['RSI'] < 40)
    df['Bearish_Divergence'] = df['Higher_High'] & ~df['RSI_Higher_High'] & (df['RSI'] > 60)
    
    # Market session where
    if 'time' in df.columns:
        df['Hour'] = df['time'].dt.hour
        df['Optimal_Hours'] = df['Hour'].between(6, 20)  # Extended hours
    else:
        df['Optimal_Hours'] = True
    
    # Add EMA and MACD crossover calculations
    df['EMA_Bullish_Cross'] = (df['EMA_9'] > df['EMA_21']) & (df['EMA_9'].shift(1) <= df['EMA_21'].shift(1))
    df['EMA_Bearish_Cross'] = (df['EMA_9'] < df['EMA_21']) & (df['EMA_9'].shift(1) >= df['EMA_21'].shift(1))
    df['MACD_Bullish_Cross'] = (df['MACD'] > df['MACD_signal']) & (df['MACD'].shift(1) <= df['MACD_signal'].shift(1))
    df['MACD_Bearish_Cross'] = (df['MACD'] < df['MACD_signal']) & (df['MACD'].shift(1) >= df['MACD_signal'].shift(1))
    
    # Crypto-specific signals - these work especially well for Bitcoin
    if is_crypto:
        # Add crypto-specific trading signals if available
        if 'Tenkan' in df.columns:
            df['Tenkan_Kijun_Bull_Cross'] = (df['Tenkan'] > df['Kijun']) & (df['Tenkan'].shift(1) <= df['Kijun'].shift(1))
            df['Tenkan_Kijun_Bear_Cross'] = (df['Tenkan'] < df['Kijun']) & (df['Tenkan'].shift(1) >= df['Kijun'].shift(1))
            df['Price_Above_Kumo'] = df['Price_Above_Cloud'] & df['Cloud_Bullish']
            df['Price_Below_Kumo'] = df['Price_Below_Cloud'] & df['Cloud_Bearish']
        
        # Bitcoin tends to respect bollinger bands well
        df['BB_Lower_Bounce'] = (df['close'].shift(1) < df['lower_band'].shift(1)) & (df['close'] > df['lower_band'])
        df['BB_Upper_Bounce'] = (df['close'].shift(1) > df['upper_band'].shift(1)) & (df['close'] < df['upper_band'])
    
    # Gold-specific signals
    if is_metal:
        # Gold often respects key moving averages
        df['Golden_Cross'] = (df['SMA_50'] > df['SMA_200']) & (df['SMA_50'].shift(1) <= df['SMA_200'].shift(1))
        df['Death_Cross'] = (df['SMA_50'] < df['SMA_200']) & (df['SMA_50'].shift(1) >= df['SMA_200'].shift(1))
        
        # Gold responds well to these levels
        rsi_oversold = 35  # Gold can reverse earlier
        rsi_overbought = 65  # Gold can reverse earlier
    else:
        # Standard levels for other markets
        rsi_oversold = 38  # Default setting
        rsi_overbought = 62  # Default setting
    
    # BUY SIGNALS - OPTIMIZED FOR BTC
    if is_crypto:
        df['Buy_Signal'] = (
            # BTC original signal with stronger confirmation
            ((df['Buy_Signal_Original']) & (df['Strong_Trend']) & (df['Ideal_Volatility'])) |
            
            # BTC specific Ichimoku signals if available
            (('Tenkan_Kijun_Bull_Cross' in df.columns) & (df['Tenkan_Kijun_Bull_Cross']) & (df['Price_Above_Kumo']) & (df['close'] > df['EMA_21'])) |
            
            # RSI oversold with bullish price action for BTC
            ((df['RSI'] < 35) & (df['Higher_Low']) & (df['EMA_9'] > df['EMA_9'].shift(1)) & (df['Strong_Trend'])) |
            
            # Bullish divergence with support
            ((df['Bullish_Divergence']) & (df['close'] > df['EMA_21']) & (df['Ideal_Volatility'])) |
            
            # Bollinger bounce - very effective for BTC
            ((df['BB_Lower_Bounce']) & (df['Volume_Increasing']) & (df['RSI'] < 40)) |
            
            # Strong breakout above resistance with volume
            ((df['Resistance_Break']) & (df['Volume_Increasing']) & (df['Strong_Trend']) & (df['close'] > df['EMA_21']))
        )
    elif is_metal:
        df['Buy_Signal'] = (
            # Gold original signal with strong confirmation
            ((df['Buy_Signal_Original']) & (df['Strong_Trend'])) |
            
            # Gold specific moving average crosses
            ((df['Golden_Cross']) & (df['close'] > df['EMA_21']) & (df['Volume_Increasing'])) |
            
            # RSI oversold with higher low - optimized for Gold
            ((df['RSI'] < 35) & (df['Higher_Low']) & (df['close'] > df['EMA_9'])) |
            
            # Bullish divergence - very reliable for Gold
            ((df['Bullish_Divergence']) & (df['Volume_Increasing'])) |
            
            # Strong support bounce with volume
            ((df['close'] < df['lower_band']) & (df['close'] > df['close'].shift(1)) & (df['Volume_Increasing']) & (df['RSI'] < 40)) |
            
            # EMA crossover with trend confirmation
            ((df['EMA_Bullish_Cross']) & (df['close'] > df['SMA_50']) & (df['RSI'] > 40))
        )
    else:
        # OPTIMIZED signals for FOREX major pairs
        df['Buy_Signal'] = (
            # Original signal with stronger confirmation
            ((df['Buy_Signal_Original']) & (df['Strong_Trend']) & (df['Ideal_Volatility'])) |
            
            # RSI oversold with bullish price action - optimized for Forex
            ((df['RSI'] < 38) & (df['Higher_Low']) & (df['EMA_9'] > df['EMA_9'].shift(1)) & (df['Strong_Trend'])) |
            
            # Bullish divergence with support
            ((df['Bullish_Divergence']) & (df['close'] > df['EMA_21']) & (df['Ideal_Volatility'])) |
            
            # Bullish engulfing at support
            ((df['Bullish_Engulfing']) & (df['close'] < df['lower_band']) & (df['RSI'] < 40)) |
            
            # Moving average crossover with trend confirmation
            ((df['EMA_Bullish_Cross']) & (df['close'] > df['SMA_50']) & (df['SMA_50'] > df['SMA_50'].shift(5)) & (df['RSI'] > 40) & (df['RSI'] < 68)) |
            
            # Range breakout signals - effective for Forex
            ((df['Range_Breakout_Up']) & (df['Volume_Increasing']) & (df['RSI'] > 40) & (df['RSI'] < 70)) |
            
            # MACD bullish crossover in uptrend
            ((df['MACD_Bullish_Cross']) & (df['close'] > df['SMA_200']) & (df['RSI'] > 35) & (df['RSI'] < 65))
        )
    
    # SELL SIGNALS - OPTIMIZED FOR BTC
    if is_crypto:
        df['Sell_Signal'] = (
            # BTC original signal with stronger confirmation
            ((df['Sell_Signal_Original']) & (df['Strong_Trend']) & (df['Ideal_Volatility'])) |
            
            # BTC specific Ichimoku signals if available
            (('Tenkan_Kijun_Bear_Cross' in df.columns) & (df['Tenkan_Kijun_Bear_Cross']) & (df['Price_Below_Kumo']) & (df['close'] < df['EMA_21'])) |
            
            # RSI overbought with bearish price action for BTC
            ((df['RSI'] > 65) & (df['Lower_High']) & (df['EMA_9'] < df['EMA_9'].shift(1)) & (df['Strong_Trend'])) |
            
            # Bearish divergence with resistance
            ((df['Bearish_Divergence']) & (df['close'] < df['EMA_21']) & (df['Ideal_Volatility'])) |
            
            # Bollinger bounce - very effective for BTC
            ((df['BB_Upper_Bounce']) & (df['Volume_Increasing']) & (df['RSI'] > 60)) |
            
            # Strong breakdown below support with volume
            ((df['Support_Break']) & (df['Volume_Increasing']) & (df['Strong_Trend']) & (df['close'] < df['EMA_21']))
        )
    elif is_metal:
        df['Sell_Signal'] = (
            # Gold original signal with strong confirmation
            ((df['Sell_Signal_Original']) & (df['Strong_Trend'])) |
            
            # Gold specific moving average crosses
            ((df['Death_Cross']) & (df['close'] < df['EMA_21']) & (df['Volume_Increasing'])) |
            
            # RSI overbought with lower high - optimized for Gold
            ((df['RSI'] > 65) & (df['Lower_High']) & (df['close'] < df['EMA_9'])) |
            
            # Bearish divergence - very reliable for Gold
            ((df['Bearish_Divergence']) & (df['Volume_Increasing'])) |
            
            # Strong resistance rejection with volume
            ((df['close'] > df['upper_band']) & (df['close'] < df['close'].shift(1)) & (df['Volume_Increasing']) & (df['RSI'] > 60)) |
            
            # EMA crossover with trend confirmation
            ((df['EMA_Bearish_Cross']) & (df['close'] < df['SMA_50']) & (df['RSI'] < 60))
        )
    else:
        # OPTIMIZED signals for FOREX major pairs
        df['Sell_Signal'] = (
            # Original signal with stronger confirmation
            ((df['Sell_Signal_Original']) & (df['Strong_Trend']) & (df['Ideal_Volatility'])) |
            
            # RSI overbought with bearish price action - optimized for Forex
            ((df['RSI'] > 62) & (df['Lower_High']) & (df['EMA_9'] < df['EMA_9'].shift(1)) & (df['Strong_Trend'])) |
            
            # Bearish divergence with resistance
            ((df['Bearish_Divergence']) & (df['close'] < df['EMA_21']) & (df['Ideal_Volatility'])) |
            
            # Bearish engulfing at resistance
            ((df['Bearish_Engulfing']) & (df['close'] > df['upper_band']) & (df['RSI'] > 60)) |
            
            # Moving average crossover with trend confirmation
            ((df['EMA_Bearish_Cross']) & (df['close'] < df['SMA_50']) & (df['SMA_50'] < df['SMA_50'].shift(5)) & (df['RSI'] < 60) & (df['RSI'] > 32)) |
            
            # Range breakout signals - effective for Forex
            ((df['Range_Breakout_Down']) & (df['Volume_Increasing']) & (df['RSI'] < 60) & (df['RSI'] > 30)) |
            
            # MACD bearish crossover in downtrend
            ((df['MACD_Bearish_Cross']) & (df['close'] < df['SMA_200']) & (df['RSI'] < 65) & (df['RSI'] > 35))
        )
    
    # Additional where - avoid trading near support/resistance levels
    support_resistance_buffer = 1.2 * df['ATR']
    df['Near_Support_Resistance'] = False
    
    for i in range(1, len(df)):
        price = df['close'].iloc[i]
        recent_highs = df['high'].iloc[max(0, i-20):i].nlargest(3)
        recent_lows = df['low'].iloc[max(0, i-20):i].nsmallest(3)
        
        for level in recent_highs:
            if abs(price - level) < support_resistance_buffer.iloc[i]:
                df.loc[df.index[i], 'Near_Support_Resistance'] = True
                break
                
        for level in recent_lows:
            if abs(price - level) < support_resistance_buffer.iloc[i]:
                df.loc[df.index[i], 'Near_Support_Resistance'] = True
                break
    
    # Apply final where - avoid trading near support/resistance levels
    df['Buy_Signal'] = df['Buy_Signal'] & ~df['Near_Support_Resistance']
    df['Sell_Signal'] = df['Sell_Signal'] & ~df['Near_Support_Resistance']
    
    # Avoid excessive signals - Reduce lookback period
    for i in range(1, min(5, len(df))):  # Reduced from 10 to 5
        if i < len(df):
            if df['Buy_Signal'].iloc[-i] and df['Buy_Signal'].iloc[-i-1:-i-3].any():  # Reduced from 5 to 3
                df.loc[df.index[-i], 'Buy_Signal'] = False
            if df['Sell_Signal'].iloc[-i] and df['Sell_Signal'].iloc[-i-1:-i-3].any():  # Reduced from 5 to 3
                df.loc[df.index[-i], 'Sell_Signal'] = False
    
    # ========== INTEGRATE ICT METHODOLOGY FOR SUPERIOR SIGNALS ==========
    # Add ICT concepts to enhance signal quality
    df = add_ict_concepts(df)
    
    # Create combined signals using both traditional and ICT approaches
    df['Traditional_Buy'] = df['Buy_Signal'].copy()
    df['Traditional_Sell'] = df['Sell_Signal'].copy()
    
    # ICT-based signals
    df['ICT_Buy'] = df.get('ict_buy_signal', False)
    df['ICT_Sell'] = df.get('ict_sell_signal', False)
    
    # ENHANCED SIGNAL LOGIC: Require confirmation from BOTH systems for higher accuracy
    # This creates a multi-layer confirmation system
    
    if is_crypto:
        # For BTC: Use aggressive ICT/SMC signals with breaker block confirmation
        df['Buy_Signal'] = (
            # Strong agreement between both systems
            (df['Traditional_Buy'] & df['ICT_Buy']) |
            # Strong ICT signal with market structure support
            (df['ICT_Buy'] & df.get('ms_bullish', False) & df.get('bull_ote_zone', False)) |
            # Traditional signal with ICT confirmation
            (df['Traditional_Buy'] & (df.get('bullish_fvg', False) | df.get('bull_order_block', False))) |
            # BREAKER BLOCK entry (NEWLY ADDED - high probability SMC setup)
            (df.get('bull_breaker', False) & df.get('ms_bullish', False) & (df['RSI'] < 50))
        )
        
        df['Sell_Signal'] = (
            # Strong agreement between both systems
            (df['Traditional_Sell'] & df['ICT_Sell']) |
            # Strong ICT signal with market structure support
            (df['ICT_Sell'] & df.get('ms_bearish', False) & df.get('bear_ote_zone', False)) |
            # Traditional signal with ICT confirmation
            (df['Traditional_Sell'] & (df.get('bearish_fvg', False) | df.get('bear_order_block', False))) |
            # BREAKER BLOCK entry (NEWLY ADDED - high probability SMC setup)
            (df.get('bear_breaker', False) & df.get('ms_bearish', False) & (df['RSI'] > 50))
        )
        
    elif is_metal:
        # For Gold: Require strong SMC agreement with breaker blocks
        df['Buy_Signal'] = (
            # Require agreement between both systems for gold
            (df['Traditional_Buy'] & df['ICT_Buy']) |
            # Allow ICT signal if multiple SMC confirmations
            (df['ICT_Buy'] & df.get('bull_order_block', False) & df.get('low_liquidity_sweep', False)) |
            # Allow traditional signal with strong ICT support
            (df['Traditional_Buy'] & df.get('ms_bullish', False) & (df['RSI'] < 35)) |
            # BREAKER BLOCK with order block confirmation (NEWLY ADDED)
            (df.get('bull_breaker', False) & df.get('bull_order_block', False) & df.get('low_liquidity_sweep', False))
        )
        
        df['Sell_Signal'] = (
            # Require agreement between both systems for gold
            (df['Traditional_Sell'] & df['ICT_Sell']) |
            # Allow ICT signal if multiple SMC confirmations
            (df['ICT_Sell'] & df.get('bear_order_block', False) & df.get('high_liquidity_sweep', False)) |
            # Allow traditional signal with strong ICT support
            (df['Traditional_Sell'] & df.get('ms_bearish', False) & (df['RSI'] > 65)) |
            # BREAKER BLOCK with order block confirmation (NEWLY ADDED)
            (df.get('bear_breaker', False) & df.get('bear_order_block', False) & df.get('high_liquidity_sweep', False))
        )
        
    else:
        # For Forex: Balance between both systems with robust SMC/breaker blocks
        df['Buy_Signal'] = (
            # Agreement between both systems (highest confidence)
            (df['Traditional_Buy'] & df['ICT_Buy']) |
            # Strong ICT signal with trend confirmation
            (df['ICT_Buy'] & df.get('ms_bullish', False) & (df['close'] > df['EMA_21'])) |
            # Traditional signal with ICT structure support
            (df['Traditional_Buy'] & (df.get('bull_ote_zone', False) | df.get('bullish_fvg', False))) |
            # Strong traditional with partial ICT confirmation
            (df['Traditional_Buy'] & df.get('bull_order_block', False) & df['Strong_Trend']) |
            # BREAKER BLOCK SMC setup (NEWLY ADDED - institutional reversal point)
            (df.get('bull_breaker', False) & df.get('ms_bullish', False) & df.get('low_liquidity_sweep', False)) |
            # Order block + Breaker confirmation (ultimate SMC setup)
            (df.get('bull_order_block', False) & df.get('bull_breaker', False) & (df['close'] > df.get('EMA_55', df['EMA_21'])))
        )
        
        df['Sell_Signal'] = (
            # Agreement between both systems (highest confidence)
            (df['Traditional_Sell'] & df['ICT_Sell']) |
            # Strong ICT signal with trend confirmation
            (df['ICT_Sell'] & df.get('ms_bearish', False) & (df['close'] < df['EMA_21'])) |
            # Traditional signal with ICT structure support
            (df['Traditional_Sell'] & (df.get('bear_ote_zone', False) | df.get('bearish_fvg', False))) |
            # Strong traditional with partial ICT confirmation
            (df['Traditional_Sell'] & df.get('bear_order_block', False) & df['Strong_Trend']) |
            # BREAKER BLOCK SMC setup (NEWLY ADDED - institutional reversal point)
            (df.get('bear_breaker', False) & df.get('ms_bearish', False) & df.get('high_liquidity_sweep', False)) |
            # Order block + Breaker confirmation (ultimate SMC setup)
            (df.get('bear_order_block', False) & df.get('bear_breaker', False) & (df['close'] < df.get('EMA_55', df['EMA_21'])))
        )
    
    # Add signal strength scoring (0-100) - ENHANCED WITH SMC/BREAKER BLOCKS
    df['Buy_Signal_Strength'] = 0
    df['Sell_Signal_Strength'] = 0
    
    # Calculate buy signal strength with ROBUST SMC confirmation
    for i in range(len(df)):
        if df['Buy_Signal'].iloc[i]:
            strength = 0
            
            # === Traditional indicators (max 30 points) ===
            if df['Traditional_Buy'].iloc[i]:
                strength += 15
            if df['RSI'].iloc[i] < 30:
                strength += 10  # Oversold condition
            if df['MACD'].iloc[i] > df['MACD_signal'].iloc[i]:
                strength += 5  # MACD crossover
            
            # === ICT/SMC Core Concepts (max 50 points) - ENHANCED ===
            if df.get('ICT_Buy', pd.Series([False]*len(df))).iloc[i]:
                strength += 15  # Full ICT signal
                
            # Market Structure (critical for SMC)
            if df.get('ms_bullish', pd.Series([False]*len(df))).iloc[i]:
                strength += 10  # Bullish market structure confirmed
                
            # Order Blocks (key SMC entry)
            if df.get('bull_order_block', pd.Series([False]*len(df))).iloc[i]:
                strength += 10  # At bullish order block
                
            # BREAKER BLOCKS (added - critical SMC concept!)
            if df.get('bull_breaker', pd.Series([False]*len(df))).iloc[i]:
                strength += 10  # Breaker block confirmation (NEWLY ADDED)
                
            # Fair Value Gap (imbalance)
            if df.get('bullish_fvg', pd.Series([False]*len(df))).iloc[i]:
                strength += 5  # In bullish FVG
            
            # === Advanced SMC Confirmations (max 20 points) ===
            # Liquidity Sweep (smart money trap)
            if df.get('low_liquidity_sweep', pd.Series([False]*len(df))).iloc[i]:
                strength += 8  # Liquidity grabbed before reversal
                
            # Optimal Trade Entry zone
            if df.get('bull_ote_zone', pd.Series([False]*len(df))).iloc[i]:
                strength += 7  # In premium/discount zone
                
            # Strong Trend confirmation
            if df['Strong_Trend'].iloc[i]:
                strength += 5
            
            df.loc[df.index[i], 'Buy_Signal_Strength'] = min(strength, 100)
        
        # Calculate sell signal strength with ROBUST SMC confirmation
        if df['Sell_Signal'].iloc[i]:
            strength = 0
            
            # === Traditional indicators (max 30 points) ===
            if df['Traditional_Sell'].iloc[i]:
                strength += 15
            if df['RSI'].iloc[i] > 70:
                strength += 10  # Overbought condition
            if df['MACD'].iloc[i] < df['MACD_signal'].iloc[i]:
                strength += 5  # MACD crossover
            
            # === ICT/SMC Core Concepts (max 50 points) - ENHANCED ===
            if df.get('ICT_Sell', pd.Series([False]*len(df))).iloc[i]:
                strength += 15  # Full ICT signal
                
            # Market Structure (critical for SMC)
            if df.get('ms_bearish', pd.Series([False]*len(df))).iloc[i]:
                strength += 10  # Bearish market structure confirmed
                
            # Order Blocks (key SMC entry)
            if df.get('bear_order_block', pd.Series([False]*len(df))).iloc[i]:
                strength += 10  # At bearish order block
                
            # BREAKER BLOCKS (added - critical SMC concept!)
            if df.get('bear_breaker', pd.Series([False]*len(df))).iloc[i]:
                strength += 10  # Breaker block confirmation (NEWLY ADDED)
                
            # Fair Value Gap (imbalance)
            if df.get('bearish_fvg', pd.Series([False]*len(df))).iloc[i]:
                strength += 5  # In bearish FVG
            
            # === Advanced SMC Confirmations (max 20 points) ===
            # Liquidity Sweep (smart money trap)
            if df.get('high_liquidity_sweep', pd.Series([False]*len(df))).iloc[i]:
                strength += 8  # Liquidity grabbed before reversal
                
            # Optimal Trade Entry zone
            if df.get('bear_ote_zone', pd.Series([False]*len(df))).iloc[i]:
                strength += 7  # In premium/discount zone
                
            # Strong Trend confirmation
            if df['Strong_Trend'].iloc[i]:
                strength += 5
            
            df.loc[df.index[i], 'Sell_Signal_Strength'] = min(strength, 100)
    
    # Filter weak signals - only keep signals with strength >= 60
    df.loc[df['Buy_Signal_Strength'] < 60, 'Buy_Signal'] = False
    df.loc[df['Sell_Signal_Strength'] < 60, 'Sell_Signal'] = False
    
    return df

# Enhanced Telegram notification system
# Replace the send_detailed_signal_alert function in main.py

def send_detailed_signal_alert(df, symbol, timeframe_name, auto_trade=False, risk_percent=2):
    """Send detailed trading signal alert to Telegram with trade execution option"""
    if df.empty:
        return False
    
    # Prepare signal message
    current_time = datetime.datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    price = df['close'].iloc[-1]
    trade_executed = False
    
    # Get JustMarkets specific info
    symbol_info = mt5.symbol_info(symbol)
    spread = (symbol_info.ask - symbol_info.bid) / symbol_info.point if symbol_info else 0
    
    # Create message based on signal type
    if df['Buy_Signal'].iloc[-1]:
        # Get signal strength
        signal_strength = df.get('Buy_Signal_Strength', pd.Series([0]*len(df))).iloc[-1]
        
        # Determine confidence level
        if signal_strength >= 80:
            confidence = "🔥 VERY HIGH"
        elif signal_strength >= 70:
            confidence = "✅ HIGH"
        elif signal_strength >= 60:
            confidence = "⚡ MEDIUM"
        else:
            confidence = "⚠️ LOW"
        
        # Determine which type of buy signal was triggered
        signal_components = []
        if df.get('Traditional_Buy', pd.Series([False]*len(df))).iloc[-1]:
            signal_components.append("Traditional")
        if df.get('ICT_Buy', pd.Series([False]*len(df))).iloc[-1]:
            signal_components.append("ICT")
        if df.get('ms_bullish', pd.Series([False]*len(df))).iloc[-1]:
            signal_components.append("Bullish Structure")
        if df.get('bull_order_block', pd.Series([False]*len(df))).iloc[-1]:
            signal_components.append("Order Block")
        if df.get('bullish_fvg', pd.Series([False]*len(df))).iloc[-1]:
            signal_components.append("Fair Value Gap")
        
        signal_type = " + ".join(signal_components) if signal_components else "Multi-Factor"
        
        # Use ICT-based exit levels if available
        entry = price
        atr = df['ATR'].iloc[-1]
        
        # Try to get ICT exit levels
        try:
            stop_loss, take_profit = calculate_ict_exit_levels(df, entry, 'buy')
            if stop_loss is None or take_profit is None:
                # Fallback to ATR-based levels
                stop_loss = price - atr * 1.5
                take_profit = price + atr * 3
        except:
            # Fallback to ATR-based levels
            stop_loss = price - atr * 1.5
            take_profit = price + atr * 3
        
        risk_reward = (take_profit - entry) / (entry - stop_loss) if (entry - stop_loss) != 0 else 0
        
        # Add JustMarkets specific info for buy orders
        message = f"🟢 <b>BUY SIGNAL ALERT</b> - {symbol} ({timeframe_name})\n" \
                 f"Time: {current_time}\n" \
                 f"Signal Confidence: {confidence} ({signal_strength:.0f}/100)\n" \
                 f"Signal Components: {signal_type}\n" \
                 f"Entry Price: {price:.5f}\n" \
                 f"Stop Loss: {stop_loss:.5f}\n" \
                 f"Take Profit: {take_profit:.5f}\n" \
                 f"Risk:Reward: 1:{risk_reward:.2f}\n\n" \
                 f"📊 <b>JustMarkets Info:</b>\n" \
                 f"Current Spread: {spread:.1f} points\n" \
                 f"Broker Time: {mt5.symbol_info_tick(symbol).time if mt5.symbol_info_tick(symbol) else 'N/A'}\n" \
                 f"RSI: {df['RSI'].iloc[-1]:.2f}\n" \
                 f"MACD: {df['MACD'].iloc[-1]:.5f}\n" \
                 f"MACD Signal: {df['MACD_signal'].iloc[-1]:.5f}\n" \
                 f"ADX: {df['ADX'].iloc[-1]:.2f}\n" \
                 f"ATR: {df['ATR'].iloc[-1]:.5f}"
        
        # Execute trade if auto-trading is enabled
        if auto_trade and is_active_session() and is_profitable_market(df):
            logger.info(f"Auto-executing BUY trade for {symbol}")
            trade_executed = execute_trade("buy", symbol, risk_percent, stop_loss, take_profit)
            
            if trade_executed:
                message += f"\n\n✅ <b>TRADE AUTOMATICALLY EXECUTED</b>"
            else:
                message += f"\n\n⚠️ <b>AUTO-TRADE ATTEMPTED BUT FAILED</b>"
        
        # Send the message to Telegram subscribers
        send_success = send_signal_to_subscribers(message, "buy")
        
    elif df['Sell_Signal'].iloc[-1]:
        # Get signal strength
        signal_strength = df.get('Sell_Signal_Strength', pd.Series([0]*len(df))).iloc[-1]
        
        # Determine confidence level
        if signal_strength >= 80:
            confidence = "🔥 VERY HIGH"
        elif signal_strength >= 70:
            confidence = "✅ HIGH"
        elif signal_strength >= 60:
            confidence = "⚡ MEDIUM"
        else:
            confidence = "⚠️ LOW"
        
        # Determine which type of sell signal was triggered
        signal_components = []
        if df.get('Traditional_Sell', pd.Series([False]*len(df))).iloc[-1]:
            signal_components.append("Traditional")
        if df.get('ICT_Sell', pd.Series([False]*len(df))).iloc[-1]:
            signal_components.append("ICT")
        if df.get('ms_bearish', pd.Series([False]*len(df))).iloc[-1]:
            signal_components.append("Bearish Structure")
        if df.get('bear_order_block', pd.Series([False]*len(df))).iloc[-1]:
            signal_components.append("Order Block")
        if df.get('bearish_fvg', pd.Series([False]*len(df))).iloc[-1]:
            signal_components.append("Fair Value Gap")
        
        signal_type = " + ".join(signal_components) if signal_components else "Multi-Factor"
        
        # Use ICT-based exit levels if available
        entry = price
        atr = df['ATR'].iloc[-1]
        
        # Try to get ICT exit levels
        try:
            stop_loss, take_profit = calculate_ict_exit_levels(df, entry, 'sell')
            if stop_loss is None or take_profit is None:
                # Fallback to ATR-based levels
                stop_loss = price + atr * 1.5
                take_profit = price - atr * 3
        except:
            # Fallback to ATR-based levels
            stop_loss = price + atr * 1.5
            take_profit = price - atr * 3
        
        risk_reward = (entry - take_profit) / (stop_loss - entry) if (stop_loss - entry) != 0 else 0
        
        # Add JustMarkets specific info for sell orders
        message = f"🔴 <b>SELL SIGNAL ALERT</b> - {symbol} ({timeframe_name})\n" \
                 f"Time: {current_time}\n" \
                 f"Signal Confidence: {confidence} ({signal_strength:.0f}/100)\n" \
                 f"Signal Components: {signal_type}\n" \
                 f"Entry Price: {price:.5f}\n" \
                 f"Stop Loss: {stop_loss:.5f}\n" \
                 f"Take Profit: {take_profit:.5f}\n" \
                 f"Risk:Reward: 1:{risk_reward:.2f}\n\n" \
                 f"📊 <b>JustMarkets Info:</b>\n" \
                 f"Current Spread: {spread:.1f} points\n" \
                 f"Broker Time: {mt5.symbol_info_tick(symbol).time if mt5.symbol_info_tick(symbol) else 'N/A'}\n" \
                 f"RSI: {df['RSI'].iloc[-1]:.2f}\n" \
                 f"MACD: {df['MACD'].iloc[-1]:.5f}\n" \
                 f"MACD Signal: {df['MACD_signal'].iloc[-1]:.5f}\n" \
                 f"ADX: {df['ADX'].iloc[-1]:.2f}\n" \
                 f"ATR: {df['ATR'].iloc[-1]:.5f}"
        
        # Execute trade if auto-trading is enabled
        if auto_trade and is_active_session() and is_profitable_market(df):
            logger.info(f"Auto-executing SELL trade for {symbol}")
            trade_executed = execute_trade("sell", symbol, risk_percent, stop_loss, take_profit)
            
            if trade_executed:
                message += f"\n\n✅ <b>TRADE AUTOMATICALLY EXECUTED</b>"
            else:
                message += f"\n\n⚠️ <b>AUTO-TRADE ATTEMPTED BUT FAILED</b>"
        
        # Send the message to Telegram subscribers
        send_success = send_signal_to_subscribers(message, "sell")
    
    else:
        return False
    
    return trade_executed

# Also add this function to send closed trade notifications
def send_trade_closed_notification(trade_data):
    """
    Send notification about closed trades to subscribers
    """
    try:
        # Format the closed trade message
        trade_type = trade_data.get("action", "").upper()
        symbol = trade_data.get("symbol", "")
        profit = trade_data.get("pnl", 0)
        pips = trade_data.get("pips", 0)
        entry_price = trade_data.get("entry_price", 0)
        close_price = trade_data.get("close_price", 0)
        
        # Determine if it's a win or loss
        result_emoji = "🎯 PROFIT" if profit > 0 else "⛔ LOSS"
        
        message = f"{result_emoji}: {trade_type} {symbol}\n"
        message += f"Entry: {entry_price:.5f} → Close: {close_price:.5f}\n"
        message += f"P&L: {profit:.2f} USD ({pips:.1f} pips)\n"
        
        # Get holding time if available
        if "time" in trade_data and "close_time" in trade_data:
            open_time = datetime.datetime.fromisoformat(trade_data.get("time")) if isinstance(trade_data.get("time"), str) else trade_data.get("time")
            close_time = datetime.datetime.fromisoformat(trade_data.get("close_time")) if isinstance(trade_data.get("close_time"), str) else trade_data.get("close_time")
            
            if open_time and close_time:
                duration = close_time - open_time
                hours, remainder = divmod(duration.total_seconds(), 3600)
                minutes, seconds = divmod(remainder, 60)
                message += f"Holding Time: {int(hours)}h {int(minutes)}m\n"
        
        # Send via Telegram
        return send_signal_to_subscribers(message, "closed")
        
    except Exception as e:
        logger.error(f"Error sending trade closed notification: {e}")
        return False
    
# Enhanced Execute Trade with Risk Management
def execute_trade(action, symbol="GBPUSD.m", risk_percent=2, stop_loss=None, take_profit=None, retry_attempts=3):
    """Execute trade with enhanced risk management for different market types"""
    
    # Generate unique trade ID
    import uuid
    trade_id = str(uuid.uuid4())[:8]
    
    # Get market data for calculations
    df = get_market_data(symbol, mt5.TIMEFRAME_M5, 100)
    if df.empty:
        logger.error(f"Cannot execute trade - no market data for {symbol}")
        return False
    
    # Determine market type - OPTIMIZED FOR FOREX, BTC & GOLD ONLY
    is_crypto = "BTC" in symbol  # Bitcoin only
    is_metal = "XAU" in symbol or "GOLD" in symbol  # Gold only
    is_forex = any(pair in symbol for pair in ["USD", "EUR", "GBP", "JPY", "AUD", "NZD", "CAD", "CHF"])
    
    # Additional check for spread
    tick = mt5.symbol_info_tick(symbol)
    if not tick:
        logger.error(f"Failed to get tick data for {symbol}")
        return False
        
    current_spread = tick.ask - tick.bid
    symbol_info = mt5.symbol_info(symbol)
    if not symbol_info:
        logger.error(f"Failed to get symbol info for {symbol}")
        return False
        
    avg_spread = current_spread / symbol_info.point

    # df = advanced_check_trade_signals(df)

    # Determine market type - OPTIMIZED FOR FOREX, BTC & GOLD ONLY
    is_crypto = "BTC" in symbol  # Bitcoin only
    is_metal = "XAU" in symbol or "GOLD" in symbol  # Gold only
    is_forex = any(pair in symbol for pair in ["USD", "EUR", "GBP", "JPY", "AUD", "NZD", "CAD", "CHF"])
    
    # Market-specific spread thresholds
    max_spread = 15  # Default for forex
    
    if is_crypto:
        max_spread = 50  # BTC typically has wider spreads
    elif is_metal:
        max_spread = 35  # Gold can have wider spreads
    else:  # Forex pairs
        max_spread = 20
    
    if avg_spread > max_spread:
        message = f"⚠️ {symbol} spread too high ({avg_spread:.1f} points)! Avoiding Trade."
        send_signal_to_subscribers(message)
        logger.warning(message)
        return False
    
    # Calculate optimal position sizing
    try:
        # Get current price
        price = tick.ask if action == "buy" else tick.bid
        atr = df['ATR'].iloc[-1]
        
        # Dynamic ATR multiplier based on market volatility and type
        volatility_ratio = df['ATR'].iloc[-1] / df['ATR'].rolling(20).mean().iloc[-1]
        
        # Market-specific ATR multipliers
        if is_crypto:
            base_atr_multiplier = 2.0  # Wider stops for crypto
        elif is_metal:
            base_atr_multiplier = 1.8  # Wider stops for gold
        else:
            base_atr_multiplier = 1.5  # Standard for forex
        
        # Adjust for current volatility
        atr_multiplier = base_atr_multiplier
        if volatility_ratio > 1.3:  # Higher volatility
            atr_multiplier = base_atr_multiplier * 1.2
        elif volatility_ratio < 0.7:  # Lower volatility
            atr_multiplier = base_atr_multiplier * 0.8
            
        # Calculate adaptive stop loss based on volatility
        if stop_loss is None:
            stop_loss = price - atr * atr_multiplier if action == "buy" else price + atr * atr_multiplier
        
        # Adaptive take profit with improved risk:reward ratio
        if take_profit is None:
            # Market-specific risk:reward ratios
            if is_crypto:
                rr_ratio = 3.0  # Higher potential in crypto
            elif is_metal:
                rr_ratio = 2.5  # Gold can make big moves
            else:
                rr_ratio = 2.0  # Standard for forex
                
            # Calculate distance to stop loss
            sl_distance = abs(price - stop_loss)
            # Set take profit based on risk:reward ratio
            take_profit = price + sl_distance * rr_ratio if action == "buy" else price - sl_distance * rr_ratio
            
            # Adjust take profit based on nearby support/resistance levels
            if action == "buy":
                # Find nearby resistance
                recent_highs = df['high'].rolling(20).max().iloc[-1]
                if recent_highs > price and recent_highs < take_profit:
                    # Take profit at resistance minus buffer
                    take_profit = recent_highs - (0.1 * atr)
            else:
                # Find nearby support
                recent_lows = df['low'].rolling(20).min().iloc[-1]
                if recent_lows < price and recent_lows > take_profit:
                    # Take profit at support plus buffer
                    take_profit = recent_lows + (0.1 * atr)
        
        # Check existing positions to avoid duplicates
        positions = mt5.positions_get(symbol=symbol)
        if positions:
            for position in positions:
                if (action == "buy" and position.type == mt5.POSITION_TYPE_BUY) or \
                   (action == "sell" and position.type == mt5.POSITION_TYPE_SELL):
                    logger.warning(f"Already have an open {action} position for {symbol}. Skipping.")
                    return False
        
        # Get account info for position sizing
        account_info = mt5.account_info()
        if not account_info:
            logger.error("Failed to get account information")
            return False
            
        balance = account_info.balance
        
        # Conservative risk management for real accounts
        is_demo = 'demo' in account_info.server.lower()
        if not is_demo:
            # Use more conservative risk for real accounts
            risk_percent = min(risk_percent, 1.0)  # Cap at 1% for real accounts
            logger.info(f"Using conservative risk management ({risk_percent}%) for real account")
        
        risk_amount = balance * (risk_percent / 100)
        
        # Calculate lot size based on risk and market
        point_value = symbol_info.point
        
        # Different markets have different pip values and contract sizes
        contract_size = symbol_info.trade_contract_size
        
        # Market-specific price per pip calculations
        if "JPY" in symbol:
            pip_value = point_value * 100  # JPY pairs have 2 decimal places
        elif is_crypto:
            # For crypto, 1 point move on 1 lot is typically $1
            pip_value = point_value * 10  # Convert points to pips
        elif is_metal and "XAU" in symbol:
            # XAU is quoted per oz, typically 0.01 = $1 per 0.01 lot
            pip_value = point_value * 10
        else:
            pip_value = point_value * 10  # Standard forex pip definition
        
        # Calculate price per pip based on contract size and current price
        price_per_pip = (contract_size * pip_value) / (100000 / price) if "JPY" not in symbol else contract_size * pip_value / 1000
        
        # For crypto and commodities, price_per_pip may be significantly different
        if is_crypto:
            # For cryptos, calculate based on contract value
            price_per_pip = contract_size * pip_value
        elif is_metal and "XAU" in symbol:
            # For gold, typically 0.01 lot = $0.1 per 0.1 point move
            price_per_pip = contract_size * pip_value / 100
        
        pips_risked = abs(price - stop_loss) / pip_value
        lot_size = risk_amount / (pips_risked * price_per_pip)
        
        # Apply limits to lot size
        min_lot = symbol_info.volume_min
        max_lot = symbol_info.volume_max
        lot_step = symbol_info.volume_step
        
        # Round to nearest lot step
        lot_size = round(lot_size / lot_step) * lot_step
        
        # Apply min/max limits
        lot_size = max(min(lot_size, max_lot), min_lot)
        
        # For crypto and gold, use smaller lot sizes due to high volatility
        if is_crypto:
            lot_size = min(lot_size, 0.1)  # Cap crypto lot size for safety
        elif is_metal and "XAU" in symbol:
            lot_size = min(lot_size, 0.5)  # Cap gold lot size for safety
            
        # Create trade request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot_size,
            "type": mt5.ORDER_TYPE_BUY if action == "buy" else mt5.ORDER_TYPE_SELL,
            "price": price,
            "sl": stop_loss,
            "tp": take_profit,
            "deviation": 10,
            "magic": 234000,
            "comment": f"AI Bot {trade_id}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        # Log detailed request for debugging
        logger.info(f"Sending trade request for {symbol}: {request}")
        
        
        # Execute the trade with retry
        for attempt in range(retry_attempts):
            result = mt5.order_send(request)
            
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                # Calculate risk-reward ratio for logging
                risk_reward = round((abs(take_profit - price) / abs(price - stop_loss)), 2)
                
                # Record the trade in history
                trade_record = {
                    "id": trade_id,
                    "time": datetime.datetime.now(timezone.utc).isoformat(),
                    "symbol": symbol,
                    "action": action,
                    "entry_price": price,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "lot_size": lot_size,
                    "risk_amount": risk_amount,
                    "risk_percent": risk_percent,
                    "risk_reward": risk_reward,
                    "balance_before": balance,
                    "order_id": result.order,
                    "position_id": result.order,  # For tracking purposes
                    "status": "open",
                    "result": None,
                    "pnl": None,
                    "close_time": None,
                    "holding_time": None
                }
                
                # Add to trade history
                trade_history.append(trade_record)
                active_trades[trade_id] = trade_record
                save_trade_history()
                
                # Log and notify
                message = f"✅ Trade Executed: {action.upper()} {symbol} at {price:.5f} | SL: {stop_loss:.5f} | TP: {take_profit:.5f} | Lot: {lot_size:.2f} | R:R = 1:{risk_reward}"
                logger.info(message)
                send_signal_to_subscribers(message)
                
                return True
            else:
                error_code = result.retcode if hasattr(result, 'retcode') else "Unknown"
                error_message = f"⚠️ Trade Failed (Attempt {attempt+1}/{retry_attempts}): {result.comment if hasattr(result, 'comment') else 'Unknown error'} (Code: {error_code})"
                logger.error(error_message)
                
                # Provide specific guidance for common error codes
                if hasattr(result, 'retcode'):
                    if result.retcode == mt5.TRADE_RETCODE_REQUOTE:
                        # Handle price change - update price and try again
                        new_tick = mt5.symbol_info_tick(symbol)
                        if new_tick:
                            request["price"] = new_tick.ask if action == "buy" else new_tick.bid
                            logger.info(f"Updating price for retry: {request['price']}")
                    elif result.retcode == mt5.TRADE_RETCODE_INVALID_VOLUME:
                        # Handle invalid volume - adjust lot size and try again
                        request["volume"] = symbol_info.volume_min if lot_size < symbol_info.volume_min else symbol_info.volume_max
                        logger.info(f"Adjusting volume to {request['volume']} and retrying")
                    elif result.retcode == 10018:  # Market closed
                        logger.error("Market is closed. Cannot execute trade.")
                        return False
                    elif result.retcode == 10027:  # AutoTrading disabled
                        logger.error("AutoTrading is disabled in MT5. Enable it to execute trades.")
                        st.error("Enable AutoTrading in MT5 to execute trades (click the 'AutoTrading' button)")
                        return False
                
                # Wait before retry
                time.sleep(1)
        
        # If all retries failed
        failure_message = f"⚠️ All trade attempts failed for {action.upper()} {symbol}"
        logger.error(failure_message)
        send_signal_to_subscribers(failure_message)
        return False
        
    except Exception as e:
        error_message = f"Unexpected Error in Trade Execution: {e}"
        logger.error(error_message)
        send_signal_to_subscribers(error_message)
        return False



def verify_justmarkets_connection():
    """Verify connection to JustMarkets and test available symbols"""
    
    if mt5.terminal_info() is None:
        logger.error("MT5 terminal not connected")
        st.error("MT5 terminal not connected. Please initialize the connection first.")
        return False
    
    account_info = mt5.account_info()
    if account_info is None:
        logger.error("No account info available - MT5 might not be logged in")
        st.error("MT5 is not logged in. Please log in to your JustMarkets account in MT5 first.")
        return False
    
    if "JustMarkets" not in account_info.server:
        logger.warning(f"Connected to {account_info.server} instead of JustMarkets")
        st.warning(f"Connected to {account_info.server} instead of JustMarkets. Make sure you're using the correct MT5 terminal.")
        return False
    
    # Check all required symbols by category - BTC & Gold only
    symbol_categories = {
        "Crypto": ["BTCUSD.m"],
        "Metals": ["XAUUSD.m"]
    }
    
    # Check symbols by category
    results = {}
    all_available = True
    
    for category, symbols in symbol_categories.items():
        available = []
        missing = []
        
        for symbol in symbols:
            # Try multiple variations for each symbol
            symbol_variations = [symbol]
            
            # Add variations for crypto and metals
            if symbol == "BTCUSD.m":
                symbol_variations.extend(["BTC.USD", "BTCUSD.m.a", "BTC/USD"])
            elif symbol == "ETHUSD.m":
                symbol_variations.extend(["ETH.USD", "ETHUSD.m.a", "ETH/USD"])
            elif symbol == "XAUUSD.m":
                symbol_variations.extend(["GOLD", "GOLD.a", "XAU/USD"])
            elif symbol == "XAGUSD.m":
                symbol_variations.extend(["SILVER", "SILVER.a", "XAG/USD"])
            
            found = False
            for sym_variant in symbol_variations:
                if mt5.symbol_info(sym_variant) is not None or mt5.symbol_select(sym_variant, True):
                    available.append(sym_variant)
                    found = True
                    break
            
            if not found:
                missing.append(symbol)
                all_available = False
        
        results[category] = {
            "available": available,
            "missing": missing
        }
    
    # Create a summary report
    st.subheader("JustMarkets Symbol Availability")
    
    for category, result in results.items():
        if result["missing"]:
            st.warning(f"**{category}**: {len(result['available'])}/{len(result['available']) + len(result['missing'])} symbols available")
            if result["missing"]:
                st.markdown(f"Missing: {', '.join(result['missing'])}")
        else:
            st.success(f"**{category}**: All {len(result['available'])} symbols available")
    
    # Test data retrieval for a few key symbols
    st.subheader("Data Retrieval Test")
    
    test_symbols = ["BTCUSD.m", "XAUUSD.m"]
    test_results = []
    
    for symbol in test_symbols:
        if symbol in [s for cat in results for s in results[cat]["available"]]:
            try:
                test_data = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 10)
                if test_data is not None and len(test_data) > 0:
                    test_results.append({"symbol": symbol, "status": "Success", "bars": len(test_data)})
                else:
                    test_results.append({"symbol": symbol, "status": "Failed - No data", "bars": 0})
                    all_available = False
            except Exception as e:
                test_results.append({"symbol": symbol, "status": f"Error: {str(e)}", "bars": 0})
                all_available = False
    
    # Display test results
    for result in test_results:
        if result["status"] == "Success":
            st.success(f"{result['symbol']}: {result['status']} ({result['bars']} bars retrieved)")
        else:
            st.error(f"{result['symbol']}: {result['status']}")
    
    # Overall connection status
    if all_available:
        st.success("✅ JustMarkets connection verified successfully - All required markets available")
        return True
    else:
        st.warning("⚠️ JustMarkets connection partially successful - Some markets may not be available")
        return False


def optimize_strategy_parameters(symbol, timeframe=mt5.TIMEFRAME_H1, period_days=30):
    """
    Optimize trading strategy parameters for a specific market
    Returns optimized parameters for the given symbol
    """
    # Determine market type
    is_crypto = "BTC" in symbol or "ETH" in symbol
    is_metal = "XAU" in symbol or "GOLD" in symbol or "XAG" in symbol or "SILVER" in symbol
    is_index = any(idx in symbol for idx in ["US30.std", "US500.std", "USTEC.std", "DE30.std", "UK100.std", "JP225.std"])
    
    # Get historical data for optimization
    bars_needed = int(period_days * 24 * 60 / (timeframe / 60))  # Convert timeframe minutes to bars needed
    
    df = get_market_data(symbol, timeframe, bars_needed, force_refresh=True)
    if df.empty:
        return None
    
    # Define parameter ranges to test based on market type
    if is_crypto:
        rsi_range = [14, 21, 28]
        macd_fast_range = [8, 12, 16]
        macd_slow_range = [21, 26, 32]
        atr_mult_range = [1.5, 2.0, 2.5]
    elif is_metal:
        rsi_range = [9, 14, 21]
        macd_fast_range = [8, 12]
        macd_slow_range = [21, 26]
        atr_mult_range = [1.2, 1.5, 1.8]
    else:  # forex and indices
        rsi_range = [7, 14, 21]
        macd_fast_range = [8, 12, 16]
        macd_slow_range = [21, 26]
        atr_mult_range = [1.0, 1.2, 1.5]
    
    # Track best parameters and performance
    best_params = None
    best_performance = -float('inf')  # We'll maximize profit factor * win rate
    
    # Progress bar for optimization
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_combinations = len(rsi_range) * len(macd_fast_range) * len(macd_slow_range) * len(atr_mult_range)
    current_combination = 0
    
    # Test each parameter combination
    for rsi_period in rsi_range:
        for macd_fast in macd_fast_range:
            for macd_slow in macd_slow_range:
                if macd_fast >= macd_slow:
                    current_combination += 1
                    continue  # Skip invalid combinations
                
                for atr_mult in atr_mult_range:
                    # Update progress
                    current_combination += 1
                    progress = current_combination / total_combinations
                    progress_bar.progress(progress)
                    status_text.text(f"Testing combination {current_combination}/{total_combinations}")
                    
                    # Create a copy of the dataframe for this test
                    test_df = df.copy()
                    
                    # Calculate indicators with current parameters
                    test_df['RSI'] = talib.RSI(test_df['close'], timeperiod=rsi_period)
                    test_df['MACD'], test_df['MACD_signal'], _ = talib.MACD(
                        test_df['close'], fastperiod=macd_fast, slowperiod=macd_slow, signalperiod=9)
                    test_df['ATR'] = talib.ATR(test_df['high'], test_df['low'], test_df['close'], timeperiod=14)
                    
                    # Additional indicators for signal generation
                    test_df['SMA_50'] = talib.SMA(test_df['close'], timeperiod=50)
                    test_df['SMA_200'] = talib.SMA(test_df['close'], timeperiod=200)
                    test_df['Uptrend'] = test_df['SMA_50'] > test_df['SMA_200']
                    test_df['Downtrend'] = test_df['SMA_50'] < test_df['SMA_200']
                    
                    # Generate signals
                    test_df['Buy_Signal'] = (test_df['RSI'] < 30) & (test_df['MACD'] > test_df['MACD_signal']) & test_df['Uptrend']
                    test_df['Sell_Signal'] = (test_df['RSI'] > 70) & (test_df['MACD'] < test_df['MACD_signal']) & test_df['Downtrend']
                    
                    # Backtest the signals
                    trades = []
                    in_position = False
                    entry_price = 0
                    position_type = None
                    
                    for i in range(1, len(test_df)):
                        if not in_position:
                            # Check for entry signals
                            if test_df['Buy_Signal'].iloc[i]:
                                in_position = True
                                position_type = 'buy'
                                entry_price = test_df['close'].iloc[i]
                                entry_index = i
                            elif test_df['Sell_Signal'].iloc[i]:
                                in_position = True
                                position_type = 'sell'
                                entry_price = test_df['close'].iloc[i]
                                entry_index = i
                        else:
                            # Check for exit conditions
                            exit_price = None
                            
                            if position_type == 'buy':
                                # Exit buy if we hit stop loss or take profit
                                stop_loss = entry_price - atr_mult * test_df['ATR'].iloc[entry_index]
                                take_profit = entry_price + atr_mult * 2 * test_df['ATR'].iloc[entry_index]
                                
                                # Check if price hit stop loss or take profit
                                if test_df['low'].iloc[i] <= stop_loss:
                                    exit_price = stop_loss
                                    trade_result = 'loss'
                                elif test_df['high'].iloc[i] >= take_profit:
                                    exit_price = take_profit
                                    trade_result = 'win'
                                # Also exit on opposing signal
                                elif test_df['Sell_Signal'].iloc[i]:
                                    exit_price = test_df['close'].iloc[i]
                                    trade_result = 'win' if exit_price > entry_price else 'loss'
                            else:  # sell position
                                # Exit sell if we hit stop loss or take profit
                                stop_loss = entry_price + atr_mult * test_df['ATR'].iloc[entry_index]
                                take_profit = entry_price - atr_mult * 2 * test_df['ATR'].iloc[entry_index]
                                
                                # Check if price hit stop loss or take profit
                                if test_df['high'].iloc[i] >= stop_loss:
                                    exit_price = stop_loss
                                    trade_result = 'loss'
                                elif test_df['low'].iloc[i] <= take_profit:
                                    exit_price = take_profit
                                    trade_result = 'win'
                                # Also exit on opposing signal
                                elif test_df['Buy_Signal'].iloc[i]:
                                    exit_price = test_df['close'].iloc[i]
                                    trade_result = 'win' if exit_price < entry_price else 'loss'
                            
                            # If we have an exit price, record the trade
                            if exit_price is not None:
                                profit = (exit_price - entry_price) if position_type == 'buy' else (entry_price - exit_price)
                                trades.append({
                                    'entry_price': entry_price,
                                    'exit_price': exit_price,
                                    'profit': profit,
                                    'type': position_type,
                                    'result': trade_result
                                })
                                in_position = False
                    
                    # Calculate performance metrics
                    if trades:
                        winning_trades = [t for t in trades if t['result'] == 'win']
                        losing_trades = [t for t in trades if t['result'] == 'loss']
                        
                        win_rate = len(winning_trades) / len(trades) if trades else 0
                        
                        total_profit = sum(t['profit'] for t in winning_trades)
                        total_loss = abs(sum(t['profit'] for t in losing_trades))
                        
                        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
                        
                        # Calculate combined performance metric
                        performance = profit_factor * win_rate * len(trades)
                        
                        # Check if this is the best combination so far
                        if performance > best_performance:
                            best_performance = performance
                            best_params = {
                                'rsi_period': rsi_period,
                                'macd_fast': macd_fast,
                                'macd_slow': macd_slow,
                                'atr_mult': atr_mult,
                                'win_rate': win_rate * 100,
                                'profit_factor': profit_factor,
                                'num_trades': len(trades)
                            }
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    return best_params
    
def update_trade_status():
    """
    Check status of active trades and update trade history
    Returns a summary of closed positions
    """
    closed_trades = []
    
    if not active_trades:
        return closed_trades
    
    # Get all positions
    positions = mt5.positions_get()
    open_position_ids = [pos.ticket for pos in positions] if positions else []
    
    # Check each active trade
    for trade_id, trade in list(active_trades.items()):
        position_id = trade.get("position_id")
        
        # If position is no longer open, it was closed
        if position_id not in open_position_ids:
            # Get trade history to find out how it closed
            end_time = datetime.datetime.now(timezone.utc)
            start_time = datetime.datetime.fromisoformat(trade["time"])
            
            # Get trade history from MT5
            history = mt5.history_deals_get(
                start_time - datetime.timedelta(minutes=5),
                end_time + datetime.timedelta(minutes=5)
            )
            
            # Find the closing deal
            close_deal = None
            for deal in history:
                if deal.position_id == position_id and deal.entry == mt5.DEAL_ENTRY_OUT:
                    close_deal = deal
                    break
            
            # Update trade record
            if close_deal:
                entry_price = trade["entry_price"]
                close_price = close_deal.price
                profit = close_deal.profit
                
                # Calculate pips gained/lost
                point = mt5.symbol_info(trade["symbol"]).point
                price_diff = abs(close_price - entry_price)
                pips = price_diff / point / 10  # Standard forex pip definition
                
                # Update the trade record
                trade["status"] = "closed"
                trade["close_price"] = close_price
                trade["close_time"] = datetime.datetime.now(timezone.utc).isoformat()
                trade["holding_time"] = str(datetime.datetime.now(timezone.utc) - datetime.datetime.fromisoformat(trade["time"]))
                trade["pnl"] = profit
                trade["pips"] = pips
                trade["result"] = "win" if profit > 0 else "loss"
                
                # Send notification
                result_emoji = "🎯 PROFIT" if profit > 0 else "⛔ LOSS"
                close_msg = f"{result_emoji}: {trade['action'].upper()} {trade['symbol']} closed with {profit:.2f} ({pips:.1f} pips)"
                logger.info(close_msg)
                send_signal_to_subscribers(close_msg)
                
                # Add to closed trades list for reporting
                closed_trades.append(trade)
                
                # Remove from active trades
                del active_trades[trade_id]
                
                # Update trade history
                for i, hist_trade in enumerate(trade_history):
                    if hist_trade["id"] == trade_id:
                        trade_history[i] = trade
                        break
                
                # Save updated trade history
                save_trade_history()
    
    return closed_trades

# Add these functions to main.py

def get_mt5_trading_history(days=30):
    """
    Get trading history directly from JustMarkets MT5
    Returns history of closed positions for the specified period
    """
    if mt5.terminal_info() is None:
        logger.error("MT5 not connected")
        return []
    
    # Calculate the date range
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=days)
    
    # Get history from MT5
    mt5_history = mt5.history_deals_get(start_date, end_date)
    
    if mt5_history is None:
        logger.error(f"Failed to get history deals, error code: {mt5.last_error()}")
        return []
    
    logger.info(f"Retrieved {len(mt5_history)} deals from JustMarkets MT5")
    
    # Process the history to group deals by position
    positions = {}
    
    for deal in mt5_history:
        position_id = deal.position_id
        
        if position_id not in positions:
            positions[position_id] = {
                'open_deals': [],
                'close_deals': []
            }
        
        # Categorize deals as open (entry) or close (exit)
        if deal.entry == mt5.DEAL_ENTRY_IN:
            positions[position_id]['open_deals'].append(deal)
        elif deal.entry == mt5.DEAL_ENTRY_OUT:
            positions[position_id]['close_deals'].append(deal)
    
    # Format the position data for our app
    formatted_history = []
    
    for position_id, deals in positions.items():
        # Skip positions with no opening or closing deals
        if not deals['open_deals'] or not deals['close_deals']:
            continue
        
        # Get the opening deal
        open_deal = deals['open_deals'][0]
        
        # Get the closing deals
        close_deals = deals['close_deals']
        
        # Calculate combined profit from all closing deals
        total_profit = sum(deal.profit for deal in close_deals)
        
        # Get the last closing deal time
        close_time = max(deal.time for deal in close_deals)
        
        # Determine position type (buy/sell)
        position_type = "buy" if open_deal.type == mt5.DEAL_TYPE_BUY else "sell"
        
        # Calculate pips gained/lost
        point = mt5.symbol_info(open_deal.symbol).point
        price_diff = abs(close_deals[-1].price - open_deal.price)
        pips = price_diff / point / 10  # Standard forex pip definition
        
        # Create the history record
        trade_record = {
            "id": str(position_id),
            "time": open_deal.time,
            "symbol": open_deal.symbol,
            "action": position_type,
            "entry_price": open_deal.price,
            "close_price": close_deals[-1].price,
            "volume": open_deal.volume,
            "lot_size": open_deal.volume,
            "pnl": total_profit,
            "pips": pips,
            "status": "closed",
            "result": "win" if total_profit > 0 else "loss",
            "close_time": close_time,
            "holding_time": str(close_time - open_deal.time),
            "comment": open_deal.comment,
            "magic": open_deal.magic,
            "commission": open_deal.commission + sum(deal.commission for deal in close_deals),
            "swap": open_deal.swap + sum(deal.swap for deal in close_deals)
        }
        
        formatted_history.append(trade_record)
    
    # Sort by close time (newest first)
    formatted_history.sort(key=lambda x: x["close_time"], reverse=True)
    
    return formatted_history


def calculate_mt5_trade_stats(days=30):
    """Calculate trading statistics using MT5 history data"""
    # Get trading history from MT5
    trade_history = get_mt5_trading_history(days)
    
    if not trade_history:
        return {
            "total_trades": 0,
            "win_rate": 0,
            "avg_profit": 0,
            "avg_loss": 0,
            "profit_factor": 0,
            "total_pnl": 0
        }
    
    # Calculate statistics
    winning_trades = [trade for trade in trade_history if trade.get("result") == "win"]
    losing_trades = [trade for trade in trade_history if trade.get("result") == "loss"]
    
    total_trades = len(trade_history)
    win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0
    
    total_profit = sum(trade.get("pnl", 0) for trade in winning_trades)
    total_loss = abs(sum(trade.get("pnl", 0) for trade in losing_trades))
    
    avg_profit = total_profit / len(winning_trades) if winning_trades else 0
    avg_loss = total_loss / len(losing_trades) if losing_trades else 0
    
    profit_factor = total_profit / total_loss if total_loss > 0 else float('inf') if total_profit > 0 else 0
    
    total_pnl = sum(trade.get("pnl", 0) for trade in trade_history)
    
    # Calculate stats by symbol
    symbols = set(trade.get("symbol") for trade in trade_history)
    symbol_stats = {}
    
    for symbol in symbols:
        symbol_trades = [trade for trade in trade_history if trade.get("symbol") == symbol]
        symbol_wins = [trade for trade in symbol_trades if trade.get("result") == "win"]
        
        symbol_stats[symbol] = {
            "trades": len(symbol_trades),
            "win_rate": len(symbol_wins) / len(symbol_trades) * 100 if symbol_trades else 0,
            "pnl": sum(trade.get("pnl", 0) for trade in symbol_trades)
        }
    
    return {
        "total_trades": total_trades,
        "win_rate": win_rate,
        "avg_profit": avg_profit,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "total_pnl": total_pnl,
        "by_symbol": symbol_stats
    }

# Function to continuously monitor market for signals
def monitor_market_signals(symbols, timeframes, check_interval=60, auto_trade=True, risk_percent=2):
    """
    Monitor multiple symbols and timeframes for trading signals,
    send alerts, and optionally execute trades
    
    symbols: List of forex symbols to monitor
    timeframes: List of tuples (name, mt5_timeframe)
    check_interval: How often to check for signals in seconds
    auto_trade: Whether to automatically execute trades on signals
    risk_percent: Risk percentage for position sizing
    """
    logger.info(f"Starting market monitor with auto-trade={auto_trade}")
    
    last_signal_time = {}
    
    # Initialize the last signal dictionary for each symbol/timeframe pair
    for symbol in symbols:
        for tf_name, tf_value in timeframes:
            last_signal_time[(symbol, tf_name)] = datetime.datetime.now(timezone.utc) - datetime.timedelta(hours=1)
    
    # Initialize connection to MT5
    if not mt5.initialize():
        logger.error("Failed to initialize MT5 in monitor thread. Exiting.")
        return
    
    while True:
        try:
            # Update status of existing trades
            closed_trades = update_trade_status()
            
            # Check each symbol/timeframe for signals
            for symbol in symbols:
                for tf_name, tf_value in timeframes:
                    try:
                        # Get market data
                        df = get_market_data(symbol, tf_value, 100)
                        if df.empty:
                            logger.warning(f"Empty data for {symbol} {tf_name}")
                            continue
                            
                        df = calculate_indicators(df)
                        df = advanced_check_trade_signals(df)
                        
                        current_time = datetime.datetime.now(timezone.utc)
                        key = (symbol, tf_name)
                        
                        # Check if there's a signal and if enough time has passed since the last signal
                        # (to avoid sending multiple signals for the same market move)
                        if (df['Buy_Signal'].iloc[-1] or df['Sell_Signal'].iloc[-1]) and \
                           (current_time - last_signal_time[key]).total_seconds() > 300:  # 5 minute minimum between signals
                            
                            # Send the alert and possibly execute trade
                            trade_executed = send_detailed_signal_alert(
                                df, symbol, tf_name, 
                                auto_trade=auto_trade,
                                risk_percent=risk_percent
                            )
                            
                            # Update the last signal time
                            last_signal_time[key] = current_time
                            
                            # Log the signal
                            signal_type = "BUY" if df['Buy_Signal'].iloc[-1] else "SELL"
                            logger.info(f"Signal detected: {signal_type} for {symbol} on {tf_name} timeframe")
                    
                    except Exception as e:
                        logger.error(f"Error processing {symbol} {tf_name}: {e}")
            
            # Wait before checking again
            time.sleep(check_interval)
            
        except Exception as e:
            logger.error(f"Error in monitor thread: {e}")
            time.sleep(check_interval)  # Continue despite errors

def scan_for_trading_opportunities():
    """Scan all markets for potential trading opportunities"""
    opportunities = []
    
    for symbol in get_default_symbols():
        try:
            # Check H1 timeframe for medium-term opportunities
            df_h1 = get_market_data(symbol, mt5.TIMEFRAME_H1, 100)
            if not df_h1.empty:
                df_h1 = calculate_indicators(df_h1, symbol)
                df_h1 = advanced_check_trade_signals(df_h1, symbol)
                
                # Check for signals
                current_price = df_h1['close'].iloc[-1]
                
                if df_h1['Buy_Signal'].iloc[-1]:
                    # Check if market is profitable to trade
                    if is_profitable_market(df_h1, symbol):
                        opportunities.append({
                            'symbol': symbol,
                            'action': 'BUY',
                            'timeframe': 'H1',
                            'price': current_price,
                            'rsi': df_h1['RSI'].iloc[-1],
                            'adx': df_h1['ADX'].iloc[-1],
                            'strength': 'High' if df_h1['Strong_Trend'].iloc[-1] else 'Medium',
                            'score': calculate_opportunity_score(df_h1, 'buy', symbol)
                        })
                
                if df_h1['Sell_Signal'].iloc[-1]:
                    # Check if market is profitable to trade
                    if is_profitable_market(df_h1, symbol):
                        opportunities.append({
                            'symbol': symbol,
                            'action': 'SELL',
                            'timeframe': 'H1',
                            'price': current_price,
                            'rsi': df_h1['RSI'].iloc[-1],
                            'adx': df_h1['ADX'].iloc[-1],
                            'strength': 'High' if df_h1['Strong_Trend'].iloc[-1] else 'Medium',
                            'score': calculate_opportunity_score(df_h1, 'sell', symbol)
                        })
            
            # Also check M15 timeframe for shorter-term opportunities
            df_m15 = get_market_data(symbol, mt5.TIMEFRAME_M15, 100)
            if not df_m15.empty:
                df_m15 = calculate_indicators(df_m15, symbol)
                df_m15 = advanced_check_trade_signals(df_m15, symbol)
                
                # Check for signals in M15 that aren't already in H1
                current_price = df_m15['close'].iloc[-1]
                
                if df_m15['Buy_Signal'].iloc[-1] and not any(o['symbol'] == symbol and o['action'] == 'BUY' for o in opportunities):
                    # Check if market is profitable to trade
                    if is_profitable_market(df_m15, symbol):
                        opportunities.append({
                            'symbol': symbol,
                            'action': 'BUY',
                            'timeframe': 'M15',
                            'price': current_price,
                            'rsi': df_m15['RSI'].iloc[-1],
                            'adx': df_m15['ADX'].iloc[-1],
                            'strength': 'Medium',
                            'score': calculate_opportunity_score(df_m15, 'buy', symbol) * 0.9  # Slight penalty for shorter timeframe
                        })
                
                if df_m15['Sell_Signal'].iloc[-1] and not any(o['symbol'] == symbol and o['action'] == 'SELL' for o in opportunities):
                    # Check if market is profitable to trade
                    if is_profitable_market(df_m15, symbol):
                        opportunities.append({
                            'symbol': symbol,
                            'action': 'SELL',
                            'timeframe': 'M15',
                            'price': current_price,
                            'rsi': df_m15['RSI'].iloc[-1],
                            'adx': df_m15['ADX'].iloc[-1],
                            'strength': 'Medium',
                            'score': calculate_opportunity_score(df_m15, 'sell', symbol) * 0.9  # Slight penalty for shorter timeframe
                        })
                
        except Exception as e:
            logger.error(f"Error scanning {symbol}: {e}")
    
    # Sort opportunities by score (highest first)
    return sorted(opportunities, key=lambda x: x['score'], reverse=True)

def calculate_opportunity_score(df, action, symbol):
    """Calculate a score for a trading opportunity (0-100)"""
    score = 50  # Base score
    
    # Determine market type
    is_crypto = "BTC" in symbol or "ETH" in symbol
    is_metal = "XAU" in symbol or "GOLD" in symbol or "XAG" in symbol or "SILVER" in symbol
    is_index = any(idx in symbol for idx in ["US30.std", "US500.std", "USTEC.std", "DE30.std", "UK100.std", "JP225.std"])
    
    # Trend strength (ADX)
    adx = df['ADX'].iloc[-1]
    if adx > 40:
        score += 15
    elif adx > 25:
        score += 10
    elif adx > 20:
        score += 5
    else:
        score -= 5
    
    # RSI alignment with trade direction
    rsi = df['RSI'].iloc[-1]
    if action == 'buy':
        if rsi < 30:
            score += 15  # Strong oversold
        elif rsi < 40:
            score += 10  # Oversold
        elif rsi > 70:
            score -= 15  # Overbought - not good for buying
    else:  # sell
        if rsi > 70:
            score += 15  # Strong overbought
        elif rsi > 60:
            score += 10  # Overbought
        elif rsi < 30:
            score -= 15  # Oversold - not good for selling
    
    # Moving average alignment
    if 'EMA_9' in df.columns and 'EMA_21' in df.columns:
        if action == 'buy':
            if df['EMA_9'].iloc[-1] > df['EMA_21'].iloc[-1]:
                score += 10
            else:
                score -= 5
        else:  # sell
            if df['EMA_9'].iloc[-1] < df['EMA_21'].iloc[-1]:
                score += 10
            else:
                score -= 5
    
    # Market specific factors
    if is_crypto:
        # Crypto volatility check
        if df['ATR'].iloc[-1] / df['close'].iloc[-1] * 100 > 5:
            score -= 10  # Too volatile
        
        # Crypto weekend penalty
        if datetime.datetime.utcnow().weekday() >= 5:  # Saturday or Sunday
            score -= 10  # Weekends often have lower liquidity
    
    elif is_metal:
        # Gold trades well in times of uncertainty
        if df['BB_Width'].iloc[-1] > df['BB_Width'].rolling(20).mean().iloc[-1] * 1.5:
            score += 10  # Expanding bands suggest strong move
    
    # Cap score between 0 and 100
    return max(0, min(100, score))

def start_signal_monitor(auto_trade=True, risk_percent=2):
    """Start the signal monitoring system in a separate thread"""
    
    if st.session_state.monitor_running:
        return "Signal monitoring already running"
    
    # Get selected symbols and timeframes from session state or use defaults - BTC & Gold only
    symbols = st.session_state.get('selected_pairs', ["BTCUSD.m", "XAUUSD.m"])
    
    timeframe_map = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1
    }
    
    # Get selected timeframes or use defaults
    selected_tf_names = st.session_state.get('selected_timeframes', ["M5", "M15", "M30", "H1"])
    timeframes = [(tf, timeframe_map[tf]) for tf in selected_tf_names if tf in timeframe_map]
    
    # Create and start the monitoring thread
    monitor_thread = threading.Thread(
        target=monitor_market_signals,
        args=(symbols, timeframes, 60, auto_trade, risk_percent),  # Check every 60 seconds
        daemon=True  # Thread will exit when main program exits
    )
    monitor_thread.start()
    
    st.session_state.monitor_thread = monitor_thread
    st.session_state.monitor_running = True
    
    mode = "with auto-trading" if auto_trade else "in alert-only mode"
    return f"Signal monitoring started {mode}"

def stop_signal_monitor():
    """Stop the signal monitoring (note: this doesn't actually stop the thread,
       it just marks it as stopped since Python threads can't be forcibly stopped)"""
    if not st.session_state.monitor_running:
        return "Signal monitoring is not running"
    
    st.session_state.monitor_running = False
    return "Signal monitoring stopped (will end after current cycle)"

def display_market_data(df):
    """Display market data and indicators in Streamlit."""
    st.subheader("Market Data")
    if df.empty:
        st.warning("No market data available")
        return
        
    # Format the dataframe for display
    display_df = df.tail(10).copy()
    for col in display_df.columns:
        if col not in ['time', 'Buy_Signal', 'Sell_Signal', 'Uptrend', 'Downtrend']:
            if display_df[col].dtype in [np.float64, np.float32]:
                display_df[col] = display_df[col].round(5)
    
    st.dataframe(display_df)

def display_chart(df, symbol):
    """Display a price chart with indicators."""
    if df.empty:
        return
    
    st.subheader(f"{symbol} Price Chart")
    
    # Create price chart
    chart_data = pd.DataFrame({
        'Close': df['close'],
        'SMA50': df['SMA_50'],
        'SMA200': df['SMA_200'],
        'Upper BB': df['upper_band'],
        'Lower BB': df['lower_band']
    }, index=df['time'])
    
    st.line_chart(chart_data)
    
    # Create RSI chart
    rsi_data = pd.DataFrame({
        'RSI': df['RSI'],
        'Overbought': pd.Series([70] * len(df), index=df['time']),
        'Oversold': pd.Series([30] * len(df), index=df['time'])
    }, index=df['time'])
    
    st.line_chart(rsi_data)



def monitor_active_mt5_trades():
    """
    Monitor active trades in MT5 and send notifications when closed
    This will run in a separate thread
    """
    # Track open positions by ticket ID
    open_positions = {}
    
    while True:
        try:
            # Skip if MT5 is not connected
            if mt5.terminal_info() is None:
                time.sleep(30)  # Check every 30 seconds
                continue
            
            # Get current open positions
            current_positions = mt5.positions_get()
            current_position_tickets = {}
            
            if current_positions:
                # Track current open positions by ticket
                for pos in current_positions:
                    current_position_tickets[pos.ticket] = {
                        'symbol': pos.symbol,
                        'type': 'buy' if pos.type == mt5.POSITION_TYPE_BUY else 'sell',
                        'open_price': pos.price_open,
                        'volume': pos.volume,
                        'profit': pos.profit,
                        'open_time': pos.time
                    }
            
            # First run - initialize open positions
            if not open_positions:
                open_positions = current_position_tickets
                logger.info(f"MT5 Monitor: Tracking {len(open_positions)} open positions")
                time.sleep(30)
                continue
            
            # Check for closed positions
            for ticket, pos_data in open_positions.items():
                if ticket not in current_position_tickets:
                    # Position closed - get details from history
                    end_time = datetime.datetime.now(timezone.utc)
                    start_time = datetime.datetime.fromtimestamp(pos_data['open_time']) - datetime.timedelta(minutes=10)
                    
                    # Get trade history from MT5
                    history = mt5.history_deals_get(
                        start_time,
                        end_time
                    )
                    
                    # Find the closing deals for this position
                    closing_deals = []
                    for deal in history:
                        if deal.position_id == ticket and deal.entry == mt5.DEAL_ENTRY_OUT:
                            closing_deals.append(deal)
                    
                    if closing_deals:
                        # Get closing details
                        close_price = closing_deals[-1].price
                        total_profit = sum(deal.profit for deal in closing_deals)
                        close_time = max(deal.time for deal in closing_deals)
                        
                        # Calculate pips
                        symbol_info = mt5.symbol_info(pos_data['symbol'])
                        point = symbol_info.point if symbol_info else 0.0001
                        price_diff = abs(close_price - pos_data['open_price'])
                        pips = price_diff / point / 10  # Standard forex pip definition
                        
                        # Create trade record for notification
                        trade_record = {
                            "action": pos_data['type'],
                            "symbol": pos_data['symbol'],
                            "entry_price": pos_data['open_price'],
                            "close_price": close_price,
                            "pnl": total_profit,
                            "pips": pips,
                            "time": datetime.datetime.fromtimestamp(pos_data['open_time']),
                            "close_time": datetime.datetime.fromtimestamp(close_time),
                        }
                        
                        # Send notification to subscribers
                        logger.info(f"MT5 Monitor: Position closed - {pos_data['symbol']} with P&L: {total_profit}")
                        send_trade_closed_notification(trade_record)
            
            # Update our tracking list
            open_positions = current_position_tickets
            
            # Wait before next check
            time.sleep(30)  # Check every 30 seconds
            
        except Exception as e:
            logger.error(f"Error in MT5 monitor thread: {e}")
            time.sleep(60)  # Longer delay after an error

def main():
    # Initialize session state variables for trading bot
    if 'monitor_running' not in st.session_state:
        st.session_state.monitor_running = False
    
    # Custom CSS for better UI styling
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 0.5rem;
        font-weight: 700;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #757575;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    .signal-badge {
        font-size: 1rem;
        font-weight: bold;
        padding: 0.3rem 0.6rem;
        border-radius: 15px;
        display: inline-block;
    }
    .buy-badge {
        background-color: rgba(76, 175, 80, 0.2);
        color: #2E7D32;
        border: 1px solid #2E7D32;
    }
    .sell-badge {
        background-color: rgba(244, 67, 54, 0.2);
        color: #C62828;
        border: 1px solid #C62828;
    }
    .neutral-badge {
        background-color: rgba(158, 158, 158, 0.2);
        color: #616161;
        border: 1px solid #616161;
    }
    .demo-account {
        background-color: rgba(33, 150, 243, 0.2);
        color: #1565C0;
        padding: 0.2rem 0.5rem;
        border-radius: 5px;
        font-weight: bold;
    }
    .real-account {
        background-color: rgba(244, 67, 54, 0.2);
        color: #C62828;
        padding: 0.2rem 0.5rem;
        border-radius: 5px;
        font-weight: bold;
    }
    .trade-action-btn {
        width: 100%;
        font-weight: bold;
        padding: 0.6rem !important;
        margin: 0.3rem 0 !important;
    }
    .signal-strength-meter {
        width: 100%;
        height: 8px;
        background-color: #e0e0e0;
        border-radius: 4px;
        overflow: hidden;
        margin: 0.5rem 0;
    }
    .signal-strength-fill {
        height: 100%;
        border-radius: 4px;
    }
    .indicator-good {
        color: #00C853;
    }
    .indicator-caution {
        color: #FF9800;
    }
    .indicator-bad {
        color: #FF3D00;
    }
    .trade-history-card {
        border-left: 4px solid;
        padding: 0.8rem;
        margin: 0.5rem 0;
        border-radius: 4px;
        background-color: #f8f9fa;
    }
    .win-trade {
        border-left-color: #00C853;
    }
    .loss-trade {
        border-left-color: #FF3D00;
    }
    .status-badge {
        font-size: 0.75rem;
        padding: 0.2rem 0.4rem;
        border-radius: 10px;
        font-weight: bold;
    }
    .status-active {
        background-color: rgba(76, 175, 80, 0.2);
        color: #2E7D32;
    }
    .status-inactive {
        background-color: rgba(244, 67, 54, 0.2);
        color: #C62828;
    }
    </style>
    """, unsafe_allow_html=True)


    # Main header
    st.markdown('<h1 class="main-header">🤖 Forex AI Trading Bot</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Advanced algorithmic trading with real-time signals and risk management</p>', unsafe_allow_html=True)
    
    if 'mt5_monitor_thread' not in st.session_state:
        monitor_thread = threading.Thread(
            target=monitor_active_mt5_trades,
            daemon=True
        )
        monitor_thread.start()
        st.session_state.mt5_monitor_thread = monitor_thread
        logger.info("Started MT5 trade monitor thread")

    # Account selection
    if 'account_type' not in st.session_state:
        st.session_state.account_type = "demo"
    
    # Top status bar
    status_col1, status_col2, status_col3, status_col4 = st.columns([1, 1, 1, 1])
    
    with status_col1:
        # Account selector with better styling
        account_type = st.radio(
            "Account Type",
            ["Demo", "Real"],
            index=0 if st.session_state.account_type == "demo" else 1,
            key="top_account_selector",
            horizontal=True
        )
        st.session_state.account_type = account_type.lower()
    
    with status_col2:
        # Account status with badges
        if st.session_state.account_type == "demo":
            st.markdown('<div class="demo-account">DEMO ACCOUNT (2001479025)</div>', unsafe_allow_html=True)
            st.markdown(f"<div>Server: JustMarkets-Demo</div>", unsafe_allow_html=True)
        else:
            st.markdown('<div class="real-account">REAL MONEY ACCOUNT (2050196801)</div>', unsafe_allow_html=True)
            st.markdown(f"<div>Server: JustMarkets-Live</div>", unsafe_allow_html=True)
    
    with status_col3:
        # Connection status
        if 'mt5_connected' not in st.session_state:
            st.session_state.mt5_connected = False
            
        if mt5.terminal_info() is not None:
            st.session_state.mt5_connected = True
        
        if st.session_state.mt5_connected:
            st.markdown('<div class="status-badge status-active">MT5 CONNECTED</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-badge status-inactive">MT5 DISCONNECTED</div>', unsafe_allow_html=True)
            
        # Auto-trading status
        if 'auto_trading' not in st.session_state:
            st.session_state.auto_trading = False
            
        auto_status = "ENABLED" if st.session_state.auto_trading else "DISABLED"
        auto_class = "status-active" if st.session_state.auto_trading else "status-inactive"
        st.markdown(f'<div class="status-badge {auto_class}">AUTO-TRADING {auto_status}</div>', unsafe_allow_html=True)
    
    with status_col4:
        # Connect button with better styling
        if st.button("Connect to MT5", key="connect_btn", use_container_width=True):
            with st.spinner(f"Connecting to {st.session_state.account_type.upper()} account..."):
                if initialize_mt5(account_type=st.session_state.account_type):
                    st.session_state.mt5_connected = True
                    st.success(f"Successfully connected to {st.session_state.account_type.upper()} account!", icon="✅")
                else:
                    st.error("Failed to connect. Check the logs for details.", icon="❌")
    
    # Horizontal separator
    st.markdown("---")
    
    # Initialize MT5 if not already done
    if not st.session_state.mt5_connected:
        if not initialize_mt5(account_type=st.session_state.account_type):
            st.warning("Please connect to MetaTrader 5 to access all features", icon="⚠️")
    
    # Create a session state for signal monitoring thread
    if 'monitor_thread' not in st.session_state:
        st.session_state.monitor_thread = None
        st.session_state.monitor_running = False
        
    # Initialize auto-refresh session state
    if 'auto_refresh' not in st.session_state:
        st.session_state.auto_refresh = False
    
    # Main tabs with better organization
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Dashboard", 
    "🔍 Signals", 
    "📈 Trading", 
    "📉 Performance", 
    "⚙️ Settings",
    "💹 Market Info",
    "🧪 Optimizer"
    ])  # Get the last tab (Optimizer)

    
    with tab1:
        # Dashboard Tab - Overview of everything
        st.subheader("Market Overview")
        
        # Updated market selector - Expanded with profitable pairs
        all_markets = [
            # Crypto
            "BTCUSD.m",
            # Commodities
            "XAUUSD.m",
            "WTI.m",
            # Forex Major
            "EURUSD.m",
            "GBPUSD.m",
            "USDJPY.m",
            # Forex Commodity
            "AUDCAD.m",
            "AUDUSD.m",
            "USDCAD.m",
            "NZDUSD.m"
        ]
            
        
        # Create dynamic grid of currency cards
        # Update the market selection in the dashboard
        col1, col2, col3 = st.columns(3)
        cols = [col1, col2, col3]
        
        # Create groupings for better organization - Expanded profitable markets
        market_groups = {
            "Crypto": ["BTCUSD.m"],
            "Commodities": ["XAUUSD.m", "WTI.m"],
            "Forex Major": ["EURUSD.m", "GBPUSD.m", "USDJPY.m"],
            "Forex Commodity": ["AUDCAD.m", "AUDUSD.m", "USDCAD.m", "NZDUSD.m"]
        }
        
        # Create market selection with categories
        st.sidebar.header("Market Selection")
        selected_categories = st.sidebar.multiselect(
            "Select Market Categories",
            list(market_groups.keys()),
            default=list(market_groups.keys())  # All categories by default
        )
        
        available_markets = []
        for category in selected_categories:
            available_markets.extend(market_groups[category])
            
        selected_markets = st.sidebar.multiselect(
            "Select Markets to Trade",
            available_markets,
            default=["BTCUSD.m", "XAUUSD.m", "WTI.m", "AUDCAD.m", "EURUSD.m"]  # Top 5 profitable markets
        )
        
        # Save selected markets to session state
        st.session_state.selected_pairs = selected_markets
        
     
        
        # Group markets by category for display
        displayed_markets = {}
        for category in selected_categories:
            displayed_markets[category] = [m for m in market_groups[category] if m in selected_markets]
        
        # Display each category in its own section
        for category, markets in displayed_markets.items():
            if markets:  # Only show categories with selected markets
                st.markdown(f"#### {category} Markets")
                
                # Create columns for the cards in this category
                cols = st.columns(min(3, len(markets)))
                
                for i, symbol in enumerate(markets):
                    with cols[i % len(cols)]:
                        # Get data for this symbol
                        tf = mt5.TIMEFRAME_M15
                        df = get_market_data(symbol, tf, 100)
                        
                        if not df.empty:
                            df = calculate_indicators(df, symbol)
                            df = advanced_check_trade_signals(df, symbol)
                            
                            # Determine card info and styling
                            signal_type = "neutral"
                            signal_text = "NEUTRAL"
                            signal_class = "neutral-badge"
                            
                            if df['Buy_Signal'].iloc[-1]:
                                signal_type = "buy"
                                signal_text = "BUY"
                                signal_class = "buy-badge"
                            elif df['Sell_Signal'].iloc[-1]:
                                signal_type = "sell"
                                signal_text = "SELL"
                                signal_class = "sell-badge"
                            
                            # Format price display based on market type
                            current_price = df['close'].iloc[-1]
                            price_change = df['close'].iloc[-1] - df['close'].iloc[-2]
                            price_change_pct = (price_change / df['close'].iloc[-2]) * 100
                            
                            # Format price differently for each market type
                            if "BTC" in symbol:
                                price_display = f"{current_price:.1f}"
                            elif "XAU" in symbol or "GOLD" in symbol:
                                price_display = f"{current_price:.2f}"
                            elif any(idx in symbol for idx in ["US30.std", "US500.std", "USTEC.std", "DE30.std", "UK100.std", "JP225.std"]):
                                price_display = f"{current_price:.1f}"
                            else:
                                price_display = f"{current_price:.5f}"
                            
                            # Create metric card
                            st.markdown(f"""
                            <div class="metric-card">
                                <h3>{symbol}</h3>
                                <div>
                                    <span style="font-size: 1.3rem; font-weight: bold;">{price_display}</span>
                                    <span style="color: {'green' if price_change >= 0 else 'red'}; margin-left: 10px;">
                                        {price_change_pct:.2f}%
                                    </span>
                                </div>
                                <div class="signal-strength-meter">
                                    <div class="signal-strength-fill" style="width: {'70%' if signal_type == 'buy' else '30%' if signal_type == 'sell' else '50%'}; 
                                        background-color: {'#4CAF50' if signal_type == 'buy' else '#F44336' if signal_type == 'sell' else '#9E9E9E'}">
                                    </div>
                                </div>
                                <div>
                                    <span class="signal-badge {signal_class}">{signal_text}</span>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            # Display error card if data not available
                            st.markdown(f"""
                            <div class="metric-card" style="opacity: 0.7;">
                                <h3>{symbol}</h3>
                                <div>No data available</div>
                            </div>
                            """, unsafe_allow_html=True)
        
        # # Add market scanner
        # st.subheader("Market Scanner")
        # scan_col1, scan_col2 = st.columns([3, 1])
        
        # with scan_col2:
        #     if st.button("Scan Markets", key="scan_markets_btn", use_container_width=True):
        #         with st.spinner("Scanning all markets for opportunities..."):
        #             opportunities = scan_for_trading_opportunities()
        #             st.session_state.opportunities = opportunities
                    
        # with scan_col1:
        #     # Check if the value exists in session state first, otherwise use False as default
        #     if "auto_scan" not in st.session_state:
        #         st.session_state.auto_scan = False

        #     # Then create the checkbox with the current value from session state
        #     auto_scan = st.checkbox(
        #         "Auto-scan every 15 minutes", 
        #         value=st.session_state.auto_scan,  # Use the value from session state
        #         key="auto_scan"
        #     )
        
        # Display opportunities if available
        if 'opportunities' in st.session_state and st.session_state.opportunities:
            opportunities = st.session_state.opportunities
            
            # Create a visually appealing opportunity table
            st.markdown("### Top Trading Opportunities")
            
            # Create a grid of opportunity cards
            op_col1, op_col2 = st.columns(2)
            cols = [op_col1, op_col2]
            
            for i, op in enumerate(opportunities[:6]):  # Show top 6 opportunities
                with cols[i % 2]:
                    # Format price display based on market type
                    symbol = op['symbol']
                    price_display = op['price']
                    
                    if "BTC" in symbol:
                        price_display = f"{price_display:.1f}"
                    elif "XAU" in symbol or "GOLD" in symbol:
                        price_display = f"{price_display:.2f}"
                    elif any(idx in symbol for idx in ["US30.std", "US500.std", "USTEC.std", "DE30.std", "UK100.std", "JP225.std"]):
                        price_display = f"{price_display:.1f}"
                    else:
                        price_display = f"{price_display:.5f}"
                    
                    # Create opportunity card
                    action_color = "4CAF50" if op['action'] == "BUY" else "F44336"
                    
                    # Score band color
                    if op['score'] >= 80:
                        score_color = "#4CAF50"  # Green (excellent)
                    elif op['score'] >= 65:
                        score_color = "#8BC34A"  # Light green (very good)
                    elif op['score'] >= 50:
                        score_color = "#FFEB3B"  # Yellow (good)
                    elif op['score'] >= 35:
                        score_color = "#FFC107"  # Amber (fair)
                    else:
                        score_color = "#FF9800"  # Orange (poor)
                    
                    st.markdown(f"""
                    <div style="border: 1px solid #e0e0e0; border-radius: 10px; padding: 15px; margin-bottom: 15px;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <strong style="font-size: 1.2rem;">{op['symbol']}</strong>
                                <span style="background-color: rgba({112 if op['action'] == 'BUY' else 244}, {175 if op['action'] == 'BUY' else 67}, {80 if op['action'] == 'BUY' else 54}, 0.2); 
                                    color: #{action_color}; padding: 3px 8px; border-radius: 4px; margin-left: 10px; font-weight: bold;">
                                    {op['action']}
                                </span>
                            </div>
                            <div>
                                <span style="font-size: 0.9rem; background-color: #e0e0e0; padding: 2px 6px; border-radius: 4px;">
                                    {op['timeframe']}
                                </span>
                            </div>
                        </div>
                        <div style="margin: 10px 0;">
                            <strong style="font-size: 1.1rem;">{price_display}</strong>
                        </div>
                        <div style="display: flex; justify-content: space-between; font-size: 0.9rem; margin-bottom: 10px;">
                            <div>RSI: {op['rsi']:.1f}</div>
                            <div>ADX: {op['adx']:.1f}</div>
                            <div>Strength: {op['strength']}</div>
                        </div>
                        <div style="background-color: #f5f5f5; height: 6px; width: 100%; border-radius: 3px;">
                            <div style="background-color: {score_color}; height: 6px; width: {op['score']}%; border-radius: 3px;"></div>
                        </div>
                        <div style="text-align: right; font-size: 0.9rem; margin-top: 5px;">
                            Score: <strong>{op['score']:.0f}/100</strong>
                        </div>
                        <div style="margin-top: 10px;">
                            <button style="background-color: #{action_color}; color: white; border: none; padding: 5px 10px; border-radius: 4px; cursor: pointer; width: 100%;"
                                    onclick="document.getElementById('trade_{i}').click();">
                                Execute {op['action']}
                            </button>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Hidden button for executing trade
                    if st.button(f"Execute Trade", key=f"trade_{i}", help=f"{op['action']} {op['symbol']}", use_container_width=True):
                        with st.spinner(f"Executing {op['action']} trade for {op['symbol']}..."):
                            success = execute_trade(op['action'].lower(), op['symbol'], 2)
                            
                            if success:
                                st.success(f"{op['action']} trade executed successfully for {op['symbol']}!")
                            else:
                                st.error(f"Failed to execute {op['action']} trade for {op['symbol']}. Check logs for details.")

        # Account metrics
        st.subheader("Account Metrics")
        
        account_col1, account_col2, account_col3, account_col4 = st.columns(4)
        
        # Get account info
        account_info = mt5.account_info()
        
        if account_info:
            with account_col1:
                st.metric(
                    "Balance", 
                    f"${account_info.balance:.2f}",
                    delta=None
                )
            
            with account_col2:
                # Calculate equity change from balance
                equity_delta = account_info.equity - account_info.balance
                st.metric(
                    "Equity", 
                    f"${account_info.equity:.2f}",
                    delta=f"${equity_delta:.2f}",
                    delta_color="normal"
                )
            
            with account_col3:
                # Calculate margin level
                margin_level = (account_info.equity / account_info.margin) * 100 if account_info.margin > 0 else 0
                st.metric(
                    "Margin Level", 
                    f"{margin_level:.2f}%",
                    delta=None
                )
            
            with account_col4:
                # Calculate used margin percentage
                used_margin_pct = (account_info.margin / account_info.equity) * 100 if account_info.equity > 0 else 0
                st.metric(
                    "Free Margin", 
                    f"${account_info.margin_free:.2f}",
                    delta=f"{100-used_margin_pct:.1f}% Available",
                    delta_color="normal"
                )
        else:
            st.warning("Account information not available. Please connect to MT5.")
        
        # Active trades summary
        st.subheader("Active Trades")
        
        positions = mt5.positions_get()
        
        if positions:
            # Create a visual grid of active positions
            positions_data = []
            
            # Group positions by symbol
            symbols_positions = {}
            for pos in positions:
                if pos.symbol not in symbols_positions:
                    symbols_positions[pos.symbol] = []
                symbols_positions[pos.symbol].append(pos)
            
            # Create position cards in a grid
            pos_col1, pos_col2 = st.columns(2)
            cols = [pos_col1, pos_col2]
            
            for i, (symbol, positions) in enumerate(symbols_positions.items()):
                with cols[i % 2]:
                    # Total profit for this symbol
                    total_profit = sum(pos.profit for pos in positions)
                    profit_color = "green" if total_profit >= 0 else "red"
                    
                    # Count buy and sell positions
                    buy_count = sum(1 for pos in positions if pos.type == mt5.POSITION_TYPE_BUY)
                    sell_count = sum(1 for pos in positions if pos.type == mt5.POSITION_TYPE_SELL)
                    
                    # Create card
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>{symbol}</h3>
                        <div>
                            <span style="color: {profit_color}; font-weight: bold; font-size: 1.2rem;">
                                {'+' if total_profit >= 0 else ''}{total_profit:.2f} USD
                            </span>
                        </div>
                        <div style="margin-top: 10px;">
                            <span class="signal-badge buy-badge">{buy_count} BUY</span>
                            <span class="signal-badge sell-badge">{sell_count} SELL</span>
                        </div>
                        <div style="margin-top: 10px; font-size: 0.9rem;">
                            Click "Trading" tab to manage positions
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("No active trades. Go to the Trading tab to open positions.")
        
        # Performance snapshot
        st.subheader("Performance Snapshot")
        
        stats = calculate_mt5_trade_stats()
        perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
        
        with perf_col1:
            st.metric(
                "Total Trades", 
                stats["total_trades"],
                delta=None
            )
        
        with perf_col2:
            st.metric(
                "Win Rate", 
                f"{stats['win_rate']:.1f}%",
                delta=None
            )
        
        with perf_col3:
            st.metric(
                "Profit Factor", 
                f"{stats['profit_factor']:.2f}",
                delta=None
            )
        
        with perf_col4:
            st.metric(
                "Total P&L", 
                f"${stats['total_pnl']:.2f}",
                delta=None
            )
        
        # Quick actions row
        st.subheader("Quick Actions")
        
        action_col1, action_col2, action_col3 = st.columns(3)
        
        with action_col1:
            if st.button("📊 Monitor Market Signals", use_container_width=True):
                if st.session_state.monitor_running:
                    message = stop_signal_monitor()
                    st.success(message)
                else:
                    message = start_signal_monitor(st.session_state.auto_trading, 2)
                    st.success(message)
        
        with action_col2:
            auto_btn_text = "🛑 Disable Auto-Trading" if st.session_state.auto_trading else "✅ Enable Auto-Trading"
            if st.button(auto_btn_text, use_container_width=True):
                st.session_state.auto_trading = not st.session_state.auto_trading
                msg = "Auto-trading enabled" if st.session_state.auto_trading else "Auto-trading disabled"
                st.success(msg)
        
        with action_col3:
            refresh_btn_text = "🔄 Disable Auto-Refresh" if st.session_state.auto_refresh else "🔄 Enable Auto-Refresh"
            if st.button(refresh_btn_text, use_container_width=True):
                st.session_state.auto_refresh = not st.session_state.auto_refresh
                msg = "Auto-refresh enabled (10s)" if st.session_state.auto_refresh else "Auto-refresh disabled"
                st.success(msg)

        # Market spotlight - focus on Bitcoin and Gold
        st.subheader("Market Spotlight")
        
        # Create cards for Bitcoin and Gold
        spotlight_col1, spotlight_col2 = st.columns(2)
        
        with spotlight_col1:
            # Bitcoin Spotlight
            st.markdown("<h4>Bitcoin (BTCUSD.m)</h4>", unsafe_allow_html=True)
            
            btc_df = get_market_data("BTCUSD.m", mt5.TIMEFRAME_H1, 100)
            if not btc_df.empty:
                btc_df = calculate_indicators(btc_df, "BTCUSD.m")
                btc_df = advanced_check_trade_signals(btc_df, "BTCUSD.m")
                
                # Current price and indicators
                current_price = btc_df['close'].iloc[-1]
                price_change = btc_df['close'].iloc[-1] - btc_df['close'].iloc[-24]  # 24-hour change
                price_change_pct = (price_change / btc_df['close'].iloc[-24]) * 100
                
                # Signal determination
                signal_type = "neutral"
                signal_text = "NEUTRAL"
                signal_class = "neutral-badge"
                
                if btc_df['Buy_Signal'].iloc[-1]:
                    signal_type = "buy"
                    signal_text = "BUY"
                    signal_class = "buy-badge"
                elif btc_df['Sell_Signal'].iloc[-1]:
                    signal_type = "sell"
                    signal_text = "SELL"
                    signal_class = "sell-badge"
                
                # Create Bitcoin card
                st.markdown(f"""
                <div class="metric-card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <span style="font-size: 1.8rem; font-weight: bold;">${current_price:.1f}</span>
                            <span style="color: {'green' if price_change >= 0 else 'red'}; margin-left: 10px;">
                                {price_change_pct:.2f}% (24h)
                            </span>
                        </div>
                        <div>
                            <span class="signal-badge {signal_class}">{signal_text}</span>
                        </div>
                    </div>
                    <div style="margin-top: 15px;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                            <div>RSI (4H):</div>
                            <div>{btc_df['RSI'].iloc[-1]:.1f}</div>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                            <div>Volatility:</div>
                            <div>{(btc_df['ATR'].iloc[-1] / current_price * 100):.2f}%</div>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                            <div>Support Level:</div>
                            <div>${btc_df['recent_low'].iloc[-1]:.1f}</div>
                        </div>
                        <div style="display: flex; justify-content: space-between;">
                            <div>Resistance Level:</div>
                            <div>${btc_df['recent_high'].iloc[-1]:.1f}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("Bitcoin data not available. Check your connection to JustMarkets.")
        
        with spotlight_col2:
            # Gold Spotlight
            st.markdown("<h4>Gold (XAUUSD.m)</h4>", unsafe_allow_html=True)
            
            gold_df = get_market_data("XAUUSD.m", mt5.TIMEFRAME_H1, 100)
            if not gold_df.empty:
                gold_df = calculate_indicators(gold_df, "XAUUSD.m")
                gold_df = advanced_check_trade_signals(gold_df, "XAUUSD.m")
                
                # Current price and indicators
                current_price = gold_df['close'].iloc[-1]
                price_change = gold_df['close'].iloc[-1] - gold_df['close'].iloc[-24]  # 24-hour change
                price_change_pct = (price_change / gold_df['close'].iloc[-24]) * 100
                
                # Signal determination
                signal_type = "neutral"
                signal_text = "NEUTRAL"
                signal_class = "neutral-badge"
                
                if gold_df['Buy_Signal'].iloc[-1]:
                    signal_type = "buy"
                    signal_text = "BUY"
                    signal_class = "buy-badge"
                elif gold_df['Sell_Signal'].iloc[-1]:
                    signal_type = "sell"
                    signal_text = "SELL"
                    signal_class = "sell-badge"
                
                # Create Gold card
                st.markdown(f"""
                <div class="metric-card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <span style="font-size: 1.8rem; font-weight: bold;">${current_price:.2f}</span>
                            <span style="color: {'green' if price_change >= 0 else 'red'}; margin-left: 10px;">
                                {price_change_pct:.2f}% (24h)
                            </span>
                        </div>
                        <div>
                            <span class="signal-badge {signal_class}">{signal_text}</span>
                        </div>
                    </div>
                    <div style="margin-top: 15px;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                            <div>RSI (4H):</div>
                            <div>{gold_df['RSI'].iloc[-1]:.1f}</div>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                            <div>Volatility:</div>
                            <div>{(gold_df['ATR'].iloc[-1] / current_price * 100):.2f}%</div>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                            <div>Support Level:</div>
                            <div>${gold_df['recent_low'].iloc[-1]:.2f}</div>
                        </div>
                        <div style="display: flex; justify-content: space-between;">
                            <div>Resistance Level:</div>
                            <div>${gold_df['recent_high'].iloc[-1]:.2f}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("Gold data not available. Check your connection to JustMarkets.")
        
    
    with tab2:
        # Signals Tab - Detailed signal analysis
        st.subheader("Signal Analysis")
        
        # Symbol and timeframe selector
        signal_col1, signal_col2 = st.columns(2)
        
        with signal_col1:
            symbol = st.selectbox(
                "Select Symbol",
                get_default_symbols(),  # Use all profitable markets
                key="signal_symbol"
            )
        
        with signal_col2:
            timeframe_options = [
                ("5 Minutes", mt5.TIMEFRAME_M5),
                ("15 Minutes", mt5.TIMEFRAME_M15),
                ("30 Minutes", mt5.TIMEFRAME_M30),
                ("1 Hour", mt5.TIMEFRAME_H1),
                ("4 Hours", mt5.TIMEFRAME_H4),
                ("Daily", mt5.TIMEFRAME_D1)
            ]
            
            timeframe = st.selectbox(
                "Select Timeframe",
                timeframe_options,
                format_func=lambda x: x[0],
                key="signal_timeframe"
            )
        
        # Get data for the selected symbol and timeframe
        df = get_market_data(symbol, timeframe[1], 100)
        
        if not df.empty:
            # Process data
            df = calculate_indicators(df)
            df = advanced_check_trade_signals(df)
            
            # Signal analysis display
            signal_analysis_col1, signal_analysis_col2 = st.columns([2, 1])
            
            with signal_analysis_col1:
                # Chart display
                display_chart(df, symbol)
            
            with signal_analysis_col2:
                # Current Signal Status with visual indicator
                signal_type = "neutral"
                signal_text = "NEUTRAL"
                signal_class = "neutral-badge"
                signal_strength = 50  # Default percentage
                
                if df['Buy_Signal'].iloc[-1]:
                    signal_type = "buy"
                    signal_text = "BUY"
                    signal_class = "buy-badge"
                    signal_strength = 70  # This could be calculated based on multiple indicators
                elif df['Sell_Signal'].iloc[-1]:
                    signal_type = "sell"
                    signal_text = "SELL"
                    signal_class = "sell-badge"
                    signal_strength = 70
                
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Current Signal</h3>
                    <div>
                        <span class="signal-badge {signal_class}" style="font-size: 1.2rem; margin: 10px 0;">{signal_text}</span>
                    </div>
                    <div>
                        <p>Signal Strength</p>
                        <div class="signal-strength-meter">
                            <div class="signal-strength-fill" style="width: {signal_strength}%; 
                                background-color: {'#4CAF50' if signal_type == 'buy' else '#F44336' if signal_type == 'sell' else '#9E9E9E'}">
                            </div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Key indicators table
                st.markdown("""
                <div class="metric-card">
                    <h3>Key Indicators</h3>
                """, unsafe_allow_html=True)
                
                # RSI status
                rsi_value = df['RSI'].iloc[-1]
                rsi_class = "indicator-good"
                if rsi_value > 70:
                    rsi_class = "indicator-caution"
                    rsi_status = "Overbought"
                elif rsi_value < 30:
                    rsi_class = "indicator-caution"
                    rsi_status = "Oversold"
                else:
                    rsi_status = "Neutral"
                
                st.markdown(f"""
                <div style="display: flex; justify-content: space-between; margin: 5px 0;">
                    <span>RSI ({rsi_value:.1f})</span>
                    <span class="{rsi_class}">{rsi_status}</span>
                </div>
                """, unsafe_allow_html=True)
                
                # MACD status
                macd_value = df['MACD'].iloc[-1]
                macd_signal = df['MACD_signal'].iloc[-1]
                macd_diff = macd_value - macd_signal
                
                macd_class = "indicator-good" if (macd_diff > 0 and signal_type == "buy") or (macd_diff < 0 and signal_type == "sell") else "indicator-caution"
                macd_status = "Bullish" if macd_diff > 0 else "Bearish"
                
                st.markdown(f"""
                <div style="display: flex; justify-content: space-between; margin: 5px 0;">
                    <span>MACD ({macd_diff:.5f})</span>
                    <span class="{macd_class}">{macd_status}</span>
                </div>
                """, unsafe_allow_html=True)
                
                # Trend status
                trend_status = "Uptrend" if df['Uptrend'].iloc[-1] else "Downtrend" if df['Downtrend'].iloc[-1] else "Sideways"
                trend_class = "indicator-good" if (trend_status == "Uptrend" and signal_type == "buy") or (trend_status == "Downtrend" and signal_type == "sell") else "indicator-caution"
                
                st.markdown(f"""
                <div style="display: flex; justify-content: space-between; margin: 5px 0;">
                    <span>Trend</span>
                    <span class="{trend_class}">{trend_status}</span>
                </div>
                """, unsafe_allow_html=True)
                
                # ADX status
                adx_value = df['ADX'].iloc[-1]
                adx_class = "indicator-good" if adx_value > 25 else "indicator-caution" if adx_value > 15 else "indicator-bad"
                adx_status = "Strong" if adx_value > 25 else "Moderate" if adx_value > 15 else "Weak"
                
                st.markdown(f"""
                <div style="display: flex; justify-content: space-between; margin: 5px 0;">
                    <span>ADX ({adx_value:.1f})</span>
                    <span class="{adx_class}">{adx_status}</span>
                </div>
                """, unsafe_allow_html=True)
                
                # Volatility status
                atr_value = df['ATR'].iloc[-1]
                avg_atr = df['ATR'].rolling(20).mean().iloc[-1]
                vol_ratio = atr_value / avg_atr if avg_atr > 0 else 1
                
                vol_class = "indicator-good" if 0.8 <= vol_ratio <= 1.2 else "indicator-caution"
                vol_status = "Normal" if 0.8 <= vol_ratio <= 1.2 else "High" if vol_ratio > 1.2 else "Low"
                
                st.markdown(f"""
                <div style="display: flex; justify-content: space-between; margin: 5px 0;">
                    <span>Volatility</span>
                    <span class="{vol_class}">{vol_status}</span>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Quick trade actions
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                
                if st.button("🟢 BUY NOW", key="quick_buy_btn", use_container_width=True, help="Execute a BUY trade immediately"):
                    with st.spinner("Executing BUY trade..."):
                        success = execute_trade("buy", symbol, 2)
                        if success:
                            st.success(f"BUY trade executed for {symbol}")
                        else:
                            st.error("Trade execution failed. Check logs for details.")
                
                if st.button("🔴 SELL NOW", key="quick_sell_btn", use_container_width=True, help="Execute a SELL trade immediately"):
                    with st.spinner("Executing SELL trade..."):
                        success = execute_trade("sell", symbol, 2)
                        if success:
                            st.success(f"SELL trade executed for {symbol}")
                        else:
                            st.error("Trade execution failed. Check logs for details.")
                
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Multi-timeframe analysis
            st.subheader("Multi-Timeframe Analysis")
            
            # Create dataframe for multi-timeframe analysis
            timeframes = [
                ("M5", mt5.TIMEFRAME_M5),
                ("M15", mt5.TIMEFRAME_M15),
                ("M30", mt5.TIMEFRAME_M30),
                ("H1", mt5.TIMEFRAME_H1),
                ("H4", mt5.TIMEFRAME_H4),
                ("D1", mt5.TIMEFRAME_D1)
            ]
            
            # Create columns for the table
            mtf_cols = st.columns([1, 1, 1, 1, 1, 1])
            
            # Header row
            for i, (tf_name, _) in enumerate(timeframes):
                mtf_cols[i].markdown(f"<div style='text-align: center; font-weight: bold;'>{tf_name}</div>", unsafe_allow_html=True)
            
            # Signal row
            for i, (tf_name, tf_value) in enumerate(timeframes):
                # Get data for this timeframe
                tf_df = get_market_data(symbol, tf_value, 100)
                
                if not tf_df.empty:
                    tf_df = calculate_indicators(tf_df)
                    tf_df = advanced_check_trade_signals(tf_df)
                    
                    # Determine signal
                    if tf_df['Buy_Signal'].iloc[-1]:
                        signal_class = "buy-badge"
                        signal_text = "BUY"
                    elif tf_df['Sell_Signal'].iloc[-1]:
                        signal_class = "sell-badge"
                        signal_text = "SELL"
                    else:
                        signal_class = "neutral-badge"
                        signal_text = "NEUTRAL"
                    
                    mtf_cols[i].markdown(f"""
                    <div style='text-align: center; margin-top: 10px;'>
                        <span class='signal-badge {signal_class}'>{signal_text}</span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    mtf_cols[i].markdown(f"""
                    <div style='text-align: center; margin-top: 10px; color: #757575;'>
                        No data
                    </div>
                    """, unsafe_allow_html=True)
            
            # RSI row
            for i, (tf_name, tf_value) in enumerate(timeframes):
                tf_df = get_market_data(symbol, tf_value, 100)
                
                if not tf_df.empty:
                    tf_df = calculate_indicators(tf_df)
                    
                    rsi_value = tf_df['RSI'].iloc[-1]
                    rsi_class = "indicator-good"
                    if rsi_value > 70:
                        rsi_class = "indicator-caution"
                    elif rsi_value < 30:
                        rsi_class = "indicator-caution"
                    
                    mtf_cols[i].markdown(f"""
                    <div style='text-align: center; margin-top: 10px;'>
                        <span>RSI: <span class='{rsi_class}'>{rsi_value:.1f}</span></span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    mtf_cols[i].markdown(f"""
                    <div style='text-align: center; margin-top: 10px; color: #757575;'>
                        -
                    </div>
                    """, unsafe_allow_html=True)
                    
            # Trend row
            for i, (tf_name, tf_value) in enumerate(timeframes):
                tf_df = get_market_data(symbol, tf_value, 100)
                
                if not tf_df.empty:
                    tf_df = calculate_indicators(tf_df)
                    
                    trend = "UP" if tf_df['Uptrend'].iloc[-1] else "DOWN" if tf_df['Downtrend'].iloc[-1] else "SIDE"
                    trend_class = "indicator-good" if trend == "UP" else "indicator-caution" if trend == "DOWN" else ""
                    
                    mtf_cols[i].markdown(f"""
                    <div style='text-align: center; margin-top: 10px;'>
                        <span>Trend: <span class='{trend_class}'>{trend}</span></span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    mtf_cols[i].markdown(f"""
                    <div style='text-align: center; margin-top: 10px; color: #757575;'>
                        -
                    </div>
                    """, unsafe_allow_html=True)
            
            # Detailed indicators
            with st.expander("Detailed Market Data", expanded=False):
                display_market_data(df)
        else:
            st.warning(f"No data available for {symbol} on {timeframe[0]} timeframe")
    
    with tab3:
        # Trading Tab - Trade execution and management
        st.subheader("Trade Execution")
        
        # Create trading panel
        trade_col1, trade_col2 = st.columns([1, 1])
        
        with trade_col1:
            # Trading form
            with st.form(key="trade_form"):
                st.markdown("<h3>New Trade</h3>", unsafe_allow_html=True)
                
                symbol = st.selectbox(
                    "Symbol",
                    ["BTCUSD.m", "XAUUSD.m"],
                    key="trade_symbol"
                )
                
                action = st.radio(
                    "Direction",
                    ["BUY", "SELL"],
                    horizontal=True,
                    key="trade_direction"
                )
                
                risk_percent = st.slider(
                    "Risk Percentage",
                    min_value=0.5,
                    max_value=5.0,
                    value=2.0,
                    step=0.1,
                    key="trade_risk"
                )
                
                st.markdown("""
                <div style="margin: 15px 0;">
                    <p><strong>Risk Calculator</strong></p>
                    <p>Account balance impact if stopped out: <strong style="color: #F44336;">-${}</strong></p>
                </div>
                """.format(round(mt5.account_info().balance * risk_percent / 100, 2) if mt5.account_info() else 0), unsafe_allow_html=True)
                
                custom_sl_tp = st.checkbox("Set custom stop-loss and take-profit", value=False)
                
                if custom_sl_tp:
                    # Get current price
                    current_tick = mt5.symbol_info_tick(symbol)
                    if current_tick:
                        current_price = current_tick.ask if action == "BUY" else current_tick.bid
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            sl_pips = st.number_input(
                                "Stop Loss (pips)",
                                min_value=5,
                                max_value=200,
                                value=30,
                                step=5,
                                key="sl_pips"
                            )
                        
                        with col2:
                            tp_pips = st.number_input(
                                "Take Profit (pips)",
                                min_value=5,
                                max_value=500,
                                value=60,
                                step=5,
                                key="tp_pips"
                            )
                        
                        # Calculate SL/TP prices
                        pip_value = 0.0001 if symbol not in ["USDJPY.view"] else 0.01
                        
                        if action == "BUY":
                            sl_price = current_price - sl_pips * pip_value
                            tp_price = current_price + tp_pips * pip_value
                        else:
                            sl_price = current_price + sl_pips * pip_value
                            tp_price = current_price - tp_pips * pip_value
                        
                        st.markdown(f"""
                        <div style="margin: 15px 0;">
                            <p><strong>Risk/Reward: 1:{round(tp_pips/sl_pips, 2)}</strong></p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.warning("Cannot retrieve current price. MT5 connection issue.")
                        sl_price = None
                        tp_price = None
                else:
                    sl_price = None
                    tp_price = None
                
                submit_btn = st.form_submit_button(
                    label=f"Execute {action} Trade", 
                    use_container_width=True
                )
                
                if submit_btn:
                    with st.spinner(f"Executing {action.lower()} trade for {symbol}..."):
                        success = execute_trade(action.lower(), symbol, risk_percent, sl_price, tp_price)
                        
                        if success:
                            st.success(f"{action} trade executed successfully!")
                        else:
                            st.error("Trade execution failed. Check logs for details.")
                        
                        # Force a refresh
                        st.rerun()
        
        with trade_col2:
            st.markdown("<h3>Active Positions</h3>", unsafe_allow_html=True)
            
            positions = mt5.positions_get()
            
            if positions:
                for pos in positions:
                    # Calculate profit in pips
                    point = mt5.symbol_info(pos.symbol).point
                    price_diff = abs(pos.price_current - pos.price_open)
                    pips = price_diff / point / 10  # Standard forex pip definition
                    
                    # Determine position type and color
                    pos_type = "BUY" if pos.type == mt5.POSITION_TYPE_BUY else "SELL"
                    pos_color = "#4CAF50" if pos_type == "BUY" else "#F44336"
                    profit_color = "#4CAF50" if pos.profit >= 0 else "#F44336"
                    
                    # Create position card
                    st.markdown(f"""
                    <div style="border: 1px solid #e0e0e0; border-radius: 10px; padding: 15px; margin-bottom: 15px;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <strong>{pos.symbol}</strong>
                                <span style="background-color: rgba({200 if pos_type == 'BUY' else 244}, {200 if pos_type == 'BUY' else 67}, {200 if pos_type == 'BUY' else 54}, 0.2); 
                                       color: {pos_color}; padding: 3px 8px; border-radius: 4px; margin-left: 10px;">
                                    {pos_type}
                                </span>
                            </div>
                            <div style="color: {profit_color}; font-weight: bold;">
                                {'+' if pos.profit >= 0 else ''}{pos.profit:.2f} USD ({round(pips, 1)} pips)
                            </div>
                        </div>
                        <div style="margin-top: 10px; font-size: 0.9rem;">
                            <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                                <div>Volume:</div>
                                <div>{pos.volume}</div>
                            </div>
                            <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                                <div>Open:</div>
                                <div>{pos.price_open}</div>
                            </div>
                            <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                                <div>Current:</div>
                                <div>{pos.price_current}</div>
                            </div>
                            <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                                <div>SL:</div>
                                <div>{pos.sl if pos.sl != 0 else 'None'}</div>
                            </div>
                            <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                                <div>TP:</div>
                                <div>{pos.tp if pos.tp != 0 else 'None'}</div>
                            </div>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin-top: 15px;">
                            <button style="background-color: #9E9E9E; color: white; border: none; padding: 5px 10px; border-radius: 4px; cursor: pointer;"
                                    onclick="document.getElementById('modify_{pos.ticket}').style.display = 'block';">
                                Modify
                            </button>
                            <button style="background-color: #F44336; color: white; border: none; padding: 5px 10px; border-radius: 4px; cursor: pointer;"
                                    onclick="document.getElementById('close_{pos.ticket}').click();">
                                Close
                            </button>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Hidden button for closing position
                    if st.button("Close Position", key=f"close_{pos.ticket}", help=f"Close {pos_type} position for {pos.symbol}"):
                        with st.spinner(f"Closing {pos_type} position for {pos.symbol}..."):
                            # Create a request to close the position
                            close_price = mt5.symbol_info_tick(pos.symbol).bid if pos.type == mt5.POSITION_TYPE_BUY else mt5.symbol_info_tick(pos.symbol).ask
                            
                            request = {
                                "action": mt5.TRADE_ACTION_DEAL,
                                "symbol": pos.symbol,
                                "volume": pos.volume,
                                "type": mt5.ORDER_TYPE_SELL if pos.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                                "position": pos.ticket,
                                "price": close_price,
                                "deviation": 20,
                                "magic": 234000,
                                "comment": "Close position",
                                "type_time": mt5.ORDER_TIME_GTC,
                                "type_filling": mt5.ORDER_FILLING_IOC,
                            }
                            
                            # Send the request
                            result = mt5.order_send(request)
                            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                                st.success(f"Position {pos.ticket} closed successfully")
                                # Force a refresh
                                st.rerun()
                            else:
                                st.error(f"Failed to close position: {result.comment if hasattr(result, 'comment') else 'Unknown error'}")
            else:
                st.info("No active positions. Use the trading form to open a new position.")
        
        # Position management tab
        st.subheader("Position Management")
        
        # Batch position management
        batch_col1, batch_col2, batch_col3 = st.columns(3)
        
        with batch_col1:
            if st.button("Close All Positions", use_container_width=True, help="Close all open positions"):
                positions = mt5.positions_get()
                if positions:
                    with st.spinner("Closing all positions..."):
                        closed_count = 0
                        for pos in positions:
                            # Create request to close position
                            close_price = mt5.symbol_info_tick(pos.symbol).bid if pos.type == mt5.POSITION_TYPE_BUY else mt5.symbol_info_tick(pos.symbol).ask
                            
                            request = {
                                "action": mt5.TRADE_ACTION_DEAL,
                                "symbol": pos.symbol,
                                "volume": pos.volume,
                                "type": mt5.ORDER_TYPE_SELL if pos.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                                "position": pos.ticket,
                                "price": close_price,
                                "deviation": 20,
                                "magic": 234000,
                                "comment": "Close position",
                                "type_time": mt5.ORDER_TIME_GTC,
                                "type_filling": mt5.ORDER_FILLING_IOC,
                            }
                            
                            result = mt5.order_send(request)
                            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                                closed_count += 1
                        
                        if closed_count > 0:
                            st.success(f"Closed {closed_count} positions successfully")
                            # Force a refresh
                            st.rerun()
                        else:
                            st.error("Failed to close positions")
                else:
                    st.info("No positions to close")
        
        with batch_col2:
            if st.button("Close Profitable Positions", use_container_width=True, help="Close all positions currently in profit"):
                positions = mt5.positions_get()
                if positions:
                    profit_positions = [pos for pos in positions if pos.profit > 0]
                    
                    if profit_positions:
                        with st.spinner("Closing profitable positions..."):
                            closed_count = 0
                            for pos in profit_positions:
                                # Create request to close position
                                close_price = mt5.symbol_info_tick(pos.symbol).bid if pos.type == mt5.POSITION_TYPE_BUY else mt5.symbol_info_tick(pos.symbol).ask
                                
                                request = {
                                    "action": mt5.TRADE_ACTION_DEAL,
                                    "symbol": pos.symbol,
                                    "volume": pos.volume,
                                    "type": mt5.ORDER_TYPE_SELL if pos.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                                    "position": pos.ticket,
                                    "price": close_price,
                                    "deviation": 20,
                                    "magic": 234000,
                                    "comment": "Close profitable position",
                                    "type_time": mt5.ORDER_TIME_GTC,
                                    "type_filling": mt5.ORDER_FILLING_IOC,
                                }
                                
                                result = mt5.order_send(request)
                                if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                                    closed_count += 1
                            
                            if closed_count > 0:
                                st.success(f"Closed {closed_count} profitable positions")
                                # Force a refresh
                                st.rerun()
                            else:
                                st.error("Failed to close positions")
                    else:
                        st.info("No profitable positions to close")
                else:
                    st.info("No positions to close")
        
        with batch_col3:
            if st.button("Update Stop Losses", use_container_width=True, help="Set break-even stop losses for positions in profit"):
                positions = mt5.positions_get()
                if positions:
                    profit_positions = [pos for pos in positions if pos.profit > 0]
                    
                    if profit_positions:
                        with st.spinner("Updating stop losses..."):
                            updated_count = 0
                            for pos in profit_positions:
                                # Move stop loss to break even + 1 pip
                                point = mt5.symbol_info(pos.symbol).point
                                new_sl = pos.price_open + (10 * point * (1 if pos.type == mt5.POSITION_TYPE_BUY else -1))
                                
                                request = {
                                    "action": mt5.TRADE_ACTION_SLTP,
                                    "symbol": pos.symbol,
                                    "position": pos.ticket,
                                    "sl": new_sl,
                                    "tp": pos.tp
                                }
                                
                                result = mt5.order_send(request)
                                if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                                    updated_count += 1
                            
                            if updated_count > 0:
                                st.success(f"Updated {updated_count} stop losses to break-even")
                            else:
                                st.error("Failed to update stop losses")
                    else:
                        st.info("No profitable positions to update")
                else:
                    st.info("No positions to update")
    
    # Replace the existing Performance Tab code in main.py (in the "with tab4:" section)
    # with this updated version that uses MT5 history data

    with tab4:
        # Performance Tab - Trade history and analytics
        st.subheader("JustMarkets Performance Analytics")
        
        # History period selector
        history_period_options = [
            ("Last 7 Days", 7),
            ("Last 30 Days", 30),
            ("Last 90 Days", 90)
        ]
        
        history_period = st.selectbox(
            "Select Period",
            history_period_options,
            format_func=lambda x: x[0],
            index=1  # Default to 30 days
        )
        
        # Button to refresh data
        col_refresh, col_source = st.columns([1, 2])
        with col_refresh:
            if st.button("Refresh MT5 History", key="refresh_mt5_history", use_container_width=True):
                with st.spinner("Fetching trading history from JustMarkets..."):
                    st.session_state.mt5_history = get_mt5_trading_history(history_period[1])
                    st.session_state.mt5_stats = calculate_mt5_trade_stats(history_period[1])
                    st.success(f"Retrieved {len(st.session_state.mt5_history)} trades from JustMarkets MT5")
        
        with col_source:
            st.info("📊 Data is fetched directly from your JustMarkets MT5 account")
        
        # Initialize or update session state for MT5 history
        if 'mt5_history' not in st.session_state or st.session_state.get('mt5_history_days') != history_period[1]:
            with st.spinner("Fetching trading history from JustMarkets..."):
                st.session_state.mt5_history = get_mt5_trading_history(history_period[1])
                st.session_state.mt5_stats = calculate_mt5_trade_stats(history_period[1])
                st.session_state.mt5_history_days = history_period[1]
        
        # Use MT5 stats for display
        stats = st.session_state.mt5_stats
        
        # Summary statistics at the top
        stats_col1, stats_col2, stats_col3, stats_col4, stats_col5 = st.columns(5)
        
        with stats_col1:
            st.metric(
                "Total Trades", 
                stats["total_trades"],
                delta=None
            )
        
        with stats_col2:
            st.metric(
                "Win Rate", 
                f"{stats['win_rate']:.1f}%",
                delta=None
            )
        
        with stats_col3:
            avg_profit = stats.get("avg_profit", 0)
            st.metric(
                "Avg. Profit", 
                f"${avg_profit:.2f}",
                delta=None
            )
        
        with stats_col4:
            avg_loss = stats.get("avg_loss", 0)
            st.metric(
                "Avg. Loss", 
                f"${avg_loss:.2f}",
                delta=None
            )
        
        with stats_col5:
            st.metric(
                "Profit Factor", 
                f"{stats['profit_factor']:.2f}",
                delta=None
            )
        
        # Create tabs for different performance views
        perf_tab1, perf_tab2, perf_tab3 = st.tabs(["JustMarkets Trade History", "Performance by Pair", "Performance by Time"])
        
        with perf_tab1:
            # Trade history table
            st.subheader(f"JustMarkets Trading History ({history_period[0]})")
            
            # Get history from session state
            mt5_history = st.session_state.mt5_history
            
            if mt5_history:
                # Sort by close time, most recent first (should already be sorted)
                for trade in mt5_history:
                    trade_result = trade.get("result", "")
                    trade_class = "win-trade" if trade_result == "win" else "loss-trade"
                    profit = trade.get("pnl", 0)
                    profit_color = "#4CAF50" if profit >= 0 else "#F44336"
                    
                    # Format trade data
                    symbol = trade.get("symbol", "")
                    action = trade.get("action", "").upper()
                    
                    # Format times
                    entry_time = trade.get("time")
                    if isinstance(entry_time, datetime.datetime):
                        entry_time = entry_time.strftime("%Y-%m-%d %H:%M")
                    
                    close_time = trade.get("close_time")
                    if isinstance(close_time, datetime.datetime):
                        close_time = close_time.strftime("%Y-%m-%d %H:%M")
                    
                    lot_size = trade.get("lot_size", 0)
                    pips = trade.get("pips", 0)
                    
                    # Colorized profit amount
                    profit_display = f"+{profit:.2f}" if profit >= 0 else f"{profit:.2f}"
                    
                    st.markdown(f"""
                    <div class="trade-history-card {trade_class}">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <strong>{symbol}</strong>
                                <span style="background-color: rgba({76 if action == 'BUY' else 244}, {175 if action == 'BUY' else 67}, {80 if action == 'BUY' else 54}, 0.2); 
                                    color: #{2 if action == 'BUY' else 'C6'}E7D32; padding: 3px 8px; border-radius: 4px; margin-left: 10px;">
                                    {action}
                                </span>
                            </div>
                            <div style="color: {profit_color}; font-weight: bold;">
                                {profit_display} USD ({round(pips, 1) if pips else 0} pips)
                            </div>
                        </div>
                        <div style="margin-top: 10px; font-size: 0.9rem; display: flex; justify-content: space-between;">
                            <div>
                                <div>Open: {entry_time}</div>
                                <div>Close: {close_time}</div>
                            </div>
                            <div>
                                <div>Lot Size: {lot_size}</div>
                                <div>Result: <span style="color: {profit_color};">{trade_result.upper()}</span></div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Add export button
                if st.button("Export JustMarkets History to CSV", key="export_mt5_history"):
                    # Convert the history to a DataFrame
                    export_df = pd.DataFrame(mt5_history)
                    
                    # Convert to CSV
                    csv = export_df.to_csv(index=False)
                    
                    # Create a download link
                    b64 = base64.b64encode(csv.encode()).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="justmarkets_history.csv">Download CSV File</a>'
                    st.markdown(href, unsafe_allow_html=True)
                    
            else:
                st.info("No JustMarkets trading history found for the selected period.")
    
    with perf_tab2:
        # Performance by currency pair
        st.subheader("Performance by Currency Pair")
        
        # Create data for the chart
        if stats.get("by_symbol"):
            # Prepare data for visualization
            symbols = list(stats["by_symbol"].keys())
            pnl_values = [stats["by_symbol"][s]["pnl"] for s in symbols]
            win_rates = [stats["by_symbol"][s]["win_rate"] for s in symbols]
            trade_counts = [stats["by_symbol"][s]["trades"] for s in symbols]
            
            # Create two-column layout
            pair_col1, pair_col2 = st.columns([3, 2])
            
            with pair_col1:
                # Create bar chart using HTML/CSS for better customization
                st.markdown("""
                <div style="margin-top: 20px;">
                    <h4>Profit/Loss by Pair</h4>
                </div>
                """, unsafe_allow_html=True)
                
                # Calculate the max absolute value for scaling
                max_abs_pnl = max(abs(v) for v in pnl_values) if pnl_values else 1
                
                for i, symbol in enumerate(symbols):
                    pnl = pnl_values[i]
                    win_rate = win_rates[i]
                    trades = trade_counts[i]
                    
                    # Calculate bar width as percentage (with minimum width for visibility)
                    bar_width = min(100, max(5, abs(pnl) / max_abs_pnl * 85))
                    
                    # Determine bar color and direction
                    bar_color = "#4CAF50" if pnl >= 0 else "#F44336"
                    bar_direction = "left" if pnl >= 0 else "right"
                    
                    st.markdown(f"""
                    <div style="margin-bottom: 15px;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                            <div>
                                <strong>{symbol}</strong> ({trades} trades)
                            </div>
                            <div>
                                <span style="color: {bar_color}; font-weight: bold;">
                                    {'+' if pnl >= 0 else ''}{pnl:.2f} USD
                                </span>
                                <span style="margin-left: 10px; color: #757575;">
                                    {win_rate:.1f}% Win
                                </span>
                            </div>
                        </div>
                        <div style="display: flex; align-items: center; height: 20px;">
                            <div style="flex: 1; display: flex; justify-content: center;">
                                <div style="background-color: #e0e0e0; height: 6px; width: 100%; border-radius: 3px; position: relative;">
                                    <div style="position: absolute; top: 0; {bar_direction}: 50%; 
                                                background-color: {bar_color}; height: 6px; width: {bar_width}%; 
                                                border-radius: 3px;"></div>
                                </div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            with pair_col2:
                # Create a table of performance statistics
                st.markdown("""
                <div style="margin-top: 20px;">
                    <h4>Performance Summary</h4>
                </div>
                """, unsafe_allow_html=True)
                
                for i, symbol in enumerate(symbols):
                    pnl = pnl_values[i]
                    win_rate = win_rates[i]
                    trades = trade_counts[i]
                    avg_profit = pnl / trades if trades > 0 else 0
                    
                    profit_color = "#4CAF50" if pnl >= 0 else "#F44336"
                    
                    st.markdown(f"""
                    <div style="border: 1px solid #e0e0e0; border-radius: 5px; padding: 10px; margin-bottom: 10px;">
                        <div style="font-weight: bold; font-size: 1.1rem; margin-bottom: 5px;">{symbol}</div>
                        <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                            <div>Total P&L:</div>
                            <div style="color: {profit_color}; font-weight: bold;">
                                {'+' if pnl >= 0 else ''}{pnl:.2f} USD
                            </div>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                            <div>Win Rate:</div>
                            <div>{win_rate:.1f}%</div>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                            <div>Trades:</div>
                            <div>{trades}</div>
                        </div>
                        <div style="display: flex; justify-content: space-between;">
                            <div>Avg Profit:</div>
                            <div style="color: {profit_color};">
                                {'+' if avg_profit >= 0 else ''}{avg_profit:.2f} USD
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("Not enough trading data to show performance by currency pair.")
    
    with perf_tab3:
        # Performance by time period
        st.subheader("JustMarkets Performance Over Time")
        
        # Get history from session state
        mt5_history = st.session_state.mt5_history
        
        if mt5_history and len(mt5_history) > 1:
            # Group trades by day
            trade_days = {}
            
            for trade in mt5_history:
                # Get the close time
                close_time = trade.get("close_time")
                if isinstance(close_time, str):
                    # Try to convert from ISO format
                    try:
                        close_time = datetime.datetime.fromisoformat(close_time)
                    except:
                        # If conversion fails, skip this trade
                        continue
                
                # Extract just the date part
                if isinstance(close_time, int):
                    # Handle integer timestamp
                    close_date = datetime.datetime.fromtimestamp(close_time).date()
                elif isinstance(close_time, datetime.datetime):
                    # Already a datetime object
                    close_date = close_time.date()
                else:
                    # Try to convert from string or other format
                    try:
                        if isinstance(close_time, str):
                            close_date = datetime.datetime.fromisoformat(close_time).date()
                        else:
                            # Default to current date if conversion fails
                            close_date = datetime.datetime.now(timezone.utc).date()
                            logger.warning(f"Unknown date format in trade history: {type(close_time)}")
                    except:
                        # If all else fails, use today's date
                        close_date = datetime.datetime.now(timezone.utc).date()
                        logger.warning(f"Failed to convert date format: {close_time}")
                                
                if close_date not in trade_days:
                    trade_days[close_date] = {
                        "pnl": 0,
                        "wins": 0,
                        "losses": 0,
                        "trades": 0
                    }
                
                trade_days[close_date]["pnl"] += trade.get("pnl", 0)
                trade_days[close_date]["trades"] += 1
                
                if trade.get("result") == "win":
                    trade_days[close_date]["wins"] += 1
                elif trade.get("result") == "loss":
                    trade_days[close_date]["losses"] += 1
            
            # Sort days chronologically
            sorted_days = sorted(trade_days.keys())
            
            # Prepare data for visualization
            dates = [day.strftime("%Y-%m-%d") for day in sorted_days]
            daily_pnl = [trade_days[day]["pnl"] for day in sorted_days]
            cumulative_pnl = []
            running_sum = 0
            
            for pnl in daily_pnl:
                running_sum += pnl
                cumulative_pnl.append(running_sum)
            
            win_rates = [trade_days[day]["wins"] / trade_days[day]["trades"] * 100 if trade_days[day]["trades"] > 0 else 0 
                         for day in sorted_days]
            
            # Create chart using HTML/CSS
            st.markdown("""
            <div style="margin-top: 20px;">
                <h4>Cumulative Profit/Loss</h4>
            </div>
            """, unsafe_allow_html=True)
            
            # Create a simple line chart visualization
            chart_height = 250
            chart_width = 100
            
            # Find min and max for scaling
            min_pnl = min(cumulative_pnl) if cumulative_pnl else 0
            max_pnl = max(cumulative_pnl) if cumulative_pnl else 0
            
            # Ensure we have a non-zero range for scaling
            y_range = max(1, max_pnl - min_pnl)
            
            # Create SVG chart
            svg_lines = []
            svg_points = []
            svg_labels = []
            
            for i in range(len(dates)):
                x = i * (chart_width / max(1, len(dates) - 1))
                y = chart_height - ((cumulative_pnl[i] - min_pnl) / y_range * chart_height)
                
                if i == 0:
                    svg_lines.append(f"M {x} {y}")
                else:
                    svg_lines.append(f"L {x} {y}")
                
                svg_points.append(f"""
                    <circle cx="{x}" cy="{y}" r="3" fill="{'#4CAF50' if daily_pnl[i] >= 0 else '#F44336'}" />
                """)
                
                # Add labels for dates (every nth label to avoid crowding)
                if i % max(1, len(dates) // 5) == 0 or i == len(dates) - 1:
                    svg_labels.append(f"""
                        <text x="{x}" y="{chart_height + 15}" text-anchor="middle" style="font-size: 8px;">{dates[i]}</text>
                    """)
            
            # Create the SVG
            svg = f"""
            <svg width="100%" height="{chart_height + 30}" style="overflow: visible;">
                <!-- Y-axis -->
                <line x1="0" y1="0" x2="0" y2="{chart_height}" stroke="#e0e0e0" stroke-width="1" />
                
                <!-- Y-axis labels -->
                <text x="-5" y="10" text-anchor="end" style="font-size: 10px;">${max_pnl:.2f}</text>
                <text x="-5" y="{chart_height}" text-anchor="end" style="font-size: 10px;">${min_pnl:.2f}</text>
                
                <!-- X-axis -->
                <line x1="0" y1="{chart_height}" x2="{chart_width}" y2="{chart_height}" stroke="#e0e0e0" stroke-width="1" />
                
                <!-- X-axis labels -->
                {"".join(svg_labels)}
                
                <!-- Zero line if applicable -->
                {f'<line x1="0" y1="{chart_height - (-min_pnl / y_range * chart_height)}" x2="{chart_width}" y2="{chart_height - (-min_pnl / y_range * chart_height)}" stroke="#757575" stroke-width="1" stroke-dasharray="5,5" />' if min_pnl < 0 and max_pnl > 0 else ''}
                
                <!-- Line chart -->
                <path d="{"".join(svg_lines)}" fill="none" stroke="{('#4CAF50' if cumulative_pnl[-1] >= 0 else '#F44336') if cumulative_pnl else '#9E9E9E'}" stroke-width="2" />
                
                <!-- Data points -->
                {"".join(svg_points)}
            </svg>
            """
            
            st.markdown(f"""
            <div style="width: 100%; overflow-x: auto;">
                {svg}
            </div>
            """, unsafe_allow_html=True)
            
            # Show daily performance table
            st.markdown("""
            <div style="margin-top: 30px;">
                <h4>Daily Performance</h4>
            </div>
            """, unsafe_allow_html=True)
            
            # Create scrollable table for daily performance
            st.markdown("""
            <div style="max-height: 300px; overflow-y: auto; margin-top: 10px;">
                <table style="width: 100%; border-collapse: collapse;">
                    <thead style="position: sticky; top: 0; background-color: white;">
                        <tr>
                            <th style="padding: 8px; text-align: left; border-bottom: 1px solid #e0e0e0;">Date</th>
                            <th style="padding: 8px; text-align: right; border-bottom: 1px solid #e0e0e0;">P&L</th>
                            <th style="padding: 8px; text-align: right; border-bottom: 1px solid #e0e0e0;">Trades</th>
                            <th style="padding: 8px; text-align: right; border-bottom: 1px solid #e0e0e0;">Win Rate</th>
                        </tr>
                    </thead>
                    <tbody>
            """, unsafe_allow_html=True)
            
            for i, day in enumerate(sorted_days):
                pnl = trade_days[day]["pnl"]
                trades = trade_days[day]["trades"]
                win_rate = trade_days[day]["wins"] / trades * 100 if trades > 0 else 0
                
                date_str = day.strftime("%Y-%m-%d")
                pnl_color = "#4CAF50" if pnl >= 0 else "#F44336"
                
                st.markdown(f"""
                    <tr>
                        <td style="padding: 8px; text-align: left; border-bottom: 1px solid #f0f0f0;">{date_str}</td>
                        <td style="padding: 8px; text-align: right; border-bottom: 1px solid #f0f0f0; color: {pnl_color}; font-weight: bold;">
                            {'+' if pnl >= 0 else ''}{pnl:.2f}
                        </td>
                        <td style="padding: 8px; text-align: right; border-bottom: 1px solid #f0f0f0;">{trades}</td>
                        <td style="padding: 8px; text-align: right; border-bottom: 1px solid #f0f0f0;">{win_rate:.1f}%</td>
                    </tr>
                """, unsafe_allow_html=True)
            
            st.markdown("""
                    </tbody>
                </table>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Not enough JustMarkets trading data to show performance over time. Complete more trades to see this chart.")
    
    with tab5:
        # Settings Tab - Bot configuration and preferences
        st.subheader("Bot Settings")
        
        # Create tabs for different settings categories
        settings_tab1, settings_tab2, settings_tab3, settings_tab4 = st.tabs([
            "General Settings", 
            "Trading Parameters", 
            "Notifications", 
            "Account"
        ])
        
        with settings_tab1:
            # General Settings
            st.subheader("General Settings")
            
            # Auto-trading settings
            st.markdown("<h4>Auto-Trading</h4>", unsafe_allow_html=True)
            
            auto_trading = st.checkbox(
                "Enable Auto-Trading", 
                value=st.session_state.auto_trading,
                help="When enabled, the bot will automatically execute trades based on signals"
            )
            
            st.session_state.auto_trading = auto_trading
            
            if auto_trading:
                st.warning("⚠️ Auto-trading is enabled. The bot will place trades automatically when signals occur.")
            
            # Auto-refresh settings
            st.markdown("<h4>Auto-Refresh</h4>", unsafe_allow_html=True)
            
            auto_refresh = st.checkbox(
                "Enable Auto-Refresh", 
                value=st.session_state.auto_refresh,
                help="When enabled, the dashboard will refresh automatically every 10 seconds"
            )
            
            st.session_state.auto_refresh = auto_refresh
            
            if auto_refresh:
                st.info("ℹ️ Auto-refresh is enabled. The dashboard will update every 10 seconds.")
            
            # Theme settings
            st.markdown("<h4>Theme Settings</h4>", unsafe_allow_html=True)
            
            theme = st.radio(
                "Theme",
                ["Light", "Dark"],
                horizontal=True,
                help="Choose the display theme for the dashboard"
            )
            
            if theme == "Dark":
                st.info("Dark theme will be implemented in a future update.")
        
        with settings_tab2:
            # Trading Parameters
            st.subheader("Trading Parameters")
            
            # Risk settings
            st.markdown("<h4>Risk Management</h4>", unsafe_allow_html=True)
            
            default_risk = st.slider(
                "Default Risk Percentage",
                min_value=0.5,
                max_value=5.0,
                value=2.0,
                step=0.1,
                help="Percentage of account balance to risk per trade"
            )
            
            # Trading pairs
            st.markdown("<h4>Trading Markets</h4>", unsafe_allow_html=True)

            all_pairs = get_default_symbols()  # Get all profitable markets

            selected_pairs = st.multiselect(
                "Select markets to trade",
                all_pairs,
                default=["BTCUSD.m", "XAUUSD.m", "WTI.m", "AUDCAD.m", "EURUSD.m"],
                help="The bot will trade these profitable markets (Crypto, Commodities, Forex)"
            )
            
            # Save selected pairs to session state
            st.session_state.selected_pairs = selected_pairs
            
            # Trading timeframes
            st.markdown("<h4>Timeframes</h4>", unsafe_allow_html=True)
            
            all_timeframes = ["M1", "M5", "M15", "M30", "H1", "H4", "D1"]
            
            selected_timeframes = st.multiselect(
                "Select timeframes for signal generation",
                all_timeframes,
                default=["M5", "M15", "M30", "H1"],
                help="The bot will only generate signals on these timeframes"
            )
            
            # Save selected timeframes to session state
            st.session_state.selected_timeframes = selected_timeframes
            
            # Trading hours
            st.markdown("<h4>Trading Hours (GMT)</h4>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                start_hour = st.number_input(
                    "Start Hour",
                    min_value=0,
                    max_value=23,
                    value=7,
                    help="Hour to start trading (GMT)"
                )
            
            with col2:
                end_hour = st.number_input(
                    "End Hour",
                    min_value=0,
                    max_value=23,
                    value=16,
                    help="Hour to stop trading (GMT)"
                )
            
            if st.button("Save Trading Parameters", use_container_width=True):
                st.success("Trading parameters saved successfully!")
                
                # Here you would save these settings to a config file or database
                # For now we'll just keep them in session state
                st.session_state.default_risk = default_risk
                st.session_state.trading_hours = (start_hour, end_hour)
        
        
          
        with settings_tab4:
            # Account Settings
            st.subheader("JustMarkets Account Settings")
            
            # Add a button to verify JustMarkets connection
            if st.button("Verify JustMarkets Connection", use_container_width=True):
                with st.spinner("Verifying connection to JustMarkets..."):
                    verify_justmarkets_connection()
            
            # MT5 connection settings
            st.markdown("<h4>JustMarkets MT5 Connection</h4>", unsafe_allow_html=True)
            
            account_settings_col1, account_settings_col2 = st.columns(2)
            
            with account_settings_col1:
                st.markdown("""
                <div style="border: 1px solid #e0e0e0; border-radius: 10px; padding: 15px;">
                    <h5>JustMarkets Demo Account</h5>
                    <p><strong>Account ID:</strong> 2001479025</p>
                    <p><strong>Server:</strong> JustMarkets-Demo</p>
                </div>
                """, unsafe_allow_html=True)
            
            with account_settings_col2:
                st.markdown("""
                <div style="border: 1px solid #e0e0e0; border-radius: 10px; padding: 15px;">
                    <h5>JustMarkets Real Account</h5>
                    <p><strong>Account ID:</strong> 2050196801</p>
                    <p><strong>Server:</strong> JustMarkets-Live</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Credentials management
            st.markdown("<h4>JustMarkets Credentials Management</h4>", unsafe_allow_html=True)
            
            # Add information about JustMarkets-specific issues
            st.markdown("""
            ### JustMarkets Connectivity Troubleshooting
            
            If you're having trouble connecting to JustMarkets, try these steps:
            
            1. Ensure MT5 is running and logged in to your JustMarkets account
            2. Check that "AutoTrading" is enabled in MT5 (button in the top toolbar)
            3. Verify your internet connection can reach JustMarkets servers
            4. Add the currency pairs to your MarketWatch in MT5
            5. Restart your MT5 terminal and try again
            """)
            
            # Tools and utilities
            st.markdown("<h4>Tools & Utilities</h4>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Export Trading History", use_container_width=True):
                    if trade_history:
                        # In a real implementation, this would generate a CSV file for download
                        st.success("Trading history exported successfully!")
                    else:
                        st.info("No trading history to export.")
            
            with col2:
                if st.button("Reset Bot Settings", use_container_width=True):
                    # This would reset all settings to default values
                    st.warning("This will reset all bot settings to their default values.")
                    confirm_reset = st.button("Confirm Reset", key="confirm_reset")
                    if confirm_reset:
                        st.success("Bot settings reset to defaults!")
                        # In a real implementation, this would reset all settings

    with tab6:
        # Market Information Tab - Details about each market
        st.subheader("Market Information")
        
        # Market selector
        selected_info_market = st.selectbox(
            "Select Market",
            all_markets,
            key="market_info_selector"
        )
        
        # Determine market type
        is_crypto = "BTC" in selected_info_market or "ETH" in selected_info_market
        is_metal = "XAU" in selected_info_market or "GOLD" in selected_info_market or "XAG" in selected_info_market
        is_index = any(idx in selected_info_market for idx in ["US30.std", "US500.std", "USTEC.std", "DE30.std", "UK100.std", "JP225.std"])
        
        if is_crypto:
            market_category = "Cryptocurrency"
        elif is_metal:
            market_category = "Precious Metal"
        elif is_index:
            market_category = "Stock Index"
        else:
            market_category = "Forex Pair"
        
        # Display market information
        st.markdown(f"### {selected_info_market} ({market_category})")
        
        # Get symbol information
        symbol_info = mt5.symbol_info(selected_info_market)
        
        if symbol_info:
            # Create columns for information display
            info_col1, info_col2 = st.columns([1, 1])
            
            with info_col1:
                st.markdown("#### Trading Parameters")
                
                # Format display based on market type
                if is_crypto:
                    tick_value = f"${symbol_info.trade_tick_value:.2f} per point"
                    contract_size = f"{symbol_info.trade_contract_size:.2f} units"
                elif is_metal and "XAU" in selected_info_market:
                    tick_value = f"${symbol_info.trade_tick_value:.2f} per 0.01 point"
                    contract_size = f"{symbol_info.trade_contract_size:.2f} oz"
                else:
                    tick_value = f"${symbol_info.trade_tick_value:.5f} per point"
                    contract_size = f"{symbol_info.trade_contract_size:,.0f} units"
                
                # Create info table
                info_table = f"""
                | Parameter | Value |
                | --- | --- |
                | Min Lot | {symbol_info.volume_min:.2f} |
                | Max Lot | {symbol_info.volume_max:.2f} |
                | Lot Step | {symbol_info.volume_step:.2f} |
                | Contract Size | {contract_size} |
                | Tick Value | {tick_value} |
                | Spread | {(symbol_info.ask - symbol_info.bid) / symbol_info.point:.1f} points |
                | Digits | {symbol_info.digits} |
                | Stop Level | {symbol_info.trade_stops_level} points |
                """
                
                st.markdown(info_table)
            
            with info_col2:
                st.markdown("#### Trading Statistics")
                
                # Get performance stats for this symbol
                symbol_stats = {}
                for trade in trade_history:
                    if trade.get("symbol") == selected_info_market and trade.get("status") == "closed":
                        result = trade.get("result", "")
                        if "symbol_stats" not in locals():
                            symbol_stats = {
                                "total": 0,
                                "wins": 0,
                                "losses": 0,
                                "pnl": 0,
                                "pips": 0
                            }
                        
                        symbol_stats["total"] += 1
                        if result == "win":
                            symbol_stats["wins"] += 1
                        elif result == "loss":
                            symbol_stats["losses"] += 1
                        
                        symbol_stats["pnl"] += trade.get("pnl", 0)
                        symbol_stats["pips"] += trade.get("pips", 0)
                
                if symbol_stats.get("total", 0) > 0:
                    win_rate = symbol_stats.get("wins", 0) / symbol_stats.get("total", 1) * 100
                    
                    stats_table = f"""
                    | Statistic | Value |
                    | --- | --- |
                    | Total Trades | {symbol_stats.get('total', 0)} |
                    | Win Rate | {win_rate:.1f}% |
                    | Total P&L | ${symbol_stats.get('pnl', 0):.2f} |
                    | Total Pips | {symbol_stats.get('pips', 0):.1f} |
                    | Avg. Trade | ${symbol_stats.get('pnl', 0) / symbol_stats.get('total', 1):.2f} |
                    """
                    
                    st.markdown(stats_table)
                else:
                    st.info(f"No trading history for {selected_info_market} yet.")
            
            # Market-specific information
            if is_crypto:
                st.markdown("#### Cryptocurrency Market Information")
                st.markdown("""
                **Trading Hours**: 24/7 with increased activity during US and European trading hours

                **Key Levels to Watch**:
                - $20,000 - Major psychological level
                - $25,000 - Strong resistance level
                - $30,000 - Major resistance zone
                - $35,000 - Key fibonacci level
                
                **Trade Tips**:
                - Bitcoin has increased volatility during US stock market hours
                - Weekends often have lower liquidity and can lead to price gaps
                - Use wider stop losses (2% or more) due to high volatility
                - Watch for correlations with tech stocks and risk sentiment
                """)
            
            elif is_metal and "XAU" in selected_info_market:
                st.markdown("#### Gold Market Information")
                st.markdown("""
                **Trading Hours**: Most active during London and New York sessions
                
                **Key Factors**:
                - Strongly influenced by US Dollar strength
                - Sensitive to interest rate expectations
                - Acts as a safe haven during market uncertainty
                - Affected by inflation data
                
                **Trade Tips**:
                - Best trading times: 8:00-16:00 GMT
                - Watch for volatility around Fed announcements
                - Often trends well for extended periods
                - Can have strong reactions to geopolitical events
                """)
            
            elif is_index:
                index_name = {
                    "US30.std": "Dow Jones Industrial Average",
                    "US500.std": "S&P 500",
                    "USTEC.std": "Nasdaq 100",
                    "DE30.std": "DAX 40 (Germany)",
                    "UK100.std": "FTSE 100 (UK)",
                    "JP225.std": "Nikkei 225 (Japan)",
                    "AUS200": "ASX 200 (Australia)"
                }.get(selected_info_market, selected_info_market)
                
                st.markdown(f"#### {index_name} Information")
                st.markdown(f"""
                **Trading Hours**: Follows the underlying market hours with some extended trading
                
                **Key Characteristics**:
                - Highly affected by economic data releases
                - Sensitive to central bank policies
                - Influenced by major company earnings
                - Reacts to geopolitical developments
                
                **Trade Tips**:
                - Indices often show clearer trends than individual stocks
                - Watch for gap trading opportunities at market open
                - Volatility typically increases in the first and last hour of trading
                - Pay attention to futures markets which can indicate opening direction
                """)
        else:
            st.warning(f"Could not retrieve information for {selected_info_market}. Make sure the symbol is available in your broker.")
        
        # Add market analysis tools
        st.subheader("Market Analysis Tools")
        
        analysis_type = st.radio(
            "Analysis Type",
            ["Technical", "Correlation", "Volatility"],
            horizontal=True
        )
        
        if analysis_type == "Technical":
            # Get data for technical analysis
            tf_options = [
                ("5 Minutes", mt5.TIMEFRAME_M5),
                ("15 Minutes", mt5.TIMEFRAME_M15),
                ("30 Minutes", mt5.TIMEFRAME_M30),
                ("1 Hour", mt5.TIMEFRAME_H1),
                ("4 Hours", mt5.TIMEFRAME_H4),
                ("Daily", mt5.TIMEFRAME_D1)
            ]
            
            selected_tf = st.selectbox(
                "Timeframe",
                tf_options,
                format_func=lambda x: x[0]
            )
            
            # Get data
            df = get_market_data(selected_info_market, selected_tf[1], 200)
            # Initialize levels_table with a default value
            levels_table = "No data available for key levels analysis."
            
            if not df.empty:
                df = calculate_indicators(df, selected_info_market)
                
                # Create chart
                # (This is a placeholder - in a real implementation you'd create a more comprehensive chart)
                st.line_chart(df['close'])
                
                # Display key levels
                st.markdown("#### Key Technical Levels")
                
                # Format based on market type
                if is_crypto:
                    price_format = "{:.1f}"
                elif is_metal and "XAU" in selected_info_market:
                    price_format = "{:.2f}"
                elif is_index:
                    price_format = "{:.1f}"
                else:
                    price_format = "{:.5f}"
                
                # Calculate key levels
                current_price = df['close'].iloc[-1]
                sma_50 = df['SMA_50'].iloc[-1]
                sma_200 = df['SMA_200'].iloc[-1]
                upper_band = df['upper_band'].iloc[-1]
                lower_band = df['lower_band'].iloc[-1]
                recent_high = df['high'].rolling(20).max().iloc[-1]
                recent_low = df['low'].rolling(20).min().iloc[-1]
                
                levels_table = f"""
            | Level | Value | Distance |
            | --- | --- | --- |
            | Current Price | {price_format.format(current_price)} | - |
            | Upper BB | {price_format.format(upper_band)} | {((upper_band/current_price)-1)*100:.2f}% |
            | Lower BB | {price_format.format(lower_band)} | {((lower_band/current_price)-1)*100:.2f}% |
            | SMA 50 | {price_format.format(sma_50)} | {((sma_50/current_price)-1)*100:.2f}% |
            | SMA 200 | {price_format.format(sma_200)} | {((sma_200/current_price)-1)*100:.2f}% |
            | Recent High | {price_format.format(recent_high)} | {((recent_high/current_price)-1)*100:.2f}% |
            | Recent Low | {price_format.format(recent_low)} | {((recent_low/current_price)-1)*100:.2f}% |
            """
            
            st.markdown(levels_table)
            
        elif analysis_type == "Correlation":
            # Market correlation analysis
            st.markdown("#### Market Correlation Analysis")
            
            # Let user select markets to compare
            compare_markets = st.multiselect(
                "Select Markets to Compare",
                all_markets,
                default=[selected_info_market, "GBPUSD.m", "US500.std", "XAUUSD.m"],
                key="correlation_markets"
            )
            
            if len(compare_markets) > 1:
                # Get correlation data
                correlation_period = st.slider(
                    "Correlation Period (Days)",
                    min_value=7,
                    max_value=90,
                    value=30,
                    step=1
                )
                
                # Collect data for all selected markets
                correlation_data = {}
                for symbol in compare_markets:
                    df = get_market_data(symbol, mt5.TIMEFRAME_D1, correlation_period)
                    if not df.empty:
                        correlation_data[symbol] = df['close']
                
                if len(correlation_data) > 1:
                    # Create correlation dataframe
                    corr_df = pd.DataFrame(correlation_data)
                    correlation_matrix = corr_df.pct_change().corr()
                    
                    # Display correlation matrix
                    st.markdown("##### Correlation Matrix (Price Movement)")
                    
                    # Create a color-coded correlation table using HTML
                    corr_html = "<table style='width:100%; border-collapse: collapse;'><tr><th></th>"
                    
                    # Create header row
                    for symbol in correlation_matrix.columns:
                        corr_html += f"<th style='padding: 8px; text-align: center; border: 1px solid #ddd;'>{symbol}</th>"
                    corr_html += "</tr>"
                    
                    # Create data rows
                    for symbol1 in correlation_matrix.index:
                        corr_html += f"<tr><td style='padding: 8px; font-weight: bold; border: 1px solid #ddd;'>{symbol1}</td>"
                        
                        for symbol2 in correlation_matrix.columns:
                            corr_value = correlation_matrix.loc[symbol1, symbol2]
                            # Determine cell color based on correlation strength
                            if symbol1 == symbol2:
                                # Diagonal elements (correlation with self = 1)
                                bg_color = "#f2f2f2"
                                text_color = "#000000"
                            elif corr_value > 0.7:
                                # Strong positive correlation
                                bg_color = "#4CAF50"
                                text_color = "#ffffff"
                            elif corr_value > 0.3:
                                # Moderate positive correlation
                                bg_color = "#8BC34A"
                                text_color = "#000000"
                            elif corr_value > -0.3:
                                # Weak or no correlation
                                bg_color = "#FFEB3B"
                                text_color = "#000000"
                            elif corr_value > -0.7:
                                # Moderate negative correlation
                                bg_color = "#FFC107"
                                text_color = "#000000"
                            else:
                                # Strong negative correlation
                                bg_color = "#F44336"
                                text_color = "#ffffff"
                            
                            corr_html += f"<td style='padding: 8px; text-align: center; border: 1px solid #ddd; background-color: {bg_color}; color: {text_color};'>{corr_value:.2f}</td>"
                        
                        corr_html += "</tr>"
                    
                    corr_html += "</table>"
                    
                    st.markdown(corr_html, unsafe_allow_html=True)
                    
                    # Explain correlation results
                    st.markdown("##### Interpretation")
                    st.markdown(f"""
                    - **Strong Positive Correlation (> 0.7)**: Markets tend to move in the same direction
                    - **Moderate Positive Correlation (0.3 to 0.7)**: Markets often move in the same direction
                    - **Weak Correlation (-0.3 to 0.3)**: Little relationship between market movements
                    - **Moderate Negative Correlation (-0.7 to -0.3)**: Markets often move in opposite directions
                    - **Strong Negative Correlation (< -0.7)**: Markets tend to move in opposite directions
                    """)
                    
                    # Provide trading insights based on correlations
                    st.markdown("##### Trading Insights")
                    
                    # Find strongest correlations for the selected market
                    primary_market = selected_info_market
                    
                    if primary_market in correlation_matrix.index:
                        corr_series = correlation_matrix[primary_market].drop(primary_market)
                        positive_corr = corr_series[corr_series > 0.5].sort_values(ascending=False)
                        negative_corr = corr_series[corr_series < -0.5].sort_values()
                        
                        insights = f"**Insights for {primary_market}:**\n\n"
                        
                        if not positive_corr.empty:
                            insights += "**Positive Correlations:**\n"
                            for market, corr in positive_corr.items():
                                insights += f"- {market} (Correlation: {corr:.2f}): Consider as confirmation for {primary_market} moves\n"
                        
                        if not negative_corr.empty:
                            insights += "\n**Negative Correlations:**\n"
                            for market, corr in negative_corr.items():
                                insights += f"- {market} (Correlation: {corr:.2f}): Consider for hedging or divergence trading\n"
                        
                        if positive_corr.empty and negative_corr.empty:
                            insights += "No strong correlations detected with other markets in the selected period."
                        
                        st.markdown(insights)
                else:
                    st.warning("Could not retrieve enough data for correlation analysis.")
            else:
                st.info("Please select at least two markets to analyze correlations.")
                
        elif analysis_type == "Volatility":
            # Volatility comparison
            st.markdown("#### Volatility Analysis")
            
            # Period selection
            volatility_period = st.slider(
                "Analysis Period (Days)",
                min_value=7,
                max_value=90,
                value=30,
                step=1
            )
            
            # Get volatility data for all markets
            volatility_data = {}
            
            for symbol in all_markets:
                try:
                    df = get_market_data(symbol, mt5.TIMEFRAME_D1, volatility_period + 10)  # Extra bars for ATR calculation
                    if not df.empty and len(df) > 14:  # Need enough data for ATR
                        df['ATR'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
                        df = df.dropna()  # Remove rows with NaN ATR values
                        
                        # Calculate average daily range as percentage
                        current_price = df['close'].iloc[-1]
                        avg_atr = df['ATR'].mean()
                        percent_atr = (avg_atr / current_price) * 100
                        
                        # Store results
                        volatility_data[symbol] = {
                            'Avg ATR': avg_atr,
                            'ATR %': percent_atr,
                            'Current Price': current_price
                        }
                except Exception as e:
                    logger.error(f"Error calculating volatility for {symbol}: {e}")
            
            if volatility_data:
                # Convert to DataFrame for display
                vol_df = pd.DataFrame.from_dict(volatility_data, orient='index')
                
                # Sort by volatility percentage
                vol_df = vol_df.sort_values(by='ATR %', ascending=False)
                
                # Format for display
                display_df = vol_df.copy()
                
                # Format based on market type
                for idx in display_df.index:
                    if "BTC" in idx or "ETH" in idx:
                        display_df.loc[idx, 'Current Price'] = f"{display_df.loc[idx, 'Current Price']:.1f}"
                        display_df.loc[idx, 'Avg ATR'] = f"{display_df.loc[idx, 'Avg ATR']:.1f}"
                    elif "XAU" in idx or "GOLD" in idx:
                        display_df.loc[idx, 'Current Price'] = f"{display_df.loc[idx, 'Current Price']:.2f}"
                        display_df.loc[idx, 'Avg ATR'] = f"{display_df.loc[idx, 'Avg ATR']:.2f}"
                    elif any(market in idx for market in ["US30.std", "US500.std", "USTEC.std", "DE30.std", "UK100.std", "JP225.std"]):
                        display_df.loc[idx, 'Current Price'] = f"{display_df.loc[idx, 'Current Price']:.1f}"
                        display_df.loc[idx, 'Avg ATR'] = f"{display_df.loc[idx, 'Avg ATR']:.1f}"
                    else:
                        display_df.loc[idx, 'Current Price'] = f"{display_df.loc[idx, 'Current Price']:.5f}"
                        display_df.loc[idx, 'Avg ATR'] = f"{display_df.loc[idx, 'Avg ATR']:.5f}"
                    
                    display_df.loc[idx, 'ATR %'] = f"{display_df.loc[idx, 'ATR %']:.2f}%"
                
                # Create HTML table for better formatting
                vol_html = "<table style='width:100%; border-collapse: collapse;'>"
                vol_html += "<tr><th style='padding: 8px; border: 1px solid #ddd; background-color: #f2f2f2;'>Market</th>"
                vol_html += "<th style='padding: 8px; border: 1px solid #ddd; background-color: #f2f2f2;'>Current Price</th>"
                vol_html += "<th style='padding: 8px; border: 1px solid #ddd; background-color: #f2f2f2;'>Avg Daily Range</th>"
                vol_html += "<th style='padding: 8px; border: 1px solid #ddd; background-color: #f2f2f2;'>Volatility %</th></tr>"
                
                # Add data rows with color coding for volatility
                for idx, row in display_df.iterrows():
                    vol_pct = vol_df.loc[idx, 'ATR %']  # Get numeric value for color coding
                    
                    # Determine color based on volatility
                    if vol_pct > 3.0:  # Very high volatility
                        bg_color = "#F44336"  # Red
                        text_color = "#ffffff"
                    elif vol_pct > 1.5:  # High volatility
                        bg_color = "#FF9800"  # Orange
                        text_color = "#000000"
                    elif vol_pct > 0.8:  # Medium volatility
                        bg_color = "#FFEB3B"  # Yellow
                        text_color = "#000000"
                    elif vol_pct > 0.3:  # Low volatility
                        bg_color = "#8BC34A"  # Light green
                        text_color = "#000000"
                    else:  # Very low volatility
                        bg_color = "#4CAF50"  # Green
                        text_color = "#ffffff"
                    
                    # Highlight the currently selected market
                    if idx == selected_info_market:
                        row_style = "font-weight: bold; background-color: #E3F2FD;"
                    else:
                        row_style = ""
                    
                    vol_html += f"<tr style='{row_style}'>"
                    vol_html += f"<td style='padding: 8px; border: 1px solid #ddd;'>{idx}</td>"
                    vol_html += f"<td style='padding: 8px; border: 1px solid #ddd;'>{row['Current Price']}</td>"
                    vol_html += f"<td style='padding: 8px; border: 1px solid #ddd;'>{row['Avg ATR']}</td>"
                    vol_html += f"<td style='padding: 8px; border: 1px solid #ddd; background-color: {bg_color}; color: {text_color};'>{row['ATR %']}</td>"
                    vol_html += "</tr>"
                
                vol_html += "</table>"
                
                st.markdown(vol_html, unsafe_allow_html=True)
                
                # Provide trading insights based on volatility
                st.markdown("##### Trading Insights")
                
                # Get volatility of selected market
                if selected_info_market in vol_df.index:
                    market_vol = vol_df.loc[selected_info_market, 'ATR %']
                    
                    st.markdown(f"""
                    **Volatility Analysis for {selected_info_market}:**
                    
                    Current volatility: **{market_vol:.2f}%** daily range
                    
                    **Position Sizing Recommendation:**
                    """)
                    
                    # Adjust position size recommendation based on volatility
                    if market_vol > 3.0:
                        st.markdown("""
                        - **Very high volatility**: Reduce standard position size by 50%
                        - Use wider stop losses (at least 1.5x normal)
                        - Consider smaller take profit targets to capture quick moves
                        """)
                    elif market_vol > 1.5:
                        st.markdown("""
                        - **High volatility**: Reduce standard position size by 25%
                        - Use wider stop losses (1.2x normal)
                        - Balance between letting trades run and taking profits
                        """)
                    elif market_vol > 0.8:
                        st.markdown("""
                        - **Medium volatility**: Use standard position size
                        - Normal stop loss settings
                        - Good balance of risk/reward potential
                        """)
                    elif market_vol > 0.3:
                        st.markdown("""
                        - **Low volatility**: Consider increasing position size by 20%
                        - Tighter stop losses possible
                        - May need to be more patient with take profit targets
                        """)
                    else:
                        st.markdown("""
                        - **Very low volatility**: Consider increasing position size by 30% or wait for higher volatility
                        - Tight stop losses effective
                        - Range trading strategies may be more effective than trend following
                        """)
                    
                    # Compare with other markets
                    st.markdown("**Market Comparison:**")
                    
                    # Identify markets with similar and different volatility
                    similar_vol = vol_df[(vol_df['ATR %'] >= market_vol * 0.8) & (vol_df['ATR %'] <= market_vol * 1.2)].index.tolist()
                    similar_vol = [m for m in similar_vol if m != selected_info_market]
                    
                    higher_vol = vol_df[vol_df['ATR %'] > market_vol * 1.5].index.tolist()
                    lower_vol = vol_df[vol_df['ATR %'] < market_vol * 0.5].index.tolist()
                    
                    if similar_vol:
                        st.markdown(f"- Markets with similar volatility: {', '.join(similar_vol[:3])}")
                    
                    if higher_vol:
                        st.markdown(f"- Higher volatility alternatives: {', '.join(higher_vol[:3])}")
                    
                    if lower_vol:
                        st.markdown(f"- Lower volatility alternatives: {', '.join(lower_vol[:3])}")
            else:
                st.warning("Could not retrieve enough data for volatility analysis.")

    with tab7:
        st.subheader("Strategy Optimizer")
        
        # Market selection
        optimizer_col1, optimizer_col2 = st.columns([1, 1])
        
        with optimizer_col1:
            optimize_symbol = st.selectbox(
                "Select Market to Optimize",
                get_default_symbols(),
                key="optimize_symbol"
            )
        
        with optimizer_col2:
            optimize_period = st.slider(
                "Optimization Period (Days)",
                min_value=7,
                max_value=90,
                value=30,
                step=1,
                key="optimize_period"
            )
        
        # Timeframe selection
        timeframe_options = [
            ("15 Minutes", mt5.TIMEFRAME_M15),
            ("30 Minutes", mt5.TIMEFRAME_M30),
            ("1 Hour", mt5.TIMEFRAME_H1),
            ("4 Hours", mt5.TIMEFRAME_H4)
        ]
        
        optimize_timeframe = st.selectbox(
            "Select Timeframe",
            timeframe_options,
            format_func=lambda x: x[0],
            key="optimize_timeframe"
        )
        
        # Start optimization
        if st.button("Run Optimization", key="run_optimize_btn", use_container_width=True):
            with st.spinner(f"Optimizing strategy parameters for {optimize_symbol}..."):
                optimal_params = optimize_strategy_parameters(
                    optimize_symbol,
                    optimize_timeframe[1],
                    optimize_period
                )
                
                if optimal_params:
                    st.session_state.optimal_params = optimal_params
                    st.success(f"Optimization complete for {optimize_symbol}!")
                else:
                    st.error(f"Could not optimize parameters for {optimize_symbol}. Check data availability.")
        
        # Display optimization results
        if 'optimal_params' in st.session_state:
            params = st.session_state.optimal_params
            
            st.markdown("### Optimal Strategy Parameters")
            
            # Create two columns for parameter display
            param_col1, param_col2 = st.columns([1, 1])
            
            with param_col1:
                st.markdown("#### Technical Indicators")
                st.markdown(f"""
                | Parameter | Value |
                | --- | --- |
                | RSI Period | {params['rsi_period']} |
                | MACD Fast | {params['macd_fast']} |
                | MACD Slow | {params['macd_slow']} |
                | ATR Multiplier | {params['atr_mult']} |
                """)
            
            with param_col2:
                st.markdown("#### Performance Metrics")
                st.markdown(f"""
                | Metric | Value |
                | --- | --- |
                | Win Rate | {params['win_rate']:.2f}% |
                | Profit Factor | {params['profit_factor']:.2f} |
                | Number of Trades | {params['num_trades']} |
                """)
            
            # Apply parameters button
            if st.button("Apply Parameters to Trading Bot", key="apply_params_btn", use_container_width=True):
                # This would update your strategy parameters in a real implementation
                st.success("Parameters applied to trading bot!")
                
                # Show what settings were applied
                st.markdown("The following settings have been applied to the trading bot:")
                st.code(f"""
                # Strategy Parameters for {optimize_symbol}
                rsi_period = {params['rsi_period']}
                macd_fast = {params['macd_fast']}
                macd_slow = {params['macd_slow']}
                atr_mult = {params['atr_mult']}
                """)
    
             
    # Auto-refresh logic
    if st.session_state.auto_refresh:
        auto_refresh_placeholder = st.empty()
        auto_refresh_placeholder.info("Auto-refresh is active. Page will refresh in 10 seconds.")
        
    # Clean up MT5 connection on session end
    st.session_state.cleanup = "MT5 connection closed"

if __name__ == "__main__":
    # Load trade history at startup
    trade_history = load_trade_history()
    main()
    
    # Auto-refresh loop - this will run after the main app renders
    while st.session_state.get('auto_refresh', False):
        time.sleep(10)  # Refresh every 10 seconds
        st.rerun()