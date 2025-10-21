import pandas as pd
import numpy as np
import talib



def add_ict_concepts(df):
    """
    Enhance dataframe with ICT (Inner Circle Trader) concepts and analysis
    """
    if df.empty or len(df) < 50:
        return df
    
    # Make a copy to avoid modifying the original dataframe
    df = df.copy()
    
    # ===== 1. Market Structure Analysis =====
    # Identify higher highs (HH), higher lows (HL), lower highs (LH), lower lows (LL)
    df['swing_high'] = False
    df['swing_low'] = False
    
    # Look for swing points (simple version - can be enhanced with more sophisticated algorithms)
    window = 5  # window to look for swings
    
    for i in range(window, len(df) - window):
        # Check for swing high
        if all(df['high'].iloc[i] > df['high'].iloc[i-j] for j in range(1, window+1)) and \
           all(df['high'].iloc[i] > df['high'].iloc[i+j] for j in range(1, window+1)):
            df.loc[df.index[i], 'swing_high'] = True
            
        # Check for swing low
        if all(df['low'].iloc[i] < df['low'].iloc[i-j] for j in range(1, window+1)) and \
           all(df['low'].iloc[i] < df['low'].iloc[i+j] for j in range(1, window+1)):
            df.loc[df.index[i], 'swing_low'] = True
    
    # Track market structure by recording the sequence of swing highs and lows
    df['last_swing_high'] = None
    df['last_swing_low'] = None
    df['ms_bullish'] = False
    df['ms_bearish'] = False
    
    last_swing_high = None
    last_swing_low = None
    prev_swing_high = None
    prev_swing_low = None
    
    for i in range(window, len(df)):
        if df['swing_high'].iloc[i]:
            if last_swing_high is not None:
                prev_swing_high = last_swing_high
            last_swing_high = df['high'].iloc[i]
            df.loc[df.index[i], 'last_swing_high'] = last_swing_high
        elif last_swing_high is not None:
            df.loc[df.index[i], 'last_swing_high'] = last_swing_high
            
        if df['swing_low'].iloc[i]:
            if last_swing_low is not None:
                prev_swing_low = last_swing_low
            last_swing_low = df['low'].iloc[i]
            df.loc[df.index[i], 'last_swing_low'] = last_swing_low
        elif last_swing_low is not None:
            df.loc[df.index[i], 'last_swing_low'] = last_swing_low
            
        # Determine market structure
        if prev_swing_high is not None and last_swing_high is not None:
            df.loc[df.index[i], 'ms_bullish'] = last_swing_high > prev_swing_high
            
        if prev_swing_low is not None and last_swing_low is not None:
            df.loc[df.index[i], 'ms_bearish'] = last_swing_low < prev_swing_low
    
    # ===== 2. Fair Value Gaps (FVG) =====
    df['bullish_fvg'] = False
    df['bearish_fvg'] = False
    
    # Bullish FVG: When a candle's low is higher than the previous candle's high
    for i in range(2, len(df)):
        if df['low'].iloc[i] > df['high'].iloc[i-2]:
            df.loc[df.index[i], 'bullish_fvg'] = True
            
    # Bearish FVG: When a candle's high is lower than the previous candle's low
    for i in range(2, len(df)):
        if df['high'].iloc[i] < df['low'].iloc[i-2]:
            df.loc[df.index[i], 'bearish_fvg'] = True
    
    # ===== 3. Order Blocks =====
    # Order blocks are candles that precede strong moves
    df['bull_order_block'] = False
    df['bear_order_block'] = False
    
    # Look for strong bullish moves
    for i in range(3, len(df)-1):
        # If there's a strong bullish move (e.g., close is significantly higher than open)
        if (df['close'].iloc[i] - df['open'].iloc[i]) > 2 * df['ATR'].iloc[i]:
            # The candle before the move is a bearish order block
            df.loc[df.index[i-1], 'bear_order_block'] = True
    
    # Look for strong bearish moves
    for i in range(3, len(df)-1):
        # If there's a strong bearish move (e.g., open is significantly higher than close)
        if (df['open'].iloc[i] - df['close'].iloc[i]) > 2 * df['ATR'].iloc[i]:
            # The candle before the move is a bullish order block
            df.loc[df.index[i-1], 'bull_order_block'] = True
    
    # ===== 4. Liquidity Sweep Detection =====
    df['high_liquidity_sweep'] = False
    df['low_liquidity_sweep'] = False
    
    # Detect liquidity sweeps: when price moves beyond a significant high/low then reverses
    for i in range(window+1, len(df)-1):
        # High sweep: price moves above recent high then back below
        if df['high'].iloc[i] > df['high'].iloc[i-1:i-window].max() and df['close'].iloc[i] < df['high'].iloc[i-1:i-window].max():
            df.loc[df.index[i], 'high_liquidity_sweep'] = True
            
        # Low sweep: price moves below recent low then back above
        if df['low'].iloc[i] < df['low'].iloc[i-1:i-window].min() and df['close'].iloc[i] > df['low'].iloc[i-1:i-window].min():
            df.loc[df.index[i], 'low_liquidity_sweep'] = True
    
    # ===== 5. Optimal Trade Entry (OTE) =====
    # In ICT methodology, the OTE is often a 50-70% retracement to an order block
    df['bull_ote_zone'] = False
    df['bear_ote_zone'] = False
    
    # Calculate 50% and 70% retracement levels for potential OTE zones
    for i in range(window+1, len(df)-1):
        if df['bull_order_block'].iloc[i-window:i].any():
            # Find the most recent bullish order block
            for j in range(i-1, i-window-1, -1):
                if df['bull_order_block'].iloc[j]:
                    # Calculate range from order block to subsequent high
                    highest_high = df['high'].iloc[j+1:i+1].max()
                    ob_low = df['low'].iloc[j]
                    ob_high = df['high'].iloc[j]
                    
                    # Calculate 50-70% retracement zone
                    retracement_50 = highest_high - (highest_high - ob_low) * 0.5
                    retracement_70 = highest_high - (highest_high - ob_low) * 0.7
                    
                    # Check if current price is in the OTE zone
                    if df['low'].iloc[i] <= retracement_50 and df['high'].iloc[i] >= retracement_70:
                        df.loc[df.index[i], 'bull_ote_zone'] = True
                    break
                    
        if df['bear_order_block'].iloc[i-window:i].any():
            # Find the most recent bearish order block
            for j in range(i-1, i-window-1, -1):
                if df['bear_order_block'].iloc[j]:
                    # Calculate range from order block to subsequent low
                    lowest_low = df['low'].iloc[j+1:i+1].min()
                    ob_low = df['low'].iloc[j]
                    ob_high = df['high'].iloc[j]
                    
                    # Calculate 50-70% retracement zone
                    retracement_50 = lowest_low + (ob_high - lowest_low) * 0.5
                    retracement_70 = lowest_low + (ob_high - lowest_low) * 0.7
                    
                    # Check if current price is in the OTE zone
                    if df['high'].iloc[i] >= retracement_50 and df['low'].iloc[i] <= retracement_70:
                        df.loc[df.index[i], 'bear_ote_zone'] = True
                    break
    
    # ===== 6. Breaker Blocks =====
    # When price breaks through an order block and then returns
    df['bull_breaker'] = False
    df['bear_breaker'] = False
    
    for i in range(window+10, len(df)-1):
        # Look back for bull order blocks
        for j in range(i-10, i-window-10, -1):
            if df['bull_order_block'].iloc[j]:
                # Check if price broke below this order block and then returned
                if (df['low'].iloc[j+1:i].min() < df['low'].iloc[j]) and (df['close'].iloc[i] > df['high'].iloc[j]):
                    df.loc[df.index[i], 'bull_breaker'] = True
                break
                
        # Look back for bear order blocks
        for j in range(i-10, i-window-10, -1):
            if df['bear_order_block'].iloc[j]:
                # Check if price broke above this order block and then returned
                if (df['high'].iloc[j+1:i].max() > df['high'].iloc[j]) and (df['close'].iloc[i] < df['low'].iloc[j]):
                    df.loc[df.index[i], 'bear_breaker'] = True
                break
    
    # ===== 7. Smart Money Concepts (SMC) Entry Logic =====
    df['ict_buy_signal'] = False
    df['ict_sell_signal'] = False
    
    for i in range(window+10, len(df)):
        # ICT Buy Signals (combination of multiple concepts)
        if (
            (df['ms_bullish'].iloc[i]) and  # Bullish market structure
            (df['bull_ote_zone'].iloc[i]) and  # In an optimal trade entry zone
            (
                df['bull_order_block'].iloc[i-window:i].any() or  # Recent bull order block
                df['bullish_fvg'].iloc[i-window:i].any()  # Recent bullish fair value gap
            ) and
            (df['low_liquidity_sweep'].iloc[i-3:i].any())  # Recent liquidity sweep of lows
        ):
            df.loc[df.index[i], 'ict_buy_signal'] = True
        
        # ICT Sell Signals (combination of multiple concepts)
        if (
            (df['ms_bearish'].iloc[i]) and  # Bearish market structure
            (df['bear_ote_zone'].iloc[i]) and  # In an optimal trade entry zone
            (
                df['bear_order_block'].iloc[i-window:i].any() or  # Recent bear order block
                df['bearish_fvg'].iloc[i-window:i].any()  # Recent bearish fair value gap
            ) and
            (df['high_liquidity_sweep'].iloc[i-3:i].any())  # Recent liquidity sweep of highs
        ):
            df.loc[df.index[i], 'ict_sell_signal'] = True
    
    # ===== 8. London/New York Session Kill Zones =====
    # The ICT methodology emphasizes trading during specific time periods
    if 'time' in df.columns and isinstance(df['time'].iloc[0], pd.Timestamp):
        # Convert to UTC if needed
        df['hour_utc'] = df['time'].dt.hour
        
        # London session kill zone (8:00-10:00 AM London / 7:00-9:00 UTC during standard time)
        df['london_kill_zone'] = df['hour_utc'].between(7, 9)
        
        # New York session kill zone (8:30-10:30 AM New York / 13:30-15:30 UTC during standard time)
        df['ny_kill_zone'] = df['hour_utc'].between(13, 16)
        
        # Enhance signals during kill zones
        df['ict_buy_signal'] = df['ict_buy_signal'] & (df['london_kill_zone'] | df['ny_kill_zone'])
        df['ict_sell_signal'] = df['ict_sell_signal'] & (df['london_kill_zone'] | df['ny_kill_zone'])
    
    return df

def ict_check_trade_signals(df, symbol="EURUSD.view"):
    """
    Generate trading signals based on ICT (Inner Circle Trader) concepts
    This can be combined with your existing advanced_check_trade_signals function
    """
    if df.empty:
        return df
    
    # First, ensure we have standard indicators
    if 'RSI' not in df.columns:
        # Calculate basic indicators if not already present
        df['RSI'] = talib.RSI(df['close'], timeperiod=14)
        df['SMA_50'] = talib.SMA(df['close'], timeperiod=50)
        df['SMA_200'] = talib.SMA(df['close'], timeperiod=200)
        df['EMA_9'] = talib.EMA(df['close'], timeperiod=9)
        df['EMA_21'] = talib.EMA(df['close'], timeperiod=21)
        df['ATR'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
    
    # Add ICT concepts to the dataframe
    df = add_ict_concepts(df)
    
    # Generate buy signals combining traditional and ICT approaches
    df['Buy_Signal'] = (
        # ICT-specific buy signals
        (df['ict_buy_signal']) |
        
        # Bull order block with liquidity sweep
        ((df['bull_order_block']) & (df['low_liquidity_sweep'].shift(1)) & (df['close'] > df['EMA_21'])) |
        
        # Bullish Fair Value Gap when price returns to it
        ((df['bullish_fvg'].shift(3)) & (df['low'] < df['low'].shift(3)) & (df['close'] > df['open']) & (df['close'] > df['EMA_9'])) |
        
        # Bullish market structure with price in OTE zone
        ((df['ms_bullish']) & (df['bull_ote_zone']) & (df['RSI'] < 40))
    )
    
    # Generate sell signals combining traditional and ICT approaches
    df['Sell_Signal'] = (
        # ICT-specific sell signals
        (df['ict_sell_signal']) |
        
        # Bear order block with liquidity sweep
        ((df['bear_order_block']) & (df['high_liquidity_sweep'].shift(1)) & (df['close'] < df['EMA_21'])) |
        
        # Bearish Fair Value Gap when price returns to it
        ((df['bearish_fvg'].shift(3)) & (df['high'] > df['high'].shift(3)) & (df['close'] < df['open']) & (df['close'] < df['EMA_9'])) |
        
        # Bearish market structure with price in OTE zone
        ((df['ms_bearish']) & (df['bear_ote_zone']) & (df['RSI'] > 60))
    )
    
    # Add confirmation filter using price action
    for i in range(1, len(df)):
        # Filter out some buy signals
        if df['Buy_Signal'].iloc[i]:
            # Cancel buy signal if price is making lower lows
            if df['low'].iloc[i] < df['low'].iloc[i-1] < df['low'].iloc[max(0, i-2)]:
                df.loc[df.index[i], 'Buy_Signal'] = False
                
            # Cancel buy signal if there's a bearish engulfing pattern
            elif df['open'].iloc[i] > df['close'].iloc[i-1] and df['close'].iloc[i] < df['open'].iloc[i-1]:
                df.loc[df.index[i], 'Buy_Signal'] = False
        
        # Filter out some sell signals
        if df['Sell_Signal'].iloc[i]:
            # Cancel sell signal if price is making higher highs
            if df['high'].iloc[i] > df['high'].iloc[i-1] > df['high'].iloc[max(0, i-2)]:
                df.loc[df.index[i], 'Sell_Signal'] = False
                
            # Cancel sell signal if there's a bullish engulfing pattern
            elif df['open'].iloc[i] < df['close'].iloc[i-1] and df['close'].iloc[i] > df['open'].iloc[i-1]:
                df.loc[df.index[i], 'Sell_Signal'] = False
    
    return df

def check_market_structure(df):
    if 'ms_bullish' in df.columns and 'ms_bearish' in df.columns:
        ms_status = "Bullish" if df['ms_bullish'].iloc[-1] else "Bearish" if df['ms_bearish'].iloc[-1] else "Neutral"
    else:
        # If ICT columns don't exist yet
        # Use a simpler method to determine market structure
        if 'SMA_50' in df.columns and 'SMA_200' in df.columns:
            ms_status = "Bullish" if df['SMA_50'].iloc[-1] > df['SMA_200'].iloc[-1] else "Bearish"
        else:
            ms_status = "Neutral"
    
    return ms_status

def hybrid_check_trade_signals(df, symbol="EURUSD.view"):
    """
    Combines both traditional indicator-based and ICT-based approaches
    for a more comprehensive signal generation system.
    """
    if df.empty:
        return df
    
    # First apply ICT concepts
    df = ict_check_trade_signals(df, symbol)
    
    # Get traditional signals from your existing function
    # Assuming advanced_check_trade_signals exists in your codebase
    # df = advanced_check_trade_signals(df, symbol)
    
    # If you want to keep both types of signals separate for comparison
    # You can rename the columns from advanced_check_trade_signals
    # df.rename(columns={'Buy_Signal': 'Indicator_Buy_Signal', 
    #                    'Sell_Signal': 'Indicator_Sell_Signal'}, inplace=True)
    
    # Create combined signals - this would combine both approaches
    # df['Buy_Signal'] = df['ict_buy_signal'] | df['Indicator_Buy_Signal']
    # df['Sell_Signal'] = df['ict_sell_signal'] | df['Indicator_Sell_Signal']
    
    # Or use a more conservative approach requiring agreement
    # df['Buy_Signal'] = df['ict_buy_signal'] & df['Indicator_Buy_Signal']
    # df['Sell_Signal'] = df['ict_sell_signal'] & df['Indicator_Sell_Signal']
    
    return df

def calculate_ict_exit_levels(df, entry_price, direction):
    """
    Calculate exit levels (stop loss and take profit) based on ICT concepts
    
    Parameters:
    df (DataFrame): Price data with ICT concepts
    entry_price (float): Entry price of the trade
    direction (str): 'buy' or 'sell'
    
    Returns:
    tuple: (stop_loss, take_profit)
    """
    if df.empty or len(df) < 10:
        return None, None
    
    current_idx = len(df) - 1
    atr = df['ATR'].iloc[-1]
    
    if direction.lower() == 'buy':
        # For buy trades
        
        # Find nearest swing low or bull order block for stop loss
        stop_candidates = []
        
        # Check for recent swing lows
        for i in range(current_idx, max(0, current_idx - 20), -1):
            if df['swing_low'].iloc[i]:
                stop_candidates.append(df['low'].iloc[i] - 5 * df['point'].iloc[i])
        
        # Check for recent bull order blocks
        for i in range(current_idx, max(0, current_idx - 20), -1):
            if df['bull_order_block'].iloc[i]:
                stop_candidates.append(df['low'].iloc[i] - 5 * df['point'].iloc[i])
        
        # If no candidates found, use ATR-based stop
        if not stop_candidates:
            stop_loss = entry_price - 2 * atr
        else:
            # Use the highest stop (closest to entry) that's below entry price
            valid_stops = [s for s in stop_candidates if s < entry_price]
            stop_loss = max(valid_stops) if valid_stops else entry_price - 2 * atr
        
        # For take profit, look for recent fair value gaps, breaker blocks or swing highs
        tp_candidates = []
        
        # Check for bearish fair value gaps (price targets)
        for i in range(current_idx, max(0, current_idx - 30), -1):
            if df['bearish_fvg'].iloc[i]:
                # Use the high of the fair value gap
                tp_candidates.append(df['high'].iloc[i])
        
        # Check for recent swing highs
        for i in range(current_idx, max(0, current_idx - 30), -1):
            if df['swing_high'].iloc[i] and df['high'].iloc[i] > entry_price:
                tp_candidates.append(df['high'].iloc[i])
        
        # If no candidates found, use ATR-based take profit
        if not tp_candidates:
            take_profit = entry_price + 3 * atr
        else:
            # Use the lowest take profit that's above entry
            valid_tps = [tp for tp in tp_candidates if tp > entry_price]
            take_profit = min(valid_tps) if valid_tps else entry_price + 3 * atr
    
    else:  # For sell trades
        # Find nearest swing high or bear order block for stop loss
        stop_candidates = []
        
        # Check for recent swing highs
        for i in range(current_idx, max(0, current_idx - 20), -1):
            if df['swing_high'].iloc[i]:
                stop_candidates.append(df['high'].iloc[i] + 5 * df['point'].iloc[i])
        
        # Check for recent bear order blocks
        for i in range(current_idx, max(0, current_idx - 20), -1):
            if df['bear_order_block'].iloc[i]:
                stop_candidates.append(df['high'].iloc[i] + 5 * df['point'].iloc[i])
        
        # If no candidates found, use ATR-based stop
        if not stop_candidates:
            stop_loss = entry_price + 2 * atr
        else:
            # Use the lowest stop (closest to entry) that's above entry price
            valid_stops = [s for s in stop_candidates if s > entry_price]
            stop_loss = min(valid_stops) if valid_stops else entry_price + 2 * atr
        
        # For take profit, look for recent fair value gaps, breaker blocks or swing lows
        tp_candidates = []
        
        # Check for bullish fair value gaps (price targets)
        for i in range(current_idx, max(0, current_idx - 30), -1):
            if df['bullish_fvg'].iloc[i]:
                # Use the low of the fair value gap
                tp_candidates.append(df['low'].iloc[i])
        
        # Check for recent swing lows
        for i in range(current_idx, max(0, current_idx - 30), -1):
            if df['swing_low'].iloc[i] and df['low'].iloc[i] < entry_price:
                tp_candidates.append(df['low'].iloc[i])
        
        # If no candidates found, use ATR-based take profit
        if not tp_candidates:
            take_profit = entry_price - 3 * atr
        else:
            # Use the highest take profit that's below entry
            valid_tps = [tp for tp in tp_candidates if tp < entry_price]
            take_profit = max(valid_tps) if valid_tps else entry_price - 3 * atr
    
    return stop_loss, take_profit

# Example usage in the main trading system:
"""
# In your main signal processing function:
df = get_market_data(symbol, timeframe, 100)
if not df.empty:
    # Add basic indicators
    df = calculate_indicators(df, symbol)
    
    # Add ICT concepts and generate signals
    df = ict_check_trade_signals(df, symbol)
    
    # OR use hybrid approach
    # df = hybrid_check_trade_signals(df, symbol)
    
    # Check for signals
    if df['Buy_Signal'].iloc[-1]:
        # Calculate ICT-based exit levels
        stop_loss, take_profit = calculate_ict_exit_levels(df, df['close'].iloc[-1], 'buy')
        # Execute trade with these levels
        execute_trade("buy", symbol, risk_percent, stop_loss, take_profit)
    
    elif df['Sell_Signal'].iloc[-1]:
        # Calculate ICT-based exit levels
        stop_loss, take_profit = calculate_ict_exit_levels(df, df['close'].iloc[-1], 'sell')
        # Execute trade with these levels
        execute_trade("sell", symbol, risk_percent, stop_loss, take_profit)
"""