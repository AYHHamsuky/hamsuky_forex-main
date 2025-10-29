# Hamsuky Forex Trading Bot

A powerful automated forex trading bot using MetaTrader 5 with advanced ICT (Inner Circle Trader) methodology and Smart Money Concepts (SMC).

## Features

- **Advanced Signal Generation**: Combines traditional technical indicators with ICT methodology
- **Smart Money Concepts**: Order blocks, fair value gaps, market structure analysis
- **Multi-Asset Support**: Forex, Gold, Bitcoin, and other markets
- **Real-time Monitoring**: Live market analysis and signal detection  
- **Telegram Integration**: Automated notifications and alerts
- **Risk Management**: Dynamic position sizing and stop-loss placement
- **JustMarkets Integration**: Optimized for JustMarkets broker

## Installation

1. **Clone the repository:**
```bash
git clone https://github.com/AYHHamsuky/hamsuky_forex-main.git
cd hamsuky_forex-main
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Install TA-Lib (if pip install fails):**
```bash
pip install ta_lib-0.6.3-cp313-cp313-win_amd64.whl
```

4. **Configure environment variables:**
   - Copy `.env.example` to `.env`
   - Add your Telegram bot token and chat ID
   - Add your MT5 account passwords

5. **Set up MetaTrader 5:**
   - Install MT5 terminal
   - Login to your JustMarkets account
   - Enable AutoTrading

## Configuration

### Environment Variables (.env file)
```env
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_chat_id
MT5_DEMO_PASSWORD=your_demo_password
MT5_LIVE_PASSWORD=your_live_password
```

### Account Settings (meta.py)
- Demo Account: 2001479025 (JustMarkets-Demo)
- Live Account: 2050196801 (JustMarkets-Live)

## Usage

### Run the Application
```bash
streamlit run main.py
```

### Features Available:
1. **Live Trading Dashboard** - Real-time signal monitoring
2. **Market Analysis** - Technical and ICT analysis
3. **Trade History** - Performance tracking
4. **Strategy Optimizer** - Parameter optimization
5. **Volatility Analysis** - Market condition assessment

## Trading Strategy

The bot uses a hybrid approach combining:

### Traditional Technical Analysis:
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- Moving Averages (EMA/SMA)
- ATR (Average True Range)

### ICT Methodology:
- **Order Blocks**: Institutional supply/demand zones
- **Fair Value Gaps**: Price imbalances
- **Market Structure**: Higher highs/lows analysis
- **Breaker Blocks**: Broken support/resistance retests
- **Premium/Discount Zones**: Value area analysis
- **Liquidity Sweeps**: Stop hunt detection

## Risk Management

- Dynamic position sizing based on account balance
- ATR-based stop losses and take profits
- Market-specific risk parameters
- Spread filtering
- Session-based trading hours

## Supported Markets

### Optimized For:
- **Forex**: EURUSD, GBPUSD, USDJPY, AUDCAD, AUDUSD, USDCAD, NZDUSD
- **Crypto**: BTCUSD
- **Commodities**: XAUUSD (Gold), WTI (Oil)

## API Integration

### Telegram Bot Setup:
1. Create a bot with @BotFather
2. Get your bot token
3. Get your chat ID
4. Add to .env file

### MetaTrader 5:
- Requires active MT5 terminal
- JustMarkets broker account
- AutoTrading enabled

## Deployment

### Local Deployment:
```bash
streamlit run main.py
```

### Cloud Deployment (Streamlit Cloud):
1. Push to GitHub
2. Connect Streamlit Cloud to repository
3. Add environment variables in Streamlit settings
4. Deploy

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

**Important**: This trading bot is for educational and research purposes. Trading forex and other financial instruments involves substantial risk and may not be suitable for all investors. Past performance is not indicative of future results. Always trade responsibly and never risk more than you can afford to lose.

## Support

For support and questions:
- GitHub Issues: [Create an issue](https://github.com/AYHHamsuky/hamsuky_forex-main/issues)
- Email: admin@hamsuky.com

## Version

Current Version: 2.0.0 (Standalone Release)
- Removed subscription management
- Optimized for standalone deployment
- Enhanced ICT methodology
- Improved performance monitoring