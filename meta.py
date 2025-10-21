import MetaTrader5 as mt5

ACCOUNT = 2050196801  # Your account number
PASSWORD = "f@tim@love5@Abu"  # Your MT5 password (keep it secret!)
SERVER = "JustMarkets-Live"  # Broker's server

# Initialize MT5 with login credentials
if not mt5.initialize(login=ACCOUNT, password=PASSWORD, server=SERVER):
    print("❌ MT5 initialization failed!", mt5.last_error())
    mt5.shutdown()
else:
    print("✅ Successfully connected to MT5!")




