import MetaTrader5 as mt5
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# JustMarkets Account Configuration
DEMO_ACCOUNT = 2001479025
LIVE_ACCOUNT = 2050196801
DEMO_SERVER = "JustMarkets-Demo"
LIVE_SERVER = "JustMarkets-Live"

# Get passwords from environment variables for security
DEMO_PASSWORD = os.getenv("MT5_DEMO_PASSWORD")
LIVE_PASSWORD = os.getenv("MT5_LIVE_PASSWORD")

def initialize_mt5_connection(account_type="demo"):
    """
    Initialize MT5 connection with proper error handling
    account_type: "demo" or "live"
    """
    if account_type.lower() == "demo":
        account = DEMO_ACCOUNT
        password = DEMO_PASSWORD
        server = DEMO_SERVER
    else:
        account = LIVE_ACCOUNT
        password = LIVE_PASSWORD
        server = LIVE_SERVER
    
    if not password:
        print(f"❌ Password not found for {account_type} account. Check your .env file.")
        return False
    
    # Initialize MT5 with login credentials
    if not mt5.initialize(login=account, password=password, server=server):
        print(f"❌ MT5 initialization failed for {account_type} account!", mt5.last_error())
        return False
    else:
        print(f"✅ Successfully connected to JustMarkets {account_type} account!")
        return True

# For backward compatibility - initialize demo by default
if __name__ == "__main__":
    initialize_mt5_connection("demo")




