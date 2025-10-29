# Deployment Guide

## GitHub Deployment Steps

### 1. Prepare Repository
```bash
# Remove any sensitive files from tracking
git rm --cached .env
git rm --cached *.log

# Add .env to .gitignore if not already there
echo ".env" >> .gitignore
echo "logs/" >> .gitignore
echo "__pycache__/" >> .gitignore
echo "*.pyc" >> .gitignore
```

### 2. Commit Clean Version
```bash
git add .
git commit -m "Clean deployment version - removed subscription management"
git push origin main
```

### 3. Local Testing
```bash
# Install dependencies
pip install -r requirements.txt

# Create .env file from example
cp .env.example .env

# Edit .env with your credentials
# Add your Telegram bot token and MT5 passwords

# Test the application
streamlit run main.py
```

### 4. Streamlit Cloud Deployment

1. **Connect Repository:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub account
   - Select the hamsuky_forex-main repository

2. **Configure Environment Variables:**
   - In Streamlit Cloud dashboard, go to your app settings
   - Add the following secrets:
   ```
   TELEGRAM_BOT_TOKEN = "your_bot_token"
   TELEGRAM_CHAT_ID = "your_chat_id"
   MT5_DEMO_PASSWORD = "your_demo_password"
   MT5_LIVE_PASSWORD = "your_live_password"
   ```

3. **Deploy:**
   - Click "Deploy"
   - Wait for the build to complete
   - Your app will be available at: `https://your-app-name.streamlit.app`

### 5. Alternative Deployment Options

#### Heroku
```bash
# Install Heroku CLI
# Create Procfile
echo "web: streamlit run main.py --server.port=$PORT --server.address=0.0.0.0" > Procfile

# Create runtime.txt
echo "python-3.11.0" > runtime.txt

# Deploy
heroku create your-app-name
heroku config:set TELEGRAM_BOT_TOKEN="your_token"
heroku config:set TELEGRAM_CHAT_ID="your_chat_id"
git push heroku main
```

#### Docker
```dockerfile
# Create Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "main.py"]
```

### 6. Environment Variables Required

- `TELEGRAM_BOT_TOKEN`: Your Telegram bot token
- `TELEGRAM_CHAT_ID`: Your Telegram chat ID
- `MT5_DEMO_PASSWORD`: MetaTrader 5 demo account password
- `MT5_LIVE_PASSWORD`: MetaTrader 5 live account password

### 7. Important Notes

- **Never commit sensitive data** like passwords or API keys
- Use environment variables for all sensitive configuration
- Test thoroughly before deploying to production
- Monitor the application for any errors after deployment
- Keep backups of your trade history data

### 8. Troubleshooting

#### Common Issues:
- **TA-Lib installation fails**: Use the provided .whl file
- **MT5 connection fails**: Ensure MT5 terminal is running and logged in
- **Streamlit errors**: Check Python version compatibility (3.8+)
- **Missing dependencies**: Run `pip install -r requirements.txt`

#### Performance Tips:
- Use smaller timeframes for faster loading
- Limit the number of symbols monitored simultaneously
- Regular cleanup of log files
- Monitor memory usage on cloud platforms