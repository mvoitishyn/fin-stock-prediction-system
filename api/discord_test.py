import discord
from binance.client import Client
import asyncio

# Optional for private endpoints
BINANCE_API_KEY = ''
BINANCE_API_SECRET = ''

DISCORD_TOKEN = 'MTM5MjE5NTg5OTU1OTc3NjM1Nw.Gxmewy.VIi1mI-Y-U4-6_jFD_BzgEdqws2UxPlsq0iHO8'
CHANNEL_ID = 1392200322167996601  # Replace with your actual channel ID (right-click channel > Copy ID)

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

binance = Client(api_key=BINANCE_API_KEY, api_secret=BINANCE_API_SECRET)

@client.event
async def on_ready():
    print(f'Logged in as {client.user}')
    channel = client.get_channel(CHANNEL_ID)
    
    if channel is None:
        print(f"❌ Channel with ID {CHANNEL_ID} not found!")
        print("Make sure:")
        print("1. The bot has access to the channel")
        print("2. The channel ID is correct")
        print("3. Developer mode is enabled in Discord")
        return
    
    try:
        # Example: BTC price fetch
        price = binance.get_symbol_ticker(symbol="BTCUSDT")
        await channel.send(f"🟢 BTC Price: {price['price']} USD")
        
        # You can schedule updates here
        await periodic_price_check(channel)
    except Exception as e:
        print(f"Error: {e}")

async def periodic_price_check(channel):
    while True:
        try:
            price = binance.get_symbol_ticker(symbol="BTCUSDT")
            await channel.send(f"🔁 BTC Price Update: {price['price']} USD")
            await asyncio.sleep(300)  # every 5 minutes
        except Exception as e:
            print(f"Error in periodic check: {e}")
            await asyncio.sleep(60)  # Wait 1 minute before retrying

client.run(DISCORD_TOKEN)