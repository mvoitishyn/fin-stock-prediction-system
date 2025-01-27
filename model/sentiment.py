import requests
from bs4 import BeautifulSoup
from openai import OpenAI
import json
import os
from dotenv import find_dotenv, load_dotenv

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def fetch_news_titles(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    titles = {
        "h2": [tag.text.strip() for tag in soup.find_all("h2")],
        "h3": [tag.text.strip() for tag in soup.find_all("h3")],
    }
    return titles
 
url_yahoo = "https://finance.yahoo.com/topic/stock-market-news/"
urL_bloomberg = "https://www.bloomberg.com/"

news_yahoo = fetch_news_titles(url_yahoo)
news_bloomberg = fetch_news_titles(urL_bloomberg)

# "Cooling Inflation Data Send Stock, Bond ETFs Higher"
test_message = news_yahoo["h2"][1]

client = OpenAI(
    api_key=OPENAI_API_KEY
)
assistant_id = "asst_vHQBv2WDnfrBHNsrsUOygUxz"

thread = client.beta.threads.create()

message = client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content=test_message
)

run = client.beta.threads.runs.create_and_poll(
thread_id=thread.id,
assistant_id=assistant_id
)

if run.status == 'completed': 
    messages = client.beta.threads.messages.list(thread_id=thread.id)
else:
    print(run.status)

def messages_json(messages):
    messages  = messages.data[0].content[0].text.value
    messages_cleaned = messages.replace("```json","").replace("\n","").replace("```","").replace("'","")
    return json.loads(messages_cleaned)

output_scores = messages_json(messages)
print(output_scores)
