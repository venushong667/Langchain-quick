from dotenv import load_dotenv
from googleapiclient.discovery import build
import pprint
import os

load_dotenv()

def google_search(search_term, api_key, cse_id, **kwargs):
    service = build("customsearch", "v1", developerKey=api_key)
    res = service.cse().list(q=search_term, cx=cse_id, **kwargs).execute()
    return res['items']

results = google_search(
    'what is google gemini', os.getenv("GOOGLE_API_KEY"), os.getenv("GOOGLE_CSE_ID"), num=10)
for result in results:
    pprint.pprint(result)