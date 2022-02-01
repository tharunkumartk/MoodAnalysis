import requests as req
import json
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer 

auth = req.auth.HTTPBasicAuth('PRvvdrGHhU7njUF6e3CA6A', 'KrWNg9xv1uET9LfSiQ5welsKktWRLQ')
data = {'grant_type': 'password',
        'username': 'Over_car_1225',
        'password': 'test_4231'}

# setup our header info, which gives reddit a brief description of our app
headers = {'User-Agent': 'MyBot/0.0.1'}

# send our request for an OAuth token
res = req.post('https://www.reddit.com/api/v1/access_token',
                    auth=auth, data=data, headers=headers)
# convert response to JSON and pull access_token value
TOKEN = res.json()['access_token']

# add authorization to our headers dictionary
headers = {**headers, **{'Authorization': f"bearer {TOKEN}"}}

# while the token is valid (~2 hours) we just add headers=headers to our requests
req.get('https://oauth.reddit.com/api/v1/me', headers=headers)

res = req.get("https://oauth.reddit.com/r/depression/hot",headers=headers,params={'limit':'100'})
depression_post_dicts = res.json()['data']['children']
depression_training_data = []
sid = SentimentIntensityAnalyzer()
for post in depression_post_dicts:
        str = post['data']['selftext']
        ss = sid.polarity_scores(str)
        entry = {
                'text':post['data']['selftext'],
                'tone_score':ss
        }
        if entry['tone_score']['compound']<-.1:
                depression_training_data.append(entry)

pretty = json.dumps(depression_training_data,indent=2)
with open("sample.json", "w") as outfile:
    outfile.write(pretty)