import requests as req
import json
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer 
import praw

reddit = praw.Reddit(client_id='PRvvdrGHhU7njUF6e3CA6A', client_secret='KrWNg9xv1uET9LfSiQ5welsKktWRLQ', user_agent='Test')
hot_depression_posts = reddit.subreddit('depression').hot(limit=100)
depression_post_list =[]
for post in hot_depression_posts:
        submission = reddit.submission(id=post.id)
        submission.comments.replace_more(limit=0)
        for top_level_comment in submission.comments.list():
                print(top_level_comment.body)
                depression_post_list.append(top_level_comment.body)

happy_training_data = []
sid = SentimentIntensityAnalyzer()
for post in depression_post_list:
        str = post
        ss = sid.polarity_scores(str)
        entry = {
                'text':str,
                'tone_score':ss
        }
        if entry['tone_score']['compound']<-.3:
                happy_training_data.append(entry)

pretty = json.dumps(happy_training_data,indent=2)
with open("depression_training_data_comments.json", "w") as outfile:
    outfile.write(pretty)