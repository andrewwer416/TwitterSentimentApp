import tweepy


def download(search):
    api_key = "/"
    api_secret = "/"
    bearer_token = "/"
    access_token = "/"
    access_token_secret = "/"
    client = tweepy.Client(bearer_token, api_key, api_secret, access_token, access_token_secret)

    query = search
    auth = tweepy.OAuth1UserHandler(api_key, api_secret, access_token, access_token_secret)
    api = tweepy.API(auth)

    response = client.search_recent_tweets(query=query, max_results=100)
    return response