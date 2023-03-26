import pandas as pd
from goose3 import Goose
import urllib
import feedparser
from enum import Enum
import numpy as np
import time

FINANCE_NEWS_DATA_FILENAME = 'news_url.csv'


class RssFeeds(Enum):
    CNBC1 = 'https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000664'
    CNBC2 = 'https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=20910258'
    CNBC3 = 'https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10001147'
    CNBC4 = 'https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=15839135'
    CNN1 = 'http://rss.cnn.com/rss/money_news_economy.rss'
    CNN2 = 'http://rss.cnn.com/rss/money_markets.rss'
    # OECD1 = 'http://www.oecd.org/economy/index.xml'
    # OECD2 = 'http://www.oecd.org/finance/index.xml'


def quoted_query(query):
    return urllib.parse.quote(query)


class CorpusBuilder:

    def __init__(self):
        self.g = Goose({'browser_user_agent': 'Mozilla', 'parser_class': 'soup'})
        self.df = None

    def add_urls(self, query_words, limit=100, add_queries=False):
        try:
            news_df = pd.read_csv(FINANCE_NEWS_DATA_FILENAME, sep='@', index_col=0)

        except Exception as e:
            print(f'Error loading {FINANCE_NEWS_DATA_FILENAME}, creating new dataframe')
            news_df = pd.DataFrame({'Title': [], 'URL': []})

        titles = []
        urls = []

        feed_urls = [rss_feed.value for rss_feed in RssFeeds]

        if add_queries:
            query_urls = ['https://news.google.com/news?hl=en&q=' + quoted_query(word) +
                          '&num=' + str(limit) + '&output=rss' for word in query_words]
        else:
            query_urls = []

        url_values = feed_urls + query_urls

        for url in url_values:
            response = feedparser.parse(url)

            entries = response['entries']

            for entry in entries:
                titles.append(entry['title'])
                urls.append(entry['link'])

        fetched_urls = pd.DataFrame({'Title': titles, 'URL': urls})

        news_df = pd.concat([news_df, fetched_urls], ignore_index=True)

        news_df.drop_duplicates(inplace=True)

        news_df.reset_index(inplace=True, drop=True)

        news_df.to_csv(FINANCE_NEWS_DATA_FILENAME, sep='@')

        self.df = news_df

    def extract_articles(self):
        if self.df is None:
            return

        wc = 0

        article_text = []

        for idx, row in self.df.iterrows():
            title, url, content = row

            print(url)

            if content is not np.nan or 'oecd' in url:
                article_text.append(content)
                continue

            try:
                article = self.g.extract(url=url)

            except Exception as e:
                continue

            article_text.append(article.cleaned_text)

            wc += len(article.cleaned_text.split(' '))

            # print(self.df['URL'][50])
            # print(article.title, article.cleaned_text)

            if 'oecd' in url:
                time.sleep(3)

        self.df['Content'] = article_text

        self.df.to_csv(FINANCE_NEWS_DATA_FILENAME, sep='@')

        print('Corpus wc: ', wc)


def main():
    corpus_builder = CorpusBuilder()

    corpus_builder.add_urls(['Crypto'])

    corpus_builder.extract_articles()


if __name__ == "__main__":
    main()

    # g = Goose({'browser_user_agent': 'Mozilla', 'parser_class': 'soup'})
