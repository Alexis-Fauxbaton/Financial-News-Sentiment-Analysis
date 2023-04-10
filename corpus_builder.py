import pandas as pd
from goose3 import Goose
import urllib
import feedparser
from enum import Enum
import numpy as np
import time

from matplotlib import pyplot as plt
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer

FINANCE_NEWS_DATA_FILENAME = 'news_url.csv'


class RssFeeds(Enum):
    CNBC1 = 'https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000664'
    CNBC2 = 'https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=20910258'
    CNBC3 = 'https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10001147'
    CNBC4 = 'https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=15839135'
    CNN1 = 'http://rss.cnn.com/rss/money_news_economy.rss'
    CNN2 = 'http://rss.cnn.com/rss/money_markets.rss'
    OECD1 = 'http://www.oecd.org/economy/index.xml'
    OECD2 = 'http://www.oecd.org/finance/index.xml'
    INVESTING = 'https://rss.app/feeds/EUoKEtkXzjbfnBJd.xml'


def get_top_n_bigram(corpus, n=None, ngrams=2):
    vec = CountVectorizer(ngram_range=(ngrams, ngrams), stop_words='english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:n]


def quoted_query(query):
    return urllib.parse.quote(query)


class CorpusBuilder:

    def __init__(self):
        self.g = Goose({'browser_user_agent': 'Mozilla', 'parser_class': 'soup'})

        try:
            self.df = pd.read_csv(FINANCE_NEWS_DATA_FILENAME, sep='@', index_col=0)
        except Exception as _:
            self.df = None

        self.corpus = None

    def add_urls(self, query_words, limit=100, add_queries=False):

        if self.df is None:
            try:
                news_df = pd.read_csv(FINANCE_NEWS_DATA_FILENAME, sep='@', index_col=0)

            except Exception as e:
                print(f'Error loading {FINANCE_NEWS_DATA_FILENAME}, creating new dataframe')
                news_df = pd.DataFrame({'Title': [], 'URL': [], 'Content': []})
        else:
            news_df = self.df

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
            # is_url_in_df = news_df['URL'].eq(url)
            # matching_indices = news_df.index[is_url_in_df].tolist()

            # if len(matching_indices) > 0 and news_df[matching_indices].loc[news_df[matching_indices]['Content'] is not np.nan & news_df[matching_indices]['Content'] != '']:

            response = feedparser.parse(url)

            entries = response['entries']

            for entry in entries:
                titles.append(entry['title'])
                urls.append(entry['link'])

        fetched_urls = pd.DataFrame({'Title': titles, 'URL': urls, 'Content': ['' for i in range(len(urls))]})

        news_df = pd.concat([news_df, fetched_urls], ignore_index=True)

        news_df.drop_duplicates(subset=['URL'], inplace=True)

        news_df.reset_index(inplace=True, drop=True)

        news_df.to_csv(FINANCE_NEWS_DATA_FILENAME, sep='@')

        self.df = news_df

    def extract_articles(self):
        if self.df is None:
            print("Corpus builder df is None, exiting...")
            return

        wc = 0

        article_text = []

        nrows = self.df.shape[0]

        for idx, row in self.df.iterrows():
            title, url, content = row

            time_ = time.time()

            if time_ % 120 == 0:
                print("Saving current corpus as checkpoint")
                pad = [np.nan for i in range(self.df.shape[0] - len(article_text))]
                article_text_tmp = article_text + pad

                self.df['Content'] = article_text_tmp

                self.df.to_csv(FINANCE_NEWS_DATA_FILENAME, sep='@')

            print(url, f'{idx} / {nrows} rows treated, {(idx / nrows) * 100:.2f} % complete')

            if content is not np.nan and content != '':
                article_text.append(content)
                continue

            '''
            elif 'oecd.org' in url:
                pad = [np.nan for i in range(self.df.shape[0] - len(article_text))]
                article_text = article_text + pad
                break
            '''

            try:
                article = self.g.extract(url=url)

            except Exception as e:
                article_text.append('Failed to parse content')
                continue

            article_text.append(article.cleaned_text)

            wc += len(article.cleaned_text.split(' '))

            # print(self.df['URL'][50])
            # print(article.title, article.cleaned_text)

            if 'oecd.org' in url:
                time.sleep(9)

        self.df['Content'] = article_text

        self.df.to_csv(FINANCE_NEWS_DATA_FILENAME, sep='@')

        print('Corpus wc: ', wc)

    def process_corpus(self):
        self.corpus = self.df.copy()

        content = self.corpus['Content']

        content = content.str.replace('\n', '', regex=False)

        content = content.str.lower()

        self.corpus['Content'] = content

        self.corpus = self.corpus.loc[self.corpus['Content'] != 'failed to parse content'].reset_index(drop=True)

        self.corpus.dropna(subset=['Content'], axis=0, inplace=True)

        content = self.corpus['Content']

        print(word_tokenize(content[0]))

        self.corpus.to_csv("corpus.csv", sep='@')

        common_words = get_top_n_bigram(content.dropna(axis=0), 40, 3)

        print(common_words)

        # print(content)

        df1 = pd.DataFrame(common_words, columns=['Text', 'count'])
        df1.groupby('Text').sum()['count'].sort_values(ascending=False).hist()
        plt.title('Top 20 bigrams in News text')
        plt.show()


def corpus_wc(df):
    wc = 0
    for idx, row in df.iterrows():
        title, url, content = row

        if content is np.nan:
            break

        wc += len(content.split(' '))

    print(f"Corpus word count: {wc}")


def main():
    corpus_builder = CorpusBuilder()

    # corpus_builder.add_urls(['Crypto'])

    corpus_builder.extract_articles()

    # corpus_wc(corpus_builder.df)

    corpus_builder.process_corpus()

    corpus_wc(corpus_builder.df)


if __name__ == "__main__":
    main()

    # g = Goose({'browser_user_agent': 'Mozilla', 'parser_class': 'soup'})
