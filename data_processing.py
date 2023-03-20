import pandas as pd


def parse_training_data(path, text_sep):
    news = []
    sentiments = []
    with open(path, 'r') as f:
        text = f.read()
        lines = text.split('\n')
        for line in lines:
            data_split = line.split(sep=text_sep)
            news.append(data_split[0])
            sentiments.append(data_split[1])

    df = pd.DataFrame({'News': news, 'Sentiment': sentiments})

    return df


def read_data(path, train_data=True, text_sep='@'):

    df = parse_training_data(path, text_sep)

    return df


def main():
    data = read_data('./FinancialPhraseBank/Sentences_50Agree.txt')


if __name__ == "__main__":
    main()
