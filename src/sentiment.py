import os
import warnings

import matplotlib.pyplot as plt
import nltk
import pandas as pd


with warnings.catch_warnings():
    warnings.simplefilter('ignore', UserWarning)
    from nltk.sentiment import vader
    try:
        nltk.data.find('sentiment/vader_lexicon/')
    except:
        nltk.download('vader_lexicon')

# https://github.com/cjhutto/vaderSentiment#about-the-scoring
scorer = vader.SentimentIntensityAnalyzer().polarity_scores

here = os.path.abspath(os.path.dirname(__file__))
mergers = pd.read_pickle(os.path.join(here, 'mergers_clean.pickle'))\
    .drop('_desc', axis=1)
mergers = mergers[mergers['desc'].str.len() > 0]

sent = mergers['desc'].apply(lambda x: scorer(x)['compound'])

if __name__ == '__main__':
    for i in ('nsmallest', 'nlargest'):
        mergers['desc'].iloc[getattr(sent, i)(10).index].to_csv(
            os.path.join(here, i + '.csv'))

    sent.index = mergers['date'].copy()
    sent.sort_index(inplace=True)
    ma = sent.rolling('90d').median()
    ma = ma[~ma.index.duplicated(keep='last')].iloc[10:]

    fig, ax = plt.subplots(figsize=(10, 5))
    ma.plot(ax=ax, marker='o', linewidth=1.5, color='#0203e2', markersize=2)
    ax.set_xlabel('')
    # Normalized, Weighted Composite Score
    ax.set_ylabel('Sentiment Score')
    ax.set_title('Rolling 90-day Median Sentiment Score: M&A News')
    plt.savefig(os.path.join(here, 'sentiment.png'))
