# flake8: ignore=E501

import os
import pickle
import re

import pandas as pd


here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'mergers.pickle'), 'rb') as f:
    mergers = pd.DataFrame(list(pickle.load(f)),
                           columns=['date', 'url', 'ticker', 'desc'])

mergers['date'] = pd.to_datetime(mergers['date'].str.replace('Today - ', ''),
                                 format='%A, %B %d, %Y')

moneysign = (u'\u0024\u00A2\u00A3\u00A4\u00A5\u058F\u060B\u09F2\u09F3'
             u'\u09FB\u0AF1\u0BF9\u0E3F\u17DB\u20A0\u20A1\u20A2\u20A3'
             u'\u20A4\u20A5\u20A6\u20A7\u20A8\u20A9\u20AA\u20AB\u20AC\u20AD'
             u'\u20AE\u20AF\u20B0\u20B1\u20B2\u20B3\u20B4\u20B5\u20B6\u20B7'
             u'\u20B8\u20B9\u20BA\u20BB\u20BC\u20BD\u20BE\u20BF\uA838\uFDFC'
             u'\uFE69\uFF04\uFFE0\uFFE1\uFFE5\uFFE6')

money = r'[+-]?(?:[A-Za-z]?[' + moneysign + ']' + r'|(?:AUD )|(?:EUR ))' + r'\d*\.?\d+[BMbm]*\b'
pct = r'(?:\d*\.)*\d+%'
parens = r'\([^)]+\)'

# 2019, 2018, 2017, ... -> 'yyyy'
year = r'\b201\d\b'

# Residual of what wasn't captured above
integer = r'\b(?<![,.])\d+(?![,.])\b|\b(?<![,.])[+-]*\b[0-9]{1,3}(,[0-9]{3})*(?![,.])\b'
decimal = r'[+-]*\b[0-9]{1,3}(,[0-9]{3})*\.[0-9]+\b|[+-]*\b[0-9]{1,3}(,[0-9]{3})*\.[0-9]*(?!\d)|[+-]*(?<!\d)\.\d+\b'
number = re.compile(r'|'.join((integer, decimal)))

etc = ' expected_to_close '
acq = ' announce_acquisition '

# Exempt ngrams containing custom stopwords.
# Avoiding complex regex here for sanity...
keyphrases = {
    r'(?:expected|scheduled|on course|on track) to (?:close|happen|occur)': etc,
    r'expects? the deal to close': etc,
    r'(?:transaction|deal) should close' : etc,
    r'announc(?:ed|ing) the (?:acquisition|purchase) of': acq,
    r'agree(?:d|ment|s) to (?:the sale|sell|buy|acquire|be acquired|purchase)': acq,
    r'(?:has )?acquired [A-Za-z]+': acq,
    r'plans to (?:buy|acquire|be acquired|purchase)': acq,
    r'is acquiring': acq,
    r'deal to acquire': acq,
    r'(?:bought|to buy) the assets': acq,
    r'(?<!efforts )to (?:acquire|purchase)': acq,
    r'letter of intent': acq,
    r'LOI': acq,
    r'finalzed the sale': acq,
    }


# In some cases, we want to avoid confuscating abbrev. with words.
# In others, the reverse is true.
mergers['_desc'] = mergers['desc'].copy()
mergers['desc'] = mergers['desc']\
    .str.replace(parens, '')\
    .str.replace(money, '-mmmm-')\
    .str.replace(pct, '-pppp-')\
    .str.replace(r'\bUS\b', 'united_states')\
    .str.replace(r'\bH\.?S\.?R(?:\b|\.)', 'hart scott rodino')\
    .str.replace(r'\b(?:Q[1234]|H[12])\b', '-qqqq-')\
    .str.lower()\
    .str.replace(r'\s+', ' ')\
    .str.replace(year, '-yyyy-')\
    .str.replace(number, '-nnnn-')\
    .str.replace(r'\bu.s.', 'united_states')\
    .str.replace(r'\bunited states\b', 'united_states')\
    .str.replace(r'\bg&a\b', 'general and administrative')\
    .str.replace(r'\b(?:dept\.?|department) of justic', 'doj')\
    .str.replace(r'\beuropean union', 'eu')\
    .str.replace(r'\bearnings per share\b', 'eps')\
    .str.replace(r'\bwall st(?:reet|\.) journal\b', 'wsj')\
    .str.replace(r'~', '')\
    .str.replace(r'(?:source:\s+)?press release', '')

for k, v in keyphrases.items():
    mergers['desc'] = mergers['desc'].str.replace(k, v)

if __name__ == '__main__':
    mergers.to_pickle(os.path.join(here, 'mergers_clean.pickle'))
