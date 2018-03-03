from itertools import chain
import os
import pickle
import time

from bs4 import BeautifulSoup, SoupStrainer
from http_request_randomizer.requests.proxy.requestProxy import RequestProxy
import requests

USER_AGENT = 'Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/27.0.1453.93 Safari/537.36'  # noqa
URL = 'https://seekingalpha.com/market-news/m-a?page={pageno}'
STRAINER = SoupStrainer('ul', attrs={'class': 'mc-list',
                                     'id': 'mc-list'})
HEADERS = {'User-Agent': USER_AGENT}


def seeking_more_alpha(soup: BeautifulSoup):
    for tag in soup.find_all('li', attrs={'class': 'date-title'}):
        date = tag.text
        deal = tag.next_sibling
        while deal and deal.get('class') != ['date-title']:
            ticker = deal.find('div', attrs={'class': 'media-left'}).text
            link = deal.find('div', attrs={'class': 'title'}).find('a').get('href')
            bullets = deal.find('div', attrs={'class': 'bullets'})
            txt = ' '.join(bullet.text for bullet in bullets.find_all('li'))
            if not txt:
                # Fall back to old paragraph structure (2011/12)
                txt = ' '.join(bullet.text for bullet in bullets.find_all('p'))
            deal = deal.next_sibling
            yield date, link, ticker, txt


# def _read_one_pg(pageno):
#     url = URL.format(pageno=pageno)
#     response = requests.get(url, headers=HEADERS)
#     return response


def read_one_pg(pageno):
    url = URL.format(pageno=pageno)
    req_proxy = RequestProxy()
    response = req_proxy.generate_proxied_request(url)
    return response


def try_pages(limit=10, debug=False, sleep=0):
    pageno = 1
    while pageno <= limit:
        if sleep:
            time.sleep(sleep)
        response = read_one_pg(pageno)
        while not response:
            if sleep:
                time.sleep(sleep)
            response = read_one_pg(pageno)
        if not response.status_code == 200:
            if debug:
                # TODO: log this
                print(response.url)
            pageno += 1
            continue
        else:
            soup = BeautifulSoup(response.text, 'html.parser',
                                 parse_only=STRAINER)
            if not soup:
                # May still get a 200 response from blank pages
                break
            else:
                pageno += 1
                yield from seeking_more_alpha(soup=soup)


if __name__ == '__main__':
    mergers = tuple(try_pages(limit=300, debug=False))
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, 'mergers.pickle'), 'wb') as f:
        pickle.dump(mergers, f, pickle.HIGHEST_PROTOCOL)
