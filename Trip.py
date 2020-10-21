import argparse
import datetime
import locale
import logging
import re
import sys
import time
from bs4  import BeautifulSoup
import requests
import pandas as pd
from selenium import webdriver
import urllib
from  geopy.geocoders import Nominatim
temp={}
page_d = []
def page_data(urlpage):
    page_d.clear()
    temp.clear()
    page = urllib.request.urlopen(urlpage)
    # parse the html using beautiful soup and store in variable 'soup'
    soup = BeautifulSoup(page, 'html.parser')
    # find results within table
    table = soup.find('div', attrs={'class': 'ui_columns is-multiline is-mobile'})
    restaurant_name=table.find('h1',attrs={'class':'ui_header h1'}).text
    review_count = table.find('span', attrs={'class': 'reviewCount cx_brand_refresh_phase2'}).text
    speciality = table.find('div', attrs={'class': 'header_links'}).text
    phone = table.find('a', attrs={'class': 'ui_link_container level_4'}).text
    popularity = table.find('span', attrs={'class': 'header_popularity popIndexValidation'}).text
    head = soup.find_all('div', attrs={'class': 'restaurants-details-card-TagCategories__categoryTitle--28rB6'})
    content= soup.find_all('div', attrs={'class': 'restaurants-details-card-TagCategories__tagText--Yt3iG'})
    address = soup.find('span', attrs={'class': 'restaurants-detail-overview-cards-LocationOverviewCard__detailLinkText--co3ei'}).text
    try:
        geolocator = Nominatim()
        loc = geolocator.geocode(address)
    except:
        loc=(None,None)

    for i in range(len(head)):
        temp[head[i].text]=content[i].text
    page_d.append(restaurant_name)
    page_d.append(review_count)
    page_d.append(speciality)
    page_d.append(phone)
    page_d.append(popularity)
    page_d.append(address)
    return page_d



class Review():
    def __init__(self, id, date, title, user, text):
        self.id = id
        self.date = date
        self.title = title
        self.user = user
        self.text = text
class TripadvisorScraper():
    def __init__(self, engine='chrome',urlpage=''):
        self.info=page_data(urlpage)

        self.language = 'en'
        self.locale_backup = locale.getlocale()[0]
        self.lookup = {}

        if engine == 'chrome':
            self.driver = webdriver.Chrome('chromedriver.exe')
        elif engine == 'firefox':
            self.driver = webdriver.Firefox()
        elif engine == 'phantomjs':
            self.driver = webdriver.PhantomJS()
        else:
            logging.warning('Engine {} not supported. Defaulting to PhantomJS.'.format(engine))
            self.driver = webdriver.Chrome()

        self.i18n = {
            'en': {
                'more_btn': 'More',
                'date_format': '%B %d, %Y'
            },
            'de': {
                'more_btn': 'Mehr',
                'date_format': '%d. %B %Y'
            }
        }

    def _parse_page(self):
        reviews = []

        try:
            self.driver.find_element_by_xpath('//span[contains(., "{}") and @class="taLnk ulBlueLinks"]'.format(self.i18n[self.language]['more_btn'])).click()
        except:
            pass

        time.sleep(2)

        review_elements = self.driver.find_elements_by_class_name('reviewSelector')
        for e in review_elements:
            try:
                id = e.get_attribute('id')
                date = e.find_element_by_class_name('ratingDate').get_attribute('title')
                date = datetime.datetime.strptime(date, self.i18n[self.language]['date_format'])
                title = e.find_element_by_class_name('quote').find_element_by_tag_name('a').find_element_by_class_name('noQuotes').text
                try:
                    user = e.find_element_by_class_name('memberOverlayLink').get_attribute('id')
                    user = user[4:user.index('-')]
                except:
                    user = None
                text = e.find_element_by_class_name('partial_entry').text.replace('\n', '')

                if id in self.lookup:
                    logging.warning('Fetched review {} twice.'.format(review_elements.id))
                else:
                    self.lookup[id] = True
                    reviews.append(Review(id, date, title, user, text))
            except:
                logging.warning('Couldn\'t fetch review.')
                pass

        return reviews

    def _set_language(self, url=''):
        if 'tripadvisor.de' in url:
            self.language = 'de'
            locale.setlocale(locale.LC_TIME, 'de_DE')
        elif 'tripadvisor.com' in url:
            self.language = 'en'
            locale.setlocale(locale.LC_TIME, 'en_US')
        else:
            logging.warn('Tripadvisor domain location not supported. Defaulting to English (.com)')





    def fetch_reviews(self, url, max_reviews=5, as_dataframe=True):
        self.lookup = {}
        reviews = []
        if not max_reviews: max_reviews = sys.maxsize
        self._set_language(url)

        if not is_valid_url(url): return logging.warning('Tripadvisor URL not valid.')
        self.driver.get(url)

        time.sleep(2)  # TODO

        while len(reviews) < max_reviews:
            reviews += self._parse_page()
            logging.info('Fetched a total of {} reviews by now.'.format(len(reviews)))
            try:
                next_button_container = self.driver.find_element_by_class_name('next')
                next_button_container.click()
            except:
                break

        locale.setlocale(locale.LC_TIME, self.locale_backup)
        if as_dataframe:
            df=pd.DataFrame.from_records([r.__dict__ for r in reviews]).set_index('id', drop=True)
            df['Restaurant']=self.info[0]
            df['Count'] = self.info[1]
            df['Speciality'] = self.info[2]
            df['Phone'] = self.info[3]
            df['Popularity'] = self.info[4]
            df['Address'] = self.info[5]
            for i in temp:
                df[i]=temp[i]
            return df
        return reviews

    def close(self):
        self.driver.quit()

def is_valid_url(url):
    URL_PATTERN = 'http(s)?:\/\/.?(www\.)?tripadvisor\.(com|de)\/Restaurant_Review.*'
    return re.compile(URL_PATTERN).match(url)

def get_language_by_url(url):
    if 'tripadvisor.de' in url: return 'de'
    elif 'tripadvisor.com' in url: return 'en'
    return None

def get_id_by_url(url):
    if not is_valid_url(url): return None
    match = re.compile('.*Restaurant_Review-g\d+-(d\d+).*').match(url)
    if match is None: return None
    return match.group(1)

pd.set_option('display.max_columns',1000)
pd.set_option('display.width',1000)
urls=[]














class TripadvisorScraper_restaurants():
    def __init__(self, engine='chrome'):
        self.language = 'en'
        self.locale_backup = locale.getlocale()[0]
        self.lookup = {}
        if engine == 'chrome':
            self.driver = webdriver.Chrome('chromedriver.exe')
        elif engine == 'firefox':
            self.driver = webdriver.Firefox()
        elif engine == 'phantomjs':
            self.driver = webdriver.PhantomJS()
        else:
            logging.warning('Engine {} not supported. Defaulting to PhantomJS.'.format(engine))
            self.driver = webdriver.Chrome()
        self.i18n = {
            'en': {'more_btn': 'More','date_format': '%B %d, %Y'},
            'de': {'more_btn': 'Mehr', 'date_format': '%d. %B %Y'}}
        web = 'https://www.tripadvisor.com/Restaurants-g295414-Karachi_Sindh_Province.html'
        self.driver.get(web)
    def _parse_page(self):
        time.sleep(2)
        j=0
        while j<2:
            review_elements = self.driver.find_elements_by_xpath("//div[@class='wQjYiB7z']/span/a")
            # print(review_elements)
            for i in review_elements:
                print(i)
                urls.append(i.get_attribute("href"))
            time.sleep(5)
            next_button_container = self.driver.find_element_by_class_name('next')
            next_button_container.click()
            j+=1
        return urls











if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Scrape restaurant reviews from Tripadvisor (.com or .de).')
    # parser.add_argument('url', help='URL to a Tripadvisor restaurant page',default=urlpage)
    # parser.add_argument('-o', '--out', dest='outfile', help='Path for output CSV file', default='reviews.csv')
    # parser.add_argument('-n', dest='max', help='Maximum number of reviews to fetch', default=sys.maxsize, type=int)
    # parser.add_argument('-e', '--engine', dest='engine', help='Driver to use', choices=['phantomjs', 'chrome', 'firefox'], default='chrome')
    # args = parser.parse_args()

    scraper = TripadvisorScraper_restaurants(engine='chrome')
    ll=scraper._parse_page()
    print(ll)
    for i in ll:
        URL_PATTERN = i
        scraper = TripadvisorScraper(engine='chrome',urlpage=i)
        df = scraper.fetch_reviews(i, 500)
        df.to_csv(page_d[0]+'.csv')
        print('Successfully fetched {} reviews.'.format(len(df.index)))
        print(df)

    scraper.close()














