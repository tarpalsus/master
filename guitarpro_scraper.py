# -*- coding: utf-8 -*-
"""
Created on Thu May 25 19:24:57 2017


"""
from bs4 import BeautifulSoup
import requests
import guitar_pro as gp
import os
import pandas as pd

def download_file(url, name, chunks=False):
    """ Split file to chunks when downloading to avoid overflows"""
    r = requests.get(url)
    with open(name, 'wb') as f:
        if chunks:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        else:
            f.write(r.content)
    return


def scrape_site(root, root_search, url_adress, band_name='Metallica'):
    """Scrape site with gp files, turn them into csv with sounds """
    i = 1
    if not os.path.exists(band_name):
        os.makedirs(band_name)
    prev_link = 'prev'
    current_link = 'first'
    seek_name = band_name.lower().replace(' ', '+')
    while(prev_link != current_link):
        prev_link = current_link
        r = requests.get(root+ root_search + seek_name + url_adress+'&page='+str(i))
        soup = BeautifulSoup(r.content, "html.parser")
        current_link = soup.find('div', class_='title').text
        links = soup.find_all('div', class_='title')
        for l in links:
            link = l.find('a')
            if link:
                print(link)
                r = requests.get(root + link['href'])
                soup = BeautifulSoup(r.content, "html.parser")
                download_link = soup.find('a', class_='dlFile')
                name =r'C:\Users\Maciek\Sound_generator\\' +  band_name + '\\' + link.text
                download_file(root + download_link['href'], name + '.gp3')
                try:
                    _, df, _ = gp.from_guitar_pro(name+'.gp3')
                    df.to_csv(name+'.csv')
                except Exception as e:
                    print('Failed to process')
                    print(str(e))
                finally:
                    os.remove(name+'.gp3')
        print('Processed page ' + str(i))
        i += 1
    return

def process_directory(path):
    """ Process downloaded files from directory"""
    for file in os.listdir(path):
        print(str(file))
        try:
            _, df, _ = gp.from_guitar_pro(os.path.join(path,file))
            df.to_csv( file.split('.')[0] + '.csv')
        except:
            print('Processing error of file ' + str(file))
    return

def scrape_site_recursive(root, url_adress):
    #To finish, probably unuseful on gptabs website
    next_site_url = soup.find_all('li', class_='next')
    if next_site_url:
        scrape_site_recursive(root, next_site_url)
    else:
        return

if __name__ == '__main__':
    root = r'http://gtptabs.com'
    root_search = r'/search/go.html?SearchForm%5BsearchString%5D='
    url_to_scrape = r'&SearchForm%5BsearchIn%5D=tab&yt0=Find'
    #r = requests.get(root+url_to_scrape)
    #soup = BeautifulSoup(r.content, "lxml")
    #links = soup.find_all('div', class_='title')
    #scrape_site(root,root_search, url_to_scrape,'Dream theater')
    path = r"C:\Users\Maciek\Sound_generator\Metallica"
    process_directory(path)
