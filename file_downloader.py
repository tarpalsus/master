# -*- coding: utf-8 -*-
"""
Created on Sat May 20 18:12:09 2017

"""

import re
import requests
from bs4 import BeautifulSoup

link = r'http://gtptabs.com/tabs/download/2391.html'

r = requests.get(link)
soup = BeautifulSoup(r.text, "html.parser")

for i in soup.find_all('a', {'class': "btn btn-info"}):
    print(re.search('http://.*\.apk', i.get('href')).group(0))