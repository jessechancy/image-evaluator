#web-scraper made for hypeauditor.com

import requests
import urllib.request
import time
from bs4 import BeautifulSoup
import csv

influencer_dict = dict()
num_pages = 20

#loops through all the pages
for page in range(num_pages):
    cur_page = str(page + 1)
    url = "https://hypeauditor.com/top-instagram/?p="+cur_page
    response = requests.get(url)
    print(response)
    soup = BeautifulSoup(response.text, 'html.parser')
    #gets all the influencer information
    for influencer in soup.findAll('tr')[1:]:
        #takes their name and their followers
        traits = influencer.find_all(class_=['bloggers-top-name', 't-a-r'])
        username = traits[1].a.string[1:]
        followers = traits[2].string
        #adding to dictionary
        influencer_dict[username] = followers

#puts the dictionary into a csv file
with open('test1.csv', 'w') as f:
    for key in influencer_dict.keys():
        f.write("%s,%s\n" %(key, influencer_dict[key]))