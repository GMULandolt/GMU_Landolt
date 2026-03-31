import requests 
from bs4 import BeautifulSoup
import csv

URL = "https://airmass.org/observatories"
r = requests.get(URL)

f = open('observatories_list.csv', 'w')

writer = csv.writer(f)

webpage = BeautifulSoup(r.content, 'html5lib')
#print(webpage.prettify())

table = webpage.find('table')

process = False

for row in table.findAll('tr'):
    if process:
        cols = row.contents
        names = cols[1].text
        country = cols[2].text
        latitude = cols[3]['data-sorttable_customkey']
        longitude = cols[4]['data-sorttable_customkey']
        elevation = cols[5].text
        lightPollution = cols[6]['data-sorttable_customkey']
        limitingMagnitude = cols[7].text
        writer.writerow([names, country, latitude, longitude, elevation, lightPollution, limitingMagnitude])
    process = True


f.close()