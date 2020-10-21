------------Beautiful Soup------------------


What is Beautiful Soup?

   Beautiful Soup is a Python library for getting data out of HTML, XML, and other markup languages. Say you’ve found some webpages that display data relevant to your research, such as date or address information, but that do not provide any way of downloading the data directly. Beautiful Soup helps you pull particular content from a webpage, remove the HTML markup, and save the information. It is a tool for web scraping that helps you clean up and parse the documents you have pulled down from the web.
--------------------
Installing Beautiful Soup:
$pip install beautifulsoup4

With sudo, the command is:
$sudo pip install beautifulsoup4

installing csv :
$pip install csv

//program to scraping a data from html file and writing data to csv file

from bs4 import BeautifulSoup
import csv

soup = BeautifulSoup (open("sample.html"), features="lxml")

final_link = soup.p.a
final_link.decompose()

f = csv.writer(open("sample.csv", "w"))
f.writerow(["Name", "Link"])    # Write column headers as the first line

links = soup.find_all('a')
for link in links:
    names = link.contents[0]
    fullLink = link.get('href')

    f.writerow([names,fullLink])
    
    
    
