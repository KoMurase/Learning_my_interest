
from selenium import webdriver
from selenium.webdriver.common.keys import Keys as keys
#from autotest_lib.client.common_lib.cros import chromedriver
import urllib
from bs4 import BeautifulSoup
import csv
path_w=r'C:\Users\mkou0\Desktop\HTML\scrap_result.html'
# Macの場合 (Chromedriveがこのプログラムを実行している同じ場所にある前提)
#driver = webdriver.Chrome(executable_path="./chromedriver") # Windowsの方はこの行をコメントアウト
#with chromedriver.chromedriver() as chromedriver_instance:
 #  driver = chromedriver_instance.driver

url="https://duet.doshisha.ac.jp/kokai/html/fi/fi050/FI05001G.html"
instance=urllib.request.urlopen(url)
soup=BeautifulSoup(instance,"html.parser")
get_words=soup.select_one("#form1 > div:nth-child(11) > div > span > table")

#print(get_words)
#print()
rows=get_words.find_all('tr')

for tr in rows:
    cols = tr.find_all('td')
    #tdはTableDataの略でありテーブルセルの内容を指定する

    for td in cols:
        data=[[x] for x in td]
        print(data)
        for wr in data:
            with open(path_w,mode='a',encoding='UTF-8') as f:
                f.write(str(wr)+'   ')
