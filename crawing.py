import requests

import pandas as pd
from bs4 import BeautifulSoup
import lxml.html as lh
import os


def crawler(purl): 
    
    url = purl
    html = requests.get(url)
    # print(html.text)
    doc = lh.fromstring(html.content)
    tr_elements = doc.xpath('//tr')
    #Create empty list
    col=[]
    i=0
    #For each row, store each first element (header) and an empty list
    ilen = len(tr_elements[0])
    for t in tr_elements[0]:
        i+=1
        name=t.text_content()
        # print (i,name)
        col.append((name,[]))
    for j in range(1,len(tr_elements)):
        #T is our j'th row
        i=0
        T=tr_elements[j]
        for t in T.iterchildren():
            data=t.text_content()
            if (len(T) == (ilen - 1) and i == 2):
                col[i][1].append('0.0000%')
                i+=1
            col[i][1].append(data) 
            i+=1

    Dict={title:column for (title,column) in col}
    df=pd.DataFrame(Dict)

    print (df)


excel_file = 'R2M.xlsx'
excel_dir = os.path.join('', excel_file)

df_from_excel = pd.read_excel(excel_dir, sheet_name = 'Sheet1', engine='openpyxl')

for index, row in df_from_excel.iterrows():
    if (row['type'] == 1):
        crawler(row['url'])


# print(df_from_excel)

# crawler('https://r2m.webzen.co.kr/gameinfo/guide/detail/446')

