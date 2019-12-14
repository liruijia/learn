import requests
from bs4 import BeautifulSoup
import time
import os


def urls(url, headers): #爬取所有url
	urlsss = []
	r = requests.get(url, headers=headers).text
	soup = BeautifulSoup(r,'lxml')
	for i in soup.find_all('h3'):  #[:-3]
		urlsss.append(i.a.get('href')) 
	return urlsss

def url_neirong(urls,headers): #每章内容的操作
	#先检查是否存在该文件夹
	if os.path.exists('D:/IT/新闻/'): 
		pass
	else:
		os.mkdir('D:/IT/新闻/')
	for q in urls:
		try:
			time.sleep(2)
			r = requests.get(q, headers=headers).text
			soup = BeautifulSoup(r,'lxml')
			for i in soup.find('div', class_="article-title"): #每章的标题
				if os.path.exists('D:/IT/新闻/' + i.get_text() +'.txt'): #检查是否已存在该文件
					print('已存在：',i.get_text())
					continue
				else:
					for i in soup.find('div', class_="article-title"): #每章的标题
						print('标题：'+ i.get_text())				
						f = open('D:/IT/新闻/'+ i.get_text() +'.txt','w',encoding='utf-8')
						f.write('标题：'+ i.get_text() + '\n')
					for i in soup.find_all('div', class_="article-source article-source-bjh"): #发布日期
						print('-'*30)
						print('日期：'+ i.find(class_="date").get_text(), end=' ');print('时间：'+ i.find(class_="time").get_text(), end=' ');print('URL:',q)
						aas = '日期：'+ i.find(class_="date").get_text()
						aad =  '时间：'+ i.find(class_="time").get_text()
						aaf = 'URL:%s'%q
						f.write(aas + '\t')
						f.write(aad + '\t')
						f.write(aaf + '\n')
						print('-'*30)
					for i in soup.find_all('div', class_="article-content"): #每章的内容
						f.write(i.get_text())
						f.close()
					print('*'*100)
		except Exception as result:
			print('网页不存在了。。。')
			print('%s',result)
			print('*'*100)

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.121 Safari/537.36'}
url = 'https://news.baidu.com/widget?id=AllOtherData&channel=internet&t=1554738238830'
print(url_neirong(urls(url,headers), headers))
print('完成！！！')
