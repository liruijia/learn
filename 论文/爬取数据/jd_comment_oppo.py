'''
我们至少要选择100多个店铺的某一个产品的评论信息，比如官方旗舰店以及各种手机专卖店

分析oppo reno 系列手机

首先利用find_all('div',class_='search-result-boxout')

然后继续定位到find_all('div',class_='title') ,最后查找第一个a标签的href属性

对于一个具体的网址：

需要获取评论人的昵称、店铺名、评论的总人数，该人的评论、发布的时间、该店铺的星级、该店铺的总的粉丝数量

比如我们爬取到的该商品的链接为https://item.jd.com/57521830334.html ，但是直接去爬取的话，是找不到相应的评论的，
要想爬取下来则要在https://item.jd.com/57521830334.html#comment该网进行爬取,而且此时我们爬取只需要直接获取productpagecomment这个JS的
内容即可

产品评论网站：https://sclub.jd.com/comment/productPageComments.action?callback=fetchJSON_comment98vv308
&productId=57521830334&score=0&sortType=5&page=0&pageSize=10&isShadowSku=0&fold=1


https://item.jd.com/57521830334.html#comment
产品网站：https://item.jd.com/productId.html#comment

获取个产品的网址的网站：
https://search.jd.com/Search?keyword=oppo%20reno&enc=utf-8&suggest=1.rem.0.0&wq=oppo%20reno&pvid=e69b2a8e30a34f6984fc293b2192d23e

'''
import urllib
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver import ActionChains
import time
import os
import http.cookiejar
import json
import pandas as pd
import csv
import codecs

class jd_comment():
    def __init__(self):
        print('开始爬取')
    def getnet_info(self):
        '''利用搜索到的结果获取每个产品的productid以及获取其下的评论个数'''
        all_info=pd.DataFrame(columns=['shop_name','comment_num','productId','good_rate','url','product'])
        #headers={'user-agent':'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.79 Safari/537.36',
        #         'cookie':'__jdu=15643226051851040841365; pinId=5rS7uzjk4iIY_10t6rX84A; pin=%E4%BD%B3%E4%BD%B3%E4%BD%B31016; unick=%E4%BD%B3%E4%BD%B3%E4%BD%B31016; _tp=q04yhIMpErvWbIVgKgQmuAFh%2Bn9sH33%2BohCLpppALCs%3D; _pst=%E4%BD%B3%E4%BD%B3%E4%BD%B31016; unpl=V2_ZzNtbUQHF0YiWkYHLxwIUWILEVkRAEFHd19ABnwbXQI3AEZeclRCFX0URldnGVoUZwcZWEFcQhJFCEdkeB5fA2AFEFlBZxVLK14bADlNDEY1WnwHBAJfF3ILQFJ8HlQMZAEUbXJUQyV1CXZUfx5ZB2QAFVxGV0oRdQlDVXIcXAdgByJtRWdzJXEBRV1%2fEWwEVwIiHxYLRBVyCkZRNhlYAmIBEV5FVkcVfAxGVX4YVQBnARVZclZzFg%3d%3d; __jdv=76161171|google-search|t_262767352_googlesearch|cpc|kwd-362776698237_0_7aebfc1be4ee4925bf3c3f7c6306a2e2|1576568529253; areaId=28; ipLoc-djd=28-2487-21646-0; shshshfpa=a2bdab8c-c101-61b7-d32e-eb0b75ce8ec7-1576568591; shshshfpb=bnrFJM6cZ7L5fQ9VdFuUtWg%3D%3D; ceshi3.com=000; TrackID=18tYjxLpdylvIgEKVu1h0V-M_YM2jE1kvxiBAmSl0qH_y_YXjg2WirTAEJhqpibK-2oI8x6-Mj4XvDpvwWHhTq9sRPgesCwGL6lo3e60bgFY; PCSYCityID=CN_620000_620100_620103; __jdc=122270672; 3AB9D23F7A4B3C9B=UQ45CY3ZSEKZC65MCK6BCU5AGQII4HW3FKN5AOSS62FC3QBXULUVRC4WODL536WNZSHM7ORWQDOEGHXV7UPG2ADRLQ; shshshfp=06031fa83bf3e7fe66a12a448b44620e; thor=58D9324435DD0AC1EEE5C46D19697B5ADF444E668FC328C46CABA0485602B287C8579C7145114310AFF2EFBDF4E49F84E8657F4743EB33CA21D9B2D2C101AD5C59B4BE4ACA48A38C1E35AB21B52DFD6F0AEDE45C11060C6BBB34809353E9A252C69B87DD6FA2B55067F455A29E2D12B8B46310BBB0E4718D6131D6D9F518C7D3B220CE27DECEC78734F08E08ABE25B20; __jda=122270672.15643226051851040841365.1564322605.1576575475.1576583300.9; __jdb=122270672.1.15643226051851040841365|9.1576583300; shshshsID=74043da9250acf14f8dde40fed669c65_1_1576583301258'}
        #    
        jj=0
        for i in range(1,6):
            try:
                url_1='https://search-x.jd.com/Search?callback=jQuery2707165&area=28&enc=utf-8&keyword=oppo+reno&adType=7&page='
                url_2=str(i)+'&ad_ids=291%3A33&xtest=new_search'
                url=url_1+url_2
                r=requests.get(url).text
                #print(r)
                response=r.lstrip('jQuery2707165(').rstrip(')')
                print('resopnse:',len(response))
                data=json.loads(response)
                for i in data['291']:
                    store_name=i['shop_link']['shop_name']
                    comment_num=i['comment_num']
                    productId=i['sku_id']
                    url=i['link_url']+'#comment'
                    product=i['ad_title']  #利用product进行后期的处理
                    good_rate=i['good_rate']
                    print(store_name,comment_num,good_rate,url)
                    all_info.loc[jj]=[store_name,comment_num,productId,good_rate,url,product]
                    jj+=1
            except Exception as result:
                print('error',result)
        all_info.to_csv('C:/Users/Administrator/Desktop/data/评论/product_info_before.csv')
        print('所有产品的基本信息已经加载完成****************')
        return all_info

    def getinfo(self,all_info):
        info=all_info.copy()
        print('the origin of len :',len(info))
        index_drop=[]
        print(info)
        for row_id,data  in all_info.iterrows():
            product_info=data['product']
            comment_num=data['comment_num']
            if 'oppo' not in product_info and 'OPPO' not in product_info:
                index_drop.append(row_id)
                print('row_id',row_id)
            else:
                if int(comment_num)==0:
                    index_drop.append(row_id)
                    print('row_id',row_id)
        print('index:',index_drop)
        info.drop(info.index[index_drop],inplace=True)
        print('the len of info :',len(info))
        info.to_csv('C:/Users/Administrator/Desktop/data/评论/product_info_after.csv')
        print('基本信息处理完成******************')
        return info 


    def _getcomment(self,productId,comment_num):
        all_comment=pd.DataFrame(columns=['user_id','user_name','referenceTime','score','productid','product_color','comment'])

        URL='https://chat1.jd.com/api/checkChat?callback=jQuery1024117&pid='+str(productId)+'&returnCharset=UTF-8&_=1576665317646'
        
        headers={'user-agent':'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.79 Safari/537.36',
                 'referer':'https://item.jd.com/'+str(productId)+'.html',
                 'cookie':'__jdu=15643226051851040841365; pinId=5rS7uzjk4iIY_10t6rX84A; pin=%E4%BD%B3%E4%BD%B3%E4%BD%B31016; unick=%E4%BD%B3%E4%BD%B3%E4%BD%B31016; _tp=q04yhIMpErvWbIVgKgQmuAFh%2Bn9sH33%2BohCLpppALCs%3D; _pst=%E4%BD%B3%E4%BD%B3%E4%BD%B31016; unpl=V2_ZzNtbUQHF0YiWkYHLxwIUWILEVkRAEFHd19ABnwbXQI3AEZeclRCFX0URldnGVoUZwcZWEFcQhJFCEdkeB5fA2AFEFlBZxVLK14bADlNDEY1WnwHBAJfF3ILQFJ8HlQMZAEUbXJUQyV1CXZUfx5ZB2QAFVxGV0oRdQlDVXIcXAdgByJtRWdzJXEBRV1%2fEWwEVwIiHxYLRBVyCkZRNhlYAmIBEV5FVkcVfAxGVX4YVQBnARVZclZzFg%3d%3d; __jdv=76161171|google-search|t_262767352_googlesearch|cpc|kwd-362776698237_0_7aebfc1be4ee4925bf3c3f7c6306a2e2|1576568529253; areaId=28; ipLoc-djd=28-2487-21646-0; shshshfpa=a2bdab8c-c101-61b7-d32e-eb0b75ce8ec7-1576568591; shshshfpb=bnrFJM6cZ7L5fQ9VdFuUtWg%3D%3D; ceshi3.com=000; TrackID=18tYjxLpdylvIgEKVu1h0V-M_YM2jE1kvxiBAmSl0qH_y_YXjg2WirTAEJhqpibK-2oI8x6-Mj4XvDpvwWHhTq9sRPgesCwGL6lo3e60bgFY; PCSYCityID=CN_620000_620100_620103; __jdc=122270672; mba_muid=15643226051851040841365; 3AB9D23F7A4B3C9B=UQ45CY3ZSEKZC65MCK6BCU5AGQII4HW3FKN5AOSS62FC3QBXULUVRC4WODL536WNZSHM7ORWQDOEGHXV7UPG2ADRLQ; _gcl_au=1.1.1366431867.1576661791; shshshfp=cee428329831c168d7622114e7ce16ab; __jda=122270672.15643226051851040841365.1564322605.1576665284.1576673751.12; JSESSIONID=60185F38B35FB8697B7C5CC932DB6B9E.s1; shshshsID=78aa37c999a13f1be43db8a9ae8ee99b_3_1576673820995; __jdb=122270672.3.15643226051851040841365|12.1576673751'}
        
        all_comment['productid']=productId
        jj=0
        comment_num=int(comment_num)
        if comment_num <=10:
            all_page=1
        elif comment_num>10000:
            all_page=1100
        else:
            all_page=int(comment_num/10)-1
        for ii in range(all_page):
            try:
                
                url_0='https://sclub.jd.com/comment/productPageComments.action?callback=fetchJSON_comment98vv308&productId='
                url_1=str(productId)+'&score=0&sortType=5&page='
                url_2=str(ii)
                url_3='&pageSize=10&isShadowSku=0&fold=1'
                final_url=url_0+url_1+url_2+url_3

                #print('i:%d'%i)
                response = requests.get(url=final_url, headers=headers)
                text=response.text
                print('response:',len(text))
                text_new=text.lstrip('fetchJSON_comment98vv308(').rstrip(';)')
                #print(text_new[:10])
                data=json.loads(text_new)
                #print('i:%d'%i)
                for i in data['comments']:
                    content = i['content']
                    user_id=i['id']
                    user_name=i['nickname']
                    referencetime=i['referenceTime']
                    score=i['score']
                    product_color=i['productColor']
                    #product_size=i['productSize']
                    #print("评论内容\n{0}".format(content))
                    all_comment.loc[jj]=[user_id,user_name,referencetime,score,None,product_color,content]
                    jj+=1
                print('product{0} 的 第{1}页评论加载完成'.format(productId,ii))
            except Exception as result:
                print('appear error:',result)
        return all_comment

    def get_all_comment(self,info):
        comment_total=pd.DataFrame(columns=['user_id','user_name','referenceTime','score','productid','product_color','comment'])
        for row_id,data in info.iterrows():
            product_id=data['productId']
            comment_num=data['comment_num']
            if product_id == '46165085511':
                continue
            comment=self._getcomment(product_id,comment_num)
            comment_total=pd.concat([comment_total,comment],axis=0)
            print('product:{0} 已经爬取完毕'.format(product_id))
        comment_total.to_csv('C:/Users/Administrator/Desktop/data/评论/comment_info_final.csv')
        return comment_total


P=jd_comment()      
all_info=P.getnet_info()        
info=P.getinfo(all_info)
comment=P.get_all_comment(info)
        
        












