'''爬取网页中中文章名'''
import requests
from lxml import etree
import codecs
html = requests.get("https://blog.csdn.net/it_xf?viewmode=contents")  #获得了网页的源代码
etree_html = etree.HTML(html.text)
#etree.HTML():构造了一个XPath解析对象并对HTML文本进行自动修正
#一般抓取数据使用的是正则表达式来做，etree提供了另一种更快速方便解析提取html页面数据的方式。
content = etree_html.xpath('//*[@id="mainBox"]/main/div[2]/div/h4/a/text()')  #在网页中获取要抓取的标题的xpath 进行复制

for each in content:
    replace = each.replace('\n', '').replace(' ', '')
    if replace == '\n' or replace == '':
        continue
    else:
        print(replace)

import requests
import json
import urllib
'''抓取图片'''
def getSogouImag(category,length,path):
    n = length
    cate = category
    imgs = requests.get('http://pic.sogou.com/pics/channel/getAllRecomPicByTag.jsp?category='+cate+'&tag=%E5%85%A8%E9%83%A8&start=0&len='+str(n))
    jd = json.loads(imgs.text)  # json.loads函数的使用，将字符串转化为字典 # json.load()函数的使用，将读取json信息
    jd = jd['all_items']   #在审查元素-选择要获取的图片然后点击Networks一直F5刷新 点击preview 其中all_items存放了所有网页里面所有图片的信息
    imgs_url = []
    for j in jd:  # 
        imgs_url.append(j['bthumbUrl'])  #其中bthumbUrl真正存放图片的照片  imgs_url获取了所有图片的照片
    m = 0
    for img_url in imgs_url:
            print('***** '+str(m)+'.jpg *****'+'   Downloading...')
            urllib.request.urlretrieve(img_url,path+str(m)+'.jpg')   #开始下载
            m = m + 1
    print('Download complete!')

#getSogouImag('壁纸',2000,'d:/下载/壁纸/')

def getqq(callback,header,params):
    gequ_info=[]
    for p in range(0,18):
        sings = requests.get("https://c.y.qq.com/soso/fcgi-bin/client_search_cp?ct=24&qqmusic_ver=1298&\
new_json=1&remoteplace=txt.yqq.song&searchid=66172054398362132&t=0&aggr=1&cr=1&catZhida=1&lossless=0&flag_qc=0&\
p="+str(p+1)+"&n=20&w=%E5%BC%A0%E6%9D%B0&g_tk=1622360077&jsonpCallback=MusicJsonCallback"+callback[p]+\
"&loginUin=2406891860&hostUin=0&format=jsonp&inCharset=utf8&outCharset=utf-8&notice=0&platform=yqq&needNewCode=0")
        response=sings.text.lstrip("MusicJsonCallback"+callback[p]+"(")
        #lstrip,rstrip用来去掉一段字符串中前面和后面相应的字符
        response=response.rstrip(")")
        response=json.loads(response)
        data=response['data']
        song_list=data['song']['list']
        for i in song_list:
            ge=str([j['title'] for j in i['singer']])
            ge=ge.lstrip("[").rstrip("]")
            gequ_info.append([i['title'],i['album']['name'],ge ,i['url']])
    return gequ_info
## 保存结果
import pandas as pd
def save(gequ_info,path):
    name=['歌曲','专辑','歌手','网址']
    data=pd.DataFrame(data=gequ_info,columns=name)
    data.to_csv(path)
    return
path='E:/歌曲.csv'
header={'user-agent':'Mozilla/5.0 \
(Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Mobile Safari/537.36',\
'referer':'https://y.qq.com/portal/search.html'}
params={'g_tk':1622360077,'jsonpCallback':'MusicJsonCallback6019933514786626','loginUin':2406891860,'hostUin':0,\
'format':'jsonp','inCharset':'utf8','outCharset':'utf-8','notice':0,'platform':'yqq','needNewCode':0}
callback=['022839168269426224','7372448589564993','14649088042252756','12838408461929673','9936533999833093','6061888127134671',\
          '7610666839764479','7398895882425602','11578261830475212','6538347624751413','2039258737826588','7297892762756342'\
          ,'13889007845047696','2644727784893195','13170824497278555','03658631599958406','8941203964685087','9316479557139787']
#gequ_info=getqq(callback,header,params)
#save(gequ_info,path)
print('\t 歌名','\t专辑','\t歌手','\t网址')
for i in gequ_info:
    print('\t'+i[0],'\t'+i[1],'\t'+i[2],'\t\t'+i[3])
        

####真正存放歌曲信息的地方但是有的时候真正存放信息的网址找不到了 只能在源码上操作了。。。。
'''
第一页的网址
https://c.y.qq.com/soso/fcgi-bin/client_search_cp?ct=24&qqmusic_ver=1298&
new_json=1&remoteplace=txt.yqq.song&searchid=62581049419093656&t=0&aggr=1&cr=1&catZhida=1&lossless=0&flag_qc=0&
p=1&n=20&w=%E5%BC%A0%E6%9D%B0&g_tk=1622360077&jsonpCallback=MusicJsonCallback022839168269426224&
loginUin=2406891860&hostUin=0&format=jsonp&inCharset=utf8&outCharset=utf-8&notice=0&platform=yqq&needNewCode=0
第二页的网址
https://c.y.qq.com/soso/fcgi-bin/client_search_cp?ct=24&qqmusic_ver=1298&
new_json=1&remoteplace=txt.yqq.song&searchid=66172054398362132&t=0&aggr=1&cr=1&catZhida=1&lossless=0&flag_qc=0&
p=2&n=20&w=%E5%BC%A0%E6%9D%B0&g_tk=1622360077&jsonpCallback=MusicJsonCallback""+str()&
loginUin=2406891860&hostUin=0&format=jsonp&inCharset=utf8&outCharset=utf-8&notice=0&platform=yqq&needNewCode=0
每一页的jsonpCallback不一样 而且p不一样 其余一样
爬歌曲链接和歌词是不一样的！！！源码里面并没有直接给出歌词的链接

在看的时候是否是网址中的参数是否是一定要的 把不要的参数可以去掉，从而实现网页的翻页---刷新

'''
import re
import codecs
def getLyric(musicid,songmid,path):  ##需要歌曲的自己的链接  歌曲的id,歌词的id
    url = 'https://c.y.qq.com/lyric/fcgi-bin/fcg_query_lyric.fcg?'  ##单个歌曲链接 单独点开的
    header = {'user-agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)\
Chrome/59.0.3071.115 Safari/537.36','referer':'https://i.y.qq.com/v8/playsong.html?ADTAG=newyqq.song&songmid={0}?'.format(songmid)
    }
    paramters = {'g_tk':5381,'musicid':musicid,'jsonpCallback':'jsonp1','uni':0,'notice':0,'nobase64':1,'songtype':0
        ,'format':'json','inCharset':'utf8','outCharset':'utf-8','notice':'0','platform':'h5','needNewCode':'1'}
    #paramters为原来网址上面的参数 该参数在将网址进行缩减的时候主要将可以将后面的参数单独写出来，因为一个网页的这些参数是一致的
    html = requests.get(url=url,params=paramters,headers=header)
    res = json.loads(html.text.lstrip('jsonp1(').rstrip(')'))
    #由于一部分歌曲是没有上传歌词，因此没有默认为空
    if 'lyric' in res.keys():  ##也可以不用太容易
        lyric = json.loads(html.text.lstrip('jsonp1(').rstrip(')'))['lyric']
    #对歌词内容做稍微清洗
        dr1 = re.compile(r'&#\d.;',re.S)        #re.S的作用是实现正字表达式的跨行 #在正则表达式里点号表示除\n的之外的所有
        dr2 = re.compile(r'\[\d+\]',re.S)
        dr3=re.compile(r'\[\w*\d+]',re.S)
        dr4=re.compile(r'\[\w\w',re.S)
        dd = dr1.sub(r'',lyric)  #re.sub函数使用正则表达式将一些符号进行替换 
        dd = dr2.sub(r'\n',dd)
        f=dr3.sub(r'',dd)
        g=dr4.sub(r'',f).replace(r']','\n')
        dd=g.replace('\n\n\n','\n')
        h=dd.split('\n')
        with codecs.open(path,'wb+',encoding='utf8') as f:
            for i in h:
                f.write(i+'\r\n')    
        return dd
    else:
        return ""

songmid='001Sh6UI3dh9mE'
musicid='4830147'

getLyric(musicid,songmid,path)
#获取所有的歌曲信息---歌名---专辑名----songmid-----musicid-------switch------张杰的歌曲信息
url='https://c.y.qq.com/soso/fcgi-bin/client_search_cp?' ##张杰的歌曲首页---所有歌曲的封面
def get_gequ_info(url,path):
    header={'referer':'https://y.qq.com/portal/search.html','user-agent':'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N)\
        AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Mobile Safari/537.36'}
    info=[]
    for i in range(1,18):
        params={'ct':24,'qqmusic_ver':1298,'new_json':1,'remoteplace':'txt.yqq.song','t':0,\
        'aggr':1,'cr':1,'catZhida':1,'lossless':0,'flag_qc':0,'p':str(i),'n':20,'w':'张杰','g_tk':1622360077,\
        'loginUin':2406891860,'hostUin':0,'format':'jsonp','inCharset':'utf8','outCharset':'utf-8','notice':0,\
        'platform':'yqq','needNewCode':0}
        html=requests.get(url,params=params,headers=header)
        response=html.text
        jes=response.lstrip("callback(").rstrip(")")
        jd=json.loads(jes)
        song_list=jd['data']['song']['list']
        for i in song_list:
            name=i['name']
            album_name=i['album']['name']
            songmid=i['mid']
            musicid=i['id']
            switch1=i['action']['switch']
            info.append([name,album_name,musicid,songmid,switch1])
    info=pd.DataFrame(info,columns=['歌曲','专辑','musicid','songmid','switch'])
    info.to_csv(path)
    return info
path='G:/音乐/张杰.csv'
info=get_gequ_info(url,path)
###获得信息  此处抓取的网址虽然可以打开,但是没法进行下载歌曲，下载下来的歌曲不合适  打开歌曲 让歌曲唱，得到相应的ma4格式的网址
#获得ma4的格式的网址之后对该网址进行分析，需要哪些参数，从主页信息获取，然后保存该链接
path1='G:/音乐/张杰'
def load__geci_(info,path1):
    geci=[]
    for index,row in info.iterrows():
        c=row['歌曲'].find(r'?')
        name=row['歌曲']
        if c!=-1:
            name=row['歌曲'].replace('?',"")  
        dd=getLyric(row['musicid'],row['songmid'],path1+"/{0}.txt".format(name))
        geci.append(dd)
def load_sing(info,path):
    ###直接获取歌曲真正的网址
    path1='http://dl.stream.qqmusic.qq.com/'
    header={'referer':'https://y.qq.com/portal/player.html',\
    'user-agent':'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Mobile Safari/537.36'}
    for index,row in info.iterrows():
        print('For '+'歌曲下载'+' Start download...')
        url='https://u.y.qq.com/cgi-bin/musicu.fcg?data={"req_0":{"module":"vkey.GetVkeyServer","method":"CgiGetVkey",\
                                                     "param":{"guid":"7208009084","songmid":["'+str(row['songmid'])+'"],\
                                                     "songtype":[0],"uin":"0","loginflag":1,"platform":"20"}}}'
        html=requests.get(url,headers=header,verify=False)
        response=json.loads(html.text)
        sip=response["req_0"]["data"]["midurlinfo"][0]["purl"]        
        urllib.request.urlretrieve(path1+sip,str(row['歌曲'])+'.m4a')
        print(str(index)+'***** '+info['歌曲'][index]+'html *****'+' Downloading...')
    return
load_sing(info,path)
#爬取数据
#爬取天气-气象数据 爬取北京城区2018年8月份到12月份的天气状况
'''
使用和上面一样的方法来获取 使用json
真实包含天气信息的网址：
http://d1.weather.com.cn/calendar_new/2018/
101010100_201812.html?_=1544789133178
header={'Referer':'http://www.weather.com.cn/weather40d/101010100.shtml',\
'User-Agent':'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko)\
Chrome/58.0.3029.110 Mobile Safari/537.36'}
'''
'''先不使用上述上述方法求，使用正则表达式获取每一天的天气情况，主要用到bs4，selenium来获取'''
#首先一天的天气情况
#eautifulsoup将复杂的HTML或者XML文件解析成复杂的树形结构，每个节点都是Python对象，
#所有的4个节点可以分为：Tag，NavigableString，BeautifulSoup，Comment。
'''
BeautifulSoup主要解析html的源码，用来抓取某个标签内容，如以下这样的一条信息
soup=BeautifulSoup('<p id="firstpara" align="center">This is paragraph <b>one</b>.</p>','html.parser') 
a=soup.p   返回<p   </p>之间的所有信息
a=a['id']  返回属性id的值  firstpara
b=a.atrrs返回属于id align的值 返回的信息主要使用字典的形式返回
a=soup.string  返回这条网址的标签的内容
Python提供了attrs方法以字典的形式返回Tag内的所有属性
html.parser是解析器，有的时候会出现某些符号被忽略的情况，这个时候使用html5lib解析器
在获取网页的源码之后，再用BeautifulSoup解析，这个时候解析得不到标签之间的内容，因此使用selenuum包来抓取网址

selenium是Python中实现操作浏览器的一个模块，其中主要实现js动态网页请求
Seesion就是服务器端开辟的一块内存空间，存放着客户端浏览器窗口的编号
获得了session对象后，要定位元素，webdriver提供了一系列的元素定位方法，常用的有以下几种方式：
id、name、class-name、link、text、partial、link、text、tag、name、xpath、cssselector

当执行完抓取操作后，必须关闭session，不然让它一直占内存会影响机器其他进程的运行
browser.close()或者browser.quit()都可以关闭session
获取浏览器的session 在使用的时候要注意chrome浏览器与chromediverse版本要相对应
在访问网站之后，可以使用上述的元素定位方法，也可以使用源码解析来做
一些函数的使用：
    webdriver.Chrome('该浏览器的安装路径') 也可以不写，直接在path中加入该浏览器的安装路径
    get（url）访问网站
    browser.page_source 获取html源码
    find_element_by_xpath('xpath') 
    然后关闭浏览器

'''
from bs4 import BeautifulSoup
from selenium import webdriver  #用来抓取王爷太动态数据
def real_time_weather(url):
    browser = webdriver.Chrome()  
    browser.get(url)#加载页面
    content = browser.page_source  #获得了网站的源码 但是格式比较乱
    browser.close()
    html = BeautifulSoup(content, "html.parser")  #解析网址源码 
    tem = html.find_all("div", class_="tem")
	# 经检查find_all方法返回的tem第一组数据为想要获取的数据
	# span区域为实时气温的数值，em区域为实时气温的单位
    result = tem[0].span.text + tem[0].em.text  

    print("实时气温：" + result)


if __name__ == "__main__":
    url_bj = "http://www.weather.com.cn/weather1d/101010100.shtml"
    real_time_weather(url_bj)

#抓取一周的天气情况
url= "http://www.weather.com.cn/weather/101010100.shtml"
browser = webdriver.Chrome()
browser.get(url)
content = browser.page_source
browser.close()
html = BeautifulSoup(content, "html.parser")
week_weather= html.find_all("div", class_="c7d")  
list_week=week_weather[0].find_all("ul",class_="t clearfix")
weather=list_week[0].find_all("li",class_="sky")
week=[]
for list1 in weather:
    day=list1.h1.text
    w=list1.find_all("p",class_="wea")[0].text
    c=list1.find_all("p",class_="tem")[0].i.text
    f1=u.find_all("p",class_="win")[0].select("span")
    if len(f1)==1:
        f2="无持续风向"
    else:
        f2=f1[0].get("title")+'-'+f1[1].get("title")
    jishu=list1.find_all("p",class_="win")[0].i.text
    week.append((day,w,c,f1+f2,jishu))


'''在myspl中建立
create database weather;
use weather;
真正的开始建立表
create table weather (
    id int primary key auto_increment,
    date1 varchar(100),
    desc1 varchar(100),
    temp1 varchar(100),
    direction1 varchar(100),
    level1 varchar(100) 
);
'''
'''
pymysql参数说明：
host(str):      MySQL服务器地址
port(int):      MySQL服务器端口号
user(str):      用户名
passwd(str):    密码
db(str):        数据库名称
charset(str):   连接编码

'''
import pymysql
con = pymysql.connect(host="127.0.0.1",user="root",password="111111",database="weather",charset="utf8",port=3306) #打开mysql

cursor = con.cursor();
#执行数据库操作
#cursor.execute(parm1,parm2)  parm1 进行正则匹配 在此基础上对字符串进行潜入处理，如果%s加上引号
#mysql中会出现oooo-oo-oo的错误日期
#当数据很多的时候直接使用executemany插入，时间只需要几秒
for result1 in week:
    print(result1)
    insertsql = "insert into weather (date1,desc1,temp1,direction1,level1) VALUES (%s,%s,%s,%s,%s)"
    cursor.execute(insertsql,result1)
con.commit()
cursor.close()
con.close()  
#爬取豆瓣分类--中国大陆--电影的电影信息
url="https://movie.douban.com/tag/#/?sort=U&range=0,10&tags=%E4%B8%AD%E5%9B%BD%E5%A4%A7%E9%99%86"
def get_info(url):
    browser=webdriver.Chrome()
    browser.get(url)
    content= browser.page_source
    browser.close()
    html = BeautifulSoup(content, "html.parser")
    info=html.find_all("div",class_="list-wp")
    movie=info[0].find_all("a")
    info_movie=[]
    for res in movie:
        f1=res.select("span")
        name=f1[1].text
        rate=f1[2].text
        url1=res.get("href")
        photo_url=f1[0].select("img")[0].get("src")
        info_movie.append((name,rate,url1,photo_url))
    return info_movie
#https://movie.douban.com/tag/#/?sort=U&range=0,10&tags=%E4%B8%AD%E5%9B%BD%E5%A4%A7%E9%99%86
#





import requests
from bs4 import BeautifulSoup
import bs4
import re
def getHTMLText(url):
    try:
        r = requests.get(url)
        r.raise_for_status()  #raise_for_status()方法可以确保访问网址确实成功，然后再让程序继续做其他事情
        r.encoding = r.apparent_encoding
        return r.text
    except:
        return ""
'''
html.find（）该方法找出来的就是源码的一段
html.find_all() 虽然也是源码，但是返回的是一个列表形式的源码
'''
def fillUnivList(ulist,html):
    count=0
    soup = BeautifulSoup(html,"html.parser")
    for tg in soup.find_all("div","pl2"):
        f1=tg.find("a")
        name=f1.text.split('\n')[1].lstrip()
        url1=f1.get("href")
        f2=tg.find_all("p",class_="pl") #pl
        date=f2[0].text.split("/")[0]
        photo=tg.find("img")
        if tg.find_all("span",re.compile("nums")): #re.compile 建立一个正则表达式
            rate = tg.find_all("span",class_="rating_nums")
            rate_num=tg.find_all("span",class_="pl")
            if photo!=None:
                ulist.append((name,date,rate[0].text,rate_num[0].text,photo.get("src"),url1))
            else:
                ulist.append((name,date,rate[0].text,rate_num[0].text,None,url1))
        else:
            if photo!=None:
                ulist.append((name,date,"暂无评价",None,photo.get("src"),url1))
            else:
                ulist.append((name,date,"暂无评价",None,None,url1))
        count+=1
    return ulist

def main():
    sumz=0
    lst=[]
    while sumz<=980:  
        lst.append(sumz)
        sumz=sumz+20
    uinfo = []
    for n in lst:
        ul=[]
        url = "https://movie.douban.com/tag/中国电影?start="+str(n)+"&type=T"
        html = getHTMLText(url)
        ulist=fillUnivList(ul, html)
        uinfo.extend(ulist)
    return uinfo
uinfo=main()
#保存
'''
 create table movie( id int primary key auto_increment,
 name varchar(100),
 date varchar(100),
    rate varchar(100),
    rate_num varchar(100),
     photo_url varchar(100),
     url1 varchar(100),);'''
con = pymysql.connect(host="127.0.0.1",user="root",password="101600ai@",database="movie",charset="utf8",port=3306) #打开mysql
cursor = con.cursor();
insertsql = "insert into movie(name,date,rate,rate_num,photo_url,movie_url) VALUES (%s,%s,%s,%s,%s,%s)"
cursor.executemany(insertsql,uinfo)
con.commit()
cursor.close()
con.close()
#利用每一个影片的网址搜索的评价信息 以及打分情况
url="https://movie.douban.com/subject/1292365/"
def get_comment(url):
    mo_if=[]
    html= getHTMLText(url)
    soup=BeautifulSoup(html,"html.parser")
    info=soup.find_all("div",id="wrapper")
    info=info[0].find_all("div",class_="grid-16-8 clearfix")
    info_movie=info[0].find_all("div",class_="subject clearfix")
    f1=info_movie[0].find_all("span")
    directior=f1[0].text
    scriptwriter=f1[3].text
    actor=str(f1[6].text.split('/')[:3]).lstrip('[').rstrip(']')
    type_movie=info_movie[0].find_all("span",property="v:genre")
    movie_type="剧情"
    for i in range(1,len(type_movie)):
        movie_type=movie_type+'-'+type_movie[i].text
    mo_if.append((directior,scriptwriter,actor,movie_type))
    sumz=0
    lst=[]
    while sumz<=980:  
        lst.append(sumz)
        sumz=sumz+20
    evaluate=[]
    for n in lst:
        url=html+'comments?start='+str(n)+'&limit=20&sort=new_score&status=P'
        soup1=BeautifulSoup(url,"html.parser")
        ifo=soup1.find_all("div",id="wrapper")
        iop=ifo[0].find_all("div",class_="grid-16-8 clearfix")
        k=iop[0].find_all("div",class_="article")
        f2=k[0].find_all("div",class_="mod-bd")
        h=f2[0].find_all("div",class_="comment-item")
        for i in h:
            f=i.find_all("span",class_="comment-info")
            if len(f)>0:
                pj=f[0].select("span")
                #print(len(pj))
                if len(pj)==3:
                    pj1=pj[0].text
                    rating=pj[1].get("title")
                    comment_time=pj[2].get("title")
                    comment=i.p.text
                    evaluate.append((pj1,rating,comment_time,comment))
                elif len(pj)==2:
                    pj1=pj[0].text
                    rating=pj[1].get("title")
                    comment=i.p.text
                    evaluate.append((pj1,rating,None,comment))
    return mo_if,evaluate
def all_evaluate(uinfo):
    all_moif=[];all_evaluate={}
    for i in uinfo:
        url=i[5]
        mo_if,evaluate=get_comment(url)
        all_moif.extend(mo_if)
        if i[0] not in all_evaluate:
            all_evaluate[i[0]]=evaluate
    return all_info,all_all_evaluate
all_info,all_all_evaluate= all_evaluate(uinfo)
