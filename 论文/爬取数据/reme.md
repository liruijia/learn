****************************************************************
爬取网易新闻---以下是以公益新闻为例子
url=http://gongyi.163.com/special/huodongsudi/

从上面的 url中获取该网站的所有的网址
首先先定位到div  class='content'
然后再定位到a的href属性中


针对的每一遍具体的文章
headers={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0.3904.108 Safari/537.36'}

下面这个的爬取是按照某一个专题进行爬取，一个专题下面有很多的文章，根据每个文章公共的class进行爬取
需要注意的爬取的时候的文章标题的获取、发布时间、以及文章的内容
文章内容: find_all('div',class_='post_text')
文章的标题:find_all('div',post_content_main)
文章的时间：find_all('div',class_='post_time_scource')  与来源  find_all('a',id='ne_article_scource')

如上的步骤我们只是爬取了该新闻下面的第一页的文档的情况，我们发现其他的网址和第一页的网址是基本上是一致的

如第19页：http://gongyi.163.com/special/huodongsudi_19/
headers=User-Agent: Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0.3904.108 Safari/537.36


保存路径C:/Users/Administrator/Desktop/github/learn/learn/论文/新闻/公益新闻/

*****************************************************************************

下面爬取关于科技新闻---首先得要观察一下标题、时间、内容得获取是否和公益新闻是否一致？？
科技新闻主页中获取所有新闻的url
首先找到所有的find_all('div','titleBar clearfix')
然后在找到的每一个titleBar中获取a的href属性


在点开一个具体得文章得时候其：
标题的获取：find_all('div',class_='post_content_main')
时间的获取：find_all('div',class_='post_time_scource')
内容的获取：find_all('div',class_='post_text')

获取多页：http://tech.163.com/special/internet_2016_09/


*****************************************************************************
下面爬取关于旅游类的新闻----http://travel.163.com/
但是在网页中找到的是加载更多，而不是下一页这种形式

在旅游网站中获取新闻的url
首先定位到find_all('div',class_='news_title')
然后在每一个找到的news_title下面，定位到h3下面，去获取标签a的href属性

对于具体的一篇文章来说，去获取其文章标题，发布时间，文章内容
标题的获取：find_all('div',class_='post_content_main')
时间的获取：find_all('div',class_='post_time_scource')
内容的获取：find_all('div',class_='post_text')

爬取更多的数据----加载更多
此时我们可以使用鼠标进行模拟，其点击“加载更多”
使用selenium包进行动态的模拟鼠标的动作

from selenium import webdriver
browser = webdriver.Chrome()

网页的请求
browser.get('http://travel.163.com/')
# page_source属性用于获取网页的源代码，然后就可以使用正则表达式，css，xpath，bs4来解析网页
print(browser.page_source)
content=browser.page_source 

browser.execute_script('window.scrollTo(0,document.body.scrollHeight)')

browser.execute_script('alert("已经到最下面了")')

browser.close()


还有一种方式是去点击加载更多
selected_div = browser.find_element_by_css_selector(".post_addmore")

ActionChains(driver).click(selected_div).perform()


不一定就是要滑倒底，只需要点击加载更多20多次足够


*****************************************************************
爬取健康类的新闻

http://jiankang.163.com/special/health_health/
获取网页的新闻的url
首先定位----find_all('div',class_='news_main_info')
然后对于每一个定位获取标签a的属性href

对于一个具体的新闻获取其文章的标题 来源 作者以及内容
标题的获取：find_all('div',class_='post_content_main')
时间的获取：find_all('div',class_='post_time_scource')
内容的获取：find_all('div',class_='post_text')

获取多页：
    实在是没有找到各个页之间的联系，因此使用点击下一页的方法来获取所有的网址
    首先要知道总共有多少页
    再点击下一页，当走到最后一步的时候下一页的状态变成了current


    首先获得总共有多少页：find_all('div',class_='bizidx_pages bizidx_news_pages')
    

**************************************************************
最后爬取关于娱乐新闻

网址：http://news.163.com/special/0001386F/rank_ent.html

获取每一篇新闻的url
    首先定位到：find_all('div',class_='tabContents active')
    然后查找tr标签的text即可

对于任意一篇文章来说：
标题的获取：find_all('div',class_='post_content_main')
时间的获取：find_all('div',class_='post_time_scource')
内容的获取：find_all('div',class_='post_text')
