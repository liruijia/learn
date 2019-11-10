from selenium import webdriver
driver=webdriver.Chrome()
driver.get('https://www.baidu.com')
driver.find_element_by_id('kw').send_keys('selenium')
driver.find_element_by_id('su').click()
pg=driver.find_element_by_xpath('//div[@id="page"]/a[last()-1]/span[last()]')
h=int(pg.text)+1
elements_list=[]
for i in range(1,h):
    elements=driver.find_elements_by_xpath('//div/h3/a')
    elements_list.extend(elements)
    df=driver.find_element_by_xpath('//div[@id="page"]/a[last()]')
    df.click()
for hj in elements_list:
    print(hj.text)
driver.quit()
