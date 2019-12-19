# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://doc.scrapy.org/en/latest/topics/item-pipeline.html
import codecs
import json
import MySQLdb.cursors
from twisted.enterprise import adbapi
import pymysql


class TaobaoTestPipeline(object):
    def process_item(self, item, spider):
        return item


class JsonPipeline(object):

    def __init__(self):
        self.file = codecs.open("jsondata.json", "w", encoding="utf-8")

    def process_item(self, item, spider):
        lines = json.dumps(dict(item), ensure_ascii=False) + "\n"
        self.file.write(lines)
        return item

    def spider_close(self):
        self.file.close()

class writeMysql(object):
    def __init__(self):
        self.client = pymysql.connect(
            host='localhost',
            port=3306,
            user='root',
            passwd='123456',
            db='new_model',
            charset='utf8'
        )
        self.cur = self.client.cursor()

    def process_item(self, item, spider):
        insert_sql, params = item.get_insert_sql()
        self.cur.execute(insert_sql, params)
        self.client.commit()
        return item

    def handle_error(self, failure, item, spider):
        print(failure)


