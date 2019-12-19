# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# https://doc.scrapy.org/en/latest/topics/items.html

import scrapy
from scrapy.loader.processors import MapCompose
from scrapy.loader.processors import Join

def parse_field(text):
    return str(text).strip()

class TaobaoTestItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    pass

class GoodsItem(scrapy.Item):

    # 评论时间
    date = scrapy.Field(
        input_processor=MapCompose(parse_field),
        output_processor=Join(),
    )

    # 评论id
    rate_id = scrapy.Field(
        input_processor=MapCompose(parse_field),
        output_processor=Join(),
    )

    # 评论内容
    content = scrapy.Field(
        input_processor=MapCompose(parse_field),
        output_processor=Join(),
    )

    # 商品链接
    link = scrapy.Field(
        input_processor=MapCompose(parse_field),
        output_processor=Join(),
    )

    # 商品id
    auc_num_id = scrapy.Field(
        input_processor=MapCompose(parse_field),
        output_processor=Join(),
    )

    def get_insert_sql(self):
        insert_sql = """
            insert into rate(date,rate_id,content,link,auc_num_id)
            values (%s,%s,%s,%s,%s)
        """

        params = (self["date"], self["rate_id"], self["content"], self["link"], self["auc_num_id"])

        return insert_sql, params