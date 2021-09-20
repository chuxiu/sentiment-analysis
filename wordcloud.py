#!/usr/bin/env python
# coding=utf-8


import os
import jieba
import numpy as np
from PIL import Image
from wordcloud import WordCloud

path = "D:/Postgraduate/WARWICK/dissertation/download/"
TEXT = path+'comment_weiya.txt'
PICTURE= path+'1.png'
# Handling Chinese fonts
FONT = 'C:/Windows/Fonts/simfang.ttf'



def cut_the_words(test=TEXT):
    with open(test, 'r',encoding='utf-8') as rp:
        content = rp.read()
    words_list = jieba.cut(content, cut_all = True)
    return ' '.join(words_list)
	
#read file function
def read_file(filename):
    with  open(filename, 'r',encoding='utf-8')as f:
        text = f.read()
        text = text.split('\n')
    return text
def create_worlds_cloud():
    background = np.array(Image.open(PICTURE))
    stopwords_path = path+'stopwords.txt'
    #stopwords = read_file(r"D:/Postgraduate/WARWICK/dissertation/download/stopwords.txt")
#     for item in ["明晚七点","明晚7","明晚7点零食节"]:
#         stopwords.append(item)
#     print(stopwords)
    f = open('D:/Postgraduate/WARWICK/dissertation/download/weiya_e.txt',encoding='utf-8')
    words = f.read()
#     words=cut_the_words() 
    wc = WordCloud(background_color="white",
                   width=1000,
                   height=700,
                   mask=background,
				   stopwords=stopwords,
                   font_path=FONT)
    wc.generate(words)
    wc.to_file('weiya.png')

if __name__ == '__main__':
    create_worlds_cloud()