import pyltp
from pyltp import Segmentor
from pyltp import SentenceSplitter
from pyltp import Postagger
import re
import numpy as np
import pandas as pd
import os

# visualization
import matplotlib.pyplot as plt
import seaborn as sns




# delete #words#
def get_new(x):
    x = re.sub('#\w+#', '', x)
    return x

# Read files, file read functions
def read_file(filename):
    with  open(filename, 'r',encoding='utf-8')as f:
        text = f.read()
        # return list 
        text = text.split('\n')
    return text


# Tokenization
def tokenize(sentence):
    # Loading models
    segmentor = Segmentor()  # Initialization Example
    # Loading models
    LTP_DIR = r'D:\Program Files\pycharm\ltp_data_v3.4.0\ltp_data_v3.4.0'  # the path of ltp model
    cws_model_path = os.path.join(LTP_DIR, 'cws.model')  # the path of `cws.model`
    segmentor = Segmentor()  # Initialization example
    segmentor.load(cws_model_path)  # Loading models
    # get segment
    words = segmentor.segment(sentence)
    words = list(words)
    # release model
    segmentor.release()
    return words

# Lexical annotation
def postagger(words):
    # Initialization example
    postagger = Postagger()
    # loading model
    postagger.load(r'D:\Program Files\pycharm\ltp_data_v3.4.0\ltp_data_v3.4.0\pos.model')
    # Lexical annotation
    postags = postagger.postag(words)
    # release model
    postagger.release()
    # return list
    postags = [i for i in postags]
    return postags

# Forming a tuple of words and lexical forms
def intergrad_word(words,postags):
    # Zip algorithm, two-by-two matching
    pos_list = zip(words,postags)
    pos_list = [ w for w in pos_list]
    return pos_list

#delete stop words
def del_stopwords(words):
    # read stop words list
    stopwords = read_file(r"stopwords.txt")
    # get the new words after deleting stop words
    new_words = []
    for word in words:
        if word not in stopwords:
            new_words.append(word)
    return new_words

# Get the words for the six weights and return the list on request
def weighted_value(request):
    result_dict = []
    if request == "one":
        result_dict = read_file(r"degree_dict\most.txt")
    elif request == "two":
        result_dict = read_file(r"degree_dict\very.txt")
    elif request == "three":
        result_dict = read_file(r"degree_dict\more.txt")
    elif request == "four":
        result_dict = read_file(r"degree_dict\ish.txt")
    elif request == "five":
        result_dict = read_file(r"degree_dict\insufficiently.txt")
    elif request == "six":
        result_dict = read_file(r"degree_dict\inverse.txt")
    elif request == 'posdict':
        result_dict = read_file(r"emotion_dict\pos_all_dict.txt")
    elif request == 'negdict':
        result_dict = read_file(r"emotion_dict\neg_all_dict.txt")
    else:
        pass
    return result_dict

print("reading sentiment dict .......")
# read sentiment dictionary
posdict = weighted_value('posdict')
negdict = weighted_value('negdict')
# read degree dictionary
# weight = 8
mostdict = weighted_value('one')
# weight = 6
verydict = weighted_value('two')
# weight =4
moredict = weighted_value('three')
# weight = 2
ishdict = weighted_value('four')
# weight =0.5
insufficientdict = weighted_value('five')
# weight = -1
inversedict = weighted_value('six')

# Different weights are given to different degree adverbs
def match_adverb(word,sentiment_value):
    # Most weight
    if word in mostdict:
        sentiment_value *= 8
    # Very weight
    elif word in verydict:
        sentiment_value *= 6
    # More weight 
    elif word in moredict:
        sentiment_value *= 4
    # ish weight
    elif word in ishdict:
        sentiment_value *= 2
    # insufficient weight 
    elif word in insufficientdict:
        sentiment_value *= 0.5
    # inverse weight 
    elif word in inversedict:
        sentiment_value *= -1
    else:
        sentiment_value *= 1
    return sentiment_value

# Score each bullet comments
def single_sentiment_score(text_sent):
    sentiment_scores = []
    sentences = text_sent
    for sent in sentences:
        words = tokenize(sent)
        seg_words = del_stopwords(words)
        # i，s Record where sentiment and degree words occur
        i = 0   # Record scanned word positions
        s = 0   # Recording the location of emotive words
        poscount = 0 # record number of positive emotion words 
        negcount = 0 # record Number of negative emotion words 
        # Find emotion words one by one
        for word in seg_words:
            # If it's a positive word
            if word in posdict:
                poscount += 1  # Number of emotive words plus one
            # Look for degree adverbs in front of emotive words
                for w in seg_words[s:i]:
                    poscount = match_adverb(w,poscount)
                s = i+1 # Recording emotion word positions
            # If it is a negative emotion word
            elif word in negdict:
                negcount +=1
                for w in seg_words[s:i]:
                    negcount = match_adverb(w,negcount)
                s = i+1
            # If it ends with an '!' or '?', it indicates the end of the sentence and looks for the emotion word 
			# before the '!' or '?' in reverse order, with a weight of +4
            elif word =='!' or  word =='！' or word =='?' or word == '？':
                for w2 in seg_words[::-1]:
                    # If positive, poscount+4
                    if w2 in posdict:
                        poscount += 4
                        break
                    # If negative, poscount+4
                    elif w2 in negdict:
                        negcount += 4
                        break
            i += 1 # Positioning of emotive words
        # Calculating sentiment values
        sentiment_score = poscount - negcount
        sentiment_scores.append(sentiment_score)
        # Check the sentiment value of each sentence
        # print('Sub-clause marks:',sentiment_score)
    sentiment_sum = 0
    for s in sentiment_scores:
        #Calculate the total score of a bullet comment 
        sentiment_sum +=s
    return sentiment_sum

# Parses all the bullet comments in research_data.xlsx
# and returns a list of elements in the (content, score) tuple
def run_score(contents):
    
    print("running scores")
    scores_list = []
    for content in contents:
        if content !='':
            score = single_sentiment_score(content)  # Call the function for each bullet comment to get the score
            scores_list.append((content, score)) 
    return scores_list


if __name__ == '__main__':
    print('Processing........')
	df = pd.read_excel("D:/Postgraduate/WARWICK/dissertation/research data.xlsx", sheet_name='Weiya', engine='openpyxl')
    df1=df.loc[:,["comment"]]

    df1["comment"] = df1["comment"].astype("str")
    df1["comment"] = df1["comment"].apply(get_new)
	#df1['comment'] = df1['comment'].str.extract(r"([\u4e00-\u9fa5]+)")
	#df = df.dropna()  # Pure emoticons deleted directly
    df1 = df1[df1["comment"].apply(len)>=4]
    df1=df1.dropna()
    print(df1.shape)
    sentences=df1["comment"].tolist()


	print(sentences)
    scores = run_score(sentences)
    al= []
    for score in scores:
        print('value',score[1])
        if score[1] < 0:
            print('category: negative')
            s = 'negative'
        elif score[1] == 0:
            print('category: neutral')
            s = 'neutral'
        else:
            print('category: positive')
            s = 'positive'
        al.append((score[0], score[1], s))
        print('content',score[1])
        df_new = pd.DataFrame(list(al))
    i = 0
    print(df_new)
    df_new.to_csv("C:/Users/cmm/爬数据/weiya1.csv", encoding="utf_8_sig")

    print('succeed.......')