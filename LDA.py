# latent Dirichlet allocation on the reviews data

# on first time run

# use the dataframe we made earlier!!!!
import sys
%matplotlib inline 
# only use for Jupyter Notebooks
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
import warnings

warnings.filterwarnings('ignore')

max_words = 500 # use only the top 500 words
k = 3 # set number of topics as 10
n_top_words = 15 # print the top 20 words for each topic

# Import stop word list from file
stpwrdpath ="D:/Postgraduate/WARWICK/dissertation/download/stopwords.txt"
with open(stpwrdpath, 'rb') as fp:
    stopword = fp.read().decode('utf-8')  

# Convert stop word lists to lists  
stpwrdlst = stopword.splitlines()

# helper function to plot topics
# see Grisel et al. 
# https://scikit-learn.org/stable/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html
def plot_top_words(model, feature_names, n_top_words, title):
    fig, axes = plt.subplots(2, 5, figsize=(30, 15), sharex=True)
    axes = axes.flatten()
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]

        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7)
        ax.set_title(f'Topic {topic_idx +1}',
                     fontdict={'fontsize': 30})
        ax.invert_yaxis()
        ax.tick_params(axis='both', which='major', labelsize=20)
        for i in 'top right left'.split():
            ax.spines[i].set_visible(False)
        fig.suptitle(title, fontsize=40)
#     plt.rcParams['font.sans-serif'] = ['SimHei'] # Step 1 Replace the sans-serif font
    plt.rcParams['axes.unicode_minus'] = False   # Step 2 Solve the problem of displaying the negative sign of the negative number of the coordinate axis

    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
    plt.show()

# vectorise the data into word counts
df=pd.read_csv('C:/Users/cmm/爬数据/Weiya1.csv')
# stopword=pd.read_csv('D:/Postgraduate/WARWICK/dissertation/download/stopwords.txt')
# stopword=stopword.to_list()
tf_vectorizer = CountVectorizer(max_features=max_words, stop_words=stpwrdlst)# Creating word pocket data structures
print(tf_vectorizer)
temp = df['content'][df['value']<0]
print(temp.shape)
tf_temp = tf_vectorizer.fit_transform(temp.tolist())
# fit LDA - we'll cover online learning later in the module
lda = LDA(n_components=k, max_iter=5, learning_method='online')#learning_method='none'/'batch'
lda.fit(tf_temp)

# get the list of words (feature names)
tf_feature_names = tf_vectorizer.get_feature_names()

# print the top words per topic
plot_top_words(lda, tf_feature_names, n_top_words, 'Topics in LDA model')