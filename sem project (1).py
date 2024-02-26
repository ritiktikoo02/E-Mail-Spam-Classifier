#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv(r"C:\Users\HP\Downloads\sms_spam.csv")


# In[3]:


df.sample(5)


# In[4]:


df. head(5)


# In[5]:


df.shape


# # cleaning
# 

# In[6]:


df.info


# In[7]:


df.sample(5)


# In[8]:


df.rename(columns={'type':'target'},inplace=True)


# In[9]:


df.sample(5)


# In[10]:


pip install -U scikit-learn


# In[11]:


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df['target'] = encoder.fit_transform(df['target'])


# In[12]:


df.sample(5)


# In[13]:


df.isnull().sum()


# In[14]:


df.duplicated().sum()


# In[15]:


df = df.drop_duplicates(keep='first')


# # eda

# In[16]:


df['target'].value_counts()


# In[17]:


pip install matplotlib


# In[18]:


import matplotlib.pyplot as plt
colors = ['green', 'red']
plt.pie(df['target'].value_counts(), labels=['ham','spam'],autopct="%0.2f",colors=colors)
plt.show()


# In[19]:


pip install nltk


# In[20]:


import nltk


# In[21]:


nltk.download('punkt')


# In[22]:


df['characters']=df['text'].apply(len)


# In[23]:


df.head(6)


# In[24]:


df['num_words'] = df['text'].apply(lambda x:len(nltk.word_tokenize(x)))


# In[25]:


df.head(6)


# In[26]:


df['num_sent'] = df['text'].apply(lambda x:len(nltk.sent_tokenize(x)))


# In[27]:


df.head(6)


# In[28]:


df[df['target'] == 0][['characters','num_words','num_sent']].describe()


# In[29]:


df[df['target'] == 1][['characters','num_words','num_sent']].describe()


# In[30]:


pip install seaborn


# In[31]:


import seaborn as sns
plt.figure(figsize=(10,6))
sns.histplot(df[df['target'] == 0]['characters'],color='red')
sns.histplot(df[df['target'] == 1]['characters'],color='green')


# In[32]:


plt.figure(figsize=(10,6))
sns.histplot(df[df['target'] == 0]['num_words'],color='red')
sns.histplot(df[df['target'] == 1]['num_words'],color='green')


# In[33]:


sns.pairplot(df,hue='target')


# In[34]:


sns.heatmap(df.corr(),annot=True,cmap="crest")


# #   DATA PREPROCESSING

# In[35]:


import nltk


# In[36]:


nltk.download('stopwords')


# In[37]:


from nltk.corpus import stopwords


# In[38]:


import string
string.punctuation


# In[39]:


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    text = y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
    
            
    return " ".join(y)


# In[40]:


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
ps.stem('loving')


# In[41]:


transform_text("I'm gonna be home soon and i don't want to talk about this stuff anymore tonight, k? I've cried enough today.")


# In[42]:


df['transformed_text'] = df['text'].apply(transform_text)


# In[43]:


df.head(6)


# In[44]:


pip install wordcloud


# In[45]:


from wordcloud import WordCloud
wc = WordCloud(width=550,height=550,min_font_size=10,background_color='black')


# In[46]:


spam_wc = wc.generate(df[df['target'] == 1]['transformed_text'].str.cat(sep=" "))


# In[47]:


plt.figure(figsize=(15,6))
plt.imshow(spam_wc)


# In[48]:


ham_wc = wc.generate(df[df['target'] == 0]['transformed_text'].str.cat(sep=" "))
plt.figure(figsize=(15,6))
plt.imshow(ham_wc)


# In[49]:


from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
cv = CountVectorizer()
tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(df['transformed_text']).toarray()


# In[50]:


#from sklearn.preprocessing import MinMaxScaler
#scaler = MinMaxScaler()
#X = scaler.fit_transform(X)


# In[51]:


y = df['target'].values


# In[52]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)


# In[53]:


from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score
gnb = GaussianNB()
mnb = MultinomialNB()
bnb = BernoulliNB()


# In[54]:


gnb.fit(X_train,y_train)
y_pred1 = gnb.predict(X_test)
print(accuracy_score(y_test,y_pred1))
print(confusion_matrix(y_test,y_pred1))
print(precision_score(y_test,y_pred1))


# In[55]:


mnb.fit(X_train,y_train)
y_pred2 = mnb.predict(X_test)
print(accuracy_score(y_test,y_pred2))
print(confusion_matrix(y_test,y_pred2))
print(precision_score(y_test,y_pred2))


# In[56]:


bnb.fit(X_train,y_train)
y_pred3 = bnb.predict(X_test)
print(accuracy_score(y_test,y_pred3))
print(confusion_matrix(y_test,y_pred3))
print(precision_score(y_test,y_pred3))


# In[57]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier


# In[58]:


svc = SVC(kernel='sigmoid', gamma=1.0)
knc = KNeighborsClassifier()
mnb = MultinomialNB()
dtc = DecisionTreeClassifier(max_depth=5)
lrc = LogisticRegression(solver='liblinear', penalty='l1')
rfc = RandomForestClassifier(n_estimators=50, random_state=2)
abc = AdaBoostClassifier(n_estimators=50, random_state=2)
bc = BaggingClassifier(n_estimators=50, random_state=2)
etc = ExtraTreesClassifier(n_estimators=50, random_state=2)
gbdt = GradientBoostingClassifier(n_estimators=50,random_state=2)


# In[59]:


clfs = {
    'SVC' : svc,
    'KN' : knc, 
    'NB': mnb, 
    'DT': dtc, 
    'LR': lrc, 
    'RF': rfc, 
    'AdaBoost': abc, 
    'BgC': bc, 
    'ETC': etc,
    'GBDT':gbdt,
   
}


# In[60]:


def train_classifier(clf,X_train,y_train,X_test,y_test):
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred)
    return accuracy,precision
    


# In[61]:


train_classifier(svc,X_train,y_train,X_test,y_test)


# In[62]:


accuracy_scores = []
precision_scores = []

for name,clf in clfs.items():
    
    current_accuracy,current_precision = train_classifier(clf, X_train,y_train,X_test,y_test)
    
    print("For ",name)
    print("Accuracy - ",current_accuracy)
    print("Precision - ",current_precision)
    
    accuracy_scores.append(current_accuracy)
    precision_scores.append(current_precision)


# In[66]:


performance_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy':accuracy_scores,'Precision':precision_scores}).sort_values('Precision',ascending=False)
performance_df


# In[67]:


# Voting Classifier
svc = SVC(kernel='sigmoid', gamma=1.0,probability=True)
mnb = MultinomialNB()
etc = ExtraTreesClassifier(n_estimators=50, random_state=2)

from sklearn.ensemble import VotingClassifier


# In[68]:


voting = VotingClassifier(estimators=[('svm', svc), ('nb', mnb), ('et', etc)],voting='soft')


# In[69]:


voting.fit(X_train,y_train)


# In[70]:


y_pred = voting.predict(X_test)
print("Accuracy",accuracy_score(y_test,y_pred))
print("Precision",precision_score(y_test,y_pred))


# In[71]:


# Applying stacking
estimators=[('svm', svc), ('nb', mnb), ('et', etc)]
final_estimator=RandomForestClassifier()


# In[72]:


from sklearn.ensemble import StackingClassifier


# In[74]:


clf = StackingClassifier(estimators=estimators, final_estimator=final_estimator)


# In[75]:


clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Accuracy",accuracy_score(y_test,y_pred))
print("Precision",precision_score(y_test,y_pred))


# In[ ]:





# In[ ]:





# In[ ]:




