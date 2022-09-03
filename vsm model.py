import tkinter as tk
from turtle import title
import nltk
import string
import os
from nltk.tokenize import word_tokenize
from collections import defaultdict
import numpy as np
import math
from nltk.stem import PorterStemmer
from numpy import dot
from numpy.linalg import norm
stoplist = []

#storing stopwords in an list
file = open(r'C:\Users\Shahriyaar\OneDrive\Desktop\ir assigment 2\Stopword-List.txt', 'r')
text = file.readlines()

for line in text:
  stoplist.append(line.strip())

files=os.listdir(r'C:\Users\Shahriyaar\OneDrive\Desktop\ir assigment 2\Abstracts')
allword=[]
inverted_index=[]
idf=defaultdict(int)
postings={}


#creating dictionary
for i in range(449): 
  inverted_index.append([])
# in this loop we open each file extract tokens, do stemming on them, create a posting list which contain each words with its docid and 
# term frequency, creted another dictionary with contain idf of each word/token. 
for i in range(448):

  docid = int(files[i].rstrip(".txt"))
  
  #reading from files
  with open (os.path.join(r'C:\Users\Shahriyaar\OneDrive\Desktop\ir assigment 2\Abstracts',files[i])) as fin:
      words = fin.read()
      
      #removing punctuations
      word=words.replace("-","  ")
      words=words.translate(words.maketrans('', '', string.punctuation))
  #tokenizing using NLTK
  
  tokens=word_tokenize(words.lower())
  stemmer=PorterStemmer()
  for i in range(len(tokens)):
    tokens[i]=stemmer.stem(tokens[i])
    
  for word in tokens:
    if word not in postings:
      postings[word]={}
    #removing stopwords from words
    if word not in stoplist and word not in inverted_index[docid]:
      if word not in allword:
        allword.append(word)

      inverted_index[docid].append(word)
      #assigning term frequency with each word for each document
      postings[word][docid]=tokens.count(word)/len(tokens)
      #calculating idf for each term
      idf[word]=math.log(448/(len(postings[word])+1),2)

      
def query_proc():
  str=entry1.get() 
  doc_vec=list()
  doc_vec.append(None)
  for i in range(1,449):
    doc_vec.append(np.zeros((len(allword),)))
    for j in range(len(allword)):
      if allword[j] in inverted_index[i]:
        doc_vec[i][j]=(postings[allword[j]][i]) * (idf[allword[j]])
  #tokenizing the entered query
  q_tokens=str.replace("-","  ")
  q_tokens=word_tokenize(q_tokens.lower())
  
  terms=[]
  for i in range(len(q_tokens)):
    if q_tokens[i] not in stoplist:

      terms.append(stemmer.stem(q_tokens[i]))
        
  q_vec=np.zeros((len(allword),))
  for i in range(len(allword)):
    if allword[i] in terms:
      q_vec[i]=terms.count(allword[i]) * (idf[allword[i]])


  print(q_vec)

  scores={}
  for i in range(len(terms)):
    for j in range(1,449):
      if j in postings[terms[i]]:
        sc=dot(q_vec, doc_vec[j])/(norm(q_vec)*norm(doc_vec[j]))
        if sc>0.001 and j not in scores:
          scores[j]=sc
  scores1=sorted(scores.items(), key =lambda x:(x[1], x[0]),reverse=True)
  sorted1=[i[0] for i in scores1]
  label1 = tk.Label(root, text=sorted1)
  label1.configure(bg='#11D3A7') #tkinter lines to add labels to add scores and doc ids
  canvas1.create_window(400, 430,width=1200, window=label1)
  print(sorted(scores.items(), key =lambda x:(x[1], x[0]),reverse=True))

#tkinter commands for gui
root= tk.Tk()
root.title('vector space model')
#set the window color
root.configure(bg='yellow')
canvas1 = tk.Canvas(root, width = 1200, height = 500)
canvas1.configure(bg='#11D3A7')
canvas1.pack()

entry1 = tk.Entry (root) 
canvas1.create_window(600, 240,width=250,height=28, window=entry1)

button1 = tk.Button(text='get query results',font=("Times New Roman", 15), command=query_proc)
canvas1.create_window(600, 280,width=150,height=28, window=button1)

root.mainloop()