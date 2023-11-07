from tkinter import *
from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
import string
import  os
import pyttsx3
import threading
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfTransformer,TfidfVectorizer
from sklearn.pipeline import Pipeline

bot=ChatBot('Bot')
trainer=ListTrainer(bot)
df=pd.read_csv('dialogs.txt',sep='\t')
a=pd.Series(df.columns)
a = a.rename({0: df.columns[0],1: df.columns[1]})
b = {'Questions':'Hi','Answers':'hello'}
c = {'Questions':'Hello','Answers':'hi'}
d= {'Questions':'how are you','Answers':"i'm fine. how about yourself?"}
e= {'Questions':'how are you doing','Answers':"i'm fine. how about yourself?"}
df = df.append(a,ignore_index=True)
df.columns=['Questions','Answers']
df = df.append([b,c,d,e],ignore_index=True)
#print(df.to_string())
#df.to_csv('chatdata.csv',index=False)
trainer.train(list(df))
def cleaner(x):
    return [a for a in (''.join([a for a in x if a not in string.punctuation])).lower().split()]

Pipe = Pipeline([
    ('bow',CountVectorizer(analyzer=cleaner)),
    ('tfidf',TfidfTransformer()),
    ('classifier',DecisionTreeClassifier())
])
Pipe.fit(df['Questions'],df['Answers'])

def botReply():
    question=questionField.get()
    question = question.capitalize()
    answer = Pipe.predict([question])[0]
    textarea.insert(END,'You:'+question+'\n \n')
    textarea.insert(END,'Bot:'+str(answer)+'\n\n')
    pyttsx3.speak(answer)

    questionField.delete(0, END)

root=Tk()

root.geometry('500x570+100+30')
root.title('ChatBot')
root.config(bg='deep pink')

logoPic =PhotoImage(file='pic.png')

logopiclabel =Label(root,image=logoPic,bg='deep pink')
logopiclabel.pack()

centerFrame=Frame(root)
centerFrame.pack()

scrollbar = Scrollbar(centerFrame)
scrollbar.pack(side=RIGHT)

textarea =Text(centerFrame,font=('times new roman',20,'bold'),height=10,yscrollcommand=scrollbar.set,wrap='word')
textarea.pack(side=LEFT)
scrollbar.config(command=textarea.yview)

questionField=Entry(root,font=('verdana',20,'bold'))
questionField.pack(pady=15,fill=X)

askPic =PhotoImage(file='ask.png')


askButton=Button(root,image=askPic,command=botReply)
askButton.pack()

def click(event):

  askButton.invoke()
  
root.bind('<Return>',click)

root.mainloop()
