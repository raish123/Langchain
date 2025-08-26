#Usually Text Document following hierarchy in Document like have header,paragraph,text line...so on
from langchain.text_splitter import RecursiveCharacterTextSplitter


#loading the text file.
with open(file=r"DocumentLoaders\sample.txt",encoding="utf-8") as file:
    #loading the content of text file
    content = file.read()
#print(content)

#creating an object  of Text Structure Based splitting by using RecursiveCharacterTextSplitter class
splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n","\n"," ",""],
    chunk_size = 200,
    chunk_overlap=0
)


#calling the built in method split_text by using RecursiveCharacterTextSplitter class
lst_chunks = splitter.split_text(content)


for i,value in enumerate(lst_chunks):
    print('index of chunk:',i+1)
    print(value)
    print('*'*50)