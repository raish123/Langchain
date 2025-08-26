#importing the module
#splitting the document into chunks based on the length of character or else we can split by words or tokens.
from langchain.text_splitter import CharacterTextSplitter

#loading the text file.
with open(file=r"DocumentLoaders\sample.txt",encoding="utf-8") as file:
    #loading the content of text file
    content = file.read()
#print(content)

#creating an object text splitter class
splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=200, #indicate that each chunks belongs to how many character.
    chunk_overlap=10,
    is_separator_regex=False,
    
)

#now generating chuks through charatertextsplitter object.
chunks = splitter.split_text(content)



for i,value in enumerate(chunks):
    print('index of chunk:',i+1)
    print(value)
    print('*'*50)