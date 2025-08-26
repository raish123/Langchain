#importing the module
#splitting the document into chunks based on the length of character or else we can split by words or tokens.
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader


#creating an object of PDFPlumberLoader class
loader = PDFPlumberLoader(
    file_path=r"DocumentLoaders\clip.pdf"
)



#now laoding the document into my working space
document = loader.load() #rtn as list of page document object

#creating an object text splitter class
splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=500, #indicate that each chunks belongs to how many character.
    chunk_overlap=10,
    is_separator_regex=False,
    
)

#now generating chuks through charatertextsplitter object.
chunks = splitter.split_documents(document)

for i,v in enumerate(chunks):
    print('chunk of document: ',i+1)
    print(v.metadata)
    print('*'*60)



