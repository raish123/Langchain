from langchain.text_splitter import RecursiveCharacterTextSplitter,Language


text = """
# 📘 Sample Markdown Document  

## 1. Introduction  
This is a sample Markdown document.  
It shows how to structure **headings, lists, tables, and code blocks**.  

---

## 2. Features  

### ✅ Lists  
- Bullet point 1  
- Bullet point 2  
  - Sub bullet point  
- Bullet point 3  

### 🔢 Numbered List  
1. First item  
2. Second item  
3. Third item  

---

## 3. Table Example  

| Feature        | Description                          | Status   |  
|----------------|--------------------------------------|----------|  
| Heading        | Title of the section                 | ✅ Done  |  
| List           | Bullet or numbered points            | ✅ Done  |  
| Table          | Organize information in rows/columns | ✅ Done  |  
| Code Block     | Show formatted code                  | ✅ Done  |  

---

## 4. Code Block Example  

```python
# Python Example
def greet(name: str):
    return f"Hello, {name}!"

print(greet("Shaik"))


"""


#creating an object of spliiter 
splitter = RecursiveCharacterTextSplitter.from_language(
    #internally they will used his own seprators based on language of programming
    language=Language.MARKDOWN,
    chunk_size = 153,
    chunk_overlap = 0,
   
    
)

#perform the splitting
lst_chunks = splitter.split_text(text)

for i,value in enumerate(lst_chunks):
    print('index of chunk:',i+1)
    print(value)
    print('*'*50)