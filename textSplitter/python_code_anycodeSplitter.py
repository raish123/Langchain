from langchain.text_splitter import RecursiveCharacterTextSplitter,Language


text = """
# 🚀 Sample Python Program using Class, Methods, and Objects

class Car:
    # Constructor method (called when object is created)
    def __init__(self, brand, model, year):
        self.brand = brand
        self.model = model
        self.year = year
    
    # Instance method
    def car_info(self):
        return f"{self.year} {self.brand} {self.model}"
    
    # Another instance method
    def start_engine(self):
        return f"The engine of {self.brand} {self.model} is now running 🚗💨"

# ---- Main Program ----
if __name__ == "__main__":
    # Creating objects of Car class
    car1 = Car("Toyota", "Corolla", 2022)
    car2 = Car("Tesla", "Model 3", 2023)

    # Calling methods using objects
    print(car1.car_info())       # Output: 2022 Toyota Corolla
    print(car1.start_engine())   # Output: Engine running
    
    print(car2.car_info())       # Output: 2023 Tesla Model 3
    print(car2.start_engine())   # Output: Engine running

"""


#creating an object of spliiter 
splitter = RecursiveCharacterTextSplitter.from_language(
    #internally they will used his own seprators based on language of programming
    language=Language.PYTHON,
    chunk_size = 300,
    chunk_overlap = 0,
   
    
)

#perform the splitting
lst_chunks = splitter.split_text(text)

for i,value in enumerate(lst_chunks):
    print('index of chunk:',i+1)
    print(value)
    print('*'*50)