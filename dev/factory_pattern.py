# implementation of factory design patter
from __future__ import annotations
from abc import ABC, abstractmethod

class Creator(ABC):
    @abstractmethod
    def factory_method(self):
        pass

    def operation(self):
        product = self.factory_method
        return f"Creator: The same creator's code has just worked with {product.operation()}"

class Product(ABC):
    @abstractmethod
    def operation(self) -> str:
        pass

class ShoesCreator(Creator):
    def factory_method(nb):
        return ShoesCreator()

class ShirtCreator(Creator):
    def factory_method(self):
        return ShirtCreator()
    

class ShoesProduct(Product):
    def operation(self) -> str:
        return "Shoes is created"
    
class ShirtProduct(Product):
    def operation(self):
        return "Shirt is created"
    

def client(creator: Creator):
    print(f"Client: I'm not aware of the creator's class, but it still works.\n"
          f"{creator.operation()}", end="")


if __name__ == "__main__":
    # Example usage of the factory design pattern
    print("Test 1 : ")
    client(ShirtProduct())
    print("\nTest 2")
    client(ShoesProduct())