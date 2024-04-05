"""For small term testing only"""
class SimpleService:
    def __init__(self, message="Hello from SimpleService!"):
        self.message = message

    def greet(self):
        print(self.message)
