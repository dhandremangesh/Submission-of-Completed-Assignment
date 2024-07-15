# parent class
class Vehicle:
    traffic_light = "Green"
    speed_limit = 60
    def start_engine(self):
        print("engine started")

# car class
class Car(Vehicle):
    def __init__(self,make):
        self.make = make

    def start_engine(self):
        return super().start_engine()
    
# bike class 
class Bike(Vehicle):
    def __init__(self,bike_type):
        self.type = bike_type   

    def start_engine(self):
        return super().start_engine()
    


if __name__=="__main__":

    #instance in car class
    car1 = Car("Maruti")
    print(car1.make)

    #instance in bike class
    bike1 = Bike("hero")
    print(bike1.type)

    #polymorphism
    car1.start_engine()
    bike1.start_engine()