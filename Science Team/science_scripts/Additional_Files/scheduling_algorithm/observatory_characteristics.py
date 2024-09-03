
class ObservatoryCharacteristics: 

    def __init__(self, name:str, latitude:float, longitude:float, timezone:str, diameter:float, elevation:int, light_pollution:int):
        self.name = name
        self.latitude = latitude
        self.longitude = longitude
        self.timezone = timezone
        self.diameter = diameter
        self.elevation = elevation
        self.light_pollution = light_pollution

def main():
    obsChr = ObservatoryCharacteristics("test",1,1,"tz",5,100,1)
    print(obsChr.name)

if __name__ == "__main__":
    main()
     