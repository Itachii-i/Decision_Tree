class Student:
    def __init__(self, number,name,location):
        self.number=number
        self.name=name
        self.location=location
    def details(self):
        print("Student number is:", self.number)
        print("student name is:", self.name)
        print("Student location is:", self.location)

s1 = Student (101,"Mario","Canada")
s2 = Student (102,"Stacy","UK")

s1.details()
s2.details()
