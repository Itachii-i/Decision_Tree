import pickle

class Employee:
    def __init__(self,eno,ename, esal):
        self.eno = eno
        self.ename = ename
        self.esal = esal

    def display(self):
        print("Number is:", self.eno)
        print("Name is:", self.ename)
        print("Salary is:", self.esal)

with open("emp.dat", "wb") as f:
    e = Employee(100, "Daniel", 10000)
    pickle.dump(e,f)
    print("Pickling of Employee object completed")
    e.display()
print()

with open("emp.dat", "rb") as f:
    obj = pickle.load(f)
    print("Print Employee info after unpickling")
    obj.display()
 