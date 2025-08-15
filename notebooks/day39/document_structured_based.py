from langchain.text_splitter import RecursiveCharacterTextSplitter, Language

text = """
class Student:
    def __init__(self, roll_no, name, marks1, marks2, marks3, age, gender):
        self.roll_no = roll_no
        self.name = name
        self.marks1 = marks1
        self.marks2 = marks2
        self.marks3 = marks3
        self.age = age
        self.gender = gender

    def display_total_score(self):
        self.total = self.marks1 + self.marks2 + self.marks3
        print("Total Marks:", self.total)
        self.percentage = self.total / 3
        print("Percentage:", self.percentage, "%")

    def __str__(self):
        return f"Roll No: {self.roll_no}, Name: {self.name}, Marks: ({self.marks1}, {self.marks2}, {self.marks3}), Age: {self.age}, Gender: {self.gender}"

students = []

n = int(input("Enter number of students: "))

for i in range(n):
    print(f"\nEnter details of student {i+1}:")
    try:
        roll_no = int(input("Roll No: "))
        name = input("Name: ")

        marks1 = int(input("Marks in Subject 1: "))
        marks2 = int(input("Marks in Subject 2: "))
        marks3 = int(input("Marks in Subject 3: "))

        if not (1 <= marks1 <= 100 and 1 <= marks2 <= 100 and 1 <= marks3 <= 100):
            print("Marks must be between 1 and 100. Student not added.")
            continue

        age = int(input("Age: "))
        gender = input("Gender (male/female): ").lower()

        if gender not in ["male", "female"]:
            print("Gender must be either 'male' or 'female'. Student not added.")
            continue

        s = Student(roll_no, name, marks1, marks2, marks3, age, gender)
        students.append(s)
        print("Student added successfully.")
    
    except ValueError:
        print("Invalid input. Please enter correct details.")

for s in students:
    print(f"\nResults for {s.name} (Roll No: {s.roll_no})")
    print(s) 
    s.display_total_score()
"""

splitter = RecursiveCharacterTextSplitter.from_language(
    language = Language.PYTHON,
    chunk_size = 333,
    chunk_overlap = 0, # helps to reatian context
)

result = splitter.split_text(text)
print(result[3])