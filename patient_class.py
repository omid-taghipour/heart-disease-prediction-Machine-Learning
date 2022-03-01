import numpy
import numpy as np


class Patient:
    target = "Not defined yet"

    def __init__(self):
        print('>>Enter the patient information<<\n')
        self.age = int(input("Enter the patient age: "))
        self.sex = int(input("Enter patient gender(0 = Female, 1 = Male): "))
        self.cp = int(input("Enter the patient chest pain type(between 0 and 4. 0 = No pain , 4 = High pain): "))
        self.trestbps = int(input("Enter the patient resting blood pressure: "))
        self.chol = int(input("Enter the patient serum cholesterol in mg/dl: "))
        self.fbs = int(input("Enter the patient fasting blood sugar (if > 120 mg/dl enter 1 otherwise enter 0): "))
        self.restecg = int(input("Enter the patient resting electron-cartographic results (0,1,2): "))
        self.thalach = int(input("Enter the patient maximum heart rate achieved: "))
        self.exang = int(input("Enter the patient exercise induced angina (0,1): "))
        self.oldpeak = float(input("Enter the patient ST depression induced by exercise relative to rest: "))
        self.slope = int(input("Enter the patient slope of the peak exercise ST segment: "))
        self.ca = int(input("Enter the patient number of major vessels (0-3) colored by fluoroscopy: "))
        self.thal = int(input("Enter the patient thalamus (3 = normal; 6 = fixed defect; 7 = reversible defect): "))

    def __str__(self):
        return f"Patient {self.id}:\n " \
               f"Age\tSex\tCp\ttrestbps\tChol\tFbs\trestecg\tthalach\texang\toldpeak\tslope\tCA\tthal\n" \
               f"{self.age}\t{self.sex}\t{self.cp}\t{self.trestbps}\t{self.chol}\t{self.fbs}\t{self.restecg}\t{self.thalach}\t{self.exang}\t{self.oldpeak}" \
               f"\t{self.slope}\t{self.ca}\t{self.thal}\n\n"

    def return_as_list(self):
        # returning values of attributes as a numpy reshaped array
        return np.asarray(
            [self.age, self.sex, self.cp, self.trestbps, self.chol, self.fbs, self.restecg, self.thalach, self.exang,
             self.oldpeak, self.slope, self.ca, self.thal]).reshape(1, -1)
