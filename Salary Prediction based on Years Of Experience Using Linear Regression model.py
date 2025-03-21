import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

years_of_experience = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]).reshape(-1, 1) 
salary = np.array([30000, 34000, 41000, 47000, 55000, 60000, 68000, 79000, 85000, 97000, 110000, 125000, 140000, 160000, 185000]) 

model = LinearRegression()
model.fit(years_of_experience, salary)

salary_predict = model.predict(years_of_experience)

choice = input("Would you like to predict salary (enter '1') or years of experience (enter '2')? ").strip().lower()

if choice == '1':
    input_experience = float(input("Enter the years of experience: "))
    predicted_salary = model.predict([[input_experience]])
    print(f"Predicted salary for {input_experience} years of experience is: {predicted_salary[0]}")
elif choice == '2':
    input_salary = float(input("Enter the salary: "))
    predicted_experience = (input_salary - intercept) / slope
    print(f"Predicted years of experience for a salary of {input_salary} is: {predicted_experience}")
else:
    print("Invalid choice!")

plt.scatter(years_of_experience, salary, marker='*', color='red', label='Actual Data')
plt.plot(years_of_experience, salary_predict, 'b-', label='Predicted Data')

if choice == 'experience':
    plt.scatter(input_experience, predicted_salary, marker='o', color='green', label='Predicted Point')
elif choice == 'salary':
    plt.scatter(predicted_experience, input_salary, marker='o', color='green', label='Predicted Point')

plt.xlabel('Years of Experience')
plt.ylabel('Salary (in thousands)')
plt.title('Years of Experience vs. Salary')
plt.legend()
plt.grid(True)
plt.show()
