import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#add your linear dataset
data = pd.read_csv("Your dataset here.csv")

# a gradient descent function for y=ax+b:
def grad_des(points, lr_m , lr_b , epochs):
    m = np.random.randn()
    b = np.random.randn()
    
    for _ in range(epochs):
        m_grad = 0
        b_grad = 0
        for i in range(len(points)):
            x = points.iloc[i].Temperature
            y = points.iloc[i].Ice_Cream_Profits
            m_grad += -(2/len(points)) * x * (y - (m * x + b))
            b_grad += -(2/len(points)) * (y - (m * x + b))
        
        m -= m_grad * lr_m
        b -= b_grad * lr_b
        
    return m, b


lr_m = 0.000005
lr_b = 0.000005
epochs = 300

m, b = grad_des(data, lr_m , lr_b, epochs)


plt.scatter(data.x, data.y, color='black')
plt.plot(data.Temperature, m * data.Temperature + b, color='red')
plt.xlabel('X label')
plt.ylabel('Y label')
plt.title('Linear Regression')
plt.show()
