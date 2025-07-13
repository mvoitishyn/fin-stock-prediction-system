import numpy as np
import pandas as pd

import borch
from borch import infer, distributions as dist

import matplotlib.pyplot as plt


price_normal = 100
price_student = 100
std = 2
df = 3

simulation_array_normal = []
simulation_array_student = []

for i in range(1000):
    
    price_normal = dist.Normal(price_normal, std).sample().item()
    simulation_array_normal.append(price_normal)

    price_student = dist.StudentT(df, price_student, std).sample().item()
    simulation_array_student.append(price_student)


plt.figure(figsize=(12, 6))
plt.plot(simulation_array_normal, label='Normal Distribution', color='blue', alpha=0.7)
plt.plot(simulation_array_student, label='Student-t Distribution', color='red', alpha=0.7)
plt.xlabel('Time Steps')
plt.ylabel('Price')
plt.title('Random Walk Simulations: Normal vs Student-t Distribution')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("simulations/plots/randomwalk_normal_student.png")


