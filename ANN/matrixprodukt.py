import time
import numpy as np

a = np.random.rand(1000000)
b = np.random.rand(1000000)

before = time.time()
c = np.dot(a, b)
after = time.time()

print("AB = " + str(round(c, 3)))
print("Tiden det tog med en matrix operation: " + str(round(1000*(after-before), 3)) + " ms")

c = 0
before = time.time()
for i in range(1000000):
    c += a[i] * b[i]

after = time.time()
print("AB = " + str(round(c, 3)))
print("Tiden det tog med en løkke: " + str(round(1000*(after-before), 3)) + " ms")
# Jeg har ganget med 1000 for at konverterer sekunder til millisekunder