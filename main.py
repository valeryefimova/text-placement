import numpy as np

mystr = "abcdx"
a = np.fromstring(mystr, np.int8) #- ord('a')
print(a)
print(len(a))


s = (a.tostring()).decode("utf-8") [1:2]
print(s)
