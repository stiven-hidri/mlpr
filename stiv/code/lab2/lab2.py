import numpy
import matplotlib.pyplot as plt

#sepal_length, sepal_width, petal_length, petal_width, family
sepls = []
sepws = []
petls = []
petws = []
classes = []

f=open("../sources/iris.csv", "r")

for line in f:
    parts=line.split(",")

    sepal_length = float(parts[0])
    sepal_width = float(parts[1])
    petal_length = float(parts[2])
    petal_width = float(parts[3])
    cl = parts[4].strip();
    
    sepls.append(sepal_length)
    sepws.append(sepal_width)
    petls.append(petal_length)
    petws.append(petal_width)
    classes.append(cl)

table = numpy.array([sepls, sepws, petls, petws, classes], dtype=str)
labes = numpy.array(["Iris-setosa","Iris-versicolor","Iris-virginica"])

# plt.hist(table[0,table[4]=="Iris-setosa"], bins = 20, density=True, ec='black', color='#080000')
# plt.hist(table[0,table[4]=="Iris-versicolor"], bins = 20, density=True, ec='green', color='#900000')
# plt.hist(table[0,table[4]=="Iris-virginica"], bins = 20, density=True, ec='blue', color='#007000')
# plt.show();

# plt.hist(table[1,table[4]=="Iris-setosa"], bins = 20, density=True, ec='black', color='#080000')
# plt.hist(table[1,table[4]=="Iris-versicolor"], bins = 20, density=True, ec='green', color='#900000')
# plt.hist(table[1,table[4]=="Iris-virginica"], bins = 20, density=True, ec='blue', color='#007000')
# plt.show();

# plt.hist(table[2,table[4]=="Iris-setosa"], bins = 20, density=True, ec='black', color='#080000')
# plt.hist(table[2,table[4]=="Iris-versicolor"], bins = 20, density=True, ec='green', color='#900000')
# plt.hist(table[2,table[4]=="Iris-virginica"], bins = 20, density=True, ec='blue', color='#007000')
# plt.show();

# plt.hist(table[3,table[4]=="Iris-setosa"], bins = 20, density=True, ec='black', color='#080000')
# plt.hist(table[3,table[4]=="Iris-versicolor"], bins = 20, density=True, ec='green', color='#900000')
# plt.hist(table[3,table[4]=="Iris-virginica"], bins = 20, density=True, ec='blue', color='#007000')
# plt.show();

d = numpy.array([sepls, sepws, petls, petws], dtype=float)
print(d-d.mean(1).reshape(d.shape[0], 1))