import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt

# import dataset
data = pd.read_csv("challenge_dataset.txt")
x_data = data[['in']].values
y_data = data[['out']].values

# train model
linaer_reg = linear_model.LinearRegression()
linaer_reg.fit(x_data, y_data)

# line constand
print "f(x) = "+str(linaer_reg.coef_[0][0])+"x + "+str(linaer_reg.intercept_[0]) # f(x) = 1.19303364419x + -3.89578087831

# plot data and regression line
plt.scatter(x_data, y_data)
plt.plot( x_data, linaer_reg.predict(x_data) )
plt.show()



# calculate all offsets
offsets = []
for index in range(len(x_data)):
    x_value = x_data[index][0]
    y_value = y_data[index][0]
    y_predicted = linaer_reg.predict(x_value)

    offset = abs(y_predicted - y_value)[0][0]
    offsets.append(offset)


# show offsets in graph
bar_order = range(len(offsets))
plt.bar(bar_order, sorted(offsets), 0.2)
plt.show()

# calculate mean
print sum(offsets) / len(offsets) # 2.19424539883
