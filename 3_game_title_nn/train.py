import pandas
import tflearn
import numpy as np
from tflearn.data_utils import VocabularyProcessor

# import data
data = pandas.read_csv("ign.csv")

# available lables
ratings = [
    'Unbearable',  # 72
    'Disaster', # 4
    'Awful', # 664
    'Painful', # 340
    'Bad', # 1269
    'Mediocre', # 1959
    'Okay', # 2945
    'Good', # 4741
    'Great', # 4773
    'Amazing', # 1804
    'Masterpiece', # 55
]

# select title and score lable from data and use it for traning
x = data['title']
# game titles is transformed to lists of 15 numbers (one to one word)
# use nltk for better transformation and predictions
word_processor = VocabularyProcessor(15)
x = np.array(list(word_processor.fit_transform(x)))

y = []
for label in data['score_phrase']:
    y.append(ratings.index(label))

# save the trained word model to use it in the predict program
word_processor.save("wordprocessor")



# find number of items in each category
def statistics():
    numbersInCategory = [0] * len(ratings)
    for labelNo in y:
        numbersInCategory[labelNo] += 1
    return numbersInCategory


# EXPERIMENT
# dublicate game titles in rating categories whith few examples
# in my understandig of the math, well known examples (categories whith many examples) will have a stonger
#   weight and others down bacause of the partial derivative
numbers = statistics()
numbers_max = max(numbers)
allDataWithDubs = []

for titleNo in range(len(x)):
    for i in range( round(numbers_max / numbers[y[titleNo]]) ):
        allDataWithDubs.append( (x[titleNo], y[titleNo]) )

x = []
y = []

for row in allDataWithDubs:
    x.append(row[0])
    y.append([0]*row[1] + [1] + [0]*(10-row[1]))


# train the model

net = tflearn.input_data([None, 15])
net = tflearn.embedding(net, input_dim=10000, output_dim=128)
net = tflearn.lstm(net, 128, dropout=.8)
net = tflearn.fully_connected(net, 11, activation='softmax')
net = tflearn.regression(net) # adam, 0.001

model = tflearn.DNN(net, tensorboard_verbose=0)
model.fit(x, y, show_metric=True,batch_size=32,
          n_epoch=10)

# save the trained word model to use it in the predict program
model.save("model.tfl")
