import tflearn
from tflearn.data_utils import VocabularyProcessor
import sys

# get all arguments
games_to_predict = sys.argv[1:]

if len(games_to_predict) is 0:
    print("Type games to predict when you run the script af arguments")
    exit()

# create and load vocal vector model
word_processor = VocabularyProcessor(15)
word_processor.restore("wordprocessor")

# create and load ML model
net = tflearn.input_data([None, 15])
net = tflearn.embedding(net, input_dim=10000, output_dim=128)
net = tflearn.lstm(net, 128, dropout=.8)
net = tflearn.fully_connected(net, 11, activation='softmax')
net = tflearn.regression(net) # adam, 0.001

model = tflearn.DNN(net, tensorboard_verbose=0)
model.load("model.tfl")

# use labels for output
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

# make a prediction for each parsed in argument, and fin the label for the output the model is most confident on
for title in games_to_predict:

    predicted = model.predict(list(word_processor.transform([title])))[0]
    max_value = max(predicted)
    max_index = predicted.index(max_value)
    # Code for debugging
    # print(predicted, max_value, max_index)

    print("I think "+title+" is a "+ratings[max_index].lower()+" game!")
