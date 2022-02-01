import numpy as np
import matplotlib.pyplot as plt 
import json
from sklearn.model_selection import train_test_split
import random

def get_input_depression(file_name):
    input = []
    for name in file_name:
        with open(file=name) as json_file:
            data = json.load(json_file)
            input = []
            for entry in data:
                input.append([entry['tone_score']['neg'],entry['tone_score']['neu'],entry['tone_score']['pos'],entry['tone_score']['compound']])
    return input


def get_input_not_depression(file_name):
    input = []
    with open(file=file_name) as json_file:
        data = json.load(json_file)
        input = []
        for entry in data:
            input.append([entry['tone_score']['neg'],entry['tone_score']['neu'],entry['tone_score']['pos'],entry['tone_score']['compound']])
    return input

input_depression = get_input_depression(['depression_training_data.json','depression_training_data_comments.json'])
input_not_depression = get_input_not_depression('happy_training_data.json')
input = np.array(input_depression+input_not_depression)
output_list = []
for k in input_depression:
    output_list.append([float(0)])
for k in input_not_depression:
    output_list.append([float(1)])
output=np.array(output_list)
X_train,X_test,y_train,y_test = train_test_split(input,output,test_size = 0.1)

# create NeuralNetwork class
class NeuralNetwork:

    # intialize variables in class
    def __init__(self,file_name):
        with open(file=file_name) as json_file:
            data = json.load(json_file) 
            self.inputs=np.array(data['inputs'])
            self.outputs=np.array(data['outputs'])
            self.weights=np.array(data['weights'])
            self.error_history = data['error_history']
            self.epoch_list = data['epoch_list']

    
    def __init__(self, inputs= None, outputs= None,file_name = None):
        if file_name != None:
            with open(file=file_name) as json_file:
                data = json.load(json_file) 
                self.inputs=np.array(data['inputs'])
                self.outputs=np.array(data['outputs'])
                self.weights=np.array(data['weights'])
                self.error_history = data['error_history']
                self.epoch_list = data['epoch_list']
        else:
            self.inputs  = inputs
            self.outputs = outputs
            # initialize weights as .50 for simplicity
            self.weights = np.array([[random.uniform(0,1)], [random.uniform(0,1)], [random.uniform(0,1)],[random.uniform(0,1)]])
            self.error_history = []
            self.epoch_list = []

    def store_model(self,file_name):
        dictd = {
            'inputs': self.inputs.tolist(),
            'outputs':self.outputs.tolist(),
            'weights': self.weights.tolist(),
            'error_history': self.error_history,
            'epoch_list': self.epoch_list
        }
        with open(file_name, "w") as outfile:
            json.dump(dictd,outfile)

    #activation function ==> S(x) = 1/1+e^(-x)
    def sigmoid(self, x, deriv=False):
        if deriv == True:
            return x * (1 - x)
        return 1 / (1 + np.exp(-x))

    # data will flow through the neural network.
    def feed_forward(self):
        self.hidden = self.sigmoid(np.dot(self.inputs, self.weights))

    # going backwards through the network to update weights
    def backpropagation(self):
        self.error  = self.outputs - self.hidden
        delta = self.error * self.sigmoid(self.hidden, deriv=True)
        self.weights += np.dot(self.inputs.T, delta)

    # train the neural net for 25,000 iterations
    def train(self, epochs=100):
        for epoch in range(epochs):
            # flow forward and produce an output
            self.feed_forward()
            # go back though the network to make corrections based on the output
            self.backpropagation()    
            # keep track of the error history over each epoch
            self.error_history.append(np.average(np.abs(self.error)))
            self.epoch_list.append(epoch)

    # function to predict output on new and unseen input data                               
    def predict(self, new_input):
        prediction = self.sigmoid(np.dot(new_input, self.weights))
        return prediction

# create neural network   
NN = NeuralNetwork(X_train, y_train)
# train neural network
NN.train()

print("trained")

plt.figure(figsize=(15,5))
plt.plot(NN.epoch_list, NN.error_history)
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.show()

NN.store_model('model.json')