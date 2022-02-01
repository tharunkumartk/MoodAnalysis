from model_creation import NeuralNetwork
from nltk.sentiment import SentimentIntensityAnalyzer



loaded_model=NeuralNetwork(file_name='model.json')
print("Loaded model from disk")

numTimes = int(input("How many entries do you have?"))
while numTimes>0:
    inp = str(input())
    sid = SentimentIntensityAnalyzer()
    ss = sid.polarity_scores(inp)
    input_data=[ss['neg'],ss['neu'],ss['pos'],ss['compound']]
    print(loaded_model.predict(input_data))
    numTimes=numTimes-1
