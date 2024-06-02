import random
import json
import torch
from model import NeuralNet
from nltk_utils import tokenize, bag_of_words
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as f:
    intents = json.load(f)

# Load model and data
FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

# Assuming you have defined NeuralNet in your model.py file
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Buddy"
print("What's up Buddy ?")

while True:
    sentence = input("Let's chat! Type 'quit' to exit: ")
    if sentence == "quit":
        break

    # Tokenize sentence
    sentence = tokenize(sentence)
    
    # Convert sentence to bag of words
    X = bag_of_words(sentence, all_words)
    
    # Convert bag of words to torch tensor
    X = np.array(X)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).float().to(device)

    # Perform model inference
    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    # Extract tag, probabilities, etc. (if needed)
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    # Print bot response based on predicted tag
    if prob.item() > 0.75:
        for intent in intents:
            if tag == intent["tag"]:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
    else:
        print(f"{bot_name}: I do not understand.")
