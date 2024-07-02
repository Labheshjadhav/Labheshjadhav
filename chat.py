import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load intents from intents.json
try:
    with open('intents.json', 'r') as file:
        intents = json.load(file)
except FileNotFoundError:
    print("Error: 'intents.json' file not found.")
    exit()

# Print loaded intents data
print(intents)  # Check if the data is loaded correctly

# Load trained model and data
FILE = "data.pth"
try:
    data = torch.load(FILE)
except FileNotFoundError:
    print("Error: Trained model file '{}' not found.".format(FILE))
    exit()

# Extract data from loaded model
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

# Print loaded tags
print(tags)

# Initialize and load the model
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

# Define bot name
bot_name = "Sam"

# Confidence threshold
CONFIDENCE_THRESHOLD = 0.75


def get_response(msg):
    print("Message:", msg)  # Print the message received from the user
    
    sentence = tokenize(msg)
    print("Tokenized sentence:", sentence)  # Print the tokenized sentence
    
    X = bag_of_words(sentence, all_words)
    print("Bag of words:", X)  # Print the bag of words
    
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]
    print("Predicted tag:", tag)  # Print the predicted tag

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    print("Confidence:", prob.item())  # Print the confidence level
    
    if prob.item() > CONFIDENCE_THRESHOLD:
        # Remove confidence level check and return response
        for intent in intents['intents']:
            if tag == intent["tag"]:
                print("Response options:", intent['responses'])  # Print the available responses
                response = random.choice(intent['responses'])
                print("Chosen response:", response)  # Print the chosen response
                return response
    else:
        return "I'm sorry, I'm not sure how to respond to that."


if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            break

        response = get_response(user_input)
        print(f"{bot_name}: {response}")
