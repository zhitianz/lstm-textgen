#Code used for ACoRL lab research.
#Dr. Licato and Zhitian Zhang
#Reference: keras example
        
from __future__ import print_function
from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
#from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import os.path

#path = get_file('nietzsche.txt', origin="https://s3.amazonaws.com/text-datasets/nietzsche.txt")
#text = open(path).read().lower()

#GLOBAL PARAMETERS
trainMode = True 
resumeMode = False #if this is true, then it looks for the most recent saved model file.
charsToGenerate = 400

#PARAMETERS FOR TRAIN MODE
text = open("emerson.txt").read().lower()
numIterations = 6000

#PARAMETERS FOR TEST MODE
seed_sentences = ["Allow me to tell you what I think of those of you here on /r/badphilosophy. You are all ", "Let me tell you, inhabitants of the future America: the meaning of life is "]
modelStoredFile = 'model_stored.json'

#check if there is an existing file to load from
startingIndex = 0
if resumeMode or not trainMode:
	for i in range(numIterations):	
		startingIndex = numIterations-1-i
		currFile = "model_weights_" + str(numIterations-1-i) + ".h5"
		if os.path.isfile(currFile):
			print("weights file found: loading from ", currFile)
			modelWeightsFile = currFile
			break
			
#modelWeightsFile = 'model_weights_18.h5' #ONLY if you want to override the auto check

print('corpus length:', len(text))

chars = set(text)
print('total chars:', len(chars)) #print number of distinct characters in the text
char_indices = dict((c, i) for i, c in enumerate(chars)) #dictionary where char_indices[i] returns the unique character assigned to index i
indices_char = dict((i, c) for i, c in enumerate(chars)) #dictionary where indices_char[c] returns the unique index assigned to character c

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 150
step = 3
sentences = [] #will contain strings of no more than maxlen characters
next_chars = [] #next_chars[i] is the character following the string sentences[i].
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))

print('Vectorization...')
X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
	#X is a 3-dimensional dataset of sentences, chars within that sentence, and
	#one-hot vectors encoding those chars. So X is of size (i,j,k) where 
	#i=num of sentences, j=the (maximum) length of the sentences, and
	#k=the size of the vector encoding chars.
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
	#y is a 2-dimensional vector where for each sentence, we have the one-hot vector
	#encoding the "next character." So with this dataset, each sentence is an example.
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1 #encode the one-hot vector
    y[i, char_indices[next_chars[i]]] = 1


# build the model: 2 stacked LSTM
if trainMode and not resumeMode:
	print('Build model...')
	model = Sequential()
	model.add(LSTM(512, return_sequences=True, input_shape=(maxlen, len(chars))))
	model.add(Dropout(0.2))
	model.add(LSTM(512, return_sequences=True))
	model.add(Dropout(0.2))
	model.add(LSTM(512, return_sequences=False))
	model.add(Dropout(0.2))
	model.add(Dense(len(chars)))
	model.add(Activation('softmax'))
else:
	if not resumeMode:
		numIterations = 2
	print('Loading model...')
	model = Sequential()
	model = model_from_json(open(modelStoredFile).read())
	model.load_weights(modelWeightsFile)
		
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')


def sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))
    
if trainMode:
	#save model to file
	json_string = model.to_json() 
	open("model_stored.json", 'w').write(json_string)

def generateOutput(diversity, seedSentence, numCharsToGenerate):
	global maxlen
	global model
	
	print()
	print('----- diversity:', diversity)

	generated = ''
	sentence = seedSentence[0: maxlen]
	while len(sentence) < maxlen:
		sentence = ' ' + sentence
	generated += sentence
	print('----- Generating with seed: "' + sentence + '"')
	sys.stdout.write(generated)

	for i in range(numCharsToGenerate):
		x = np.zeros((1, maxlen, len(chars)))
		for t, char in enumerate(sentence):
			if char in char_indices:
				x[0, t, char_indices[char]] = 1.
			else:
				x[0, t, char_indices['.']] = 1.

		preds = model.predict(x, verbose=0)[0]
		next_index = sample(preds, diversity)
		next_char = indices_char[next_index]

		generated += next_char
		sentence = sentence[1:] + next_char

		sys.stdout.write(next_char)
		sys.stdout.flush()
	print()

# train the model, output generated text after each iteration
for iteration in range(1, numIterations):
    print()
    print('-' * 50)
    if trainMode or resumeMode:
        print('Iteration', iteration+startingIndex)
        model.fit(X, y, batch_size=128, nb_epoch=1)
        #save weights to file
        #http://keras.io/faq/#how-can-i-save-a-keras-model
        model.save_weights("model_weights_" + str(iteration+startingIndex) + ".h5")
    	start_index = random.randint(0, len(text) - maxlen - 1)
    	for diversity in [0.2, 0.5, 1.0, 1.2]:
    		seedSentence = text[start_index: start_index + maxlen]
    		generateOutput(diversity,seedSentence,charsToGenerate)
    else:
    	for seedSentence in seed_sentences:
    		for diversity in [0.2, 0.5, 1.0, 1.2]:
    			generateOutput(diversity,seedSentence.lower(),charsToGenerate)
