# Sentence Predicting Neural Network
The task is to predict the fourth word in sequence given the preceding tri-gram, e.g., trigram: `Neural nets are', fourth word: `awesome'.
A database of articles were parsed to store sample fourgrams restricted to a vocabulary size of 250 words. The file assign2_data2.mat 
contains training samples for input and output. 
The input layer has 3 neurons corresponding to the trigram entries. An embedding matrix R (250xD) is used to linearly map each single word
onto a vector representation of length D. The same embedding matrix is used for each input word in the trigram, without considering the
sequence order. The hidden layer uses a sigmoidal activation function on each of P hidden-layer neurons. The output layer predicts a
separate response zi for each of 250 vocabulary words, and the probability of each word is estimated via a soft-max operation.

Different D and P values are tested to improve the model's accuracy and then the model is tested for a few trigrams.
