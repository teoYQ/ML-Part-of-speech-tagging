# ML-Part-of-speech-tagging
Implementing the hidden markov model using viterbi algorithm

# How to run the code
Download Test.py and Functions.py and ensure that both files are in the same folder. The list of functions implemented in Functions.py are mentioned below.
Open Test.py and edit the first 4 lines to your respective paths. All parts run on the assumption that we are taking the file to train data is called "train" and file to test our code is "dev.in"

For part5, please scroll to the bottom in Test.py to edit the number of epochs and the learning rate. Currently it is set to 10 and 0.0005.

Run Test.py to get results for parts2-5.

-------------------------------------------------------------------------------------------------------------------------------


List of functions
1) loadfile(path)   		      :   reads the file in path and split them into words and labels

2) smooth(data,k)   		      :   takes in the dataset as data and threshold as k. If a word
		    	  	                    less than k times, we replace it with 'unk'

3) init_values(data)		      :   Takes in a dataset and returns list of words replaced by UNK,
		      		                    counts of each label and 2 empty dataframes for emission and
		      		                    transition

4) emission(dfB,data,labels)	 :   Takes in empty emission matrix created by init_values(data),
				                           data and all the labels. Returns a filled emission matrix.

5) sentiment(emis) 		         :   returns a dictionary of words and corresponding label

6) transition(dfA,data,labels) :   like emission, this takes in an empty transition matrix, 
				                           dataset we used in init and the list of labels in dataset.
				                           Returns a filled transiition matrix

7) viterbi(y,A,B,pi=None)	     : Takes in a sequence, transition matrix, emission matrix,
				                         initial probabilities (if any).
				                         This functions computes the most probable label for each 
				                         word in sequence using the matrices.
				                         Returns list of labels in word and index form, score of the
				                         most probable path thus far and the previous best preceeding nodes.

8) viterbi_under(y,A,B,pi=None) : Same as above, just that this time we factor in underflow due 
				                          to multiplications of multiple small numbers

9) viterbi_kbest(y,A,B,pi=None,k): Same as viterbi (number 7) but takes in another parameter k,
				                           where k causes the function to receive the Kth most probable
				                           label sequence.
10)viterbi_kunder(y,A,B,pi=None,k): Same as above, just that we factor in underflow as mentioned
				                            earlier

11) train_perceptron(train,trans,
	emis,pi,epoch,lr)	              : Takes in the training dataset, initial transition and emiss-
				                             ion matrix, intial probabilities, epoch and learning rate
				    returns the new best transition and emission matrix and the scores
