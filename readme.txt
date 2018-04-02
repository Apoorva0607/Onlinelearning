Goal is to build a classifier that identifies which among these are phishing websites.Using Phishing data set from the UCI Machine Learning repository to study this.The data has been preprocessed into a standard format.
Using the training/development/testfiles called phishing.train, phishing.dev and phishing.test. 
These files are in the LIBSVM format, where each row is a single training example. The format of the each rowin the data is:<label> <index1>:<value1> <index2>:<value2> ...Here<label>denotes the label for that example. The rest of the elements of the rowis a sparse vector denoting the feature vector. For example, if the original feature vector is[0;0;1;2;0;3], this would be represented as3:1 4:2 6:3. That is, only the non-zero entriesof the feature vector are stored.

This directory contains -
1. Directory Perceptron that contains python source code.
2. Shell script 'run.sh' used to run the program. 
3. Directory Dataset that contains data set.

To execute the program, execute the run.sh script. 
./run.sh
just in case if this doesnt work try: bash run.sh
