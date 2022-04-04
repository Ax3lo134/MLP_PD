The data is extracted from the 'SxRx.txt' files. They are the output of sensors from 10 patients with Parkinson's disease and have locomotor deficiencies 
due to the disease.

Each data series is a matrix in which: the first column is the the point in time for each sensor's data acquisition , columns 2-10 are the values of the acceleration
recorded by sensors and column 11 represents the labels: 0 is unrelated, 1 is no syndrome, 2 is syndrome.

The script implements a multilayer perceptron and solves a classification problem. It tries to label when a patient has problems with locomotion or not.

There are comments for every line written in main.py.
