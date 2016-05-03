# Recognizing Digits with a Computer

## Data
Data can be downloaded from the kaggle competition page on digit recognition.
It can be put into any directory you choose.
The path where the data is located should be placed in a file called `datapath.txt`.
The path should then be read from this file using

    with open('datapath.txt') as f:
    	 datapath=f.readlines()[0].rstrip()

