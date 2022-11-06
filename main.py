import numpy as np
import pandas as pd
from train import gradient_descent
from test import test_random_image

k = 2

while(k == 2):
    # Loading training dataset
    data_train = pd.read_csv('Datasets/train.csv')
    data_train = np.array(data_train)

    m, n = data_train.shape
    print("TRAINING WITH -")
    print("Number of images : "+str(m))
    print("Number of pixels per image : "+str(n-1))
    print()

    # Framing training data
    data_train = data_train.T
    Y_train = data_train[0]
    X_train = data_train[1:n]
    X_train = X_train / 255. # Normalising pixel values

    #TRAIN
    alpha = float(input("Enter learning rate\n"))
    cycles = int(input("Enter the number of learning cycles\n"))
    W1, b1, W2, b2 = gradient_descent(X_train, Y_train, alpha, cycles)

    # Loading test dataset
    test_data = pd.read_csv('Datasets/test.csv')
    data = np.array(test_data)

    #TEST
    m, n = data.shape

    print()
    print("TESTING WITH -")
    print("Number of images : "+str(m))
    print("Number of pixels per image : "+str(n))
    print()

    k = 1

    # testing data
    data_test = data.T
    while(k == 1):
        #TEST
        index = int(input("Enter select a random image number from 0 to 27999 \n"))
        print(data_test.shape)
        current_img = data_test[:, index, None]
        X_test = current_img / 255.  # Normalising pixel values
        test_random_image(X_test, W1, b1, W2, b2)
        k = int(input("Press 1 to try again, 2 to re-train the neural network or press any other key to exit\n"))












