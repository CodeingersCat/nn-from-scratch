from funcs import forward_prop
from train import get_predictions, get_accuracy
from matplotlib import pyplot as plt

def test_random_image(X, W1, b1, W2, b2):
    current_img = X
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, current_img)
    current_img = current_img.reshape((28, 28)) * 255
    predictions = get_predictions(A2)
    print(predictions)
    plt.gray()
    plt.imshow(current_img, interpolation='nearest')
    plt.show()
