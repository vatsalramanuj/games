import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import pickle

data = pd.read_csv(r"/home/ace/Documents/New_Stuff/Machine_learning/train.csv")
data.head()
data = np.array(data)
m, n = data.shape
np.random.shuffle(data)

data_dev = data[:1000].T # transposed too
# Separating pixel values and true numbers
Y_dev = data_dev[0] # 1st row
X_dev = data_dev[1:]/255 # other rows - pixel values

data_train = data[1000:].T
Y_train = data_train[0] 
X_train = data_train[1:]/255


def init_params():
    # W - weights
    # b - biases
    hidden_layer_size = 64
    W1 = np.random.rand(hidden_layer_size, 784) - 0.5 # random values between -0.5 to 0.5
    b1 = np.random.rand(hidden_layer_size, 1) - 0.5
    W2 = np.random.rand(10, hidden_layer_size) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2

def ReLU(Z):
    """
    0 if num < 0
    num if num >= 0
    """
    return np.maximum(0, Z)

def softmax(Z):
    """
    returns probabilities between 0 and 1 for each prediction between 0 to 9
    """
    expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return expZ / np.sum(expZ, axis=0, keepdims=True)

def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def one_hot(Y):
    """
    turn numbers into array with 1 at the number of row 
    """
    one_hot_Y = np.zeros((Y.size, Y.max() + 1)) # initiate an empty 2d array with Y.size number of rows and (here) 10 columns
    one_hot_Y[np.arange(Y.size), Y] = 1 # iterating using lists as indexes. the "Y" in teh 2nd index is if th real number is say 5, there will be a 1 at the 5th index.
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def deriv_ReLU(Z):
    """
    differentiation of ReLU
    """
    return (Z > 0).astype(int) # boolean when converted to numbers maps True to 1 and False to 0

def back_prop(Z1, A1, Z2, A2, W2, X, Y):
    m = Y.size # == len(Y)
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = W2.T.dot(dZ2) * deriv_ReLU(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    """alpha (learning rate) is the hyperparameter - set by user, not automatically set by algo"""
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1    
    W2 = W2 - alpha * dW2  
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2

def get_predictions(A2):
    """return prediction with highest probablity"""
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y)/Y.size * 100

def compute_loss(A2, Y):
    m = Y.size
    Yh = one_hot(Y)
    return -(1/m) * np.sum(Yh * np.log(A2 + 1e-8))

def gradient_descent(X_train, Y_train, num_iters = 500, alpha = 0.1):
    W1, b1, W2, b2 = init_params()
    for iters in range(num_iters):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X_train)
        dW1, db1, dW2, db2 = back_prop(Z1, A1, Z2, A2, W2, X_train, Y_train)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        # bookkeeping:
        if (iters+1)%50 == 0:
            print("Iteration:", iters+1)
            print("Accuracy:", get_accuracy(get_predictions(A2), Y_train), "%")
            print("Loss:", compute_loss(A2, Y_train))
    return W1, b1, W2, b2


def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

def test_prediction(index, W1, b1, W2, b2):
    """Visualizing"""
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()



# # Training step - run only when training or changing hyperparameters

# W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 500, 0.3)
# with open("model_params.pkl", "wb") as f:
#     pickle.dump((W1, b1, W2, b2), f)


# with open("model_params.pkl", "rb") as f:
#     W1, b1, W2, b2 = pickle.load(f)


# # test_prediction(10, W1, b1, W2, b2)

# dev_predictions = make_predictions(X_dev, W1, b1, W2, b2)
# print("Accuracy: {:.2f}".format(get_accuracy(dev_predictions, Y_dev)), "%")