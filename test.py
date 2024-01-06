import keras
import numpy as np
from sampling import sampling
from loss_func import loss_func
import matplotlib.pyplot as plt
encoder=keras.models.load_model('VAE_ENCODER.h5',custom_objects={'sampling':sampling,'vae_loss':loss_func})
(X_train,Y_train),(X_test,Y_test)=keras.datasets.mnist.load_data()
X_train=X_train.astype("float32")/255.0
X_test=X_test.astype("float32")/255.0
X_test=np.reshape(X_test,newshape=(X_test.shape[0],X_train.shape[1],X_train.shape[2],1))
def plot_test_data( data, labels):
    mu= encoder.predict(data)
    print(mu.shape)
    plt.figure(figsize=(10, 10))
    plt.scatter(mu[:, 0], mu[:, 1], c=labels)
    plt.colorbar()
    plt.xlabel("mu[0]")
    plt.ylabel("mu[1]")
    plt.show()





plot_test_data( X_test, Y_test)