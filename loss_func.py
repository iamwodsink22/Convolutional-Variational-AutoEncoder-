import keras
import tensorflow as tf
def loss_func(mu,log_var):
    def vae_reconstruct_loss(y_true,y_predict):
        factor=1000
        
        
        reconstruct_loss=keras.backend.mean(keras.backend.square(y_true-y_predict),axis=[1,2,3])
        return factor*reconstruct_loss
    def kl_loss(mu,log_var):
        return -0.5*keras.backend.sum(1.0+log_var-keras.backend.square(mu)-keras.backend.exp(log_var),axis=1)
        
    def kl_loss_metric(y_predict,y_true):
        return -0.5*keras.backend.sum(1.0+log_var-keras.backend.square(mu)-keras.backend.exp(log_var),axis=1)
        
    def vae_loss(y_predict,y_true):
        reconstruct_loss=vae_reconstruct_loss(y_true,y_predict)
        k_loss=kl_loss(y_true,y_predict)
        return reconstruct_loss+k_loss
    
    return vae_loss