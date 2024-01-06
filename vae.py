import tensorflow as tf
import keras
import numpy as np
from loss_func import loss_func
from sampling import sampling
img=28
channel=1
latent_dim=2
x=keras.layers.Input(shape=(img,img,channel),name='encoder_input')
encoder_conv1=keras.layers.Conv2D(filters=1,kernel_size=(3,3),strides=1,padding='same')(x)
encoder_norm1=keras.layers.BatchNormalization()(encoder_conv1)
encoder_leru1=keras.layers.LeakyReLU()(encoder_norm1)
encoder_conv2=keras.layers.Conv2D(filters=32,kernel_size=(3,3),strides=2,padding='same')(encoder_leru1)
encoder_norm2=keras.layers.BatchNormalization()(encoder_conv2)
encoder_leru2=keras.layers.LeakyReLU()(encoder_norm2)
encoder_conv3=keras.layers.Conv2D(filters=64,kernel_size=(3,3),strides=2,padding='same')(encoder_leru2)
encoder_norm3=keras.layers.BatchNormalization()(encoder_conv3)
encoder_leru3=keras.layers.LeakyReLU()(encoder_norm3)
encoder_conv4=keras.layers.Conv2D(filters=64,kernel_size=(3,3),strides=1,padding='same')(encoder_leru3)
encoder_norm4=keras.layers.BatchNormalization()(encoder_conv4)
encoder_leru4=keras.layers.LeakyReLU()(encoder_norm4)
before_flatten = keras.backend.int_shape(encoder_leru4)[1:]
print(before_flatten)
flattened=keras.layers.Flatten()(encoder_leru4)
print(encoder_leru4.shape)
mu=keras.layers.Dense(units=latent_dim)(flattened)
log_var=keras.layers.Dense(units=latent_dim)(flattened)
encoder_output=keras.layers.Lambda(sampling)([mu,log_var])
print(encoder_output.shape)


encoder=keras.models.Model(x,encoder_output,name='encoder_model')



y=keras.Input(shape=(latent_dim))
decoder_dense=keras.layers.Dense(units=np.prod(before_flatten))(y)
decoder_reshaped=keras.layers.Reshape(target_shape=before_flatten)(decoder_dense)
decoder_conv1=keras.layers.Conv2DTranspose(filters=64,kernel_size=(3,3),strides=1,padding='same')(decoder_reshaped)
decoder_norm1=keras.layers.BatchNormalization()(decoder_conv1)
decoder_leru1=keras.layers.LeakyReLU()(decoder_norm1)
decoder_conv2=keras.layers.Conv2DTranspose(filters=64,kernel_size=(3,3),strides=2,padding='same')(decoder_leru1)
decoder_norm2=keras.layers.BatchNormalization()(decoder_conv2)
decoder_leru2=keras.layers.LeakyReLU()(decoder_norm2)
decoder_conv3=keras.layers.Conv2DTranspose(filters=64,kernel_size=(3,3),strides=2,padding='same')(decoder_leru2)
decoder_norm3=keras.layers.BatchNormalization()(decoder_conv3)
decoder_leru3=keras.layers.LeakyReLU()(decoder_norm3)
decoder_conv4=keras.layers.Conv2DTranspose(filters=1,kernel_size=(3,3),strides=1,padding='same')(decoder_leru3)
decoder_norm4=keras.layers.BatchNormalization()(decoder_conv4)
decoder_output=keras.layers.LeakyReLU()(decoder_norm4)
decoder=keras.models.Model(y,decoder_output,name='decoder_model')


vae_input=keras.Input(shape=(img,img,channel))
vae_encoder_output=encoder(vae_input)
vae_decoder_output=decoder(vae_encoder_output)
vae=keras.models.Model(vae_input,vae_decoder_output)
vae.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005), loss=loss_func(mu, log_var))
(X_train,Y_train),(X_test,Y_test)=keras.datasets.mnist.load_data()
X_train=X_train.astype("float32")/255.0
X_test=X_test.astype("float32")/255.0
X_train=np.reshape(X_train,newshape=(X_train.shape[0],X_train.shape[1],X_train.shape[2],1))
X_test=np.reshape(X_test,newshape=(X_test.shape[0],X_train.shape[1],X_train.shape[2],1))

vae.fit(X_train, X_train, epochs=20, batch_size=32, shuffle=True, validation_data=(X_test, X_test))
encoder.save("VAE_ENCODER.h5")
decoder.save("VAE_DECODER.h5")
vae.save("VAE.h5")






    