
def sampling(mu_logvar):
    import keras
    
    
    epsilon=keras.backend.random_normal(shape=keras.backend.shape(mu_logvar[0]),stddev=1,mean=0)
    return mu_logvar[0]+keras.backend.exp(mu_logvar[1]/2)*epsilon
    