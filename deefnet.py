
from keras.models import Model
from keras.layers import  add,Activation,Multiply ,Reshape,GlobalAveragePooling2D,Concatenate,AveragePooling2D, DepthwiseConv2D,Dense,Conv2D,Flatten,MaxPooling2D,Dropout,BatchNormalization, Input,ZeroPadding2D



import tensorflow as tf


def lipi(classes=None):
    input_shape=(64,64,1)
    X_input= Input(input_shape)
    
    #x = ZeroPadding2D((3,3))(X_input)
    
    x1 = Conv2D(64, (3, 3), activation = 'relu', padding='same',)(X_input)
    x = Conv2D(64, (3, 3), activation = 'relu', padding='same',)(x1)
    x = Conv2D(64, (3, 3), activation = 'relu', padding='same',)(x)
    
    x1 = DepthwiseConv2D(3,strides=(1,1),padding='same',depth_multiplier=1,data_format='channels_last',activation=None,use_bias=False)(x1)
    x1 = BatchNormalization(axis=-1)(x1)
    x1=Activation('relu')(x1)
    x1 = Conv2D(64,(1,1),strides=(1,1),padding='same',data_format='channels_last',activation=None,use_bias=False)(x1)
    x = Concatenate()([x,x1])


    x =MaxPooling2D()(x)
    
    
    x1 = Conv2D(128, (3, 3), activation = 'relu', padding='same',)(x)
    x = Conv2D(128, (3, 3), activation = 'relu', padding='same',)(x1)
    x = Conv2D(128, (3, 3), activation = 'relu', padding='same',)(x)
    
    x1 = DepthwiseConv2D(3,strides=(1,1),padding='same',depth_multiplier=1,data_format='channels_last',activation=None,use_bias=False)(x1)
    x1 = BatchNormalization(axis=-1)(x1)
    x1=Activation('relu')(x1)
    x1 = Conv2D(128,(1,1),strides=(1,1),padding='same',data_format='channels_last',activation=None,use_bias=False)(x1)
    x = Concatenate()([x,x1])
     

    
    x =MaxPooling2D()(x)
      
    x1 = Conv2D(512, (3, 3), activation = 'relu', padding='same',)(x)
    x = Conv2D(512, (3, 3), activation = 'relu', padding='same',)(x1)
    x = Conv2D(512, (3, 3), activation = 'relu', padding='same',)(x)
    
    x1 = DepthwiseConv2D(3,strides=(1,1),padding='same',depth_multiplier=1,data_format='channels_last',activation=None,use_bias=False)(x1)
    x1 = BatchNormalization(axis=-1)(x1)
    x1=Activation('relu')(x1)
    x1 = Conv2D(512,(1,1),strides=(1,1),padding='same',data_format='channels_last',activation=None,use_bias=False)(x1)
    x = Concatenate()([x,x1])

    x =GlobalAveragePooling2D()(x)
    x = Dense(classes, activation = 'softmax')(x)
    model= Model(X_input,x)
    return model











    
