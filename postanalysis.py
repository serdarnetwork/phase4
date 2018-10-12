#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 11:53:08 2018

@author: neuro
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 20:01:29 2018

@author: neuro
"""

import matplotlib.pyplot as plt
import numpy as np
import os 

import dataload25 as dl
#import dataload25_t as dlt
import numpy as np
w=60
l=11
k=10
p=0
e=4
train_x, train_y, val_x, val_y, Ns = dl.prep(w)
#train_x, train_y, val_x, val_y, Ns = dlt.prep(w)



from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop
from keras.layers import Bidirectional,Convolution1D, Dense, Activation, Dropout
from scipy import fftpack

def spectrum(inputdata, Fs):
    pspecvals = fftpack.fft(inputdata)[0:len(inputdata) // 2]
    maxfreq = Fs / 2.0
    pspecaxis = np.linspace(0.0, maxfreq, len(pspecvals), endpoint=False)
    return pspecaxis, pspecvals

#model = Sequential()
#
#
#model.add(Convolution1D(nb_filter=64, filter_length=5, padding='same',input_shape=(None, 1)))
#model.add(Activation('relu'))
#model.add(Dropout(rate=0.5))
#model.add(Convolution1D(nb_filter=64, filter_length=5, padding='same'))
#model.add(Dropout(rate=0.5))
#model.add(Activation('relu'))
#model.add(Convolution1D(nb_filter=64, filter_length=5, padding='same'))
#model.add(Dropout(rate=0.5))
#model.add(Activation('relu'))
#model.add(Convolution1D(nb_filter=1, filter_length=5, padding='same'))



model = Sequential()


model.add(Convolution1D(nb_filter=k, filter_length=5, padding='same',input_shape=(None, 1)))
#    model.add(BatchNormalization())
model.add(Dropout(rate=p))
model.add(Activation('relu'))

for layer in range(l-2):
    model.add(Convolution1D(nb_filter=k, filter_length=5, padding='same'))
#        model.add(BatchNormalization())
    model.add(Dropout(rate=p))
    model.add(Activation('relu'))

model.add(Convolution1D(nb_filter=1, filter_length=5, padding='same'))

model.summary()

#model.add(layers.TimeDistributed(layers.Dense(1)))
#model.add((layers.Dense(31)))

#model.add(layers.LSTM(100,
#                     return_sequences=True))
#model.add(layers.LSTM(100))
    
#model.compile(optimizer=RMSprop(), loss='mse')
#
#history = model.fit(train_x,train_y,
#                    batch_size=128,
##                    steps_per_epoch=500,
#                    epochs=1,
#                    shuffle=True,
#                    validation_data=(val_x, val_y))
##                    validation_split=0.2)
model.load_weights('model_f10l10.h5')


YPred2=model.predict(val_x)
N=21601
rnn_length=w
length=(N-rnn_length-1)

K=13
val_x=val_x[-K*length:,:,:]
val_y=val_y[-K*length:,:,:]
YPred2=YPred2[-K*length:,:,:]


YPred2_val=YPred2.reshape(K,-1,(w+1),1)
val_y_val=val_y.reshape(K,-1,(w+1),1)
val_x_val=val_x.reshape(K,-1,(w+1),1)

Nv=val_y_val.shape[1]
Ns=K
output_pred=np.zeros([Ns,Nv+100])
output_real=np.zeros([Ns,Nv+100])
output_raw=np.zeros([Ns,Nv+100])

for k in range(Ns):
    print(k)
    for i in range(0,Nv):
        output_pred[k,i:i+(w+1)]+=YPred2_val[k,i,:,0]
        output_real[k,i:i+(w+1)]+=val_y_val[k,i,:,0]
        output_raw[k,i:i+(w+1)]+=val_x_val[k,i,:,0]

output_pred/=(w+1)
output_real/=(w+1)
output_raw/=(w+1)

error_s=output_real-output_pred
sq_error_s=(np.mean(np.square(error_s),axis=1))

error2_s=output_raw-output_real
sq_error2_s=(np.mean(np.square(error2_s),axis=1))


import getphase as gp3 
thephase_pred_temp = gp3.phasefromwave(output_pred[1,:], 25,debug=True)
thephase_pred_temp = gp3.phasefromwave(output_pred[1,:], 25,debug=True)

thephase_pred_temp = gp3.phasefromwave(output_pred[2,:], 25)
thephase_pred_temp = gp3.phasefromwave(output_pred[3,:], 25)
thephase_pred_temp = gp3.phasefromwave(output_pred[4,:], 25)

thephase_pred=np.zeros([K,thephase_pred_temp.shape[0]])
thephase_real=np.zeros([K,thephase_pred_temp.shape[0]])
thephase_raw=np.zeros([K,thephase_pred_temp.shape[0]])


#for i in range(7):
c2=0
#for c,i in enumerate(range(K)):
#    print(i)
#    if (i != 9) and (i != 10) and (i != 11):
#        thephase_pred[c2,:] = gp3.phasefromwave(output_pred[i,:], 25)
#        thephase_raw[c2,:] = gp3.phasefromwave(output_raw[i,:], 25)
#        thephase_real[c2,:] = gp3.phasefromwave(output_real[i,:], 25)
#        c2=c2+1

for c,i in enumerate(range(K)):
    print(i)
    thephase_pred[c2,:] = gp3.phasefromwave(output_pred[i,:], 25)
    thephase_raw[c2,:] = gp3.phasefromwave(output_raw[i,:], 25)
    thephase_real[c2,:] = gp3.phasefromwave(output_real[i,:], 25)
    c2=c2+1

#error1=np.square(thephase_pred-thephase_real)
#error2=np.square(thephase_raw-thephase_real)

#
#for i in range(K2):
#    for j in range(error1.shape[1]):
#        error1[i,j]=math.fmod(error1[i,j]+math.pi,2*math.pi)-math.pi
#        error2[i,j]=math.fmod(error2[i,j]+math.pi,2*math.pi)-math.pi 

thedifference =np.fmod(np.unwrap(thephase_real, axis=1) - np.unwrap(thephase_pred, axis=1) + np.pi, 2.0 * np.pi) - np.pi
thedifference = thedifference - np.mean(thedifference, axis=1)
td=np.transpose(thedifference)
td=td-np.mean(td,axis=0)
themse = np.mean(np.square(td), axis=0)
#themse = np.mean(np.square(thedifference), axis=1)
thedifference_r =np.fmod(np.unwrap(thephase_real, axis=1) - np.unwrap(thephase_raw, axis=1) + np.pi, 2.0 * np.pi) - np.pi
thedifference_r = thedifference_r - np.mean(thedifference_r, axis=1)
td_r=np.transpose(thedifference_r)
td_r=td_r-np.mean(td_r,axis=0)
themse_r = np.mean(np.square(td_r), axis=0)

for i in range(td.shape[1]):
    plt.close()
    plt.plot(td[:,i])
    plt.plot(td_r[:,i])
    plt.legend([ 'pred','raw'])
    plt.savefig('phase_'+str(i)+'.jpg')
    plt.close()
    
    plt.show()