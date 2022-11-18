from keras.layers import Input, Conv1D, Dense, Activation, Dropout, MaxPooling1D, Flatten,concatenate,GlobalMaxPool1D
from keras import Model
from tensorflow.keras.optimizers import (RMSprop, Adam, SGD)
from keras.models import load_model
from tensorflow.keras.utils import plot_model
from keras import regularizers
from keras.regularizers import (l1, l2, l1_l2)
from keras.callbacks import (EarlyStopping, ModelCheckpoint)
from tensorflow.keras.optimizers import (RMSprop, Adam, SGD)
from keras.layers import (Input, Dense, Dropout, Flatten, BatchNormalization,
                                     Conv1D, Conv2D, MaxPooling1D, MaxPooling2D,GlobalAveragePooling1D,
                                     LSTM, GRU, Embedding, Bidirectional, Concatenate, Multiply)

def get_DeepPhoPred_model(trainX1, trainX2):
    ### Head-1:
    input1 = Input(shape=trainX1[0].shape)
    x = Conv1D(filters=64, kernel_size=5, padding='same', kernel_regularizer=l2(l=0.01))(input1)
    #x = BatchNormalization()(x)
    x = Dropout(rate=0.50)(x)

    shortcut = x
    # calculate the number of filters the input has
    filters = x.shape[-1]
    # the squeeze operation reduces the input dimensionality
    # here we do a global average pooling across the filters, which
    # reduces the input to a 1D vector
    x = GlobalAveragePooling1D()(x)
    # reduce the number of filters (1 x 1 x C/r)
    x = Dense(filters // 16, activation="relu", kernel_initializer="he_normal", use_bias=False)(x)

    # the excitation operation restores the input dimensionality
    x = Dense(filters, activation="sigmoid", kernel_initializer="he_normal", use_bias=False)(x)

    # multiply the attention weights with the original input
    x = Multiply()([shortcut, x])

    head1 = Flatten()(x)

    x = Model(inputs=[input1], outputs=[head1])

    ### Head-2:
    input2 = Input(shape=trainX2[0].shape)

    y = Conv1D(filters=64, kernel_size=5, padding='same', kernel_regularizer=l2(l=0.01))(input2)
    y = Dropout(rate=0.5)(y)

    shortcut2 = y
    # calculate the number of filters the input has
    filters2 = y.shape[-1]
    # the squeeze operation reduces the input dimensionality
    # here we do a global average pooling across the filters, which
    # reduces the input to a 1D vector
    y = GlobalAveragePooling1D(keepdims=True)(y)
    # reduce the number of filters (1 x 1 x C/r)
    y = Dense(filters2 // 16, activation="relu", kernel_initializer="he_normal", use_bias=False)(y)

    # the excitation operation restores the input dimensionality
    y = Dense(filters2, activation="sigmoid", kernel_initializer="he_normal", use_bias=False)(y)

    # multiply the attention weights with the original input
    y = Multiply()([shortcut2, y])

    # y = Conv1D(filters=16, kernel_size=3, padding='same', activation='relu', kernel_regularizer=l2(l=0.01))(y)
    # y = BatchNormalization()(y)
    # y = Dropout(rate=0.50)(y)

    head2 = Flatten()(y)
    y = Model(inputs=[input2], outputs=[head2])

    combined = concatenate([x.output, y.output])

    # z= Dense(units=2048, activation='relu')(combined)
    # output2 = BatchNormalization()(output2)
    # z = Dropout(rate=0.5)(z)

    z = Dense(units=1024, activation='relu')(combined)
    z = Dropout(rate=0.5)(z)

    z = Dense(units=512, activation='relu')(z)
    z = Dropout(rate=0.5)(z)

    # z= Dense(units=256, activation='relu')(z)
    # output2 = BatchNormalization()(output2)
    # z = Dropout(rate=0.5)(z)

    z = Dense(units=128, activation='relu')(z)
    z = Dropout(rate=0.5)(z)


    z = Dense(units=16, activation='relu')(z)
    z = Dropout(rate=0.5)(z)

    z = Dense(1, activation='sigmoid')(z)

    # our model will accept the inputs of the two branches and
    # then output a single value
    return Model(inputs=[x.input, y.input], outputs=z)

import numpy as np
npzfile = np.load('balance_data/train_smote_balanced_21_S.npz', allow_pickle=True)
x_train= npzfile['arr_0']
y_train= npzfile['arr_1']
x_test = npzfile['arr_2']
y_test = npzfile['arr_3']

from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.20, random_state=10)

from utilities import get_pssm_spd
trainX1, trainX2 = get_pssm_spd(x_train)
valX1, valtX2 = get_pssm_spd(x_val)
testX1, testX2 = get_pssm_spd(x_test)

import tensorflow.keras.backend as K
model = get_DeepPhoPred_model()
model.summary()
plot_model(model, to_file='model-combined.png', show_shapes=True, show_layer_names=False, expand_nested=True)

# patient early stopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
# save the best model
mc = ModelCheckpoint('best_model_combined.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
# Compile Model:
from tensorflow.keras.optimizers import (RMSprop, Adam, SGD)
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy',metrics=['accuracy'])

history=model.fit(x=[trainX1, trainX2], y=y_train, validation_data=([valX1, valtX2], y_val), epochs=100, batch_size=32, callbacks=[es,mc], verbose = 1)


from matplotlib import pyplot as plt
#plot training and validation accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
#plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','validation'], loc='lower left')
plt.savefig('accuracy_curve_combined.png')
plt.show()


#plot training and validation loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.plot()
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','validation'], loc='upper right')
plt.savefig('loss_curve_combined.png')
plt.show()

from tensorflow.keras.models import load_model
model = load_model('best_model_combined.h5')
probabilities_ind = model.predict([testX1,testX2])
predicted_classes_ind = probabilities_ind >= 0.5
predicted_classes_ind = predicted_classes_ind.astype(int)

from evalution_metrics import performance_result
performance_result(y_test,predicted_classes_ind, probabilities_ind)
