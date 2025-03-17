from tensorflow.keras import optimizers
from tensorflow.keras.layers import Input, Conv1D, Conv2D, Dense, MaxPool1D, MaxPool2D, GlobalAveragePooling1D, GlobalAveragePooling2D
import datetime
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# job 1 = Model Training
# job 2 = Model Testing

job = 1

Model_name = 'Model-B2'

Data_path = 'Project Data Path1'
Result_path = 'Project Data Path2'

if job == 1:
    # Load Data
    Time_data_V = np.load(f'{Data_path}/Time_data_V.npy', allow_pickle=True)
    Time_data_C1 = np.load(f'{Data_path}/Time_data_C1.npy', allow_pickle=True)
    Time_data_C2 = np.load(f'{Data_path}/Time_data_C2.npy', allow_pickle=True)
    Time_data_C3 = np.load(f'{Data_path}/Time_data_C3.npy', allow_pickle=True)

    # FFT_data_V = np.load(f'{Data_path}/FFT_data_V.npy', allow_pickle=True)
    # FFT_data_C1 = np.load(f'{Data_path}/FFT_data_C1.npy', allow_pickle=True)
    # FFT_data_C2 = np.load(f'{Data_path}/FFT_data_C2.npy', allow_pickle=True)
    # FFT_data_C3 = np.load(f'{Data_path}/FFT_data_C3.npy', allow_pickle=True)
    #
    # MFCC_data_V = np.load(f'{Data_path}/MFCC_data_V.npy', allow_pickle=True)
    # MFCC_data_C1 = np.load(f'{Data_path}/MFCC_data_C1.npy', allow_pickle=True)
    # MFCC_data_C2 = np.load(f'{Data_path}/MFCC_data_C2.npy', allow_pickle=True)
    # MFCC_data_C3 = np.load(f'{Data_path}/MFCC_data_C3.npy', allow_pickle=True)

    Y_train = np.load(f'{Data_path}/Y_train.npy')
    print(f"Time_data_V shape : {Time_data_V.shape}\n"
          f"Time_data_C1 shape : {Time_data_C1.shape}\n"
          f"Time_data_C2 shape : {Time_data_C2.shape}\n"
          f"Time_data_C3 shape : {Time_data_C3.shape}\n"
          
          # f"FFT_data_V shape : {FFT_data_V.shape}\n"
          # f"FFT_data_C1 shape : {FFT_data_C1.shape}\n"
          # f"FFT_data_C2 shape : {FFT_data_C2.shape}\n"
          # f"FFT_data_C3 shape : {FFT_data_C3.shape}\n"
          # 
          # f"MFCC_data_V shape : {MFCC_data_V.shape}\n"
          # f"MFCC_data_C1 shape : {MFCC_data_C1.shape}\n"
          # f"MFCC_data_C2 shape : {MFCC_data_C2.shape}\n"
          # f"MFCC_data_C3 shape : {MFCC_data_C3.shape}\n"
          
          f"Y_train shape : {Y_train.shape}")

    #individual learning structure
    Time_V_Input = Input(shape=(900,1))
    T_V_layer = Conv1D(16, 3, padding="same", activation='relu')(Time_V_Input)
    T_V_layer = Conv1D(16, 3, padding="same", activation='relu')(T_V_layer)
    T_V_layer = MaxPool1D(2)(T_V_layer)
    T_V_layer = Conv1D(32, 3, padding="same", activation='relu')(T_V_layer)
    T_V_layer = Conv1D(32, 3, padding="same", activation='relu')(T_V_layer)
    T_V_layer = MaxPool1D(2)(T_V_layer)
    T_V_layer = Conv1D(64, 3, padding="same", activation='relu')(T_V_layer)
    T_V_layer = Conv1D(64, 3, padding="same", activation='relu')(T_V_layer)
    T_V_layer = MaxPool1D(2)(T_V_layer)
    T_V_layer = Conv1D(128, 3, padding="same", activation='relu')(T_V_layer)
    T_V_layer = Conv1D(128, 3, padding="same", activation='relu')(T_V_layer)
    T_V_layer = MaxPool1D(2)(T_V_layer)
    T_V_layer = Conv1D(128, 3, padding="same", activation='relu')(T_V_layer)
    T_V_layer = Conv1D(128, 3, padding="same", activation='relu')(T_V_layer)
    T_V_layer = MaxPool1D(2)(T_V_layer)
    T_V_layer = GlobalAveragePooling1D()(T_V_layer)

    Time_C1_Input = Input(shape=(900,1))
    T_C1_layer = Conv1D(16, 3, padding="same", activation='relu')(Time_C1_Input)
    T_C1_layer = Conv1D(16, 3, padding="same", activation='relu')(T_C1_layer)
    T_C1_layer = MaxPool1D(2)(T_C1_layer)
    T_C1_layer = Conv1D(32, 3, padding="same", activation='relu')(T_C1_layer)
    T_C1_layer = Conv1D(32, 3, padding="same", activation='relu')(T_C1_layer)
    T_C1_layer = MaxPool1D(2)(T_C1_layer)
    T_C1_layer = Conv1D(64, 3, padding="same", activation='relu')(T_C1_layer)
    T_C1_layer = Conv1D(64, 3, padding="same", activation='relu')(T_C1_layer)
    T_C1_layer = MaxPool1D(2)(T_C1_layer)
    T_C1_layer = Conv1D(128, 3, padding="same", activation='relu')(T_C1_layer)
    T_C1_layer = Conv1D(128, 3, padding="same", activation='relu')(T_C1_layer)
    T_C1_layer = MaxPool1D(2)(T_C1_layer)
    T_C1_layer = Conv1D(128, 3, padding="same", activation='relu')(T_C1_layer)
    T_C1_layer = Conv1D(128, 3, padding="same", activation='relu')(T_C1_layer)
    T_C1_layer = MaxPool1D(2)(T_C1_layer)
    T_C1_layer = GlobalAveragePooling1D()(T_C1_layer)

    Time_C2_Input = Input(shape=(900,1))
    T_C2_layer = Conv1D(16, 3, padding="same", activation='relu')(Time_C2_Input)
    T_C2_layer = Conv1D(16, 3, padding="same", activation='relu')(T_C2_layer)
    T_C2_layer = MaxPool1D(2)(T_C2_layer)
    T_C2_layer = Conv1D(32, 3, padding="same", activation='relu')(T_C2_layer)
    T_C2_layer = Conv1D(32, 3, padding="same", activation='relu')(T_C2_layer)
    T_C2_layer = MaxPool1D(2)(T_C2_layer)
    T_C2_layer = Conv1D(64, 3, padding="same", activation='relu')(T_C2_layer)
    T_C2_layer = Conv1D(64, 3, padding="same", activation='relu')(T_C2_layer)
    T_C2_layer = MaxPool1D(2)(T_C2_layer)
    T_C2_layer = Conv1D(128, 3, padding="same", activation='relu')(T_C2_layer)
    T_C2_layer = Conv1D(128, 3, padding="same", activation='relu')(T_C2_layer)
    T_C2_layer = MaxPool1D(2)(T_C2_layer)
    T_C2_layer = Conv1D(128, 3, padding="same", activation='relu')(T_C2_layer)
    T_C2_layer = Conv1D(128, 3, padding="same", activation='relu')(T_C2_layer)
    T_C2_layer = MaxPool1D(2)(T_C2_layer)
    T_C2_layer = GlobalAveragePooling1D()(T_C2_layer)

    Time_C3_Input = Input(shape=(900,1))
    T_C3_layer = Conv1D(16, 3, padding="same", activation='relu')(Time_C3_Input)
    T_C3_layer = Conv1D(16, 3, padding="same", activation='relu')(T_C3_layer)
    T_C3_layer = MaxPool1D(2)(T_C3_layer)
    T_C3_layer = Conv1D(32, 3, padding="same", activation='relu')(T_C3_layer)
    T_C3_layer = Conv1D(32, 3, padding="same", activation='relu')(T_C3_layer)
    T_C3_layer = MaxPool1D(2)(T_C3_layer)
    T_C3_layer = Conv1D(64, 3, padding="same", activation='relu')(T_C3_layer)
    T_C3_layer = Conv1D(64, 3, padding="same", activation='relu')(T_C3_layer)
    T_C3_layer = MaxPool1D(2)(T_C3_layer)
    T_C3_layer = Conv1D(128, 3, padding="same", activation='relu')(T_C3_layer)
    T_C3_layer = Conv1D(128, 3, padding="same", activation='relu')(T_C3_layer)
    T_C3_layer = MaxPool1D(2)(T_C3_layer)
    T_C3_layer = Conv1D(128, 3, padding="same", activation='relu')(T_C3_layer)
    T_C3_layer = Conv1D(128, 3, padding="same", activation='relu')(T_C3_layer)
    T_C3_layer = MaxPool1D(2)(T_C3_layer)
    T_C3_layer = GlobalAveragePooling1D()(T_C3_layer)

    # FFT_V_Input = Input(shape=(900,1))
    # FFT_V_layer = Conv1D(16, 3, padding="same", activation='relu')(FFT_V_Input)
    # FFT_V_layer = Conv1D(16, 3, padding="same", activation='relu')(FFT_V_layer)
    # FFT_V_layer = MaxPool1D(2)(FFT_V_layer)
    # FFT_V_layer = Conv1D(32, 3, padding="same", activation='relu')(FFT_V_layer)
    # FFT_V_layer = Conv1D(32, 3, padding="same", activation='relu')(FFT_V_layer)
    # FFT_V_layer = MaxPool1D(2)(FFT_V_layer)
    # FFT_V_layer = Conv1D(64, 3, padding="same", activation='relu')(FFT_V_layer)
    # FFT_V_layer = Conv1D(64, 3, padding="same", activation='relu')(FFT_V_layer)
    # FFT_V_layer = MaxPool1D(2)(FFT_V_layer)
    # FFT_V_layer = Conv1D(128, 3, padding="same", activation='relu')(FFT_V_layer)
    # FFT_V_layer = Conv1D(128, 3, padding="same", activation='relu')(FFT_V_layer)
    # FFT_V_layer = MaxPool1D(2)(FFT_V_layer)
    # FFT_V_layer = Conv1D(128, 3, padding="same", activation='relu')(FFT_V_layer)
    # FFT_V_layer = Conv1D(128, 3, padding="same", activation='relu')(FFT_V_layer)
    # FFT_V_layer = MaxPool1D(2)(FFT_V_layer)
    # FFT_V_layer = GlobalAveragePooling1D()(FFT_V_layer)
    #
    # FFT_C1_Input = Input(shape=(900,1))
    # FFT_C1_layer = Conv1D(16, 3, padding="same", activation='relu')(FFT_C1_Input)
    # FFT_C1_layer = Conv1D(16, 3, padding="same", activation='relu')(FFT_C1_layer)
    # FFT_C1_layer = MaxPool1D(2)(FFT_C1_layer)
    # FFT_C1_layer = Conv1D(32, 3, padding="same", activation='relu')(FFT_C1_layer)
    # FFT_C1_layer = Conv1D(32, 3, padding="same", activation='relu')(FFT_C1_layer)
    # FFT_C1_layer = MaxPool1D(2)(FFT_C1_layer)
    # FFT_C1_layer = Conv1D(64, 3, padding="same", activation='relu')(FFT_C1_layer)
    # FFT_C1_layer = Conv1D(64, 3, padding="same", activation='relu')(FFT_C1_layer)
    # FFT_C1_layer = MaxPool1D(2)(FFT_C1_layer)
    # FFT_C1_layer = Conv1D(128, 3, padding="same", activation='relu')(FFT_C1_layer)
    # FFT_C1_layer = Conv1D(128, 3, padding="same", activation='relu')(FFT_C1_layer)
    # FFT_C1_layer = MaxPool1D(2)(FFT_C1_layer)
    # FFT_C1_layer = Conv1D(128, 3, padding="same", activation='relu')(FFT_C1_layer)
    # FFT_C1_layer = Conv1D(128, 3, padding="same", activation='relu')(FFT_C1_layer)
    # FFT_C1_layer = MaxPool1D(2)(FFT_C1_layer)
    # FFT_C1_layer = GlobalAveragePooling1D()(FFT_C1_layer)
    #
    # FFT_C2_Input = Input(shape=(900,1))
    # FFT_C2_layer = Conv1D(16, 3, padding="same", activation='relu')(FFT_C2_Input)
    # FFT_C2_layer = Conv1D(16, 3, padding="same", activation='relu')(FFT_C2_layer)
    # FFT_C2_layer = MaxPool1D(2)(FFT_C2_layer)
    # FFT_C2_layer = Conv1D(32, 3, padding="same", activation='relu')(FFT_C2_layer)
    # FFT_C2_layer = Conv1D(32, 3, padding="same", activation='relu')(FFT_C2_layer)
    # FFT_C2_layer = MaxPool1D(2)(FFT_C2_layer)
    # FFT_C2_layer = Conv1D(64, 3, padding="same", activation='relu')(FFT_C2_layer)
    # FFT_C2_layer = Conv1D(64, 3, padding="same", activation='relu')(FFT_C2_layer)
    # FFT_C2_layer = MaxPool1D(2)(FFT_C2_layer)
    # FFT_C2_layer = Conv1D(128, 3, padding="same", activation='relu')(FFT_C2_layer)
    # FFT_C2_layer = Conv1D(128, 3, padding="same", activation='relu')(FFT_C2_layer)
    # FFT_C2_layer = MaxPool1D(2)(FFT_C2_layer)
    # FFT_C2_layer = Conv1D(128, 3, padding="same", activation='relu')(FFT_C2_layer)
    # FFT_C2_layer = Conv1D(128, 3, padding="same", activation='relu')(FFT_C2_layer)
    # FFT_C2_layer = MaxPool1D(2)(FFT_C2_layer)
    # FFT_C2_layer = GlobalAveragePooling1D()(FFT_C2_layer)
    #
    # FFT_C3_Input = Input(shape=(900,1))
    # FFT_C3_layer = Conv1D(16, 3, padding="same", activation='relu')(FFT_C3_Input)
    # FFT_C3_layer = Conv1D(16, 3, padding="same", activation='relu')(FFT_C3_layer)
    # FFT_C3_layer = MaxPool1D(2)(FFT_C3_layer)
    # FFT_C3_layer = Conv1D(32, 3, padding="same", activation='relu')(FFT_C3_layer)
    # FFT_C3_layer = Conv1D(32, 3, padding="same", activation='relu')(FFT_C3_layer)
    # FFT_C3_layer = MaxPool1D(2)(FFT_C3_layer)
    # FFT_C3_layer = Conv1D(64, 3, padding="same", activation='relu')(FFT_C3_layer)
    # FFT_C3_layer = Conv1D(64, 3, padding="same", activation='relu')(FFT_C3_layer)
    # FFT_C3_layer = MaxPool1D(2)(FFT_C3_layer)
    # FFT_C3_layer = Conv1D(128, 3, padding="same", activation='relu')(FFT_C3_layer)
    # FFT_C3_layer = Conv1D(128, 3, padding="same", activation='relu')(FFT_C3_layer)
    # FFT_C3_layer = MaxPool1D(2)(FFT_C3_layer)
    # FFT_C3_layer = Conv1D(128, 3, padding="same", activation='relu')(FFT_C3_layer)
    # FFT_C3_layer = Conv1D(128, 3, padding="same", activation='relu')(FFT_C3_layer)
    # FFT_C3_layer = MaxPool1D(2)(FFT_C3_layer)
    # FFT_C3_layer = GlobalAveragePooling1D()(FFT_C3_layer)
    #
    # MFCC_V_Input = Input(shape=(90, 10, 1))
    # MFCC_V_layer = Conv2D(16, (3, 3), padding="same", activation='relu')(MFCC_V_Input)
    # MFCC_V_layer = Conv2D(16, (3, 3), padding="same", activation='relu')(MFCC_V_layer)
    # MFCC_V_layer = MaxPool2D((2,2),1)(MFCC_V_layer)
    # MFCC_V_layer = Conv2D(32, (3, 3), padding="same",activation='relu')(MFCC_V_layer)
    # MFCC_V_layer = Conv2D(32, (3, 3), padding="same",activation='relu')(MFCC_V_layer)
    # MFCC_V_layer = MaxPool2D((2,2),1)(MFCC_V_layer)
    # MFCC_V_layer = Conv2D(64, (3, 3), padding="same", activation='relu')(MFCC_V_layer)
    # MFCC_V_layer = Conv2D(64, (3, 3), padding="same", activation='relu')(MFCC_V_layer)
    # MFCC_V_layer = MaxPool2D((2,2),1)(MFCC_V_layer)
    # MFCC_V_layer = Conv2D(128, (3, 3), padding="same", activation='relu')(MFCC_V_layer)
    # MFCC_V_layer = Conv2D(128, (3, 3), padding="same", activation='relu')(MFCC_V_layer)
    # MFCC_V_layer = MaxPool2D((2, 2), 1)(MFCC_V_layer)
    # MFCC_V_layer = Conv2D(128, (3, 3), padding="same", activation='relu')(MFCC_V_layer)
    # MFCC_V_layer = Conv2D(128, (3, 3), padding="same", activation='relu')(MFCC_V_layer)
    # MFCC_V_layer = MaxPool2D((2, 2), 1)(MFCC_V_layer)
    # MFCC_V_layer = GlobalAveragePooling2D()(MFCC_V_layer)
    #
    # MFCC_C1_Input = Input(shape=(90, 10, 1))
    # MFCC_C1_layer = Conv2D(16, (3, 3), padding="same", activation='relu')(MFCC_C1_Input)
    # MFCC_C1_layer = Conv2D(16, (3, 3), padding="same", activation='relu')(MFCC_C1_layer)
    # MFCC_C1_layer = MaxPool2D((2,2),1)(MFCC_C1_layer)
    # MFCC_C1_layer = Conv2D(32, (3, 3), padding="same", activation='relu')(MFCC_C1_layer)
    # MFCC_C1_layer = Conv2D(32, (3, 3), padding="same", activation='relu')(MFCC_C1_layer)
    # MFCC_C1_layer = MaxPool2D((2,2),1)(MFCC_C1_layer)
    # MFCC_C1_layer = Conv2D(64, (3, 3), padding="same", activation='relu')(MFCC_C1_layer)
    # MFCC_C1_layer = Conv2D(64, (3, 3), padding="same", activation='relu')(MFCC_C1_layer)
    # MFCC_C1_layer = MaxPool2D((2,2),1)(MFCC_C1_layer)
    # MFCC_C1_layer = Conv2D(128, (3, 3), padding="same", activation='relu')(MFCC_C1_layer)
    # MFCC_C1_layer = Conv2D(128, (3, 3), padding="same", activation='relu')(MFCC_C1_layer)
    # MFCC_C1_layer = MaxPool2D((2, 2), 1)(MFCC_C1_layer)
    # MFCC_C1_layer = Conv2D(128, (3, 3), padding="same", activation='relu')(MFCC_C1_layer)
    # MFCC_C1_layer = Conv2D(128, (3, 3), padding="same", activation='relu')(MFCC_C1_layer)
    # MFCC_C1_layer = MaxPool2D((2, 2), 1)(MFCC_C1_layer)
    # MFCC_C1_layer = GlobalAveragePooling2D()(MFCC_C1_layer)
    #
    # MFCC_C2_Input = Input(shape=(90, 10, 1))
    # MFCC_C2_layer = Conv2D(16, (3, 3), padding="same", activation='relu')(MFCC_C2_Input)
    # MFCC_C2_layer = Conv2D(16, (3, 3), padding="same", activation='relu')(MFCC_C2_layer)
    # MFCC_C2_layer = MaxPool2D((2,2),1)(MFCC_C2_layer)
    # MFCC_C2_layer = Conv2D(32, (3, 3), padding="same",activation='relu')(MFCC_C2_layer)
    # MFCC_C2_layer = Conv2D(32, (3, 3), padding="same",activation='relu')(MFCC_C2_layer)
    # MFCC_C2_layer = MaxPool2D((2,2),1)(MFCC_C2_layer)
    # MFCC_C2_layer = Conv2D(64, (3, 3), padding="same", activation='relu')(MFCC_C2_layer)
    # MFCC_C2_layer = Conv2D(64, (3, 3), padding="same", activation='relu')(MFCC_C2_layer)
    # MFCC_C2_layer = MaxPool2D((2,2),1)(MFCC_C2_layer)
    # MFCC_C2_layer = Conv2D(128, (3, 3), padding="same", activation='relu')(MFCC_C2_layer)
    # MFCC_C2_layer = Conv2D(128, (3, 3), padding="same", activation='relu')(MFCC_C2_layer)
    # MFCC_C2_layer = MaxPool2D((2, 2), 1)(MFCC_C2_layer)
    # MFCC_C2_layer = Conv2D(128, (3, 3), padding="same", activation='relu')(MFCC_C2_layer)
    # MFCC_C2_layer = Conv2D(128, (3, 3), padding="same", activation='relu')(MFCC_C2_layer)
    # MFCC_C2_layer = MaxPool2D((2, 2), 1)(MFCC_C2_layer)
    # MFCC_C2_layer = GlobalAveragePooling2D()(MFCC_C2_layer)
    #
    # MFCC_C3_Input = Input(shape=(90, 10, 1))
    # MFCC_C3_layer = Conv2D(16, (3, 3), padding="same", activation='relu')(MFCC_C3_Input)
    # MFCC_C3_layer = Conv2D(16, (3, 3), padding="same", activation='relu')(MFCC_C3_layer)
    # MFCC_C3_layer = MaxPool2D((2,2),1)(MFCC_C3_layer)
    # MFCC_C3_layer = Conv2D(32, (3, 3), padding="same",activation='relu')(MFCC_C3_layer)
    # MFCC_C3_layer = Conv2D(32, (3, 3), padding="same",activation='relu')(MFCC_C3_layer)
    # MFCC_C3_layer = MaxPool2D((2,2),1)(MFCC_C3_layer)
    # MFCC_C3_layer = Conv2D(64, (3, 3), padding="same", activation='relu')(MFCC_C3_layer)
    # MFCC_C3_layer = Conv2D(64, (3, 3), padding="same", activation='relu')(MFCC_C3_layer)
    # MFCC_C3_layer = MaxPool2D((2,2),1)(MFCC_C3_layer)
    # MFCC_C3_layer = Conv2D(128, (3, 3), padding="same",activation='relu')(MFCC_C3_layer)
    # MFCC_C3_layer = Conv2D(128, (3, 3), padding="same",activation='relu')(MFCC_C3_layer)
    # MFCC_C3_layer = MaxPool2D((2,2),1)(MFCC_C3_layer)
    # MFCC_C3_layer = Conv2D(128, (3, 3), padding="same", activation='relu')(MFCC_C3_layer)
    # MFCC_C3_layer = Conv2D(128, (3, 3), padding="same", activation='relu')(MFCC_C3_layer)
    # MFCC_C3_layer = MaxPool2D((2,2),1)(MFCC_C3_layer)
    # MFCC_C3_layer = GlobalAveragePooling2D()(MFCC_C3_layer)

    Dense_layer = tf.keras.layers.concatenate([T_C1_layer, T_C2_layer, T_C3_layer, T_V_layer])
    output = Dense(5, activation='softmax')(Dense_layer)

    # Define the model
    model = tf.keras.Model(inputs=[Time_V_Input, Time_C1_Input, Time_C2_Input, Time_C3_Input], outputs=output)
    model.summary()

    # Compile the model
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=optimizers.Adam(learning_rate=0.001),
                  metrics=['accuracy'])


    #train the model
    history = model.fit([Time_data_V, Time_data_C1, Time_data_C2, Time_data_C3], Y_train,
                         epochs=50,
                         batch_size=32,
                         validation_split=0.1,
                         validation_steps=4)
    # Save model
    model.save(f'{Result_path}/{Model_name}.h5')

    # Visualize training results
    train_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    val_loss = history.history['val_loss']
    train_loss = history.history['loss']

    plt.subplot(2, 1, 1)
    plt.plot(train_acc, '-bo', label='Training acc')
    plt.plot(val_acc, '-ro', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(['train', 'validation'])

    plt.subplot(2, 1, 2)
    plt.plot(train_loss, '-bo', label='train loss')
    plt.plot(val_loss, '-ro', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(['train', 'validation'])

    a1 = pd.DataFrame(train_acc)
    a2 = pd.DataFrame(val_acc)
    a3 = pd.DataFrame(train_loss)
    a4 = pd.DataFrame(val_loss)
    result = pd.concat([a1, a2, a3, a4], axis=1)
    result.to_csv(f'{Result_path}/{Modle_name}_Train_History.csv', index=False)
    plt.savefig(f'{Result_path}/{Modle_name}_Train_History.tiff')
    plt.show()

if job == 2:
    # Load the pre-trained model
    model = load_model(f'{Result_path}/{Modle_name}.h5')

    # Load test data
    Test_time_data_V = np.load(f'{Data_path}/Test_time_data_V.npy', allow_pickle=True)
    Test_time_data_C1 = np.load(f'{Data_path}/Test_time_data_C1.npy', allow_pickle=True)
    Test_time_data_C2 = np.load(f'{Data_path}/Test_time_data_C2.npy', allow_pickle=True)
    Test_time_data_C3 = np.load(f'{Data_path}/Test_time_data_C3.npy', allow_pickle=True)

    # Test_FFT_data_V = np.load(f'{Data_path}/Test_FFT_data_V.npy', allow_pickle=True)
    # Test_FFT_data_C1 = np.load(f'{Data_path}/Test_FFT_data_C1.npy', allow_pickle=True)
    # Test_FFT_data_C2 = np.load(f'{Data_path}/Test_FFT_data_C2.npy', allow_pickle=True)
    # Test_FFT_data_C3 = np.load(f'{Data_path}/Test_FFT_data_C3.npy', allow_pickle=True)
    #
    # Test_MFCC_data_V = np.load(f'{Data_path}/Test_MFCC_data_V.npy', allow_pickle=True)
    # Test_MFCC_data_C1 = np.load(f'{Data_path}/Test_MFCC_data_C1.npy', allow_pickle=True)
    # Test_MFCC_data_C2 = np.load(f'{Data_path}/Test_MFCC_data_C2.npy', allow_pickle=True)
    # Test_MFCC_data_C3 = np.load(f'{Data_path}/Test_MFCC_data_C3.npy', allow_pickle=True)

    Y_test = np.load(f'{Data_path}/Y_test.npy')

    print(f"Test_time_data_V shape : {Test_time_data_V.shape}\n"
          f"Test_time_data_C1 shape : {Test_time_data_C1.shape}\n"
          f"Test_time_data_C2 shape : {Test_time_data_C2.shape}\n"
          f"Test_time_data_C3 shape : {Test_time_data_C3.shape}\n"

          # f"Test_FFT_data_V shape : {Test_FFT_data_V.shape}\n"
          # f"Test_FFT_data_C1 shape : {Test_FFT_data_C1.shape}\n"
          # f"Test_FFT_data_C2 shape : {Test_FFT_data_C2.shape}\n"
          # f"Test_FFT_data_C3 shape : {Test_FFT_data_C3.shape}\n"
          # 
          # f"Test_MFCC_data_V shape : {Test_MFCC_data_V.shape}\n"
          # f"Test_MFCC_data_C1 shape : {Test_MFCC_data_C1.shape}\n"
          # f"Test_MFCC_data_C2 shape : {Test_MFCC_data_C2.shape}\n"
          # f"Test_MFCC_data_C3 shape : {Test_MFCC_data_C3.shape}\n"

          f"Y_test shape : {Y_test.shape}")

    # Evaluate model on the test data
    score = model.evaluate([Test_time_data_V, Test_time_data_C1, Test_time_data_C2, Test_time_data_C3],
                            Y_test, verbose=0)
    print(f"Test loss: {score[0]}\n Test accuracy: {score[1]}")

    # Save test results to CSV
    a1 = pd.DataFrame(score)
    a1.to_csv(f'{Result_path}/{Modle_name}_Noisetest_result.csv',
              index=False, mode='a', header=False)

    # Generate and display confusion matrix
    Y_test_categorical = tf.keras.utils.to_categorical(Y_test, 5)
    y_pred = model.predict([Test_time_data_V, Test_time_data_C1, Test_time_data_C2, Test_time_data_C3])
    np.save(f'{Result_path}/{Modle_name}_predict(5)_for t-sne.npy', y_pred)
    YY_test = np.argmax(Y_test_categorical, axis=1)
    YY_pred = np.argmax(y_pred, axis=1)

    label = ['0', '1', '2', '3', '4']
    report = confusion_matrix(YY_test, YY_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=report, display_labels=label)
    disp.plot(cmap='Blues')
    plt.savefig(f'{Result_path}/{Modle_name}_ConfusionMatrix_NoiseX.tiff')

    ## Test for different noise levels
    noiselist = [0.25, 0.5, 0.75, 1, 1.25, 1.5]
    for noise in noiselist:
        noise = round(noise, 2)
        print(f"noise: {noise}")

        Test_time_data_V = np.load(f'{Data_path}/Test_time_data_V_Noise{noise}.npy', allow_pickle=True)
        Test_time_data_C1 = np.load(f'{Data_path}/Test_time_data_C1_Noise{noise}.npy', allow_pickle=True)
        Test_time_data_C2 = np.load(f'{Data_path}/Test_time_data_C2_Noise{noise}.npy', allow_pickle=True)
        Test_time_data_C3 = np.load(f'{Data_path}/Test_time_data_C3_Noise{noise}.npy', allow_pickle=True)

        # Test_FFT_data_V = np.load(f'{Data_path}/Test_FFT_data_V_Noise{noise}.npy', allow_pickle=True)
        # Test_FFT_data_C1 = np.load(f'{Data_path}/Test_FFT_data_C1_Noise{noise}.npy', allow_pickle=True)
        # Test_FFT_data_C2 = np.load(f'{Data_path}/Test_FFT_data_C2_Noise{noise}.npy', allow_pickle=True)
        # Test_FFT_data_C3 = np.load(f'{Data_path}/Test_FFT_data_C3_Noise{noise}.npy', allow_pickle=True)
        #
        # Test_MFCC_data_V = np.load(f'{Data_path}/Test_MFCC_data_V_Noise{noise}.npy', allow_pickle=True)
        # Test_MFCC_data_C1 = np.load(f'{Data_path}/Test_MFCC_data_C1_Noise{noise}.npy', allow_pickle=True)
        # Test_MFCC_data_C2 = np.load(f'{Data_path}/Test_MFCC_data_C2_Noise{noise}.npy', allow_pickle=True)
        # Test_MFCC_data_C3 = np.load(f'{Data_path}/Test_MFCC_data_C3_Noise{noise}.npy', allow_pickle=True)

        Y_test = np.load(f'{Data_path}/Y_test.npy')

        print(f"Test_time_data_V shape : {Test_time_data_V.shape}\n"
              f"Test_time_data_C1 shape : {Test_time_data_C1.shape}\n"
              f"Test_time_data_C2 shape : {Test_time_data_C2.shape}\n"
              f"Test_time_data_C3 shape : {Test_time_data_C3.shape}\n"

              # f"Test_FFT_data_V shape : {Test_FFT_data_V.shape}\n"
              # f"Test_FFT_data_C1 shape : {Test_FFT_data_C1.shape}\n"
              # f"Test_FFT_data_C2 shape : {Test_FFT_data_C2.shape}\n"
              # f"Test_FFT_data_C3 shape : {Test_FFT_data_C3.shape}\n"
              # 
              # f"Test_MFCC_data_V shape : {Test_MFCC_data_V.shape}\n"
              # f"Test_MFCC_data_C1 shape : {Test_MFCC_data_C1.shape}\n"
              # f"Test_MFCC_data_C2 shape : {Test_MFCC_data_C2.shape}\n"
              # f"Test_MFCC_data_C3 shape : {Test_MFCC_data_C3.shape}\n"

              f"Y_test shape : {Y_test.shape}")

        # Evaluate model on the test data
        score = model.evaluate([Test_time_data_V, Test_time_data_C1, Test_time_data_C2, Test_time_data_C3],
                               Y_test, verbose=0)
        print(f"Test loss: {score[0]}\n Test accuracy: {score[1]}")

        # Save test results to CSV
        a1 = pd.DataFrame(score)
        a1.to_csv(f'{Result_path}/{Modle_name}_Noisetest_result.csv',
                  index=False, mode='a', header=False)

        # Generate and display confusion matrix
        Y_test_categorical = tf.keras.utils.to_categorical(Y_test, 5)
        y_pred = model.predict([Test_time_data_V, Test_time_data_C1, Test_time_data_C2, Test_time_data_C3])
        np.save(f'{Result_path}/{Modle_name}_predict(5)_Noise{noise}_for t-sne.npy', y_pred)
        YY_test = np.argmax(Y_test_categorical, axis=1)
        YY_pred = np.argmax(y_pred, axis=1)

        label = ['0', '1', '2', '3', '4']
        report = confusion_matrix(YY_test, YY_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=report, display_labels=label)
        disp.plot(cmap='Blues')
        plt.savefig(f'{Result_path}/{Modle_name}_ConfusionMatrix_Noise{noise}.tiff')

    plt.show()
