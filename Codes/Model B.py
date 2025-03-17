from tensorflow.keras import optimizers, layers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, Input, MaxPool2D,GlobalAveragePooling2D, Dense, concatenate, ZeroPadding2D, LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Flatten
import datetime
import tensorflow as tf
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# job 1 = Model Training
# job 2 = Model Testing

job = 1

Model_name = 'Model-1'

Data_path = 'Project Data Path1'
Result_path = 'Project Data Path2'

if job == 1:
    # Load Data // Data-level fusion data
    Time_data = np.load(f'{Data_path}/Train_time_data.npy', allow_pickle=True)
    # FFT_data = np.load(f'{Data_path}/Train_FFT_data.npy', allow_pickle=True)
    # MFCC_data = np.load(f'{Data_path}/Train_MFCC_data.npy', allow_pickle=True)
    Y_train = np.load(f'{Data_path}/Y_train.npy')
    print(f"Time_data shape : {Time_data.shape}\n"
          # f"FFT_data shape : {FFT_data.shape}\n"
          # f"MFCC_data shape : {MFCC_data.shape}\n"
          f"Y_train shape : {Y_train.shape}")

    ##Feature compression block-----------------------------------------------------------------------------------------
    Time_Input = Input(shape=(170,170,4))
    layer_A = Conv2D(16, (3, 3), 2, padding="same", activation="relu")(Time_Input)
    layer_A = ZeroPadding2D(((0, 1),(0, 1)))(layer_A)
    layer_A = Conv2D(32, (3, 3), 2, padding="same", activation="relu")(layer_A)
    layer_A = ZeroPadding2D(((1, 0),(1, 0)))(layer_A)
    layer_A = MaxPool2D((2, 2))(layer_A)
    layer_A = ZeroPadding2D(((4, 4)))(layer_A)
    layer_A = Conv2D(8, (3, 3),padding="same", activation="relu")(layer_A)

    # FFT_Input = Input(shape=(30, 30, 4))
    # MFCC_Input = Input(shape=(30, 30, 4))
    # feature_level_fusion = concatenate([layer_A, FFT_Input, MFCC_Input], axis=-1)

    ##Multidomain enhancement block-------------------------------------------------------------------------------------
    layer_X = Conv2D(16, (3, 3), padding="same", activation='relu')(layer_A)
    layer_X = Conv2D(16, (3, 3), padding="same", activation='relu')(layer_X)
    layer_X = MaxPool2D((2, 2))(layer_X)
    layer_X = Conv2D(32, (3, 3), padding="same", activation='relu')(layer_X)
    layer_X = Conv2D(32, (3, 3), padding="same", activation='relu')(layer_X)
    layer_X = MaxPool2D((2, 2))(layer_X)
    layer_X = Conv2D(64, (3, 3), padding="same", activation='relu')(layer_X)
    layer_X = Conv2D(64, (3, 3), padding="same", activation='relu')(layer_X)
    layer_X = MaxPool2D((2, 2))(layer_X)
    layer_X = GlobalAveragePooling2D()(layer_X)
    outputs = Dense(5, 'softmax')(layer_X)

    # Define the model
    model = Model(inputs=Time_Input, outputs=outputs)
    model.summary()

    # Compile the model
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=optimizers.Adam(learning_rate=0.001),
                  metrics=['accuracy'])

    # Train the model
    history = model.fit([Time_data, FFT_data, MFCC_data], Y_train,
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
    Test_time_data = np.load(f'{Data_path}/Test_time_data.npy', allow_pickle=True)
    # Test_FFT_data = np.load(f'{Data_path}/Test_FFT_data.npy', allow_pickle=True)
    # Test_MFCC_data = np.load(f'{Data_path}/Test_MFCC_data.npy', allow_pickle=True)
    Y_test = np.load(f'{Data_path}/Y_test.npy')
    print(f"Test_time_data shape : {Test_time_data.shape}\n"
          # f"Test_FFT_data shape : {Test_FFT_data.shape}\n"
          # f"Test_MFCC_data shape : {Test_MFCC_data.shape}\n"
          f"Y_test shape : {Y_test.shape}")

    # Evaluate model on the test data
    score = model.evaluate(Test_time_data, Y_test, verbose=0)
    print(f"Test loss: {score[0]}\n Test accuracy: {score[1]}")

    # Save test results to CSV
    a1 = pd.DataFrame(score)
    a1.to_csv(f'{Result_path}/{Modle_name}_Noisetest_result.csv',
              index=False, mode='a', header=False)

    # Generate and display confusion matrix
    Y_test_categorical = tf.keras.utils.to_categorical(Y_test, 5)
    y_pred = model.predict(Test_time_data)
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

        Test_time_data = np.load(f'{Data_path}/Test_time_data_Noise{noise}.npy', allow_pickle=True)
        # Test_FFT_data = np.load(f'{Data_path}/Test_FFT_data_Noise{noise}.npy', allow_pickle=True)
        # Test_MFCC_data = np.load(f'{Data_path}/Test_MFCC_data_Noise{noise}.npy', allow_pickle=True)
        Y_test = np.load(f'{Data_path}/Y_test.npy')
        print(f"Test_time_data shape : {Test_time_data.shape}\n"
              # f"Test_FFT_data shape : {Test_FFT_data.shape}\n"
              # f"Test_FFT_data shape : {Test_MFCC_data.shape}\n"
              f"Y_test shape : {Y_test.shape}")

        # Evaluate model on the test data
        score = model.evaluate([Test_time_data], Y_test, verbose=0)
        print(f"Test loss: {score[0]}\n Test accuracy: {score[1]}")

        # Save test results to CSV
        a1 = pd.DataFrame(score)
        a1.to_csv(f'{Result_path}/{Modle_name}_Noisetest_result.csv',
                  index=False, mode='a', header=False)

        Y_test_categorical = tf.keras.utils.to_categorical(Y_test, 5)
        y_pred = model.predict([Test_time_data])
        np.save(f'{Result_path}/{Modle_name}_predict(5)_Noise{noise}_for t-sne.npy', y_pred)
        YY_test = np.argmax(Y_test_categorical, axis=1)
        YY_pred = np.argmax(y_pred, axis=1)

        label = ['0', '1', '2', '3', '4']
        report = confusion_matrix(YY_test, YY_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=report, display_labels=label)
        disp.plot(cmap='Blues')
        plt.savefig(f'{Result_path}/{Modle_name}_ConfusionMatrix_Noise{noise}.tiff')

    plt.show()
