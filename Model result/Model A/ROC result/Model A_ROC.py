import numpy as np
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import pandas as pd
from sklearn.metrics import roc_auc_score

Model_name = 'Cross-level network #1'

Data_path = 'Project Data Path1'
Result_path = 'Project Data Path2'

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

y_score = model.predict([Test_time_data_V, Test_time_data_C1, Test_time_data_C2, Test_time_data_C3])
print(y_score.shape)

# ROC curves and AUC calculations by class (OvR method)
fpr = dict()
tpr = dict()
roc_auc = dict()

num_classes = y_score.shape[1]

for class_idx in range(num_classes):
    fpr[class_idx], tpr[class_idx], _ = roc_curve((Y_test == class_idx).astype(int), y_score[:, class_idx])
    roc_auc[class_idx] = auc(fpr[class_idx], tpr[class_idx])

# Create Result DataFrame
roc_auc_df = pd.DataFrame.from_dict(roc_auc, orient='index', columns=['AUC'])
fpr_df = pd.DataFrame.from_dict(fpr, orient='index')
tpr_df = pd.DataFrame.from_dict(tpr, orient='index')

for class_idx in range(num_classes):
    class_result = pd.DataFrame({
        'FPR': fpr[class_idx],
        'TPR': tpr[class_idx],
        'AUC': [roc_auc[class_idx]] * len(fpr[class_idx])})
    class_result.to_csv(f'{Result_path}/{Model_name}_ROC_result_class_{class_idx}.csv', index=False)

#################################################################################

noise = str(0.25)
noiselist = [0.25, 0.5, 0.75, 1, 1.25, 1.5]

for noise in noiselist:
    noise = round(noise, 2)
    print(f"noise: {noise}")

    # Load test data
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

    y_score = model.predict([Test_time_data_V, Test_time_data_C1, Test_time_data_C2, Test_time_data_C3])
    print(y_score.shape)

    # ROC curves and AUC calculations by class (OvR method)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    num_classes = y_score.shape[1]

    for class_idx in range(num_classes):
        fpr[class_idx], tpr[class_idx], _ = roc_curve((Y_test == class_idx).astype(int), y_score[:, class_idx])
        roc_auc[class_idx] = auc(fpr[class_idx], tpr[class_idx])

    # Create Result DataFrame
    roc_auc_df = pd.DataFrame.from_dict(roc_auc, orient='index', columns=['AUC'])
    fpr_df = pd.DataFrame.from_dict(fpr, orient='index')
    tpr_df = pd.DataFrame.from_dict(tpr, orient='index')

    for class_idx in range(num_classes):
        class_result = pd.DataFrame({
            'FPR': fpr[class_idx],
            'TPR': tpr[class_idx],
            'AUC': [roc_auc[class_idx]] * len(fpr[class_idx])})
        class_result.to_csv(f'{Result_path}/{Model_name}_Noise_{noise}_ROC_result_class_{class_idx}.csv', index=False)

    #################################################################################