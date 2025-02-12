# **Cross-Level Multidomain Fusion Fault Diagnosis**

Rotating machinery plays a critical role in industrial systems, yet diagnosing faults in noisy environments remains challenging due to complex signal characteristics and diverse fault patterns. This study proposes a **Cross-Level Multidomain Fusion Fault Diagnosis**, utilizing various signal processing techniques to enhance robustness and accuracy in noisy conditions through the fusion of multimodal data.

## Key Features

- **Multimodal Input**: Vibration and current data are used as input sources, representing both time-domain and frequency-domain signals for comprehensive analysis.

- **Signal Processing Techniques**: Utilize various signal processing techniques, transforming the outputs into 2D images for effective feature extraction:
  - **Time Domain**: Recurrence Plot (RP) is applied for capturing dynamic temporal patterns.
  - **Frequency Domain**: Fast Fourier Transform (FFT) is used for frequency-domain feature extraction.
  - **Time-Frequency Domain**: Mel-frequency cepstral coefficients (MFCC) are employed to capture both time and frequency features simultaneously.

- **Convolutional Feature Extraction**: Leverages convolutional neural networks (CNNs) specifically designed for image feature extraction, ensuring robust and effective feature processing across multiple domains.

- **Cross-Level Fusion**: Utilizes both data-level fusion and feature-level fusion approach, enhancing fault diagnosis by combining raw data and extracted features.

- **Real-World Validation**: Evaluated on real-world datasets from industrial rotating machinery, demonstrating superior fault diagnosis performance even under noisy conditions and with diverse fault types.

## Dataset

The dataset covers four fault modes of rotating machines, including the normal state:
- Shaft misalignment
- Bearing failure
- Belt sagging
- Rotor imbalance

### Data Overview:
The dataset includes vibration and current data from three phases(R,S,T), with fault classifications for each.

- **Train Data:** 45,000 samples per phase (vibration and current data from each phase)
- **Test Data:** 5,000 samples per phase (vibration and current data from each phase)

## Multidomain Data Creation

This project involves generating multidomain data representations from vibration and current signals. Various signal processing techniques are applied to the vibration and current data, converting each dataset into a meaningful 2D representation for effective feature extraction and fault diagnosis.

### Time Domain
In the time domain, **Recurrence Plot (RP)** is applied to the raw signal. This method transforms the time-domain data into a 2D image that captures the dynamic temporal patterns of the signal, making it easier to identify potential faults.

### Frequency Domain
In the frequency domain, **Fast Fourier Transform (FFT)** is applied to the signal. FFT efficiently transforms the signal into its frequency components, allowing for detailed analysis of the signal’s spectral features. The resulting frequency data is then stacked row-wise to form a 2D representation, enabling further analysis for fault detection through frequency-domain features.

### Time-Frequency Domain
In the time-frequency domain, **Mel-frequency cepstral coefficients (MFCC)** are used. MFCC is a feature extraction technique inspired by the human auditory system, designed to capture the perceptually relevant features of the signal. MFCC generates a 2D image that simultaneously captures both time and frequency characteristics of the signal, providing a rich representation of the signal’s features across both domains.

### Final Data Shape

After applying the signal processing techniques, the final data shape for each domain is as follows:

- **Training Data**

  - **Time Domain Train Data**: 45,000 time samples with shape  `(45000, 170, 170, 4)`
  - **Frequency Domain Train Data**: 45,000 FFT samples with shape  `(45000, 30, 30, 4)`
  - **Time-Frequency Domain Train data**: 45,000 MFCC samples with shape  `(45000, 30, 30, 4)`

- **Test Data**

  - **Time Domain Test data**: 45,000 time samples with shape  `(5000, 170, 170, 4)`
  - **Frequency Domain Test Data**: 45,000 FFT samples with shape  `(5000, 30, 30, 4)`
  - **Time-Frequency Domain Test data**: 45,000 MFCC samples with shape  `(5000, 30, 30, 4)`

 ## Model Training and Testing Code

The main code for the **cross-Level multidomain fusion project** is named **Model D**. Additionally, models using only time-domain data and those employing only feature-level fusion are included for comparison.

<p align="center">
  <img src="https://github.com/user-attachments/assets/cc052024-d72e-4eed-96cb-3c90f7fb7791" alt="Image" />
</p>

The code for both training and testing the model is organized within the `/Codes` directory. Each model consists of two main components:

- **JOB1**: corresponds to the training code.
- **JOB2**: corresponds to the testing code, which also includes the noise testing.

## Model Results
The following details summarize the results from the **cross-level multidomain fusion fault diagnosis** project. Each model's training history, noise test results, ROC curve, t-SNE visualization, and associated .h5 file are provided. **Each model was independently trained five times** to ensure robustness and reliability of the results. **The test results are presented as the average and standard deviation of the five runs**. For the ROC curve and t-SNE visualization, the model closest to the average performance was selected to present the results.

- **Model Result Files**: Located in `/Model results/Model name`

Inside each directory, you can find:
- Train result
  - The model's `.h5` file
  - Training history graphs
- test result
  - Confusion matrices
- ROC result
  - One-vs-Rest (OvR) ROC curves for each class
- t-SNE result
  - Prediction results `.npy file` from the final layer
  - t-SNE visualization

## Noise test result
The noise levels were set to 22.8, -16.9, -13.3, -10.8, -8.8, and -7.3 dB (ref=1). The model accuracy results for each noise level are presented in the figure below.

<p align="center">
  <img src="https://github.com/user-attachments/assets/9eef47ef-1849-49f8-8f75-286f8f2d5440" alt="Image" />
</p>

For additional details, you can explore the respective files in the specified directory.
