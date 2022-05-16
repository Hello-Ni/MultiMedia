# Prerequisites
## Audio features
blues、classical、country、disco、hophop、jazz、metal、pop、reggas、rock
## Environments
python 3.9.6
### require packages
1. TensorFlow
2. Librosa
3. Numpy
4. Scikit-learn
5. JSON
6. Pytdub
## Execution
**Data preprocessing**
```
python preprocessing.py
```
**Model training**
> Already generate model model_RNN_LSTM to save the training time
```
python model.py
```
**Prediction**
> ex: python predict.py test_song.wav
```
python predict.py "File Path"

```
# Theory
For audio and music feature extraction in machine learning, we usually take mel-frequency cepstral coefficients(MFCCs) as extraction from song or audio.

<img src="https://i.imgur.com/W5zQd7V.jpg" width="600" height="300" />

But MFCC feature extraction is a way to extract only relevant information from an audio. In other words, the digital format not give us too much information about audio or song.Hences, we used Fast Fourier Transform(FFT) to represent the audio in **frequency domain**. The following figure displays the audio data in frequency and time doamins,called **Sepectogram**

<img src="https://i.imgur.com/3HA5Kkd.jpg" width="600" height="300" />

# Data Preprocessing
# Model
# Accuracy/Loss
After servel time of model training, epochs=50 is most stable accuracy/loss in training preocess.

validation accuracy: 0.824800144958496
validation loss: 0.73860031786463

**Accuracy**

<img src="https://i.imgur.com/uRR1zhn.png" width="600" height="300" />

**Loss**
<img src="https://i.imgur.com/epoW3O8.png" width="600" height="300" />