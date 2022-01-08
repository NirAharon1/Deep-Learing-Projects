# Welcome to my Deep Learing Projects

###  <span style="color:red">Project 1 - Deepfake voice detection text</span>


![Project Image](https://github.com/NirAharon1/Deep-Learing-Projects/blob/main/WavToSpecToClassification.PNG)


### Goals
Training neural net to classify voice calls into two categories spoofed or bona fide audio files.

Voice spoofing is a major security concern and a breeding ground for deep fake attacks and as such, the field of finding automatic solutions is in high demand nowadays. 


### Dataset
The ASVspoof 2019 challenge provides 25Gb of audio data. 
The LA subset of the provided dataset includes bona fide speech and different kinds of TTS and VC spoofing attacks. 
Training and Development sets share the same 6 attacks (A01-A06), consisting of 4 TTS and 2 VC algorithms. 
In the test set, there are 11 unknown attacks (A07- A15, A17, A18) including combinations of different TTS and VC attacks. 
The test set also includes two attacks (A16, A19) that use the same algorithms as two of the attacks (A04, A06) in the training set but are trained with different data. 

link to ASVspoof 2019 challenge: https://www.asvspoof.org/index2019.html

### utility functions
The initial input was ten-second audio, I extract to most important one second from the hole ten seconds using NumPy trapezoidal rule:

```python
def get_1_sec(self,audio_index):
    """return the loudest 1 second from the audio data"""
    sound_data,sample_rate = self.audio_data(audio_index)
    sum_of_sq = []
    step = 1000 
    max_arg = 0
    for i in range(0,len(sound_data)-sample_rate,step):
        sum_of_sq.append(np.trapz(abs(sound_data[i:(i+sample_rate):100])))
        max_arg = step*(np.argmax(sum_of_sq))
    return sound_data[max_arg : max_arg+sample_rate]</p>
```

Then I convert the audio with mel-frequency cepstrum

```python
librosa.feature.mfcc()
```
### Net architecture
the net architecture was simple and made by three 2D convolutions layers following max-pull to four linear dense layers with RELU activation function  
