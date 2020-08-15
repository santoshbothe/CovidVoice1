from python_speech_features import mfcc
from python_speech_features import logfbank
import scipy.io.wavfile as wav
import numpy

(rate,sig) = wav.read("./CovidVoice1/english.wav")
mfcc_feat = mfcc(sig,rate)
fbank_feat = logfbank(sig,rate)
a=fbank_feat
b=mfcc_feat
c=fbank_feat[1:3,:]
d=mfcc_feat[1:3,:]
#MFCC
numpy.savetxt("VoiceFeatureFbank.csv", c, delimiter=",")
numpy.savetxt("VoiceFeatureMFCCFl.csv", d, delimiter=",")