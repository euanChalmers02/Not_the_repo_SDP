import numpy as np
import librosa
from scipy import signal
import sounddevice as sd
import time
import pyttsx3

from Setup import Setup

# adjustable output parameters (add these to the setup class??)
beep_pause = 800  # ms (can we standardise these)
beep_duration = 0.4  # sec
st = Setup(1280, 720, 79, 41)  # this should be cache and passed as an arg??? Not done due to unit tests
engine = pyttsx3.init()
engine.setProperty('rate', 200)

# static helper function
def closest_value(input_list, input_value):
    arr = np.asarray(input_list)
    i = (np.abs(arr - input_value)).argmin()
    return arr[i]

class Sound:
    RECHERCHE_SOURCE_FILES = '/Users/euanchalmers/Desktop/SoundProject/Sound_Files/IRC_1002/COMPENSATED/WAV'
    BEEP_SOUND_ONE = '/Users/euanchalmers/Desktop/SoundProject/Sound_Files/beep-10.wav'

    def convert_to_file(self):
        angle, elev = st.find_the_file_two(self.coord)

        angle = str(angle)
        elev = str(elev)

        if len(angle) != 3:
            add = 3 - len(angle)
            add = '0' * add
            bearing = add + angle
        else:
            bearing = angle

        if len(elev) != 3:
            add = 3 - len(elev)
            add = '0' * add
            elevation = add + elev
        else:
            elevation = elev

        return '/IRC_1002_C/IRC_1002_C_R0195_T' + str(bearing) + '_P' + str(elevation) + '.wav'

    def __init__(self, coord, distance, text, beep):
        self.coord = coord
        self.file = self.convert_to_file()
        self.distance = distance  # how to convey distance (increase frequency of beeps)
        self.text = text
        self.beep = beep
        self.Bin_Max = None
        self.freq = None  # this will be either true or false
        print('this is the file',self.file)

    # add distance later
    def create_3d(self):
        HRIR_RE, fs_H0 = librosa.load(self.RECHERCHE_SOURCE_FILES + self.file, sr=48000, mono=False)
        print('Sample rate = ', fs_H0)
        print('data dimentions = ', HRIR_RE.shape)

        [src_o, fs_s0] = librosa.load(self.BEEP_SOUND_ONE, mono=True, sr=48000)

        s_0_L = signal.fftconvolve(src_o, HRIR_RE[0, :])  # source left
        s_0_R = signal.fftconvolve(src_o, HRIR_RE[1, :])  # right

        Bin_Max = np.vstack([s_0_L, s_0_R]).transpose()
        Bin_Max = Bin_Max / np.max(np.abs(Bin_Max))
        # final end product

        # Plays the sound within the control system loop
        self.Bin_Max = Bin_Max
        self.freq = fs_s0

    def play(self):
        sd.play(self.Bin_Max, self.freq)
        sd.sleep(int(beep_duration * beep_pause))
        sd.stop()

    def textToSpeech(self):
        if self.text == "":
            return
        engine.say(self.text)
        engine.runAndWait()
        print('engine is finish')

        # this i currently the hold up between the text and the beeps
        # possibly lower latency
        # pyttsx3.speak("I will speak this text")
