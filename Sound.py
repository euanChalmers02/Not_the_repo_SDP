import numpy as np
import librosa
import pyttsx3
from scipy import signal
import sounddevice as sd
import time


# import pyttsx3

# adjustable output parameters
beep_pause = 800
beep_duration = 0.4

# static helper function
def closest_value(input_list, input_value):
    arr = np.asarray(input_list)
    i = (np.abs(arr - input_value)).argmin()
    return arr[i]


def convert_to_file(coord):
    pass


def coord_to_bearing(x_val):
    bearing = 360-(((x_val-640)/640)*90) if x_val > 640 else 90-((x_val/640)*90)
    print("-- BEARING:", bearing)
    return bearing


def coord_to_elevation(y_val):
    if y_val < 180:
        elevation = 45
    elif y_val > 540:
        elevation = 315
    else:
        elevation = 0

    print("-- ELEVATION:", elevation)
    return elevation


class Sound:
    # Lea's paths:
    # RECHERCHE_SOURCE_FILES = '/Users/neo/Documents/SoundProject/Sound_Files/IRC_1002/COMPENSATED/WAV/'
    # TEST_SOURCE_SOUND = '/Users/neo/Documents/SoundProject/CantinaBand3.wav'
    # BEEP_SOUND_ONE = '/Users/neo/Documents/SoundProject/Sound_Files/beep-10.wav'

    # Euan's paths:
    # RECHERCHE_SOURCE_FILES = '/Users/euanchalmers/Desktop/SoundProject/Sound_Files/IRC_1002/COMPENSATED/WAV'
    # BEEP_SOUND_ONE = '/Users/euanchalmers/Desktop/SoundProject/Sound_Files/beep-10.wav'

    # TODO: need to make these relative

    RECHERCHE_SOURCE_FILES = '/Users/neo/Documents/SoundProject/Sound_Files/IRC_1002/COMPENSATED/WAV/'
    BEEP_SOUND_ONE = '/Users/neo/Documents/SoundProject/Sound_Files/beep-10.wav'

    def __init__(self, coord, distance, text, beep):
        self.coord = coord
        self.file = convert_to_file(coord)
        self.distance = distance  # how to convey distance (increase frequency of beeps)
        self.text = text
        self.beep = beep
        self.Bin_Max = None
        self.freq = None  # this will be either true or false
        self.bearing = coord_to_bearing(coord[0])
        self.elevation = coord_to_elevation(coord[1])

    # finds the best datset sound to use based on your inputs

    # setup the arrays in a json file or predone?
    def parse_input(self):
        # get all the diffrent items in the datset
        ELEVATIONS_IN_DATSET = [0, 15, 30, 45, 60, 75, 90, 315, 330, 345]
        ANGLES_IN_DATASET = []

        counter = 0
        for y in range(int(360 / 15) + 1):
            ANGLES_IN_DATASET.append(counter)
            counter = counter + 15

        angle = str(closest_value(ANGLES_IN_DATASET, self.bearing))
        elev = str(closest_value(ELEVATIONS_IN_DATSET, self.elevation))

        if len(angle) != 3:
            add = 3 - len(angle)
            add = '0' * add
            self.bearing = add + angle
        else:
            self.bearing = angle

        if len(elev) != 3:
            add = 3 - len(elev)
            add = '0' * add
            self.elevation = add + elev
        else:
            self.elevation = elev

        print(ANGLES_IN_DATASET)
        # print(self.bearing)

    def driver(self, file):
        HRIR_RE, fs_H0 = librosa.load(self.RECHERCHE_SOURCE_FILES + file, sr=48000, mono=False)
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
        print("playing")

    # add distance and elevation later
    def create_3d(self):
        start = (time.time())

        self.parse_input()
        base_file = '/IRC_1002_C/IRC_1002_C_R0195_T' + str(self.bearing) + '_P' + str(self.elevation) + '.wav'
        file = base_file
        self.driver(file)
        end = (time.time())
        print(end - start)

    def play(self):
        sd.play(self.Bin_Max, self.freq)
        sd.sleep(int(beep_duration * beep_pause))
        sd.stop()

    def textToSpeech(self):
        if self.text == "":
            return

        engine = pyttsx3.init()
        engine.say(self.text)
        engine.runAndWait()
        engine.stop()
