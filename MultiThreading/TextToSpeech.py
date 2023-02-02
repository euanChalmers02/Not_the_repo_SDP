import pyttsx3
MSG_CACHE_PATH = '/Users/euanchalmers/Desktop/SoundProject/Sound_Files/recorded_msg'

def save_msg_to_cache(input_text, file_name):
    if 'wav' not in file_name:
        file_name = file_name + '.wav'

    engine = pyttsx3.init()
    # engine.say("this is a test save of an audio recording")
    print('written to...', MSG_CACHE_PATH + "/" + file_name)
    engine.save_to_file(input_text, str(MSG_CACHE_PATH + "/" + file_name))
    # engine.runAndWait()
    engine.stop()

def play_msg_cache(file_name):
    if 'wav' not in file_name:
        file_name = file_name + '.wav'
    [y, sr] = librosa.load(MSG_CACHE_PATH + "/" + file_name)

    sd.play(y.transpose(), sr)

if __name__ == '__main__':
    save_msg_to_cache('Pause this now pls', 'pause')