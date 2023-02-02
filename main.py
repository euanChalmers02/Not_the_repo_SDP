from Sound import Sound
# import TextToSpeech.py as TX

if __name__ == '__main__':
    # example sound
    print('started')
    rx = Sound(220, 0, "Hellow this is a test speach", True)  # add each of the params for a sound here

    # we might want to cache each of the beeps instead to speed up the process
    # can otimise by loading in most meta data into the system on startup
    rx.create_3d()

    for y in range(3):
        rx.play()

    # rx.textToSpeech()

    # TX.save_msg_to_cache('HI MY NAME IS EUAN', 'euan')
    # TX.play_msg_cache('euan')
