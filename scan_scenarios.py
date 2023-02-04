import time
import unittest

from Sound import Sound
from TextToSpeech import *

"""
Use https://www.mobilefish.com/services/record_mouse_coordinates/record_mouse_coordinates.php
to get "realistic" coords from test images.
object coordinates:
1 - [583,281], [1097,60]  
2 - [245,188], [914,274], [1179,258]
3 - [308,253]
4 - [249,348], [749,230], [986,253]
object type:
1 - [H, ST]
2 - [T, E, LT]
3 - [ST]
4 - [H, ST, LT]
"""

num_beeps = 3
pause_length = 1
# three trained signs; generic announcements
# generic_announcements = {
#     edhelp_sign: "Ed Help sign",
#     exit_sign: "Exit sign",
#     toilet_sign: "Toilet sign",
#     long_text: "Long text"
# }
edhelp_sign = "Ed Help sign"
exit_sign = "Exit sign"
toilet_sign = "Toilet sign"
long_text = "Long text"

def play_sounds(all_objects):
    for o in all_objects:
        o.create_3d()
        o.textToSpeech()
        for _ in range(num_beeps):
            o.play()
        time.sleep(pause_length)

def load_cache_sounds():
    TextToSpeech.save_msg_to_cache(edhelp_sign, "edhelp_sign")


class scan_scenarios(unittest.TestCase):
    # real-image scenarios
    def test_real_lib_1(self):
        load_cache_sounds()

        obj1 = Sound([583,281], 0, "Object 1. " + edhelp_sign, True)
        obj2 = Sound([1097,60], 0, "Object 2. " + "Library Cafe", True)
        all_objects = [obj1, obj2]

        play_sounds(all_objects)


    def test_real_lib_2(self):
        obj1 = Sound([245,188], 0, "Object 1. " + toilet_sign, True)
        obj2 = Sound([914,274], 0, "Object 2. " + exit_sign, True)
        obj3 = Sound([1179, 258], 0, "Object 3. " + long_text, True)
        all_objects = [obj1, obj2, obj3]

        play_sounds(all_objects)

    def test_real_cinema_3(self):
        obj1 = Sound([308,253], 0, "Object 1. " + "Cinema City", True)
        all_objects = [obj1]

        play_sounds(all_objects)

    def test_real_restaurant_4(self):
        obj1 = Sound([249,348], 0, "Object 1. " + "Il Calcio bistro", True)
        obj2 = Sound([986, 253], 0, "Object 2. " + edhelp_sign, True)
        obj3 = Sound([749, 230], 0, "Object 3. " + long_text, True)
        all_objects = [obj1, obj2, obj3]

        play_sounds(all_objects)

    # other synthetic scenarios.
    """
    - too many objects (5+)
    - 
    """

if __name__ == '__main__':
    unittest.main()
