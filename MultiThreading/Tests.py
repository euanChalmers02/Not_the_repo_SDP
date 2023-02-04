import time
import unittest

import TextToSpeech
from Sound import Sound

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

# def load_cache_sounds():
#     TextToSpeech.save_msg_to_cache(edhelp_sign, "edhelp_sign")


class scan_scenarios(unittest.TestCase):
    # real-image scenarios
    def test_real_lib_1(self):
        # load_cache_sounds()

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
    - all L
    - all R
    - all top
    - all bottom
    """

    # the need for pause/skip button
    # or the danger of recognising too much text?
    def test_too_much_information(self):
        obj1 = Sound([79,470], 0, "Object 1. " + "Il Calcio bistro", True)
        obj2 = Sound([65,291], 0, "Object 2. " + edhelp_sign, True)
        obj3 = Sound([348,195], 0, "Object 3. " + long_text, True)
        obj4 = Sound([700,279], 0, "Object 3. " + exit_sign, True)
        obj5 = Sound([1076,440], 0, "Object 3. " + toilet_sign, True)
        obj6 = Sound([937,148], 0, "Object 3. " + long_text, True)
        all_objects = [obj1, obj2, obj3, obj4, obj5, obj6]

        play_sounds(all_objects)

    def test_all_left(self):
        obj1 = Sound([151, 113], 0, "Object 1. " + edhelp_sign, True)
        obj2 = Sound([186, 266], 0, "Object 2. " + "Library Cafe", True)
        obj3 = Sound([293, 503], 0, "Object 2. " + "Library Cafe", True)
        all_objects = [obj1, obj2, obj3]

        play_sounds(all_objects)

    def test_all_centre(self):
        obj1 = Sound([421, 419], 0, "Object 1. " + edhelp_sign, True)
        obj2 = Sound([623, 117], 0, "Object 2. " + "Library Cafe", True)
        obj3 = Sound([644, 392], 0, "Object 2. " + "Library Cafe", True)
        all_objects = [obj1, obj2, obj3]

        play_sounds(all_objects)

    # the need for a "read long text" button:
    # starts reading long text automatically
    def test_reads_everything(self):
        obj1 = Sound([249, 348], 0, "Object 1. " + "Il Calcio bistro", True)
        obj2 = Sound([986, 253], 0, "Object 2. " + edhelp_sign, True)
        obj3 = Sound([749, 230], 0, "Object 3. " +
                    "On successful completion of this course, you should be able to: "+
                    "1. Working as members of a team in designing and implementing a complex and multi-faceted system"+
                    "2. Planning and monitoring the effort of a project to meet milestones and deadlines, within a limited time scale"+
                    "3. Drawing together knowledge and understanding of wide areas of software and hardware systems"+
                    "4. Demonstrating and presenting the outcome from a practical project"+
                    "5. Documenting the feasibility, design and development of a potential product", True)
        all_objects = [obj1, obj2, obj3]

        play_sounds(all_objects)

    # test/simulate with buttons.
    # three long texts to choose from
    def test_choose_a_text_to_read(self):
        obj1 = Sound([249, 348], 0, "Object 1. " + long_text, True)
        obj2 = Sound([986, 253], 0, "Object 2. " + long_text, True)
        obj3 = Sound([749, 230], 0, "Object 3. " + long_text, True)
        text1 = "On successful completion of this course, you should be able to: "\
                    "1. Working as members of a team in designing and implementing a complex and multi-faceted system"\
                    "2. Planning and monitoring the effort of a project to meet milestones and deadlines, within a limited time scale"\
                    "3. Drawing together knowledge and understanding of wide areas of software and hardware systems"\
                    "4. Demonstrating and presenting the outcome from a practical project"\
                    "5. Documenting the feasibility, design and development of a potential product"
        text2 = ""
        text3 = ""

if __name__ == '__main__':
    unittest.main()
