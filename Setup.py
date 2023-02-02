from Sound import closest_value


class Setup:
    def __init__(self, DEFAULT_CAMERA_WIDTH, DEFAULT_CAMERA_HEIGHT, DEFAULT_FIELD_OF_VIEW_WIDTH,
                 DEFAULT_FIELD_OF_VIEW_HEIGHT):

        self.all_angles_hoz = []

        x = 0
        self.all_angles_hoz.append(x)
        while x < DEFAULT_FIELD_OF_VIEW_WIDTH / 2:
            x = x + 15
            self.all_angles_hoz.append(x)

        self.all_angles_hoz.sort(reverse=True)

        x = 360
        while x > 360 - DEFAULT_FIELD_OF_VIEW_WIDTH / 2:
            x = x - 15
            self.all_angles_hoz.append(x)

        self.all_angle_ver = []

        x = 0
        self.all_angle_ver.append(x)
        while x < DEFAULT_FIELD_OF_VIEW_HEIGHT / 2:
            x = x + 15
            self.all_angle_ver.append(x)

        self.all_angle_ver.sort(reverse=True)

        part2 = []

        x = 345
        part2.append(x)
        while x > 345 - DEFAULT_FIELD_OF_VIEW_HEIGHT / 2:
            x = x - 15
            part2.append(x)

        part2.sort()
        self.all_angle_ver = self.all_angle_ver + part2

        # print(self.all_angle_ver)
        # print(self.all_angles_hoz)

        split_h = DEFAULT_CAMERA_WIDTH / len(self.all_angles_hoz)
        split_v = DEFAULT_CAMERA_HEIGHT / len(self.all_angle_ver)

        self.pixels_h = []
        counter_h = 0
        for y in range(len(self.all_angles_hoz)):
            self.pixels_h.append(round(counter_h))
            counter_h = counter_h + split_h

        self.pixels_v = []
        counter_v = 0
        for y in range(len(self.all_angle_ver)):
            self.pixels_v.append(round(counter_v))
            counter_v = counter_v + split_v

        # print(self.pixels_h)
        # print(self.pixels_v)

    def find_the_file_two(self, coord):
        index_h = closest_value(self.pixels_h, coord[0])
        index_v = closest_value(self.pixels_v, coord[1])

        return self.all_angles_hoz[index_h], self.all_angle_ver[index_v]
