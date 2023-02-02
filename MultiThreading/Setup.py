import numpy as np

scale_factor = 5

# static helper function
def closest_value_index(input_list, input_value):
    arr = np.asarray(input_list)
    i = (np.abs(arr - input_value)).argmin()
    return i


class Setup:
    def __init__(self, DEFAULT_CAMERA_WIDTH, DEFAULT_CAMERA_HEIGHT, DEFAULT_FIELD_OF_VIEW_WIDTH,
                 DEFAULT_FIELD_OF_VIEW_HEIGHT):

        self.all_angles_hoz = []

        width_fov_scaled = (DEFAULT_FIELD_OF_VIEW_WIDTH / 2) * scale_factor
        height_fov_scaled = (DEFAULT_FIELD_OF_VIEW_HEIGHT / 2) * scale_factor

        print('sclqaed_fatcors')
        print(width_fov_scaled,height_fov_scaled)

        x = 0
        self.all_angles_hoz.append(x)
        while x < width_fov_scaled:
            x = x + 15
            self.all_angles_hoz.append(x)

        self.all_angles_hoz.sort(reverse=True)

        x = 360
        while x > 360 - width_fov_scaled:
            x = x - 15
            self.all_angles_hoz.append(x)

        self.all_angle_ver = []

        x = 0
        self.all_angle_ver.append(x)
        while x < height_fov_scaled:
            x = x + 15
            self.all_angle_ver.append(x)

        self.all_angle_ver.sort(reverse=True)

        part2 = []

        x = 345
        part2.append(x)
        while x > 345 - height_fov_scaled:
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

        print(self.all_angle_ver)
        print(self.all_angles_hoz)

    def find_the_file_two(self, coord):
        index_h = closest_value_index(self.pixels_h, coord[0])
        index_v = closest_value_index(self.pixels_v, coord[1])

        return self.all_angles_hoz[index_h], self.all_angle_ver[index_v]