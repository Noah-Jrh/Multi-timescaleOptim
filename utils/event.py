"""
author   : lujunwei
contact  : lujunwei1995@outlook.com
file     : EventVision.py
time     : 2019-09-22 18:37
"""
# import cv2
import numpy as np
from scipy import io


class Events(object):
    """
    Temporal Difference events.

    data: a NumPy Record Array with the following named fields.
        x: pixel x coordinate, unsigned 16bit int.
        y: pixel y coordinate, unsigned 16bit int.
        p: polarity value, boolean. False=off, True=on.
        ts: timestamp in microseconds, unsigned 64bit int.
    width: The width of the frame.
    height: The height of the frame.
    """

    def __init__(self, num_events, width, height, label=None):
        """
        Initialize the class Events.

        :param num_events: number of events this instance will initially contain
        :param width: width of event image.
        :param height: height of event image.
        """
        self.data = np.rec.array(None,
                                 dtype=[('x', np.int16),
                                        ('y', np.int16),
                                        ('p', np.int8),
                                        ('ts', np.int32)],
                                 shape=(num_events,))
        self.width = width
        self.height = height
        self.label = label

    def denoise_td(self, tau_m=200e3, th=10, radius=1):
        """
        Uses a filter on the event data and does not modify instance data.

        radius: the radius of filter.
        tau_m: milliseconds.
        th: threshold.
        """

        dt = 10  # microseconds
        lut_1 = np.exp(-np.arange(0, 10 * tau_m + dt, dt, np.float_) / tau_m)
        v_0 = 1 / np.max(lut_1)

        # Get Gauss filter
        sigma = 1.68
        y, x = np.mgrid[-radius:radius + 1, -radius:radius + 1]
        filter = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2)) / (2 * np.pi * sigma ** 2) * 10
        filter[radius, radius] = 0

        image_size = (self.height + 2 * radius, self.width + 2 * radius)
        t_last = np.zeros(image_size)
        K_1 = np.zeros(image_size)

        valid_indices = np.zeros(len(self.data), dtype=np.bool_)

        for index, (x, y, p, ts) in np.ndenumerate(self.data):
            x, y, ts = x + radius, y + radius, ts / dt
            x1, x2, y1, y2 = x - radius, x + radius + 1, y - radius, y + radius + 1

            delta_t = ts - t_last[y1:y2, x1:x2]
            t_last[y1:y2, x1:x2] = ts

            Sc_1 = np.exp(-delta_t / tau_m)
            K_1[y1:y2, x1:x2] = Sc_1 * K_1[y1:y2, x1:x2] + v_0 * filter

            valid_indices[index] = 1 if K_1[y, x] > th else 0

        self.data = self.data[valid_indices]

    # def show_td(self, frame_length=24e3, wait_delay=1):
    #     """
    #     Displays the TD events (change detection ATIS or DVS events)

    #     frame_length: milliseconds.
    #     wait_delay: milliseconds.
    #     """
    #     frame_start = self.data[0].ts
    #     frame_end = frame_start + frame_length

    #     cv2.startWindowThread()

    #     while frame_start < self.data.ts[-1]:
    #         frame_data = self.data[(frame_start <= self.data.ts) & (
    #                 self.data.ts < frame_end)]

    #         if frame_data.size > 0:
    #             td_img = np.ones((self.height, self.width, 3), dtype=np.uint8)
    #             for x, y, p, _ in frame_data:
    #                 if p == 0:
    #                     td_img[y, x] = [255, 0, 0]  # negative: red
    #                 else:
    #                     td_img[y, x] = [0, 0, 255]  # positive: blue
    #             img = cv2.resize(td_img, (240, 240),
    #                              interpolation=cv2.INTER_AREA)
    #             cv2.imshow('Event Image', img)
    #             cv2.waitKey(wait_delay)

    #         frame_start = frame_end
    #         frame_end = frame_start + frame_length

    #     cv2.destroyAllWindows()
    #     return

    def sort_order(self):
        """
        Generate data sorted by ascending ts
        Does not modify instance data
        Will look through the struct events, and sort all events by the field 'ts'.
        In other words, it will ensure events_out.ts is monotonically increasing,
        which is useful when combining events from multiple recordings.
        """
        # chose mergesort because it is a stable sort, at the expense of more memory usage
        # self.data = np.sort(self.data, order='ts', kind='mergesort')
        # TODO: sort
        idx = np.argsort(self.data.ts, kind='mergesort')
        self.data = self.data[idx]
        return self.data
        # events_out = np.sort(self.data, order='ts', kind='mergesort')
        # return events_out

    def apply_refraction(self, refrac_time):
        """
        Implements a refractory period for each pixel.
        Does not modify instance data
        In other words, if an event occurs within 'refrac_time' microseconds of
        a previous event at the same pixel, then the second event is removed
        us_time: time in millisecond
        """
        last_time = np.zeros((self.height, self.width),
                             dtype=np.int32) - refrac_time
        valid_indices = np.ones(self.data.size, dtype=np.bool_)

        for index, (x, y, p, ts) in np.ndenumerate(self.data):
            # TODO:
            # if ts - last_time[y, x] < refrac_time:
            if ts - last_time[y - 1, x - 1] <= refrac_time:
                valid_indices[index] = 0
            else:
                last_time[y - 1, x - 1] = ts

        self.data = self.data[valid_indices]

        return self.data


def read_dataset(file, mode):
    size = [34, 34]
    """
    Reads in the TD events contained in the N-MNIST/N-CALTECH101 dataset file.

    file: file of the events.
    """
    if 'slice' in mode:
        if 'mnist-dvs' in mode:
            size = [32, 32]
        else:
            size = [128, 128]
        data = io.loadmat(file, squeeze_me=True, struct_as_record=False)
        TD_list = []
        # 最后一个slice为空
        for islice in range(len(data['TD'])):
            if data['TD'][islice].x.size > 0:
                n_events = data['TD'][islice].x.shape[0]
                TD = Events(np.int_(n_events), size[0], size[1])
                TD.data.x = data['TD'][islice].x
                TD.data.y = data['TD'][islice].y
                TD.data.p = data['TD'][islice].p
                TD.data.ts = data['TD'][islice].ts
            else:
                continue
        TD_list.append(TD)

        return TD_list
    else:
        if 'nmnist' in mode:
            datafile = open(file, 'rb')
            raw_data = np.fromfile(datafile, dtype=np.uint8)
            raw_data = np.uint32(raw_data)
            datafile.close()

            TD = Events(np.int_(raw_data.size / 5), size[0], size[1])
            TD.data.x = raw_data[0::5]
            TD.data.y = raw_data[1::5]
            TD.data.p = (raw_data[2::5] & 128) >> 7
            TD.data.ts = ((raw_data[2::5] & 127) << 16) | (raw_data[3::5] << 8) | (raw_data[4::5])
        else:
            if 'mnist-dvs' in mode:
                size = [32, 32]
            else:
                size = [128, 128]
            data = io.loadmat(file, squeeze_me=True, struct_as_record=False)
            noEvents = data['TD'].x.shape[0]
            TD = Events(np.int_(noEvents), size[0], size[1])
            TD.data.x = data['TD'].x
            TD.data.y = data['TD'].y
            TD.data.p = data['TD'].p
            TD.data.ts = data['TD'].ts
        return TD


if __name__ == '__main__':
    pass
    # test_td = read_dataset("/Users/lulu/dataset/nmnist/0/00002.bin")
    # test_td.show_td(5e3)
