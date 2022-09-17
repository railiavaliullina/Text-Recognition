from easydict import EasyDict

cfg = EasyDict()

cfg.dataset_path = 'E:/datasets/University/mjsynth/mnt/ramdisk/max/90kDICT32px/'
cfg.annotation_files_path = {'train': cfg.dataset_path + 'annotation_train.txt',
                             'test': cfg.dataset_path + 'annotation_test.txt',
                             'val': cfg.dataset_path + 'annotation_val.txt'}

cfg.vertical_crop_h = 32
cfg.horizontal_size_thr = 500
cfg.mean = 118.6253  # 118.8218

cfg.alphabet = {'-': 0, 'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8, 'I': 9, 'J': 10, 'K': 11,
                'L': 12, 'M': 13, 'N': 14, 'O': 15, 'P': 16, 'Q': 17, 'R': 18, 'S': 19, 'T': 20, 'U': 21, 'V': 22,
                'W': 23, 'X': 24, 'Y': 25, 'Z': 26, 'a': 27, 'b': 28, 'c': 29, 'd': 30, 'e': 31, 'f': 32, 'g': 33,
                'h': 34, 'i': 35, 'j': 36, 'k': 37, 'l': 38, 'm': 39, 'n': 40, 'o': 41, 'p': 42, 'q': 43, 'r': 44,
                's': 45, 't': 46, 'u': 47, 'v': 48, 'w': 49, 'x': 50, 'y': 51, 'z': 52, '0': 53, '1': 54, '2': 55,
                '3': 56, '4': 57, '5': 58, '6': 59, '7': 60, '8': 61, '9': 62}
a = 1
