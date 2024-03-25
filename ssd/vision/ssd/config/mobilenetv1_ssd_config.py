
import numpy as np

from vision.utils.box_utils import SSDSpec, SSDBoxSizes, generate_ssd_priors


image_size = 300
image_mean = np.array([127, 127, 127])  # RGB layout
image_std = 128.0
iou_threshold = 0.45
center_variance = 0.1
size_variance = 0.2

specs = [
    SSDSpec(19, 16, SSDBoxSizes(60, 105), [3, 2]),
    SSDSpec(10, 32, SSDBoxSizes(105, 150), [3, 2]),
    SSDSpec(5, 64, SSDBoxSizes(150, 195), [3, 2]),
    SSDSpec(3, 100, SSDBoxSizes(195, 240), [3, 2]),
    SSDSpec(2, 150, SSDBoxSizes(240, 285), [3, 2]),
    SSDSpec(1, 300, SSDBoxSizes(285, 330), [3, 2])
]


priors = generate_ssd_priors(specs, image_size)
'''
import numpy as np

from vision.utils.box_utils import SSDSpec, SSDBoxSizes, generate_ssd_priors

image_size = 300
image_mean = np.array([127, 127, 127])  # RGB layout
image_std = 128.0
iou_threshold = 0.45
center_variance = 0.2
size_variance = 0.2

specs = [
    SSDSpec(19, 16, SSDBoxSizes(105, 60), [2, 3]),
    SSDSpec(10, 32, SSDBoxSizes(150, 105), [2, 3]),
    SSDSpec(5, 64, SSDBoxSizes(195, 150), [2, 3]),
    SSDSpec(3, 100, SSDBoxSizes(240, 195), [2, 3]),
    SSDSpec(2, 150, SSDBoxSizes(285, 245), [2, 3]),
    SSDSpec(1, 300, SSDBoxSizes(330, 285), [2, 3])
]


priors = generate_ssd_priors(specs, image_size)

import numpy as np

from vision.utils.box_utils import SSDSpec, SSDBoxSizes, generate_ssd_priors


image_size = 300
image_mean = np.array([127, 127, 127])  # RGB layout
image_std = 128.0
iou_threshold = 0.45
center_variance = 2
size_variance = 0

specs = [
    SSDSpec(19, 16, SSDBoxSizes(175, 195), [2, 3]),
    SSDSpec(10, 32, SSDBoxSizes(195, 210), [2, 3]),
    SSDSpec(5, 64, SSDBoxSizes(235, 255), [2, 3]),
    SSDSpec(3, 100, SSDBoxSizes(275, 295), [2, 3]),
    SSDSpec(2, 150, SSDBoxSizes(295, 315), [2, 3]),
    SSDSpec(1, 300, SSDBoxSizes(315, 335), [2, 3])
]


priors = generate_ssd_priors(specs, image_size)'''
