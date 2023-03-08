# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# Reference:
# UM-MAE: https://github.com/implus/UM-MAE
# --------------------------------------------------------

import random
import math

import PIL
from einops.einops import rearrange
import numpy as np
import torchvision.transforms as transforms


class RandomMaskingGenerator:
    def __init__(self, input_size, mask_ratio, regular=False, block=False):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2

        self.height, self.width = input_size

        self.num_patches = self.height * self.width
        self.num_mask = int(mask_ratio * self.num_patches)
        self.regular = regular
        self.block = block

        if block:
            assert mask_ratio == 0.75

        if regular:
            assert mask_ratio == 0.75

            candidate_list = []
            while True:  # add more
                for j in range(4):
                    candidate = np.ones(4)
                    candidate[j] = 0
                    candidate_list.append(candidate)
                if len(candidate_list) * 4 >= self.num_patches * 2:
                    break
            self.mask_candidate = np.vstack(candidate_list)
            print('using regular, mask_candidate shape = ',
                  self.mask_candidate.shape)

    def __repr__(self):
        repr_str = "Mask: total patches {}, mask patches {}, regular {}, block {}".format(
            self.num_patches, self.num_mask, self.regular, self.block
        )
        return repr_str

    def __call__(self):
        # returns a (num_patches,) binary mask
        if self.regular:
            mask = self.mask_candidate.copy()
            np.random.shuffle(mask)
            mask = rearrange(mask[:self.num_patches // 4], '(h w) (p1 p2) -> (h p1) (w p2)',
                             h=self.height // 2, w=self.width // 2, p1=2, p2=2)
            mask = mask.flatten()

        elif self.block:
            visible_width = int(np.sqrt(self.num_patches - self.num_mask))
            total_width = int(np.sqrt(self.num_patches))
            mask = np.ones((total_width, total_width))

            left_x = np.random.randint(low=0, high=total_width // 2)
            up_y = np.random.randint(low=0, high=total_width // 2)

            mask[up_y: up_y + visible_width, left_x: left_x + visible_width] = 0
            mask = mask.flatten()

        else:
            mask = np.hstack([
                np.zeros(self.num_patches - self.num_mask),  # 0 is visible
                np.ones(self.num_mask),  # 1 is masked
            ])
            np.random.shuffle(mask)

        return mask


class MaskTransform(object):
    def __init__(self, args, is_train=True):
        if is_train:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),  # 3 is bicubic
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            # val transform
            if args.input_size <= 224:
                crop_pct = 224 / 256
            else:
                crop_pct = 1.0
            size = int(args.input_size / crop_pct)

            self.transform = transforms.Compose([
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(args.input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        if not hasattr(args, 'mask_regular'):
            args.mask_regular = False
        if not hasattr(args, 'mask_block'):
            args.mask_block = False

        self.masked_position_generator = RandomMaskingGenerator(
            args.token_size, args.mask_ratio, args.mask_regular, args.mask_block,
        )

    def __call__(self, image):
        return self.transform(image), self.masked_position_generator()

    def __repr__(self):
        repr = "(MaskTransform,\n"
        repr += "  transform = %s,\n" % str(self.transform)
        repr += "  Masked position generator = %s,\n" % str(self.masked_position_generator)
        repr += ")"
        return repr
