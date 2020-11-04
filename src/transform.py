import albumentations as A

def get_transform(is_train):
    if is_train:
        return A.Compose(
        [
            A.Identity()            
        ])
    else:
        return A.Compose(
        [
            A.Identity()
        ])

"""
def get_transform(height, width, mappings):
    scale = random.randint(2, 4)
    return A.Compose([
        OneOf([
            JpegCompression(quality_lower=20, quality_upper=70, p=0.5),
            Downscale(scale_min=0.25, scale_max=0.50, interpolation=1, p=0.5),
            Resize(height//scale,width//scale, interpolation=1, p=1.0)
        ], p=0.6),
        HorizontalFlip(p=0.5),
        A.augmentations.transforms.GaussNoise(p=0.2),
        A.RandomBrightnessContrast(p=0.2),    
        A.RandomGamma(p=0.2),    
        A.CLAHE(p=0.2),
        A.ChannelShuffle(p=0.2),
        A.MultiplicativeNoise(multiplier=[0.5, 1.5], elementwise=True, p=0.1),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.2),
    ], p=0.9,
    additional_targets=mappings)
"""
