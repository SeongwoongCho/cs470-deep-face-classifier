from utils import seed_everything
import albumentations

seed_everything(42)

def get_transform(is_train):
    if is_train:
        return albumentations.Compose(
        [   
            albumentations.Resize(224,224),
            albumentations.OneOf([
                albumentations.JpegCompression(quality_lower=20, quality_upper=70, p=0.5),
                albumentations.Downscale(scale_min=0.25, scale_max=0.50, interpolation=1, p=0.5),
            ], p=0.6),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.VerticalFlip(p=0.5),
            albumentations.GaussNoise(p=0.2),
            albumentations.RandomBrightnessContrast(p=0.2),    
            albumentations.RandomGamma(p=0.2),    
            albumentations.CLAHE(p=0.2),
            albumentations.ChannelShuffle(p=0.2),
            albumentations.MultiplicativeNoise(multiplier=[0.5, 1.5], elementwise=True, p=0.1),
            albumentations.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.2),     
            albumentations.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0)
        ])
    else:
        return albumentations.Compose(
        [
            albumentations.Resize(224,224),
            albumentations.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0)
        ])
