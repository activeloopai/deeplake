from hub.core.augment.transforms import *

def transform_image(comp_func):
    """
    Returns a trasformed tensor. Takes in a hub compute object.

    Args:
        comp_func: Hub compute function.
    """
    return comp_func.func(*comp_func.args)


def run_transform(image, transform_name, magnitude):
    """
    Runs a transform on an image and returns the image.

    Args:
        image: A numpy array. 
        transform_name: Name of the transform to apply.
        magnitude: Magnitude of the transform to apply.
    """
    if transform_name == "TranslateX":      #kept between -1 and 1.
        return transform_image(translate(image, magnitude, 0))
    elif transform_name == "TranslateY":     #kept between -1 and 1.
        return transform_image(translate(image, 0, magnitude))
    
    elif transform_name == "Rotate":    # expects degrees positive is anticlockwise
        return transform_image(scale_rotate(image, magnitude, 1))
    
    elif transform_name == "Brightness":        #0 black 1 orig, greater than 0 expected
        return transform_image(adjust_brightness_contrast(image, magnitude, 0))

    elif transform_name == "Color":         #between 0 and 1 grayscale if 0 and orig if 1
        return transform_image(adjust_saturation(image, magnitude))

    elif transform_name == "Contrast":      #mag = 1 implies pixel=pixel + 1*mean_piexel_val; expected greater than 0 
        return transform_image(adjust_brightness_contrast(image, 1, magnitude))

    elif transform_name == "Posterize":
        if not isinstance(magnitude, int):
            raise ValueError(f"Expected integer type for Posterize got {type(magnitude)}")
        return transform_image(posterize(image, magnitude))

    elif transform_name == "Solarize":     #choose pixel value above which to invert pixel
        return transform_image(solarize(image, magnitude))

    elif transform_name == "Equalize":      #no parameter needed equalizes pizel magnitudes
        return transform_image(equalize(image))

    elif transform_name == "ShearX":        #shear angle range depends on the dimension of image
        return transform_image(shearX(image, magnitude))

    elif transform_name == "ShearY":        #shear angle range depends on the dimension of image
        return transform_image(shearY(image, magnitude))
    elif transform_name == "Invert":    
        return transform_image(invert(image))


