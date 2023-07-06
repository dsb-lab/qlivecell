import numpy as np
from scipy.ndimage import zoom

# Assuming your original image is stored in a NumPy array called 'image'
# with dimensions (1, 3)

# Define the original resolutions
image = np.asarray([[1.0,2.0,3.0],
                    [3.0,2.0,1.0]])

resolutions = (1.0, 2.0)  # (Resolution of first dimension, Resolution of second dimension)

# Compute the target resolutions (making the first dimension same as the second dimension)
target_resolutions = (resolutions[1], resolutions[1])

# Compute the zoom factor for the first dimension
zoom_factor = target_resolutions[0] / resolutions[0]

# Perform the interpolation using zoom function
isotropic_image = zoom(image, (10/2,1), order=1)

# 'isotropic_image' will now have isotropic resolutions in both dimensions

# Optionally, you can also update the resolutions information
# to reflect the new isotropic values
isotropic_resolutions = (target_resolutions[1], target_resolutions[1])

# You can access the new resolutions as:
isotropic_first_resolution = isotropic_resolutions[0]
isotropic_second_resolution = isotropic_resolutions[1]