# Arcunator

#### Project Requirments

Print out the template_aruco.pdf template, cut out 4 Aruco markers with different IDs, and place them in the corners of the monitor so that the markers are visible. IT IS ESSENTIAL TO KEEP THE WHITE MARGIN AROUND THE ARUCO MARKERS. Record a short video of the monitor showing these markers. In each frame of the video, detect the markers and project the image `UTB.jpg` with adjusted perspective into the corners of the monitor. Create your own implementation of a function for image transformation using perspective transformation. You may not use the built-in function `cv2.warpPerspective()`. In your function, use a linear interpolation scheme or the nearest neighbour method. Save the resulting video and embed it as part of your Jupyter notebook.    

#### Dev Note

Instead of a monitor, in my kolej, I have a spot where a TV should be, I put the aruco markers there and did the task. I used a custom image instead of the UTB logo. The script also has an if else condition, that prefers the use of GPU for processing, since on a notebook CPU, processing can be quite compute-intesive, especially with the unoptimized implementation of image transform.

# Preprocessing

Also implemented is a helper.ipynb, my phone records in 4k 60fps, its an iphone and the format is .MOV, this is not ideal for developing, and in initial testing, a short 20 second video took sometimes up to 30 seconds to complete. The helper notebook, is used to lower the resoloution, and convert the .MOV into .mp4, and at that point can be processe3d by the arcunator application


#### System Architechture

The two main orchestrators for arcunator are the following classes.

`detector.py`: Handles the detection of the ArUco markers, extracts their center coordinates, sorts the corners to align with the source image, and applies the masking logic to blend the warped image naturally into the original frame.

`transform.py`: Calculates the perspective transformation matrix and maps the source image to the destination frame using inverse warping.

#### Image transformation

The requirment was to implement a Direct Linear Transformation (DLT) algorithm for homography and a custom inverse mapping function for perspective image warping. 

*The Homography Matrix (SVD)*

To project our source image into the 3D space represented by the 2D video frame, we need to find a 3×3 transformation matrix H. The matrix looks like: 

$$
\begin{bmatrix} w \cdot u \\\\ w \cdot v \\\\ w \end{bmatrix} = \begin{bmatrix} h_{11} & h_{12} & h_{13} \\\\ h_{21} & h_{22} & h_{23} \\\\ h_{31} & h_{32} & h_{33} \end{bmatrix} \begin{bmatrix} x \\\\ y \\\\ 1 \end{bmatrix}
$$

By expanding this and isolating the variables, we can form a system of linear equations. For each of the 4 ArUco markers, we generate two rows in our matrix A that looks like: 

$$
A_i = \begin{bmatrix} x_i & y_i & 1 & 0 & 0 & 0 & -u_i x_i & -u_i y_i & -u_i \\\\ 0 & 0 & 0 & x_i & y_i & 1 & -v_i x_i & -v_i y_i & -v_i \end{bmatrix}
$$

Stacking the equations for all 4 points gives us an 8×9 matrix A with the soloution being the last row of $(V^t)$


*Perspective Warp & Inverse Mapping*

Inverse Warping is used to prevent "holes" in the output image due to floating-point rounding. The steps are as follows.

1. **Coordinate Grid Generation**: A grid of coordinates is generated for the *destination* bounding box.

2. **Inverse Transformation**: We multiply these destination coordinates by the inverse of our homography matrix $H^{-1}$ to find the exact, continuous $(x, y)$ source coordinates they map back to.

3. **Normalization**: We divide by the $Z$ coordinate (homogeneous division) to bring the points back into 2D Cartesian space.

*Bilinear Interpolation*

The inverse transformation results in floating-point coordinates (e.g., source pixel at $x=4.2, y=9.8$), we cannot map to a single discrete pixel. The function implements **Bilinear Interpolation** to blend the colors of the 4 nearest neighboring pixels. 

If our floating point coordinate falls between $(x_0, y_0)$ and $(x_1, y_1)$, the color is weighted based on its distance to each neighbor:
* $w_a = (x_1 - x) \cdot (y_1 - y)$
* $w_b = (x_1 - x) \cdot (y - y_0)$
* $w_c = (x - x_0) \cdot (y_1 - y)$
* $w_d = (x - x_0) \cdot (y - y_0)$

The final pixel value is the sum of the 4 neighbor pixels multiplied by their respective weights.


I used this medium article for insporation. AI was used for debugging and verifyig mathmatical implementaion.

https://medium.com/@gausic10/creating-augmented-reality-experiences-with-opencv-a-step-by-step-guide-63f9b757707f

**RESULT VIDEO IS ATTACHED IN data/video-output**









