## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./out/undistort_output.png "Undistorted"
[image2]: ./out/road_transformed.png "Road Transformed"
[image3]: ./out/filtered_image.png "Binary Example"
[image4]: ./out/filtered_roi.png
[image5]: ./out/warped_undistorted.png "Warp Example"
[image6]: ./out/color_fit_lines.png "Fit Visual"
[image7]: ./out/example_output.png "Output"
[video1]: ./out/project_video_out.mp4 "Video"
[video2]: ./out/challenge_video_out.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

### Camera Calibration

#### 1. Camera Calibration.

The code for this step is contained in the function calibrate_from_images() or in lines #25 through #63 of the file called `laneIdentification.py`).  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image. `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

####

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]
The distortion correction is applied by simply calling the function of opencv undistort passing as parameters the output of calibrate_from_images() discussed previously.

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines # through # in `laneIdentification.py`).  Here's an example of my output for this step.
![alt text][image3]
The filtering is processed through 2 classes Thresholding and ColorFiltering, each producing a binary image with the method binary_out combining all processing steps in the class. Both binary images are combined in one with a pixelwise or condition to produce a combined binary image.
In Thresholding I combine a simple x-gradient filtering with a magnitude and direction filtering. The sobel thresholding is applied in an artificially created image comprising of the Red channel of the initial image, and the Cr channel of the transformation cv2.COLOR_RGB2YCrCb. I found the latter to perform relatively well on images with intense shadows like in the first challenge video.
In ColorFiltering I pick out the yellows and whites in the mehtod yellow_white_filter. For the yellows I use the Z channel of the cv2.COLOR_RGB2XYZ transform, and for the whites a cube of 40 out of 255 bits - upper area - in the cv2.COLOR_RGB2HSV transform along the value axis. Additionally, I apply thresholding in the Hue, Saturation and Red channels in the method hls_filtering.
After the filtering process an additionall ROI is picked up with the function region_of_interest(). A sample image of the resulting binary image for further processing is ![alt text][image4]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 461 through 474 in the file `laneIdentification.py`, together with the class ImgTrans in lines 129 to 153, which holds the data relevant for the warping transform.  The `warper()` function takes as inputs an image (`img`), and uses the transformation matrix calculated in ImgTrans. I chose the hardcode the source and destination points, by choosing fixed positions in the image and requesting them to construct parallel lines in the transformed image. The rule is contained in ImgTrans class and the source - destination points are:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 455, 547      | 210, 547      | 
| 839, 547      | 1106, 547     |
| 210, 717      | 210, 717      |
| 1106, 717     | 1106, 717     |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image5]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The lane-line pixels are computed in a two step proccess, first applying the find_lane_pixels() (or the search_around_poly()) function as in the course material. In the find_lane_pixels() I added an additional step moving each window in the same turn in the middle of detected pixels and picking up the active pixels in the new area.
In the second step I calculate the polynomial coefficients from the previously found data, for each line in the function fit_polynomial(). The coefficients are calculated for both pixel and meter values. A sample result is: 

![alt text][image6]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The radius of curvature is calculating in the function measure_curvature() in `laneIdentification.py`, by simply applying the formula from the course. The calculation is done in pixels as the transformation in meters significantly deteriorates the result, which further would lead sanity checks to fail due to curvature computations.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `laneIdentification.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image7]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

The video processing pipeline is summarized in the function lane_identification() in the file `laneIdentification.py`. The input image is corrected, then fitlered and either search_around_poly() or find_lane_pixels() is used depending on if both lines were detected correctly in the last cycle. Then the polynomial is fitted and the curvature is measured, with the sanity checks following.

Each line is represented by a class Line containing all the relevant data. Polynomial coefficients in meters and pixels the fitted curves and the curvature of each line are represented with a CircularBuffer type which I defined to average the data, for filtering out bad frames. The filtering proceeds individually for each line. Furthermore for curvature are averaged 20 frames after a comment I found in knowledge base as those values vary significantly. Each of the first frames is used to fill the buffers, instead of filling them with the values of the first frame.

Here's a [link to my video result](./out/project_video_out.mp4)
and also a link to the [first challenge video result](./out/challenge_video_out.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I found that a dedicated search of white and yellow color is improving the lane detection on images of average difficulty, and applied those steps in Thresholding and ColorFiltering class. The problem of those channels is that they destroy the binary output for difficult images with extreme light or shadow like in the harder challenge video. I would like to receive a suggestion of which color spaces would perform the best filtering on difficult images.

Furhtermore I think that thresholding a matrix of stacked color channels gives better results, although it deteriorates the speed significantly. This proccess can provide an additional filtering method, as the output binary image can be produced from the sum of the proccessed matrix along the 3rd dimension and choosing say values >1 or >2 instead ov >0. The Cr channel of the YCrCb color space seems to pick up well features in shaded images but it doesn't seem to provide a big benefit in very difficult images. In such images the Hue channel seemed to perform well.

All in all the filtering seems to be the most important step in the proccess, and an adaptive channel selection - or/and adaptive filtering thresholds - could be applied depending on some feature detection of each image. However this could not be real-time applicable.

Another improvement could be to dynamically adapt the region-of-interest-limits and the window size in the find_lanes_pixels algorithm. The weakest point among all pipeline steps is for me the warping as it can blow-up the entire process with the pixel distortion it inserts. I would like to receive a comment about what would be a state of the art approach in this step.
