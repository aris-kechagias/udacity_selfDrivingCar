import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import cv2
import numpy as np
from moviepy.editor import VideoFileClip

""" Utility """


def compare_two_img(img1, img2, img2_title):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(img1)
    ax1.set_title('Original Image', fontsize=40)
    ax2.imshow(img2)
    ax2.set_title(img2_title, fontsize=40)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()


""" *******************************************************************************************
                                        Calibration and data setup
    ******************************************************************************************* """


def calibrate_from_images(nx=9, ny=6, debug=False, demo=False):
    """
    Camera Calibration: find the camera calibration parameters, from the dataset images
    :param ny: number of vertical corners
    :param nx: number of horizontal corners
    :param debug: plot all calibrating images
    :param demo: plot the chessboard corners found
    :return: mtx, dist
    """
    images = glob.glob('camera_cal/calibration*.jpg')  # Read calibration images
    dims = mpimg.imread('camera_cal/calibration1.jpg').shape[1::-1]

    # Arrays to store object points and image points from all images
    obj_pts = []  # 3D points in real world space
    img_pts = []  # 2D points in image plane

    # Prepare object points
    obj_p = np.zeros((ny * nx, 3), np.float32)
    obj_p[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)  # x, y coordinates

    for f_name in images:
        img = mpimg.imread(f_name)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        if ret:
            img_pts.append(corners)
            obj_pts.append(obj_p)
            if debug:  # draw and display corners
                img = cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
                plt.figure()
                plt.imshow(img)
                plt.show()

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, dims, None, None)
    if demo:
        img_new = mpimg.imread('camera_cal/calibration1.jpg')
        undistorted = cv2.undistort(img_new, mtx, dist, None, mtx)
        compare_two_img(img_new, undistorted, img2_title='Undistorted Image')
    return mtx, dist, dims


class CircularBuffer:
    def __init__(self, size):
        self.maxsize = size
        self._data = np.array([False])
        self._cursize = 0

    def add_data(self, data):
        temp = np.array(data)
        temp_sz = temp.ndim if temp.ndim == 1 else np.size(temp, axis=0)
        if self._cursize == 0:
            self._data = temp
            self._cursize += temp_sz
        if self._cursize + temp_sz <= self.maxsize:
            self._data = np.vstack((self._data, temp))
            self._cursize += temp_sz
        else:
            raise BufferError(f"The buffer max size {self.maxsize} is exceeded")

    def append(self, value):
        self._data[-1] = value
        self._data = np.roll(self._data, 1, axis=0)

    def get_last(self):
        return self._data[0]

    def get_mean(self):
        return np.sum(self._data, axis=0) / np.size(self._data, axis=0)

    def get_median(self):
        temp = np.sort(self._data)
        return np.median(temp, axis=0)

    def is_full(self):
        return self.maxsize == self._cursize

    def is_empty(self):
        return self._cursize == 0


class Line:  # Class containing the characteristics of each line detection
    def __init__(self, img_shape, frame_avg=6, curv_avg=20):
        self.frame_avg = frame_avg
        self.curv_avg = curv_avg
        self.detected = False  # was the line detected in the last iteration?
        self.line_base_pos = None  # distance in meters of vehicle center from the line
        self.poly_last = [np.array([False])]  # latest polynomial coefficients
        self.poly_buff = CircularBuffer(self.frame_avg)  # average polynomial coefficients over the last n iterations
        self.poly_last_m = [np.array([False])]  # latest polynomial coefficients in meters
        self.poly_buff_m = CircularBuffer(self.frame_avg)  # average polynomial coefficients over n iterations in meters
        self.fittedx_last = None  # x values of the last fit of the line
        self.fittedx_buff = CircularBuffer(self.frame_avg)  # x values of the last n fits of the line
        self.y_space = np.linspace(0, img_shape[0] - 1, img_shape[0])
        self.allx = None  # x values for detected line pixels
        self.ally = None  # y values for detected line pixels
        self.curvature_last = None
        self.curvature_buff = CircularBuffer(self.curv_avg)  # radius of curvature of the line in some units


""" Global data """


class ImgTrans:
    _warp_ul = [455, 547]
    _warp_ur = [839, 547]
    _warp_dl = [210, 717]
    _warp_dr = [1106, 717]
    _src_points = np.float32([_warp_ul, _warp_ur, _warp_dl, _warp_dr])
    _dest_points = np.float32([[_warp_dl[0], _warp_ul[1]], [_warp_dr[0], _warp_ur[1]], _warp_dl, _warp_dr])
    x_rel = 890  # relevant pixels in x dimension
    y_rel = 720  # relevant pixels in y dimension
    x_dist_max = 1170
    x_dist_min = 710
    x_margin = 710  # allowed pixel deviation along x
    ym_per_pix = 16.5 / y_rel  # projected_lane_length
    xm_per_pix = 3.7 / x_rel  # lane width (m)
    # _src_points = np.float32([[585, 460], [203, 720], [1127, 720], [695, 460]])
    # _dest_points = np.float32([[320, 0], [320, 720], [960, 720], [960, 0]])
    # x_rel = 640  # relevant pixels in x dimension
    # y_rel = 720  # relevant pixels in y dimension
    # x_dist_max = 800
    # x_dist_min = 490
    # x_margin = 710  # allowed pixel deviation along x
    # ym_per_pix = 30 / y_rel  # projected_lane_length
    # xm_per_pix = 3.7 / x_rel  # lane width (m)
    warp_mat = cv2.getPerspectiveTransform(_src_points, _dest_points)
    warp_mat_inv = cv2.getPerspectiveTransform(_dest_points, _src_points)


matrix, distance, img_dims = calibrate_from_images()
l_line = Line(img_dims)
r_line = Line(img_dims)


""" *******************************************************************************************
                                        Pipeline
    ******************************************************************************************* """


def map_lane(img, warped):
    """

    :param img:
    :param warped:
    :return:
    """
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    x_fitted_l = l_line.fittedx_buff.get_last() if l_line.fittedx_buff.is_full() else l_line.fittedx_last
    pts_left = np.array([np.transpose(np.vstack([x_fitted_l, l_line.y_space]))])

    x_fitted_r = r_line.fittedx_buff.get_last() if r_line.fittedx_buff.is_full() else r_line.fittedx_last
    pts_right = np.array([np.flipud(np.transpose(np.vstack([x_fitted_r, r_line.y_space])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space
    warp_inv = cv2.warpPerspective(
        color_warp, ImgTrans.warp_mat_inv, (warped.shape[1], warped.shape[0]), flags=cv2.INTER_LINEAR)

    # Combine the result with the original image
    result = cv2.addWeighted(img, 1, warp_inv, 0.3, 0)

    # Calculate curvature and offset in meters
    y_eval = np.max(l_line.y_space) * ImgTrans.ym_per_pix
    l_a, l_b, l_c = l_line.poly_buff_m.get_median()
    l_curvature_m = (((1 + (2*l_a*y_eval + l_b)**2)**1.5) / np.absolute(2 * l_a))
    r_a, r_b, r_c = r_line.poly_buff_m.get_median()
    r_curvature_m = (((1 + (2*r_a*y_eval + r_b)**2)**1.5) / np.absolute(2*r_a))
    curvature = int((l_curvature_m + r_curvature_m) / 2)
    # The car's offset from the the lane center is the difference between image and lane's centers.
    offset = l_line.fittedx_buff.get_mean()[-1] \
             + (r_line.fittedx_buff.get_mean()[-1] - l_line.fittedx_buff.get_mean()[-1])/2 - result.shape[1]/2

    font_sz = 2
    red = (255, 0, 0)
    green = (0, 255, 0)
    txt = f"Curvature: {curvature}(m) Lane_center_offset: {format(offset*ImgTrans.xm_per_pix,'.3f')}(m)"
    txt_color = green if (l_line.detected and r_line.detected) else red
    cv2.putText(result, txt, (250, 80), cv2.FONT_HERSHEY_PLAIN, font_sz, (0, 0, 255), 2, cv2.LINE_AA)
    detect = "D"
    l_d = (300, 600)
    l_d_color = green if l_line.detected else red
    cv2.putText(result, detect, l_d, cv2.FONT_HERSHEY_PLAIN, font_sz, l_d_color, 4, cv2.LINE_AA)
    r_d = (1100, 600)
    r_d_color = green if r_line.detected else red
    cv2.putText(result, detect, r_d, cv2.FONT_HERSHEY_PLAIN, font_sz, r_d_color, 4, cv2.LINE_AA)

    return result


def accepted_deviation(new_val, given, percent=180):
    """

    :param given:
    :param new_val:
    :param percent:
    :return:
    """
    min_acceptable = 2
    result = True
    if np.abs(new_val) > min_acceptable and np.abs(given) > min_acceptable:
        if np.abs(np.abs(new_val) - np.abs(given)) * 100 / np.abs(given) > percent:
            result = False

    return result


def sanity_checks():
    """

    :return:
    """
    accepted = True
    # Horizontal line distance is approximately right
    bottom_x_dist = (r_line.fittedx_last[-int(len(r_line.fittedx_last) / 30)]
                     - l_line.fittedx_last[-int(len(l_line.fittedx_last) / 30)])
    if bottom_x_dist > ImgTrans.x_dist_max or bottom_x_dist < ImgTrans.x_dist_min:
        accepted = False
    if accepted:  # Roughly parallel
        mid_x_dist = (r_line.fittedx_last[-int(len(r_line.fittedx_last) / 10)]
                      - l_line.fittedx_last[-int(len(l_line.fittedx_last) / 10)])
        if bottom_x_dist - mid_x_dist > ImgTrans.x_margin:
            accepted = False
    if not accepted:
        l_line.detected = False
        r_line.detected = False
    else:
        l_line.detected = True
        r_line.detected = True
    if accepted:  # Similar curvature with previous frames
        if not accepted_deviation(l_line.curvature_last, l_line.curvature_buff.get_median()):
            l_line.detected = False
        if not accepted_deviation(r_line.curvature_last, r_line.curvature_buff.get_median()):
            r_line.detected = False


def measure_curvature(meters=False):
    """
    Calculates the curvature of polynomial functions in pixels.
    :return:
    """
    # Start by generating our fake example data
    # Make sure to feed in your real data instead in your project!
    if meters:
        left_fit = l_line.poly_last_m
        right_fit = r_line.poly_last_m
        y_eval = np.max(l_line.y_space) * ImgTrans.ym_per_pix
    else:
        left_fit = l_line.poly_last
        right_fit = r_line.poly_last
        y_eval = np.max(l_line.y_space)  # y-value where we want radius of curvature: bottom of the image

    l_line.curvature_last = (((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0]))
    r_line.curvature_last = (((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0]))


def fit_polynomial(valid_l_ind, valid_r_ind):
    """

    :param valid_l_ind:
    :param valid_r_ind:
    :return:
    """
    # Extract left and right line pixel positions
    left_x = l_line.allx[valid_l_ind]
    left_y = l_line.ally[valid_l_ind]
    right_x = r_line.allx[valid_r_ind]
    right_y = r_line.ally[valid_r_ind]

    l_line.poly_last = np.polyfit(left_y, left_x, 2)
    r_line.poly_last = np.polyfit(right_y, right_x, 2)

    l_line.poly_last_m = ImgTrans.xm_per_pix*np.asarray([l_line.poly_last[0]/(ImgTrans.ym_per_pix**2),
                                                        (l_line.poly_last[1]/ImgTrans.ym_per_pix), l_line.poly_last[2]])
    r_line.poly_last_m = ImgTrans.xm_per_pix*np.asarray([r_line.poly_last[0]/(ImgTrans.ym_per_pix**2),
                                                        (r_line.poly_last[1]/ImgTrans.ym_per_pix), r_line.poly_last[2]])

    # Generate x and y values for plotting
    try:
        l_line.fittedx_last = (l_line.poly_last[0] * l_line.y_space ** 2 + l_line.poly_last[1] * l_line.y_space
                               + l_line.poly_last[2])
        r_line.fittedx_last = (r_line.poly_last[0] * r_line.y_space ** 2 + r_line.poly_last[1] * r_line.y_space
                               + r_line.poly_last[2])
    except TypeError:  # Avoids an error if `left` and `r_line.current_fit` are still none or incorrect
        print('The function failed to fit a line!')
        l_line.fittedx_last = 1 * l_line.y_space ** 2 + 1 * l_line.y_space
        r_line.fittedx_last = 1 * r_line.y_space ** 2 + 1 * r_line.y_space


def search_around_poly(bin_warped, demo=False):
    """
    :param bin_warped:
    :param demo:
    :return:
    """
    margin = 150  # HYPER-PARAMETER: width of the margin around the previous polynomial to search
    l_a, l_b, l_c = l_line.poly_buff.get_mean()
    r_a, r_b, r_c = r_line.poly_buff.get_mean()
    # Set the area of search based on activated x-values within the +/- margin of our polynomial function
    l_lane_idx = (
            (l_line.allx > (l_a*(l_line.ally**2) + l_b*l_line.ally + l_c - margin))
            & (l_line.allx < (l_a*(l_line.ally**2) + l_b*l_line.ally + l_c + margin)))
    r_lane_idx = (
            (r_line.allx > (r_a*(r_line.ally**2) + r_b*r_line.ally + r_c - margin))
            & (r_line.allx < (r_a*(r_line.ally**2) + r_b*r_line.ally + r_c + margin)))
    if demo:  # Visualization #
        fit_polynomial(l_lane_idx, r_lane_idx)
        out_img = np.dstack((bin_warped, bin_warped, bin_warped)) * 255
        window_img = np.zeros_like(out_img)
        out_img[l_line.ally[l_lane_idx], l_line.allx[l_lane_idx]] = [255, 0, 0]
        out_img[r_line.ally[r_lane_idx], r_line.allx[r_lane_idx]] = [0, 0, 255]

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([l_line.fittedx_last - margin, l_line.y_space]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([l_line.fittedx_last + margin, l_line.y_space])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([r_line.fittedx_last - margin, r_line.y_space]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([r_line.fittedx_last + margin, r_line.y_space])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
        plt.imshow(result)
        plt.show()

    return l_lane_idx, r_lane_idx


def find_lane_pixels(bin_warped, demo=False):
    """
    :param bin_warped:
    :param demo:
    :return:
    """
    # HYPER-PARAMETERS
    n_window = 9  # Number of sliding windows
    margin: int = 80  # Width of the windows +/- margin
    min_pix = 30  # Minimum number of pixels found to recenter window

    histogram = np.sum(bin_warped[bin_warped.shape[0] // 2:, :], axis=0)

    out_img = np.dstack((bin_warped, bin_warped, bin_warped)) * 255

    midpoint = np.int(histogram.shape[0] // 2)
    # Current positions to be updated window in n_window
    left_current_x = np.argmax(histogram[:midpoint])  # Left-histogram half peak
    right_current_x = midpoint + np.argmax(histogram[midpoint:])  # Right-histogram half peak

    window_height = np.int(bin_warped.shape[0] // n_window)  # cover entire y-axis

    empty_wins_l = 0
    empty_wins_r = 0
    empty_wins_max = 3

    # Empty lists to receive left and right lane pixel indices
    l_lane_idx = []
    r_lane_idx = []

    # Step through the windows one by one
    for window in range(n_window):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = bin_warped.shape[0] - (window + 1) * window_height
        win_y_high = win_y_low + window_height
        win_left_x_low = left_current_x - margin
        win_left_x_high = left_current_x + margin
        win_right_x_low = right_current_x - margin
        win_right_x_high = right_current_x + margin

        # Pre-calculation: nonzero pixels in x and y within the window
        good_left_inds = ((l_line.ally >= win_y_low) & (l_line.ally < win_y_high) &
                          (l_line.allx >= win_left_x_low) & (l_line.allx < win_left_x_high)).nonzero()[0]
        good_right_inds = ((r_line.ally >= win_y_low) & (r_line.ally < win_y_high) &
                           (r_line.allx >= win_right_x_low) & (r_line.allx < win_right_x_high)).nonzero()[0]

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > min_pix:
            left_current_x = np.int(np.mean(l_line.allx[good_left_inds]))
            empty_wins_l = 0
        else:
            empty_wins_l += 1
        if len(good_right_inds) > min_pix:
            right_current_x = np.int(np.mean(r_line.allx[good_right_inds]))
            empty_wins_r = 0
        else:
            empty_wins_r += 1

        # Center window for pixels of pre-calculation
        win_left_x_low = left_current_x - margin
        win_left_x_high = left_current_x + margin
        win_right_x_low = right_current_x - margin
        win_right_x_high = right_current_x + margin

        good_left_inds = ((l_line.ally >= win_y_low) & (l_line.ally < win_y_high) &
                          (l_line.allx >= win_left_x_low) & (l_line.allx < win_left_x_high)).nonzero()[0]
        good_right_inds = ((r_line.ally >= win_y_low) & (r_line.ally < win_y_high) &
                           (r_line.allx >= win_right_x_low) & (r_line.allx < win_right_x_high)).nonzero()[0]

        # Append the indices to the lists only if the conditions are met.
        if (win_left_x_low >= 0 and empty_wins_l <= 1) or empty_wins_l <= empty_wins_max:
            l_lane_idx.append(good_left_inds)
        if (win_right_x_high <= bin_warped.shape[1] and empty_wins_r <= 1) or empty_wins_r <= empty_wins_max:
            r_lane_idx.append(good_right_inds)

        if demo:  # Draw the windows on the visualization image
            cv2.rectangle(out_img, (win_left_x_low, win_y_low), (win_left_x_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(out_img, (win_right_x_low, win_y_low), (win_right_x_high, win_y_high), (0, 255, 0), 2)

    try:  # Concatenate the arrays of indices (previously was a list of lists of pixels)
        l_lane_idx = np.concatenate(l_lane_idx)
        r_lane_idx = np.concatenate(r_lane_idx)
    except ValueError:  # Avoids an error if the above is not implemented fully
        pass

    if demo:  # Visualization
        fit_polynomial(l_lane_idx, r_lane_idx)
        out_img[l_line.ally[l_lane_idx], l_line.allx[l_lane_idx]] = [255, 0, 0]
        out_img[r_line.ally[r_lane_idx], r_line.allx[r_lane_idx]] = [0, 0, 255]

        plt.plot(l_line.fittedx_last, l_line.y_space, color='yellow')
        plt.plot(r_line.fittedx_last, r_line.y_space, color='yellow')
        plt.imshow(out_img)
        plt.show()

    return l_lane_idx, r_lane_idx


def warper(img):
    """

    :param img:
    :return:
    """
    out = cv2.warpPerspective(img, ImgTrans.warp_mat, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
    nonzero = out.nonzero()
    l_line.allx = np.array(nonzero[1])
    l_line.ally = np.array(nonzero[0])
    r_line.allx = l_line.allx
    r_line.ally = l_line.ally

    return out


def region_of_interest(img):
    """
    Applies an image mask. Only keeps the region of the image defined by the polygon formed from `vertices`.
    The rest of the image is set to black. `vertices` should be a numpy array of integer points.
    :param img:
    :return: masked image
    """
    im_dim_y, im_dim_x = img.shape
    view_depth = 0.36
    y_margin = 0.92
    x_margin = 0.28
    vertices = np.array([[(0, int(im_dim_y*y_margin)),  # 650
                          (0, int(im_dim_y*y_margin-60)),
                          (int(im_dim_x*x_margin), int(im_dim_y*(1-view_depth))),  # 1050 -1100
                          (int(im_dim_x*(1-x_margin)), int(im_dim_y*(1-view_depth))),
                          (im_dim_x, int(im_dim_y*y_margin-60)),
                          (im_dim_x, int(im_dim_y*y_margin))]], dtype=np.int32)
    mask = np.zeros_like(img)
    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    cv2.fillPoly(mask, vertices, ignore_mask_color)
    return cv2.bitwise_and(img, mask)


class Thresholding:  # Sobel Thresholding
    def __init__(self, img):
        cr = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)[:, :, 1]
        # s = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)[:, :, 1]
        self.rgb_im = np.dstack([img[:, :, 0],  cr])  # img[:, :, 1],

    def abs_sobel_thresh(self, orient='x', sobel_kernel=3, thresh=(0, 255)):
        # cv2.COLOR_RGB2GRAY if mpimg.imread(); cv2.COLOR_BGR2GRAY if cv2.imread()
        if orient == 'x':
            sobel_dir = cv2.Sobel(self.rgb_im, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        elif orient == 'y':
            sobel_dir = cv2.Sobel(self.rgb_im, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        else:
            return np.copy(self.rgb_im)

        abs_sobel = np.absolute(sobel_dir)
        scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))  # Scale to 8-bit (0-255) then convert to np.uint8

        binary_output = np.zeros_like(scaled_sobel)
        binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
        return binary_output

    def gradient_mag(self, sobel_kernel=3, thresh=(0, 255)):
        # gradients
        sobel_x = cv2.Sobel(self.rgb_im, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobel_y = cv2.Sobel(self.rgb_im, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

        sobel_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        scaled_sobel = np.uint8(255 * sobel_mag / np.max(sobel_mag))

        binary_out = np.zeros_like(scaled_sobel)
        binary_out[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
        return binary_out

    def gradient_dir(self, sobel_kernel=3, thresh=(0, np.pi / 2)):
        sobel_x = cv2.Sobel(self.rgb_im, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobel_y = cv2.Sobel(self.rgb_im, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

        direction = np.arctan2(np.absolute(sobel_x), np.absolute(sobel_y))

        binary_output = np.zeros_like(direction)
        binary_output[(direction >= thresh[0]) & (direction <= thresh[1])] = 1
        return binary_output

    def binary_out(self, demo=False):
        k_size = 5  # Choose a larger odd number to smooth gradient measurements
        gradient_x = self.abs_sobel_thresh(orient='x', sobel_kernel=k_size, thresh=(18, 100))
        gradient_y = self.abs_sobel_thresh(orient='y', sobel_kernel=k_size, thresh=(18, 100))
        magnitude = self.gradient_mag(sobel_kernel=k_size, thresh=(30, 100))
        direction = self.gradient_dir(sobel_kernel=k_size, thresh=(1.4, 1.7))
        combined = np.zeros_like(direction)
        combined[(gradient_x == 1) | ((magnitude == 1) & (direction == 1))] = 1  # & (gradient_y == 1)
        if demo:
            f, ((ax1, ax2, ax3), (ax4, ax5, _)) = plt.subplots(2, 3, figsize=(24, 9))
            f.tight_layout()
            ax1.imshow(np.bitwise_or(gradient_x, gradient_x))
            ax2.imshow(np.bitwise_or(gradient_y, gradient_y))
            ax3.imshow(np.bitwise_or(magnitude, magnitude))
            ax4.imshow(direction)
            ax5.imshow(combined)
            ax1.set_title('gradient_x', fontsize=50)
            ax2.set_title('gradient_y', fontsize=50)
            ax3.set_title('magnitude', fontsize=50)
            ax4.set_title('direction', fontsize=50)
            ax5.set_title('combined', fontsize=50)
            plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
            plt.show()
        return combined.sum(axis=2)


class ColorFiltering:
    def __init__(self, img):
        self.rgb_img = img

    def hls_filter(self, s_thresh=(130, 250), h_thresh=(90, 100), r_thresh=(200, 255)):  # 130, 140
        """
        Color filtering to eliminate light and color variance of the frames.
        :param s_thresh: 90,254
        :param h_thresh: 50,102
        :param r_thresh:
        :return:
        """
        r_channel = self.rgb_img[:, :, 0]
        hls = cv2.cvtColor(self.rgb_img, cv2.COLOR_RGB2HLS)
        h_channel = hls[:, :, 0]
        s_channel = hls[:, :, 2]

        s_binary = np.zeros_like(s_channel)  # Threshold saturation channel
        s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
        h_binary = np.zeros_like(h_channel)  # Threshold hue channel
        h_binary[(s_channel >= h_thresh[0]) & (h_channel <= h_thresh[1])] = 1
        r_binary = np.zeros_like(r_channel)  # Threshold red channel
        r_binary[(r_channel >= r_thresh[0]) & (r_channel <= r_thresh[1])] = 1
        return s_binary, h_binary, r_binary

    def yellow_white_filter(self, z_thresh=(235, 253), white_range=40):  # 10
        """
        Color filtering for picking-up lanes in yellow and white colors
        :param z_thresh:
        :param white_range:
        :return: binary_out
        """
        xyz = cv2.cvtColor(self.rgb_img, cv2.COLOR_RGB2XYZ)[:, :, 2]
        yellow_mask = cv2.inRange(xyz, z_thresh[0], z_thresh[1])
        yellow_mask[yellow_mask > 0] = 1

        hsv = cv2.cvtColor(self.rgb_img, cv2.COLOR_RGB2HSV)
        white_low = np.array([0, 0, 255 - white_range])
        white_high = np.array([255, white_range, 255])
        white_mask = cv2.inRange(hsv, white_low, white_high)
        white_mask[white_mask > 0] = 1
        return yellow_mask, white_mask

    def binary_out(self, demo=False):
        s_bin, h_bin, r_bin = self.hls_filter()
        yellow, white = self.yellow_white_filter()
        combined = np.zeros_like(s_bin)  # Binary image with all channels stacked

        combined[((s_bin == 1) | (h_bin == 1) | (r_bin == 1)) | ((yellow == 1) | (white == 1))] = 1
        if demo:
            f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(24, 9))
            f.tight_layout()
            ax1.imshow(s_bin)
            ax2.imshow(h_bin)
            ax3.imshow(r_bin)
            ax4.imshow(yellow)
            ax5.imshow(white)
            ax6.imshow(combined)
            ax1.set_title('saturation', fontsize=50)
            ax2.set_title('hue', fontsize=50)
            ax3.set_title('red', fontsize=50)
            ax4.set_title('yellow', fontsize=50)
            ax5.set_title('white', fontsize=50)
            ax6.set_title('combined', fontsize=50)
            plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
            plt.show()
        return combined


def lane_identification(img, demo=False):
    """
    :param img:
    :param demo:
    :return:
    """
    corrected = cv2.undistort(img, matrix, distance, None, matrix)

    # Color/gradient Filtering
    filtered = np.zeros_like(img[:, :, 0])
    filtered[(Thresholding(corrected).binary_out() >= 1) | (ColorFiltering(corrected).binary_out() >= 1)] = 1

    # Apply the ROI
    masked = region_of_interest(filtered)

    # Perspective transform
    warped = warper(masked)

    # Detect lane lines
    if l_line.detected and r_line.detected:
        l_lane_idx, r_lane_idx = search_around_poly(warped)
    else:
        l_lane_idx, r_lane_idx = find_lane_pixels(warped)

    fit_polynomial(l_lane_idx, r_lane_idx)

    measure_curvature()

    if r_line.fittedx_buff.is_full() and r_line.curvature_buff.is_full():
        sanity_checks()

    # Fill buffers until they are full
    if not r_line.fittedx_buff.is_full():
        l_line.fittedx_buff.add_data(l_line.fittedx_last)
        l_line.poly_buff.add_data(l_line.poly_last)
        l_line.poly_buff_m.add_data(l_line.poly_last_m)
        r_line.fittedx_buff.add_data(r_line.fittedx_last)
        r_line.poly_buff.add_data(r_line.poly_last)
        r_line.poly_buff_m.add_data(r_line.poly_last_m)
    if not r_line.curvature_buff.is_full():
        l_line.curvature_buff.add_data([l_line.curvature_last])
        r_line.curvature_buff.add_data([r_line.curvature_last])

    # Append to circular buffers and plot
    if r_line.fittedx_buff.is_full():
        if l_line.detected:
            l_line.line_base_pos = filtered.shape[1]/2 - l_line.fittedx_last[0]  # The camera is mounted at car's center
            l_line.fittedx_buff.append(l_line.fittedx_last)
            l_line.poly_buff.append(l_line.poly_last)
            l_line.poly_buff_m.append(l_line.poly_last_m)
        if r_line.detected:
            r_line.line_base_pos = r_line.fittedx_last[0] - filtered.shape[1]/2
            r_line.fittedx_buff.append(r_line.fittedx_last)
            r_line.poly_buff.append(r_line.poly_last)
            r_line.poly_buff_m.append(r_line.poly_last_m)
    if r_line.curvature_buff.is_full():
        if l_line.detected:
            l_line.curvature_buff.append(l_line.curvature_last)
        if r_line.detected:
            r_line.curvature_buff.append(r_line.curvature_last)

    # Print lanes and info
    stacked = map_lane(corrected, warped)

    if demo:
        corrected = cv2.undistort(img, matrix, distance, None, matrix)
        # src2 = np.array([[[455, 547], [839, 547], [1106, 717], [210, 717]]], np.int32)
        # # src2 = np.array([[[585, 460], [203, 720], [1127, 720], [695, 460]]], np.int32)
        # corrected2 = cv2.polylines(corrected, [src2], isClosed=True, color=(255, 0, 0), thickness=4)
        # warped2 = cv2.warpPerspective(corrected2, ImgTrans.warp_mat, (im_dim_x, im_dim_y), flags=cv2.INTER_LINEAR)
        # compare_two_img(corrected2, warped2, img2_title='Warped & Undistorted Image')

        # Color/gradient Filtering
        color_filtered_img = ColorFiltering(corrected).binary_out(demo=False)
        threshold_img = Thresholding(corrected).binary_out(demo=False)
        filtered = np.zeros_like(color_filtered_img)
        filtered[(threshold_img >= 1) | (color_filtered_img == 1)] = 1
        masked = region_of_interest(filtered)
        # plt.imshow(masked)
        # plt.show()

        warped = cv2.warpPerspective(masked, ImgTrans.warp_mat, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
        nonzero = warped.nonzero()
        l_line.allx = np.array(nonzero[1])
        l_line.ally = np.array(nonzero[0])
        r_line.allx = l_line.allx
        r_line.ally = l_line.ally

        find_lane_pixels(warped, demo=True)
        return 0
    return stacked


""" *******************************************************************************************
                                        Main
    ******************************************************************************************* """

# # Read in and correct the image
# image = cv2.imread('test_images/straight_lines1.jpg')
# lane_identification(image, demo=True)

# video_name = 'harder_challenge_video.mp4'
# video_name = 'challenge_video.mp4'
video_name = 'project_video.mp4'
challenge_output = f"output_videos/{video_name}"
# To try a shorter subclip of the video add .subclip(start_second,end_second) to the end of the line below
# start_second and end_second are integers representing the start and end of the subclip
clip = VideoFileClip(f"{video_name}")  # .subclip(15, 26)
clip_out = clip.fl_image(lane_identification)
clip_out.write_videofile(challenge_output, audio=False)
