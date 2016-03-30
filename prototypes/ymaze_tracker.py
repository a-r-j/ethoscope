__author__ = 'diana'


from ethoscope.hardware.input.cameras import MovieVirtualCamera
from ethoscope.trackers.trackers import *
from ethoscope.roi_builders.target_roi_builder import TargetGridROIBuilder
from ethoscope.core.monitor import Monitor
from ethoscope.core.data_point import DataPoint
from ethoscope.utils.io import SQLiteResultWriter
from ethoscope.drawers.drawers import DefaultDrawer
import cv2
import numpy as np
import optparse
import logging
import traceback


class YMazeTracker(BaseTracker):

    def __init__(self, roi, data=None):
        self._accum = None
        self._alpha = 0.001
        super(YMazeTracker, self).__init__(roi, data)



    def _filter_contours(self, contours, min_area =50, max_area=200):
        out = []
        for c in contours:
            if c.shape[0] < 6:
                continue
            area = cv2.contourArea(c)
            if not min_area < area < max_area:
                continue

            out.append(c)
        return out
    def _find_position(self, img, mask,t):
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return self._track(img, grey, mask, t)


    def _track(self, img,  grey, mask, t):
        if self._accum is None:
            self._accum = grey.astype(np.float64)

        frame_float64 = grey.astype(np.float64)
        cv2.accumulateWeighted(frame_float64, self._accum, self._alpha)
        bg = self._accum.astype(np.uint8)

        diff = cv2.subtract(bg, grey)
        cv2.medianBlur(grey, 7, grey)
        _, bin_im = cv2.threshold(diff, 100, 255, cv2.THRESH_BINARY)

        contours,hierarchy = cv2.findContours(bin_im,
                                              cv2.RETR_EXTERNAL,
                                              cv2.CHAIN_APPROX_SIMPLE)

        contours= self._filter_contours(contours)

        if len(contours) != 1:
            raise NoPositionError
        hull = contours[0]

        (_,_) ,(w,h), angle = cv2.minAreaRect(hull)

        M = cv2.moments(hull)
        x = int(M['m10']/M['m00'])
        y = int(M['m01']/M['m00'])
        if w < h:
            angle -= 90
            w,h = h,w
        angle = angle % 180

        h_im = min(grey.shape)
        max_h = 2*h_im
        if w>max_h or h>max_h:
            raise NoPositionError
        x_var = XPosVariable(int(round(x)))
        y_var = YPosVariable(int(round(y)))
        w_var = WidthVariable(int(round(w)))
        h_var = HeightVariable(int(round(h)))
        phi_var = PhiVariable(int(round(angle)))

        out = DataPoint([x_var, y_var, w_var, h_var, phi_var])

        return [out]


class Ymaze(TargetGridROIBuilder):
    _vertical_spacing =  0
    _horizontal_spacing =  0
    _n_rows = 1
    _n_cols = 1
    _top_margin =  0.05
    _bottom_margin = 0.05
    _left_margin = 0.05
    _right_margin = 0.05

    def _find_blobs(self, im, scoring_fun):

        # Read image
        grey = cv2.cvtColor(im, cv2.IMREAD_GRAYSCALE)
        # Set up the detector with default parameters.
        detector = cv2.SimpleBlobDetector()
        print "Hello"

        # Detect blobs.
        keypoints = detector.detect(grey)

        # Draw detected blobs as red circles.
        # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
        im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # Show keypoints
        cv2.imshow("Keypoints", im_with_keypoints)
        cv2.waitKey(0)
        # grey= cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        # rad = int(self._adaptive_med_rad * im.shape[1])
        # if rad % 2 == 0:
        #     rad += 1
        #
        # med = np.median(grey)
        # scale = 255/(med)
        # cv2.multiply(grey,scale,dst=grey)
        # bin = np.copy(grey)
        # score_map = np.zeros_like(bin)
        # for t in range(0, 255,5):
        #     cv2.threshold(grey, t, 255,cv2.THRESH_BINARY_INV,bin)
        #     if np.count_nonzero(bin) > 0.7 * im.shape[0] * im.shape[1]:
        #         continue
        #     contours, h = cv2.findContours(bin,cv2.RETR_EXTERNAL,cv2.cv.CV_CHAIN_APPROX_SIMPLE)
        #     bin.fill(0)
        #     for c in contours:
        #         score = scoring_fun(c, im)
        #         if score >0:
        #             cv2.drawContours(bin,[c],0,score,-1)
        #     cv2.add(bin, score_map,score_map)
        # return score_map



if __name__ == "__main__":

    parser = optparse.OptionParser()
    parser.add_option("-o", "--output", dest="out", help="the output file (eg out.csv   )", type="str",default=None)
    parser.add_option("-i", "--input", dest="input", help="the output video file", type="str")
    #
    parser.add_option("-r", "--result-video", dest="result_video", help="the path to an optional annotated video file."
                                                                "This is useful to show the result on a video.",
                                                                type="str", default=None)

    parser.add_option("-d", "--draw-every",dest="draw_every", help="how_often to draw frames", default=0, type="int")

    parser.add_option("-m", "--mask", dest="mask", help="the mask file with 3 targets", type="str")

    (options, args) = parser.parse_args()

    option_dict = vars(options)

    logging.basicConfig(level=logging.INFO)


    logging.info("Starting Monitor thread")

    cam = MovieVirtualCamera(option_dict ["input"], use_wall_clock=False)

    #my_image = cv2.imread(option_dict['mask'])
    #print option_dict['mask']

    # accum = []
    # for i, (_, frame) in enumerate(cam):
    #     accum.append(frame)
    #     if i  >= 5:
    #         break

    #accum = np.median(np.array(accum),0).astype(np.uint8)
    # cv2.imshow('window', my_image)
    roi_builder = Ymaze()
    rois = roi_builder.build(cam)

    logging.info("Initialising monitor")

    cam.restart()

    metadata = {
                             "machine_id": "None",
                             "machine_name": "None",
                             "date_time": cam.start_time, #the camera start time is the reference 0
                             "frame_width":cam.width,
                             "frame_height":cam.height,
                             "version": "whatever"
                              }
    draw_frames = False
    if option_dict["draw_every"] > 0:
        draw_frames = True

    drawer = DefaultDrawer('~/Desktop/test_video.MP4', draw_frames = True)

    monit = Monitor(cam, YMazeTracker, rois)


    with SQLiteResultWriter(option_dict["out"], rois) as rw:
        monit.run(None, drawer)


    logging.info("Stopping Monitor")
