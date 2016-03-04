from os import path
# from threading import Thread
import traceback
import logging
import time
from ethoscope.web_utils.control_thread import ControlThread, ExperimentalInformations
from ethoscope.utils.description import DescribedObject
import os
import tempfile
import shutil
import multiprocessing
import glob
import os
import datetime

class PiCameraProcess(multiprocessing.Process):
    _VIDEO_CHUNCK_DURATION = 30 * 10
    def __init__(self, stop_queue,video_prefix, video_root_dir, img_path, width, height, fps, bitrate):
        self._stop_queue = stop_queue
        self._img_path = img_path
        self._resolution = (width, height)
        self._fps = fps
        self._bitrate = bitrate
        self._video_prefix = video_prefix
        self._video_root_dir = video_root_dir
        super(PiCameraProcess, self).__init__()

    def _make_video_name(self, i):
        return '%s_%04d.h264' % (self._video_prefix, i)

    def _write_video_index(self):
        index_file = os.path.join(self._video_root_dir, "index.html")
        all_video_files = [y for x in os.walk(self._video_root_dir) for y in glob.glob(os.path.join(x[0], '*.h264'))]

        with open(index_file, "w") as index:
            for f in all_video_files:
                index.write(f + "\n")

    def run(self):
        import picamera
        i = 0

        try:
            with picamera.PiCamera() as camera:
                camera.resolution = self._resolution
                camera.framerate = self._fps
                camera.start_recording(self._make_video_name(i), bitrate=self._bitrate)
                self._write_video_index()
                start_time = time.time()
                i += 1
                while True:
                    camera.wait_recording(2)
                    camera.capture(self._img_path, use_video_port=True)
                    if time.time() - start_time >= self._VIDEO_CHUNCK_DURATION:
                        camera.split_recording(self._make_video_name(i))
                        self._write_video_index()
                        start_time = time.time()
                        i += 1
                    if not self._stop_queue.empty():
                        self._stop_queue.get()
                        self._stop_queue.task_done()
                        break

                camera.wait_recording(1)
                camera.stop_recording()

        except Exception as e:
            logging.error("Error or starting video record:" + traceback.format_exc(e))


class VideoRecorder(DescribedObject):
    _description  = {   "overview": "A video simple recorder",
                            "arguments": [
                                {"type": "number", "name":"width", "description": "The width of the frame","default":1280, "min":480, "max":1980,"step":1},
                                {"type": "number", "name":"height", "description": "The height of the frame","default":960, "min":360, "max":1080,"step":1},
                                {"type": "number", "name":"fps", "description": "The target number of frames per seconds","default":25, "min":1, "max":25,"step":1},
                                {"type": "number", "name":"bitrate", "description": "The target bitrate","default":200000, "min":0, "max":10000000,"step":1000}
                               ]}

    def __init__(self, video_prefix, video_dir, img_path,width=1280, height=960,fps=25,bitrate=200000):

        self._stop_queue = multiprocessing.JoinableQueue(maxsize=1)
        self._p = PiCameraProcess(self._stop_queue, video_prefix, video_dir, img_path, width, height,fps, bitrate)


    def run(self):
        self._is_recording = True
        self._p.start()
        while self._p.is_alive():
            time.sleep(.25)
    def stop(self):
        self._is_recording = False
        self._stop_queue.put(None)
        self._stop_queue.close()
        self._p.join(10)









class ControlThreadVideoRecording(ControlThread):

    _evanescent = False
    _option_dict = {

        "recorder":{
                "possible_classes":[VideoRecorder],
            },
        "experimental_info":{
                        "possible_classes":[ExperimentalInformations],
                }
     }
    for k in _option_dict:
        _option_dict[k]["class"] =_option_dict[k]["possible_classes"][0]
        _option_dict[k]["kwargs"] ={}


    _tmp_last_img_file = "last_img.jpg"
    _dbg_img_file = "dbg_img.png"
    _log_file = "ethoscope.log"

    def __init__(self, machine_id, name, version, ethoscope_dir, data=None, *args, **kwargs):

        # for FPS computation
        self._last_info_t_stamp = 0
        self._last_info_frame_idx = 0
        self._recorder = None

        now = time.time()
        date_time = datetime.datetime.fromtimestamp(now)
        formated_time = date_time.strftime('%Y-%m-%d_%H-%M-%S')
        device_id = machine_id
        device_name = name
        file_prefix = "%s_%s" % (formated_time, device_id)
        self._video_root_dir = ethoscope_dir
        self._output_video_full_prefix = os.path.join(ethoscope_dir,
                                      device_id,
                                      device_name,
                                      formated_time,
                                      file_prefix
                                      )

        try:
            os.makedirs(os.path.dirname(self._output_video_full_prefix))
        except OSError:
            pass

        self._tmp_dir = tempfile.mkdtemp(prefix="ethoscope_")

        #todo add 'data' -> how monitor was started to metadata
        self._info = {  "status": "stopped",
                        "time": time.time(),
                        "error": None,
                        "log_file": os.path.join(ethoscope_dir, self._log_file),
                        "dbg_img": os.path.join(ethoscope_dir, self._dbg_img_file),
                        "last_drawn_img": os.path.join(self._tmp_dir, self._tmp_last_img_file),
                        "id": machine_id,
                        "name": name,
                        "version": version,
                        "experimental_info": {}
                        }

        self._parse_user_options(data)
        super(ControlThread, self).__init__()

    def _update_info(self):
        if self._recorder is None:
            return
        self._last_info_t_stamp = time.time()


    def _parse_one_user_option(self,field, data):

        try:
            subdata = data[field]
        except KeyError:
            logging.warning("No field %s, using default" % field)
            return None, {}

        Class = eval(subdata["name"])
        kwargs = subdata["arguments"]

        return Class, kwargs


    def run(self):

        try:
            self._info["status"] = "initialising"
            logging.info("Starting Monitor thread")

            self._info["error"] = None


            self._last_info_t_stamp = 0
            self._last_info_frame_idx = 0

            ExpInfoClass = self._option_dict["experimental_info"]["class"]
            exp_info_kwargs = self._option_dict["experimental_info"]["kwargs"]
            self._info["experimental_info"] = ExpInfoClass(**exp_info_kwargs).info_dic
            self._info["time"] = time.time()


            logging.info("Start recording")

            RecorderClass = self._option_dict["recorder"]["class"]
            recorder_kwargs = self._option_dict["recorder"]["kwargs"]
            self._recorder = RecorderClass(video_prefix = self._output_video_full_prefix,
                                           video_dir = self._video_root_dir,
                                           img_path=self._info["last_drawn_img"],**recorder_kwargs)

            self._info["status"] = "recording"
            self._recorder.run()
            logging.warning("recording RUN finished")


        except Exception as e:
            self.stop(traceback.format_exc(e))

        #for testing purposes
        if self._evanescent:
            import os
            self.stop()
            os._exit(0)


    def stop(self, error=None):

        if error is not None:
            logging.error("Recorder closed with an error:")
            logging.error(error)
        else:
            logging.info("Recorder closed all right")

        self._info["status"] = "stopping"
        self._info["time"] = time.time()
        self._info["experimental_info"] = {}

        logging.info("Stopping monitor")
        if self._recorder is not None:
            logging.warning("Control thread asking recorder to stop")
            self._recorder.stop()

            self._recorder = None

        self._info["status"] = "stopped"
        self._info["time"] = time.time()
        self._info["error"] = error

