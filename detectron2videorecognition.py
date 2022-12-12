detectron2 @ https://github.com/facebookresearch/detectron2.git

import detectron2
from detectron2.engine import DefaultPredictor
from detectron2.utils.logger import setup_logger
#setup_logger()
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2 import model_zoo


def __init__(self, model_type = "objectDetection"):
    self.cfg = get_cfg()
    self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
    self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
    self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    self.cfg.MODEL.DEVICE = "cpu" # cpu or cuda

    self.predictor = DefaultPredictor(self.cfg)


def videoInput(self, uploaded_video):
    video = cv2.VideoCapture(uploaded_video)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_per_second = video.get(cv2.CAP_PROP_FPS)
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    timestamp = datetime.timestamp(datetime.now())
    video_path = os.path.join('data/outputs',str(timestamp)+image_file.name)
    output_path = os.path.join('data/outputs', os.path.basename(video_path))

    fourcc = cv2.VideoWriter_fourcc(*'mvp4')
    video_writer = cv2.VideoWriter('output.mp4', fourcc, fps=float(frames_per_second), frameSize=(width, height), isColor=True)

    v = VideoVisualizer(MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), ColorMode.IMAGE)

    def runVideo(video, maxFrames):
        readFrames = 0
        while True:
            hasFrame, fame = video.read()
            if not hasFrame:
                break
            output = self.predictor(frame)

            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            visualization = v.draw_instance_predictions(frame, outputs['instances'].to('cpu'))
            visualization = cv2.cvtColor(visualization.get_image(),cv2.COLOR_RGB2BGR)
            yield visualization

            readFrames += 1
            if readFrames > maxFrames:
                break

    num_frames = 200
    for visualization in tqdm.tqdm(ruVideo(video, num_frames), total = num_frames):
        video_writer.write(visualization)
    video.release()
    video_writer.release()
    cv2.destroyAllWindows()

def videoFunc(x):
    detector = Detector(model_type=x)
    uploaded_video = st.file_uploader("Upload video", type=['mp4', 'mpeg', 'mov'])
    if uploaded_video is not None:
        video = uploaded_video.name
        with open (video, mode = 'wb') as f:
            f.write(uploaded_video.read())

        st_video = open(video, 'rb')
        video_bytes = open('ouput.mp4','rb')
        st.video(video_bytes)
        st.write('Uploaded Video')
        detector.videoInput(video)
        st_video = open('output.mp4','rb')
        video_bytes = st_video.read()
        st.video(video_bytes)
        st.write("Detected Video")
