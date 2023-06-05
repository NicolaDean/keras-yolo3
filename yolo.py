# -*- coding: utf-8 -*-
"""
Class definition of YOLO_v3 style detection model on image and video
"""

import colorsys
import os
from timeit import default_timer as timer

import numpy as np
import tensorflow.python.keras.backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body, yolo_loss, box_iou
from yolo3.utils import letterbox_image
import os
import tensorflow as tf


def compute_map():
            '''
            TODO
            MAP tutorial
            https://learnopencv.com/mean-average-precision-map-object-detection-model-evaluation-metric/
            IOU tutorial
            #https://www.kaggle.com/code/vbookshelf/keras-iou-metric-implemented-without-tensor-drama
            Per ogni box della true voglio trovare dei box sovrapposti con stessa classe
            (per ogni box prendo in considerazione solo il miglior iou! (per evitare fake))

            Se un box rimane senza match => False Negative
            
            Se iou > 0.5 con stessa classe => TP
            Se iou < 0.5 con stessa classe => FP
            Se iou == 0 FN
        
            TN non si puo calcolare

            P  =  TP/(TP + FP)  #Precision
            R = TP / (TP + FN)  #Recall
            
            '''
            pass

def compute_F1_score(y_true_boxes,y_true_classes,out_boxes, out_classes,iou_th=0.5):

    out_boxes   = out_boxes.numpy().tolist()
    out_classes = out_classes.numpy().tolist()
    TP = 0
    FP = 0
    FN = 0
    #TN Do not exist in Object detection => Is the background

    precision = 0
    recall    = 0
    f1_score  = 0

    #For each True label extract the best matching box (Same class and best IOU)
    for b_true,class_true in zip(y_true_boxes,y_true_classes):
        iou_tmp = iou_th
        curr_sel_idx = None
        idx = 0
        for b_pred,pred_class in zip(out_boxes,out_classes):

            #print(f"[{pred_class}] vs [{class_true}]")
            if pred_class == class_true:
                tmp = box_iou(b_true,b_pred)
                #Check if this BB is a best match than the previous one
                if tmp > iou_tmp:
                    iou_tmp = tmp
                    curr_sel_idx = idx
            idx += 1
        #If match found remove it from expected Bounding Box
        if iou_tmp > iou_th:
            out_boxes.pop(curr_sel_idx)
            out_classes.pop(curr_sel_idx)
            TP += 1 #True Positive are golden box with match
        else:
            FN += 1 #False Negative are golden box without match
    
    #False Positive are all Resulted box without a match in the golden list.
    #if len(out_boxes) != 0:
    FP = len(out_boxes)

    #Compute Precision, Recall, F1_score
    if TP + FP != 0:
        precision = TP / (TP + FP) 
    if TP + FN != 0:
        recall    = TP / (TP + FN)
    if precision + recall != 0:
        f1_score  = (2*precision*recall)/(precision + recall)

    print("-"*20)
    print(f"Sample statistics => TP [{TP}] , FP [{FP}], FN [{FN}]")
    print(f"precison: [{precision}] , recall: [{recall}], f1_score: [{f1_score}]")
    print("-"*20)

    return precision,recall,f1_score

    ''' 
    #For each Expected label extract the best matching box with True Label(Same class and best IOU)
    for b_pred,pred_class in zip(out_boxes,out_classes):
        iou_tmp = iou_th
        curr_sel_idx = None
        idx = 0
        for b_true,class_true in not_matching_golden:
    '''

    

def compute_iou(y_true_boxes,y_true_classes,out_boxes, out_scores, out_classes):

    #TODO FIX => DO THE OPPOSITE => FOR EACH PREDICTION CHECK FOR A LABEL (and return the list of max iou for each box)
    #This will help in FP,TP,FN computation => therfore help computing Precision,Recall,F1 and finaly MAP score
    iou_res = 0
    i = 0
    #For each True label extract the best matching box (Same class and best IOU)
    for b_true,class_true in zip(y_true_boxes,y_true_classes):
        iou_tmp = 0

        for b_pred,pred_score,pred_class in zip(out_boxes,out_scores,out_classes):
            #print(f"[{pred_class}] vs [{class_true}]")
            if pred_class == class_true:
                tmp = box_iou(b_true,b_pred)
                if tmp > iou_tmp:
                    iou_tmp = tmp
        #print(f"Label [{i}] has IOU =  [{iou_tmp}]")
        iou_res += iou_tmp
        i += 1
    if len(y_true_classes) != 0:
        return iou_res / len(y_true_classes)
    else:
        return None

class YOLO(object):
    _defaults = {
        "model_path": 'model_data/yolo.h5',
        "anchors_path": 'model_data/yolo_anchors.txt',
        "classes_path": 'model_data/coco_classes.txt',
        "score" : 0.3,
        "iou" : 0.45,
        "model_image_size" : (416, 416),
        "gpu_num" : 1,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        print(self.class_names)
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        #self.input_image_shape = (2,)
        self.input_image_shape =  [2, 2]
        if self.gpu_num>=2:
            self.yolo_model = tf.keras.utils.multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,len(self.class_names), self.input_image_shape,score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image,y_true = None):
        start = timer()

        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                                image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)

        image_data = np.array(boxed_image, dtype='float32')

        print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
            
        yolo_out = self.yolo_model.predict(image_data)
        

        '''
        if y_true:
            print(f"Banana = {self.class_names}")
            argss = [yolo_out[0],yolo_out[1],yolo_out[2],y_true[0],y_true[1],y_true[2]]
            loss = yolo_loss(argss,self.anchors,len(self.class_names),0.5)
            print(f"LOSS = [{loss}]")
        '''

        self.input_image_shape =  [image.size[1], image.size[0]]
        out_boxes, out_scores, out_classes = yolo_eval(yolo_out, self.anchors,len(self.class_names), self.input_image_shape,score_threshold=self.score, iou_threshold=self.iou)
        y_true = tf.constant(y_true,dtype=tf.float32)
        #print("PRED")
        #print(out_boxes)
        #print("YTRUE")
        #print(y_true[0])
        if y_true:
            y_true          = np.hsplit(y_true[0],[4,5])
            y_true_boxes    = y_true[0]
            y_true_classes  = y_true[1]

            iou_score = compute_iou(y_true_boxes,y_true_classes,out_boxes, out_scores, out_classes)
            #iou_score = compute_iou(out_boxes,out_classes,out_boxes, out_scores, out_classes) => SHOUD OUTPUT 1 (Testing purpose)

            print(f"=========[ IOU = {iou_score} ]=========")

            print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        #font = ImageFont.truetype(font='font/FiraMono-Medium.otf',size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        font = font = ImageFont.load_default()
        thickness = (image.size[0] + image.size[1]) // 300


        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom))

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw

        end = timer()
        print(end - start)

        if not y_true:
            return image,out_boxes, out_scores, out_classes
        return image

    def close_session(self):
        self.sess.close()

def detect_video(yolo, video_path, output_path=""):
    import cv2
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC    = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps       = vid.get(cv2.CAP_PROP_FPS)
    video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if output_path != "" else False
    if isOutput:
        print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    while True:
        return_value, frame = vid.read()
        image = Image.fromarray(frame)
        image = yolo.detect_image(image)
        result = np.asarray(image)
        
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", result)
        if isOutput:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    yolo.close_session()

