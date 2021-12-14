import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import matplotlib.pyplot as plt
import numpy as np
import cv2
from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer
import os

from matplotlib import cm
import numpy.linalg as lin

from PIL import Image

import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO

from operator import itemgetter

print("채점할 모의고사")
print("1. 2021 고3 9월 수학")
print("2. 2020 고3 9월 수학")
print("3. 2019 고3 9월 수학")
a = input()
if(a == "1"):
  print("선택 과목 입력 ex. 확통미적")
  b = input()
  if(b == "확통미적"):
    answer = [3,4,4,1,2,3,2,1,2,5,3,1]
elif(a == "2"):
  answer = []
elif(a == "3"):
  answer = []


cfg = get_cfg()
cfg.merge_from_file("./detectron2_repo/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.DATASETS.TRAIN = ("test-papers",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.02
cfg.SOLVER.MAX_ITER = 300
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128 
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

cfg.MODEL.WEIGHTS = "/content/drive/My Drive/test-paper-detector/detectron2/output/model_final.pth" 
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 
cfg.DATASETS.TEST = ("test-papers", )
predictor = DefaultPredictor(cfg)

PATH_TO_TEST_IMAGES_DIR = 'data/segmentation_images/test'
TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, '{}_seg.jpg'.format(i)) for i in range(115,121) ] ## 원본이미지

page = 0
for image_path in TEST_IMAGE_PATHS:  ## 각 이미지에 대해 segmentation 후 투시변환
  page += 1
  im = cv2.imread(image_path)
  outputs = predictor(im)
  v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2) #1.2
  v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
  cv2_imshow(v.get_image()[:, :, ::-1])

  ins = outputs["instances"]
  pred_masks = ins.get_fields()["pred_masks"]

  upA = np.empty((0,2), int)
  upB = np.array([])
  rightA = np.empty((0,2), int)
  rightB = np.array([])
  leftA = np.empty((0,2), int)
  leftB = np.array([])
  downA = np.empty((0,2), int)
  downB = np.array([])

  #up
  for i in range(0, int(pred_masks.size()[1]/3),2): #y
    for j in range(0, pred_masks.size()[2],7): #x
      if(pred_masks[0][i][j] == True):
        if (pred_masks[0][i-1][j] == False and pred_masks[0][i+1][j] == True):
          if(len(upA) == 0 or upA[-1][0] == j):
            upA = np.append(upA, np.array([[j, 1]]), axis=0)
            upB = np.append(upB, np.array([i]))
            continue

          if(abs((upB[-1] - i) / (upA[-1][0] - j)) < 3):
            upA = np.append(upA, np.array([[j, 1]]), axis=0)
            upB = np.append(upB, np.array([i]))

            if(len(upA) == 1):
              upA.pop(0)
              upB.pop(0)
  #down
  for i in range(int((pred_masks.size()[1]*2)/3), pred_masks.size()[1],2): #y
    for j in range(0, pred_masks.size()[2],7): #x
      if(pred_masks[0][i][j] == True):
        if (pred_masks[0][i+1][j] == False and pred_masks[0][i-1][j] == True):
          if(len(downB) == 0 or downA[-1][0] == j):
            downA = np.append(downA, np.array([[j, 1]]), axis=0)
            downB = np.append(downB, np.array([i]))

            continue
          if(abs((downB[-1] - i )/ (downA[-1][0] - j)) < 3):
            downA = np.append(downA, np.array([[j, 1]]), axis=0)
            downB = np.append(downB, np.array([i]))

            if(len(downA) == 1):
              downA.pop(0)
              downB.pop(0)
  #right
  for i in range(0, pred_masks.size()[1],10): #y
    for j in range(int((pred_masks.size()[2]*3)/4), pred_masks.size()[2],2): #x
      if(pred_masks[0][i][j] == True):
        if (pred_masks[0][i][j+1] == False and pred_masks[0][i][j-1] == True):
          if(len(rightA) == 0 or rightA[-1][0] == j):
            rightA = np.append(rightA, np.array([[j, 1]]), axis=0)
            rightB = np.append(rightB, np.array([i]))
            continue

          if(abs((rightB[-1] - i )/ (rightA[-1][0] - j)) > 6):
            rightA = np.append(rightA, np.array([[j, 1]]), axis=0)
            rightB = np.append(rightB, np.array([i]))
            
            if(len(rightA) == 1):
              rightA.pop(0)
              rightB.pop(0)

  #left
  for i in range(0, pred_masks.size()[1],10): #y
    for j in range(0, int(pred_masks.size()[2]/4),2): #x
      if(pred_masks[0][i][j] == True):   
        if (pred_masks[0][i][j-1] == False and pred_masks[0][i][j+1] == True):
          if(len(leftA) == 0 or leftA[-1][0] == j):
            leftA = np.append(leftA, np.array([[j, 1]]), axis=0)
            leftB = np.append(leftB, np.array([i]))
            continue
            
          if(abs((leftB[-1] - i )/ (leftA[-1][0] - j)) > 6):
            leftA = np.append(leftA, np.array([[j, 1]]), axis=0)
            leftB = np.append(leftB, np.array([i]))

  X_up = np.linalg.inv(upA.T.dot(upA)).dot(upA.T).dot(upB)
  X_right = np.linalg.inv(rightA.T.dot(rightA)).dot(rightA.T).dot(rightB)
  X_left = np.linalg.inv(leftA.T.dot(leftA)).dot(leftA.T).dot(leftB)
  X_down = np.linalg.inv(downA.T.dot(downA)).dot(downA.T).dot(downB)

  upandleftA = np.array([[-X_up[0], 1], [-X_left[0], 1]])
  upandleftB = np.array([X_up[1], X_left[1]])
  itpoint_1 = np.linalg.solve(upandleftA,upandleftB)

  if(itpoint_1[0] < 0):
    itpoint_1[0] = 0
  if(itpoint_1[1] < 0):
    itpoint_1[1] = 0

  upandrightA = np.array([[-X_up[0], 1], [-X_right[0], 1]])
  upandrightB = np.array([X_up[1], X_right[1]])
  itpoint_2 = np.linalg.solve(upandrightA,upandrightB)

  if(itpoint_2[0] > pred_masks.size()[2]):
    itpoint_2[0] = pred_masks.size()[2]
  if(itpoint_2[1] < 0):
    itpoint_2[1] = 0

  downandleftA = np.array([[-X_down[0], 1], [-X_left[0], 1]])
  downandleftB = np.array([X_down[1], X_left[1]])
  itpoint_3 = np.linalg.solve(downandleftA,downandleftB)

  if(itpoint_3[0] < 0):
    itpoint_3[0] = 0
  if(itpoint_3[1] > pred_masks.size()[1]):
    itpoint_3[1] = pred_masks.size()[1]

  downandrightA = np.array([[-X_down[0], 1], [-X_right[0], 1]])
  downandrightB = np.array([X_down[1], X_right[1]])
  itpoint_4 = np.linalg.solve(downandrightA,downandrightB)

  if(itpoint_4[0] > pred_masks.size()[2]):
    itpoint_4[0] = pred_masks.size()[2]
  if(itpoint_4[1] > pred_masks.size()[1]):
    itpoint_4[1] = pred_masks.size()[1]

  pts1 = np.float32([itpoint_1, itpoint_2, itpoint_4, itpoint_3])
  w1 = abs(itpoint_1[0] - itpoint_2[0])
  w2 = abs(itpoint_3[0] - itpoint_4[0])
  h1 = abs(itpoint_1[1] - itpoint_3[1])
  h2 = abs(itpoint_2[1] - itpoint_4[1])
  width = max([w1, w2])
  height = max([h1, h2])

  pts2 = np.float32([[0,0], [width-1, 0], [width-1, height-1], [0, height-1]])

  mtrx = cv2.getPerspectiveTransform(pts1, pts2)

  result = cv2.warpPerspective(im, mtrx, (int(width), int(height)))
  plt.imshow(result)
  save = Image.fromarray(result, 'RGB')
  save.save("data/result/image_page{}.jpg".format(page)) ## 각각 투시변환한 이미지 페이지별로 저장


def draw_bounding_boxes(img, output_dict,count):
    boxlist = []
    height, width, _ = img.shape
 
    obj_index = output_dict['detection_scores'] > 0.9
    
    scores = output_dict['detection_scores'][obj_index]
    boxes = output_dict['detection_boxes'][obj_index]
    classes = output_dict['detection_classes'][obj_index]

    for i in range(len(boxes)):
      boxlist.append([boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]])
    
    boxlist.sort(key=lambda x:x[1])
    print(boxlist)

    if(len(boxlist) >= 3):
      if(len(boxlist) == 4):
        if(boxlist[0][0] > boxlist[1][0]):
          boxlist[0], boxlist[1] = boxlist[1], boxlist[0]

        if(boxlist[2][0] > boxlist[3][0]):
          boxlist[2], boxlist[3] = boxlist[3], boxlist[2]
      else:
        if(boxlist[0][0] > boxlist[1][0]):
          boxlist[0], boxlist[1] = boxlist[1], boxlist[0]
    print(boxlist)
 
    count1 = 0
    for box in boxlist:
        count1 += 1

        cut_img = img[int(box[0] * height):int(box[2] * height), int(box[1] * width):int(box[3] * width)].copy()
        save_img = Image.fromarray(cut_img)
        save_img.save("data/result/question/image_page{0}{1}.jpg".format(count,count1)) ## 각 페이지 속 문제들 저장
        
    return img

PATH_TO_FROZEN_GRAPH = 'research/inference_graph/frozen_inference_graph.pb'
 
detection_graph = tf.Graph()
with detection_graph.as_default():
 
    od_graph_def = tf.compat.v1.GraphDef()
 
    with tf.compat.v2.io.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as f:
 
        serialized_graph = f.read()
        od_graph_def.ParseFromString(serialized_graph)
 
        tf.import_graph_def(od_graph_def, name = "")
 
def run_inference_for_single_image(image, graph):
    with tf.compat.v1.Session(graph = graph) as sess:
 
        input_tensor = graph.get_tensor_by_name('image_tensor:0')
        
        target_operation_names = ['num_detections', 'detection_boxes',
                                  'detection_scores', 'detection_classes', 'detection_masks']
        tensor_dict = {}
        for key in target_operation_names:
            op = None
            try:
                op = graph.get_operation_by_name(key)
                
            except:
                continue
 
            tensor = graph.get_tensor_by_name(op.outputs[0].name)
            tensor_dict[key] = tensor
 
        if 'detection_masks' in tensor_dict:
            detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
 
        output_dict = sess.run(tensor_dict, feed_dict = {input_tensor : [image]})
            
        output_dict['num_detections'] = int(output_dict['num_detections'][0])
        output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
        output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
        output_dict['detection_scores'] = output_dict['detection_scores'][0]
 
        return output_dict
   
PATH_TO_TEST_IMAGES_DIR = 'data/result'
TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image_page{}.jpg'.format(i)) for i in range(1,page+1) ] ## 각 페이지 별 seg이미지 가져오기


cnt = 0

for image_path in TEST_IMAGE_PATHS: 
    cnt += 1  
    print(cnt) 
    image_np = cv2.imread(image_path)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    height, width, _ = image_np.shape
    
    output_dict = run_inference_for_single_image(image_np, detection_graph)
    draw_bounding_boxes(image_np, output_dict, cnt)


import glob
#from glob import glob
output = glob.glob('data/result/question/*.jpg') # 모든 문제들 

###################
###  항목 감지  ###

PATH_TO_FROZEN_GRAPH = 'research/inference_graph_choice/frozen_inference_graph.pb'
 
detection_graph = tf.Graph()
with detection_graph.as_default():
 
    od_graph_def = tf.compat.v1.GraphDef()
 
    with tf.compat.v2.io.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as f:
 
        serialized_graph = f.read()
        od_graph_def.ParseFromString(serialized_graph)
 
        tf.import_graph_def(od_graph_def, name = "")
print('\n계산 그래프 로드 완료...\n')
 
 
def run_inference_for_single_image_choice(image, graph):
    with tf.compat.v1.Session(graph = graph) as sess:
 
        input_tensor = graph.get_tensor_by_name('image_tensor:0')
        
        target_operation_names = ['num_detections', 'detection_boxes',
                                  'detection_scores', 'detection_classes', 'detection_masks']
        tensor_dict = {}
        for key in target_operation_names:
            op = None
            try:
                op = graph.get_operation_by_name(key)
                
            except:
                continue
 
            tensor = graph.get_tensor_by_name(op.outputs[0].name)
            tensor_dict[key] = tensor
 
        if 'detection_masks' in tensor_dict:
            detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
 
        output_dict = sess.run(tensor_dict, feed_dict = {input_tensor : [image]})
            
        output_dict['num_detections'] = int(output_dict['num_detections'][0])
        output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
        output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
        output_dict['detection_scores'] = output_dict['detection_scores'][0]
 
        return output_dict

def draw_bounding_boxes_choice(img, output_dict, class_info, number):
    height, width, _ = img.shape
    obj_index = output_dict['detection_scores'] > 0.5
    
    scores = output_dict['detection_scores'][obj_index]
    boxes = output_dict['detection_boxes'][obj_index]
    classes = output_dict['detection_classes'][obj_index]
    count = 0

    boxlist = []

    
    for box, cls in zip(boxes, classes):
        boxlist.append([box[0], box[1], box[2], box[3], cls])

    option = 0
    cp = 0
    for i in range(len(boxlist)):
      for j in range(len(boxlist)):
        if(boxlist[i][2] < boxlist[j][0]):
          cp += 1
    if(cp > 6 and cp < 8): # 두줄
      option = 2
    elif(cp == 8): #세줄
      option = 3

    temp = []
    if (option == 0):
      temp = boxlist
      temp.sort(key=lambda x:x[1])

    elif (option == 2): #y축 배열 후 해당 행에서 다시 비교
      boxlist.sort(key=lambda x:x[0])
      for i in range(5):
        if(boxlist[i][4] == 2):
          if(i == 3 or i == 4):
            temp.append(boxlist[0])
            temp.append(boxlist[1])
            temp.append(boxlist[2])
            del boxlist[0]
            del boxlist[0]
            del boxlist[0]
            boxlist.sort(key=lambda x:x[1])
            temp.append(boxlist[0])
            temp.append(boxlist[1])
            break

          elif(i == 0 or i == 1 or i == 2):
            del boxlist[3]
            del boxlist[3]
            boxlist.sort(key=lambda x:x[1])
            temp.append(boxlist[0])
            temp.append(boxlist[1])
            temp.append(boxlist[2])
            break

    else: #y축 배열 후 해당 행에서 다시 비교
      boxlist.sort(key=lambda x:x[0])
      for i in range(5):
        if(boxlist[i][4] == 2):
          
          if(i == 0 or i == 1):
            del boxlist[2]
            del boxlist[2]
            boxlist.sort(key=lambda x:x[1])
            temp.append(boxlist[0])
            temp.append(boxlist[1])
            temp.append(boxlist[2])
            break

          elif(i == 2 or i == 3):
            temp.append(boxlist[0])
            temp.append(boxlist[1])
            del boxlist[0]
            del boxlist[0]
            del boxlist[2]
            boxlist.sort(key=lambda x:x[1])
            temp.append(boxlist[0])
            temp.append(boxlist[1])
            break

          else:
            temp = boxlist
            break

    user_answer = 0
    for box in temp:
        # draw bounding box
        count += 1
        if(box[4] == 2):
          print(count)
          user_answer = count

        img = cv2.rectangle(img,
                            (int(box[1] * width), int(box[0] * height)),
                            (int(box[3] * width), int(box[2] * height)), class_info[box[4]][1], 2)
        
    save = Image.fromarray(img, 'RGB')   
    save.save("data/result/question/image_{}.jpg".format(number))

    return user_answer


class_info = {}
class_info = {1: ['not-choice', (255,255,0)], 
              2: ['choice', (0,0,255)]}

number = 0
user_answer = []
for image_path in output: ## 잘린 문제들 각각 모두   
    image_np = cv2.imread(image_path)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    height, width, _ = image_np.shape
    
    number += 1
    print(number,"번 :", end=" ")
    
    output_dict = run_inference_for_single_image_choice(image_np, detection_graph)
    user_answer.append(draw_bounding_boxes_choice(image_np, output_dict,class_info,number))

for i in range(len(answer)):
  if(user_answer[i] != answer[i]):
    print(i,"번의 정답: ", answer[i], "선택한 답 : ", user_answer[i])