# Scoring-Paper-Tests-with-RCNN

## Summary

 최근 태블릿PC로 공부하는 학생들이 증가하고 있습니다.그래서 태블릿피씨를 활용한 모바일 앱들이 증가하고 있으며, 특히 학생들을 위한 기출문제 풀이를 위한 어플서비스가 개발되고 있습니다. 서비스 기능에는 답을 입력해 자동으로 채점해주는 서비스들이 제공되어 학생들은 좀더 편리하게 공부를 할 수 있습니다. 
 
 그러나 여전히 학생들은 종이 모의고사를 통해 시험을 치르고 교과서와 문제집은 모두 종이로 되어있어  종이 문제지와 시험지를 위한 서비스가 있다면 좋겠다는 생각을 했습니다. 그래서 종이시험지를 위한 ai서비스를 기획해 학생들이 편리하게 학습할 수 있는 그 범위를 확장하고자 했고, 종이시험지 자동 채점 프로그램을 기획하게 되었습니다.

## 개발환경
- Python 3.7.12
- Tensorflow 2.7.0
- Pytorch 1.9.1
- torchvision 0.10.1

## Table of contents

1. [프로그램 동작 과정](https://github.com/Nayeon12/Scoring-Paper-Tests-with-RCNN#1-%ED%94%84%EB%A1%9C%EA%B7%B8%EB%9E%A8-%EB%8F%99%EC%9E%91-%EA%B3%BC%EC%A0%95)
2. [설치](https://github.com/Nayeon12/Scoring-Paper-Tests-with-RCNN#2-%EC%84%A4%EC%B9%98) 
3. [데이터 준비](https://github.com/Nayeon12/Scoring-Paper-Tests-with-RCNN#3-%EB%8D%B0%EC%9D%B4%ED%84%B0-%EC%A4%80%EB%B9%84)
>- step 1. 데이터 수집
>- step 2. 데이터 라벨링
>- step 3. csv파일 통합
>- step 4. TFRecord 파일 생성
>- step 5. Label map 생성
4. [모델 학습](https://github.com/Nayeon12/Scoring-Paper-Tests-with-RCNN#4-%EB%AA%A8%EB%8D%B8-%ED%95%99%EC%8A%B5)
>- step 1. 배경 분리 segmentation 모델
>- step 2. 문제 감지 모델
>- step 3. 항목 감지 모델
5. [모델 테스트](https://github.com/Nayeon12/Scoring-Paper-Tests-with-RCNN#5-%EB%AA%A8%EB%8D%B8-%ED%85%8C%EC%8A%A4%ED%8A%B8)
6. [추가자료](https://github.com/Nayeon12/Scoring-Paper-Tests-with-RCNN#6-%EC%B6%94%EA%B0%80%EC%9E%90%EB%A3%8C)

## 1. 프로그램 동작 과정

![image](https://user-images.githubusercontent.com/49023717/146561507-993b8cd9-0d70-490b-a438-4d5dedcb5dff.png)

1. 먼저 배경을 포함해 사용자는 시험지를 촬영하게 됩니다. 
2. 배경을 시험지와 분리합니다. 
3. 시험지 속 문제들을 인식해 문제별로 자릅니다.
4. 문제 속 선택항목을 인식하여 사용자가 몇번을 선택했는지 판단하게 됩니다. 
5. 최종적으로 프로그램은 사용자가 선택한 답을 번호와 함께 출력하고 틀린문제를 추가로 출력하면서 프로그램을 종료하게 됩니다.


## 2. 설치

[Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection)를 git clone해 설치를 진행합니다. 디렉토리는 아래와 같이 구성하고, 포함된 데이터셋과 config 파일은 아래에서 설명하겠습니다.

```bash
├── research
│   ├── object_detection
│   │   ├── samples
│   │   │   ├── configs
│   │   │   │   ├── faster_rcnn_inception_resnet_v2_atrous_coco.config
│   │   │   │   └── checkpoint
│   │   ├── data
│   │   │   ├── question.record
│   │   │   ├── choice.record
│   │   │   ├── question_labelmap.pbtxt
│   │   │   └── choice_labelmap.pbtxt
│   ├── faster_rcnn_inception_resnet_v2_atrous_coco_2017_11_08
│   │   ├── frozen_inference_graph.pb
│   │   └── model.ckpt
│   ├── train.py
│   ├── export_inference_graph.py
│   └── generate_tfrecord.py
```

## 3. 데이터 준비

### step 1. 데이터 수집

두 종류의 시험지 데이터를 수집했습니다. 첫번째는 배경과 시험지를 분리하는 모델을 학습하기 위해 사용될 직접 촬영한 이미지, 두번째는 수능 시험지 pdf파일을 jpg 파일로 저장한 이미지들입니다.

![image](https://user-images.githubusercontent.com/49023717/146562098-3c92be17-4c25-4df0-bc94-bda31aff0402.png)

- 배경 분리를 위한 촬영 이미지 : 다양한 환경에서의 시험지로 학습하기 위해 지인들에게 부탁해 수집한 이미지들입니다.

![image](https://user-images.githubusercontent.com/49023717/146562155-fa6e6606-28f4-40f4-bf38-e5b39a1c742f.png)

- 문제 / 항목 감지를 위한 pdf 파일 이미지 : 데이터 augmentation 을 포함해 증강된 수능 시험지 데이터 총 800장을 수집했습니다.

### step 2. 데이터 라벨링

LabelImg 라벨링툴로 라벨링을 진행했습니다. 뒤에서 언급되지만 배경을 분리하기 위한 모델과 문제/항목을 감지하기 위한 모델이 각각 pytorch와 tensorflow로 학습되어 필요한 데이터가 json 과 TFRecord파일로 다릅니다.

![image](https://user-images.githubusercontent.com/49023717/146563173-700c22a7-df05-45eb-b63b-872dffc5f9c6.png)
![image](https://user-images.githubusercontent.com/49023717/146563229-5606bca3-5ffe-4432-8359-26e0b9883181.png)
- 배경을 분리하기 위한 이미지들의 라벨링과 json 파일들
> 학습을 위한 통합 json 파일을 만드는 과정은 https://hansonminlearning.tistory.com/52?category=935564 참고한 블로그 링크를 남기겠습니다.

![image](https://user-images.githubusercontent.com/49023717/146563076-38bcf3c4-8aa8-4105-bc6d-87c4f6b346d1.png)
![image](https://user-images.githubusercontent.com/49023717/146563000-808f52c6-435b-4af5-86b4-991c9e085bde.png)
- 문제와 항목을 감지하기 위한 이미지들의 라벨링과 xml 파일들

### step 3. csv 파일 통합

깃허브에 업로드된 xml to csv to tfrecord.ipynb 소스코드를 통해 xml파일을 csv파일로 만들어줍니다. 

![image](https://user-images.githubusercontent.com/49023717/146563558-8e8e4e6a-f397-4161-a3ff-6e080680a6ff.png)

각 이미지 파일에 대한 xml 파일을 csv파일로 변환해 TFRecord 생성을 위한 포맷을 맞추어 줍니다.
>(filename, width, height, class, xmin, ymin, xmax, ymax)

### step 4. TFRecord 파일 생성

이 단계에서도 깃허브에 업로드된 xml to csv to tfrecord.ipynb 소스코드를 통해 쉽게 생성할 수 있습니다. 앞에서 생성한 csv파일을 넣어 아래 명령어를 실행합니다.

`! python generate_tfrecord.py --csv_input=data/.../train_labels.csv --output_path=data/train.record`

### step 5. Label map 생성

문제 감지와 항목 감지에서는 두종류의 Label map이 필요합니다. 문제 감지를 위해서는 'question'이라는 하나의 클래스만 포함하며, 항목 감지를 위해서는 선택하지 않은 항목과 선택한 항목 두가지를 감지해야 하므로 'not-choice', 'choice' 두개의 클래스가 필요합니다.


## 4. 모델 학습

### step 1. 배경 분리 segmentation 모델

![image](https://user-images.githubusercontent.com/49023717/146564708-c421dd6d-fdad-493d-9172-b607aed5df8e.png)

오른쪽 그림처럼 시험지영역을 검출하기 위해 segmentation을 진행해야 했고, segmentation을 위한 마스크 이미지를 검출하기 위해 mask rcnn모델을 사용했습니다. 

![image](https://user-images.githubusercontent.com/49023717/146564802-e28629ba-caa8-4083-b0f8-6d621a99485e.png)

Pytorch의 Detectron2 라이브러리를 사용해 학습을 진행했습니다. 생성한 json 파일과 이미지들을 미리학습된 모델의 객체로 추가하고 학습을 진행합니다.

### step 2. 문제 감지 모델

![image](https://user-images.githubusercontent.com/49023717/146568227-70840b44-da74-481a-b93f-fbd78d9cde24.png)

문제 / 항목 감지 모델 모두 Faster RCNN모델을 사용했습니다. Faster rcnn 은 convolutional feature map이 Region Proposal단계 에서도 쓰이도록 하여 속도 측면에서 성능을 향상시킨 모델입니다. Tensorflow Object Detection API에서 제공되는 모델 중 속도는 느리지만 가장 성능이 좋은 모델을 선택했습니다.

앞에서 생성한 데이터 파일들을 위 디렉토리 구성으로 셋팅하면 학습을 진행하게 됩니다. 디렉토리에는 미리 학습된 Faster rcnn inception resnet v2 atrous coco 모델의 체크포인트와 추론그래프가 포함됩니다. 미리 학습된 모델은 [model download](https://github.com/tensorflow/models/blob/r1.12.0/research/object_detection/g3doc/detection_model_zoo.md)에서 가능합니다. 또한 API에서 제공되는 [faster_rcnn_inception_resnet_v2_atrous_coco.config](https://github.com/Nayeon12/Scoring-Paper-Tests-with-RCNN/tree/main/model)파일을 수정해주어야 합니다. 

---
> 10 num_classes: 1<br/>
> 108 fine_tune_checkpoint: "faster_rcnn_inception_resnet_v2_atrous_coco_2017_11_08/model.ckpt"<br/>
> 123 input_path: "object_detection/data/question_train.record"<br/>
> 125 label_map_path: "object_detection/data/question_labelmap.pbtxt"<br/>
> 137 input_path: "object_detection/data/question_val.record"<br/>
> 139 label_map_path: "object_detection/data/question_labelmap.pbtxt"
---

수정해주었으면 제공되는 train.py 파일을 통해 학습을 진행합니다.

`! python train.py --train_dir=training/ --pipeline_config_path=.../samples/faster_rcnn_inception_resnet_v2_atrous_coco.config`

![ezgif com-gif-maker](https://user-images.githubusercontent.com/49023717/144734682-5317a371-c87e-4935-86e8-e9d1da6980c6.gif)

loss값이 0.0~0.2에 계속 머물러 학습을 종료 했고 1328 step을 진행했습니다.

### step 3. 항목 감지 모델

항목감지 모델도 문제 감지 모델을 학습하는 과정과 동일합니다. 여기서는 클래스 두개를 감지해야 하므로 config 파일의 num_classes: 2 로 수정해주면 됩니다.

![그림1](https://user-images.githubusercontent.com/49023717/146566345-791bdb44-b099-4c88-920a-e54da364843a.gif)

1097 step 진행했으며, loss 값이 0.2~0.3에 계속 머물러 종료했습니다.

## 5. 모델 테스트

모델 테스트를 위해서는 다음과 같은 과정이 필요합니다.

1. 추론 그래프 추출
2. 추론 그래프를 사용하여 객체 검출

추론그래프를 추출하기 위해서는 제공되는 export_inference_graph.py을 사용하면 됩니다. 

`! python export_inference_graph.py
--input_type image_tensor
--pipeline_config_path training/faster_rcnn_inception_resnet_v2_atrous_coco.config
--trained_checkpoint_prefix samples/model.ckpt-xxxxx
--output_directory inference_graph`

학습 진행 후 실행폴더에는 체크포인트가 저장되는데, 그때 저장된 스텝을 xxxx에 작성해 실행하면 됩니다.

추론그래프를 생성하였다면 test set으로 추론 결과를 시각화 하게 됩니다.
자세한 코드는 [Inference](https://github.com/Nayeon12/Scoring-Paper-Tests-with-RCNN/blob/main/code/Inference_Question_Detection.ipynb)를 참고하면 됩니다.

![image](https://user-images.githubusercontent.com/49023717/146567410-6bb6a01f-0f87-436b-af11-2bb7337604a6.png)
- 문제 감지후 문제별로 잘려 저장된 모습

![image](https://user-images.githubusercontent.com/49023717/146567231-5c6efc2b-b402-4041-b8c0-866bedf8d868.png)
- 선택한 항목과 선택하지 않은 항목이 bounding box로 감지되는 모습


## 6. 추가자료

시연 동영상과 좀 더 자세한 내용을 담은 pdf파일은 아래 첨부합니다.

[종이시험지 자동채점 프로그램 시연동영상](https://youtu.be/nVRXpLfXRB0)<br/>
[종이시험지 자동채점 프로그램.pdf](https://github.com/Nayeon12/Scoring-Paper-Tests-with-RCNN/files/7736076/_.pdf)<br/>
[종이시험지 자동채점 프로그램 구현 블로그정리](https://velog.io/@nayeon_p00/series/2021CapstoneDesign)



