# object_detection_study

## yolo v3 구현(model, train, test)
> 작성중
- pytorch 이용
### 1. yolo v3 모델 구현
- yolo_model.py 부분
- yolo v3 architecture
<img src="https://user-images.githubusercontent.com/54797864/106993063-3aeca800-67bd-11eb-92ae-4a90d60b26c0.png"  width="800" height="370">
<img src="https://user-images.githubusercontent.com/54797864/106993079-463fd380-67bd-11eb-8147-43c229cbcf56.png"  width="500" height="530">

[참조1](http://datahacker.rs/tensorflow2-0-yolov3/)
[참조2](https://www.programmersought.com/article/2967815530/)

### 2. data 불러오기
#### dataset 사용 방법
- PASCAL VOC 2007 data 사용
1. [데이터 다운 주소](http://host.robots.ox.ac.uk/pascal/VOC/)에 들어가서 The VOC2007 Challenge의 train, val, test data 다운로드
2. voc_label.py 실행
#### data 관련 파일 설명
- data/voc/voc_label.py : voc data 라벨링 작업하는 파일
- config/voc.data : voc data의 class 정보, 경로 저장된 파일
- utils/datasets.py : data 전처리, dataset 불러오는 함수 정의된 파일

## object detection 성능
https://blueskyvision.tistory.com/465
