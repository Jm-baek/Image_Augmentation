imgaug는 이미지 증강 관련된 라이브러리에서 매우 유명하다~!<br/>
나의 상황에 맞게 사용할 수 있도록 코드를 작성해 보았다.

## 참고 사이트
1. https://github.com/aleju/imgaug
2. https://imgaug.readthedocs.io/en/latest/

## 코드 설명
이미지 파일 폴더와 라벨 파일 폴더 각각 존재해야합니다.  
이 부분이 불편하면 코드 수정해서 같은 파일에서 작업이 되도록해도 괜찮습니다.

## 수정 부분
  + 마지막 text 파일 저장하는 부분에서 조금 더 깔끔한? 저장이 필요합니다.(텍스트 파일 코드 작성이 아직 미숙합니다.)
  + cv2를 사용하여 이미지를 불러오는 과정에서 RGB로 변경했지만 BGR로 바뀌는 현상 문제점.(변경을 한 번 더 해줬다.)

## 주의사항
  + 본 코드는 파일 안의 이미지 한 장, 텍스트 파일 한 장씩 변형하고 있습니다..
  + 따라서,augment_image를 사용해야합니다.(augment_images 사용했다가.. 몇 일 고생했습니다.)
  + 본 코드 작성시, **yolo format** 형태로 bounding box 값을 가지고 있었습니다.  
    yolo format -> voc format(xmin, ymin, xmax, ymax) -> yolo format


**ps**
코딩 실력이 오르면 더 깔끔하게 작성한 코드로 수정하겠습니다.
