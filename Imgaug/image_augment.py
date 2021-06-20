import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from imgaug import augmenters as iaa
import imgaug as ia


'''yolo bbox에서 min max bbox로 변환하고 imgaug 작업 후, 다시 yolo로 변환하는 코드입니다.'''

def img_wid_height(img_path, img_name):
    """ height, width , channel = image.shape """
    # cv2.IMREAD_COLOR는 Color 읽어옴
    # cv2.imread 는 RGB가 아닌 BGR로 읽어온다.
    image = cv2.cvtColor(cv2.imread(os.path.join(img_path, img_name)), cv2.COLOR_BGR2RGB)
    img_height = image.shape[0]
    img_width = image.shape[1]

    return image, img_height, img_width

def bbox_annotation(file_path, file_name):
    """ text 파일 안의 bbox 값을 하나씩 불러오기 """
    with open(os.path.join(file_path, file_name), 'r') as f:
        bbox = []
        text = f.readlines()
        for txt in text:
            new_txt = txt.strip()
            # 리스트 안의 문자열 형태를 정수 형태로 변환
            new_txt = list(new_txt.split(' '))   # 리스트 안에 문자열 형태
            new_txt = list(map(float, new_txt))  # 리스트 안에 정수 형태
            bbox.append(new_txt)

    return bbox

def denorm_box(width, height, x, y, w, h):
    """ width, height는 image의 값  / yolo에서 min, max로 변환 """
    xmax = int((x*width) + (w * width)/2.0)
    xmin = int((x*width) - (w * width)/2.0)
    ymax = int((y*height) + (h * height)/2.0)
    ymin = int((y*height) - (h * height)/2.0)

    return xmin, xmax, ymin, ymax

def convert(size, box, label):
    """ xmin ymin xmax ymax -> yolo format """
    dw = 1./size[0]
    dh = 1./size[1]

    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]

    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh

    return x, y, w, h, label

# color 설정
blue = (255, 0, 0)
green = (0, 255, 0)
red = (0, 0, 255)
white = (255, 255, 255)

# 경로 설정
img_path = r''     # 이미지 폴더 경로
txt_path = r''           # 라벨 폴더 경로
write_path = r''         # 최종 저장 폴더 경로

for txt_file, img_file in zip(os.listdir(txt_path), os.listdir(img_path)):
    if not img_file.endswith('ini'):                                              # desktop.ini 파일이 계속 잡히는 문제 발생
        image, img_height, img_width = img_wid_height(img_path, img_file)         # image 불러오기
        bbox = bbox_annotation(txt_path, txt_file)                                # bounding box 불러오기

        # 파일 안의 bounding box 개수만큼 생성되서 추가가 필요합니다.
        ia_bounding_boxes = []
        for bboxx in bbox:
            xmin, xmax, ymin, ymax = denorm_box(img_width, img_height, bboxx[1], bboxx[2], bboxx[3],bboxx[4])  # denormalized bouding box 함수
            ia_bounding_boxes.append(ia.BoundingBox(x1=xmin, y1=ymin, x2=xmax, y2=ymax, label='person'))
        bbs = ia.BoundingBoxesOnImage(ia_bounding_boxes, shape=image.shape)

        '''단일 증강 코드'''
        # seq = iaa.AllChannelsCLAHE(clip_limit=6)                          # 명암 선명하게(seq)
        # seq = iaa.SigmoidContrast(gain=10, cutoff=0.25)                   # 이미지 밝게(sig)
        # seq = iaa.MotionBlur(k=7)                                         # 이미지 뒤틀림(mot)    # 영상 흔들림 대비
        # seq = iaa.ChannelShuffle(0.35)                                    # 색상 변화(cha)       # 여러 색상 대비
        # seq = iaa.Fliplr(1)                                               # 좌우 반전(hor)
        # seq = iaa.Flipud(1)                                               # 상하 반전(ver)
        seq = iaa.Affine(scale=(0.8, 0.8))                                # 이미지 줄이기(scale)
        # seq = iaa.Sharpen(alpha=0.7, lightness=1.0)                       # 테두리가 조금 생긴다(sharp)
        # seq = iaa.pillike.FilterFindEdges()                               # 테두리 강화
        # seq = iaa.pillike.FilterEdgeEnhanceMore()                         # 테두리 강화 어둡게 변함
        # seq = iaa.Rotate((30))                                            # roatet는 bounding box 위치가 조금 이상한 것 같다.

        '''한 번에 여러 증강 작업을 원할 경우'''
        # seq = iaa.Sequential([
        #    iaa.Affine(scale=(0.5, 0.5)),
        #    iaa.SigmoidContrast(gain=10, cutoff=0.25)
        #    등등~ 원하는 증강 코드 추가하기.
        # ])

        '''기존 이미지, 바운딩 박스 값 변경'''
        img_aug = seq.augment_image(image)                   # 변경된 이미지
        bbs_aug = seq.augment_bounding_boxes(bbs)            # 변경된 bounding box

        '''BGR -> RGB 변경 필요(왜?? 위에서 변경을 했는뎁...흠..)'''
        cv2.imwrite(os.path.join(write_path, 'scale0.8'+img_file), cv2.cvtColor(img_aug, cv2.COLOR_BGR2RGB))

        '''min, ymin, xmax, ymax -> centerX, centerY , width, height(yolo) 변경'''
        with open(os.path.join(write_path, 'scale0.8'+txt_file), 'a') as f:           # 텍스트 파일 이름 변경 필요!!
            for idx in range(len(bbs_aug.bounding_boxes)):
                bb_box = bbs_aug.bounding_boxes[idx]
                x, y, w, h, label = convert((img_width, img_height), (int(bb_box.x1), int(bb_box.x2), int(bb_box.y1), int(bb_box.y2)), 0)  # 순서 정확하게!!!
                f.write('{} {} {} {} {} \n'.format(0, x, y, w, h))


#-----참고 Note-----
# 바운딩 박스 값 확인하기
# 1. using CV2 라이브러리
# img_bbox = cv2.rectangle(image, (ia_bounding_boxes[0][0], ia_bounding_boxes[0][1]), (ia_bounding_boxes[0][2], ia_bounding_boxes[0][3]), red, 3)
# plt.imshow(image)
# plt.show()

# 2. using imgaug 라이브러리
# img_after = bbs_aug.draw_on_image(img_aug, size=5, color= blue)
# plt.imshow(img_after)
# plt.show()





