# encoding: utf-8

import cv2
from Tiny import PPYOLO_Detector

if __name__ == '__main__':
    if True:
        model_path = "./ppyolotiny.nb"  # 模型参数
        img_path = "./train.jpg"  # 自己的预测图像
        label_list = ['car', 'bus', 'person', 'bike', 'truck', 'motor', 'train', 'rider', 'traffic sign',
                      'traffic light']
        input_size = [224, 224]  # 输入图像大小
        img_means = [0.485, 0.456, 0.406]  # 图片归一化均值
        img_stds = [0.229, 0.224, 0.225]  # 图片归一化方差
        threshold = 0.5  # 预测阈值
        num_thread = 2  # ARM CPU工作线程数
        # work_mode = PowerMode.LITE_POWER_NO_BIND  # ARM CPU工作模式
        max_bbox_num = 10  # 每帧最多标注数

        detector = PPYOLO_Detector(
            nb_path=model_path,
            label_list=label_list,
            input_size=input_size,
            img_means=img_means,
            img_stds=img_stds,
            threshold=threshold,
            num_thread=num_thread,
        )

        # img = plt.imread(img_path)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (256, 256))
        detector.get_input_img(img)
        boxes = detector.get_output_img()  # 框
        for i in boxes:
            label = i[0]
            # 类别得分
            score = i[1]
            print(label_list[int(label)], " score: ", score, "label: ", i[2], i[3], i[4], i[5])
    else:
        print("PPYOLO-Tiny loading...")
        model_path = "./ppyolotiny.nb"  # 模型参数
        label_list = ['car', 'bus', 'person', 'bike', 'truck', 'motor', 'train', 'rider', 'traffic sign',
                      'traffic light']
        input_size = [224, 224]  # 输入图像大小
        img_means = [0.485, 0.456, 0.406]  # 图片归一化均值
        img_stds = [0.229, 0.224, 0.225]  # 图片归一化方差
        threshold = 0.5  # 预测阈值
        num_thread = 2  # ARM CPU工作线程数
        # work_mode = PowerMode.LITE_POWER_NO_BIND  # ARM CPU工作模式
        max_bbox_num = 10  # 每帧最多标注数
        detector = PPYOLO_Detector(
            nb_path=model_path,
            label_list=label_list,
            input_size=input_size,
            img_means=img_means,
            img_stds=img_stds,
            threshold=threshold,
            num_thread=num_thread,
        )
        capture = cv2.VideoCapture(1)
        while True:
            ret, frame = capture.read()
            # cv2.imshow("capture", frame)

            img = cv.resize(frame, (256, 256))
            detector.get_input_img(img)
            boxes = detector.get_output_img()  # 框
            # print(boxes)
            for i in boxes:
                label = i[0]
                # 类别得分
                score = i[1]
                print(label_list[int(label)], " score: ", score, "label: ", i[2], i[3], i[4], i[5])
