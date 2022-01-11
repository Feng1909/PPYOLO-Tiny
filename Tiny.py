# encoding: utf-8

from paddlelite.lite import *
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from time import time
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
from PIL import ImageEnhance


class PPYOLO_Detector(object):

    def __init__(self, nb_path=None,  # nb路径
                 label_list=None,  # 类别list
                 input_size=[320, 320],  # 输入图像大小
                 img_means=[0., 0., 0.],  # 图片归一化均值
                 img_stds=[0., 0., 0.],  # 图片归一化方差
                 threshold=0.1,  # 预测阈值
                 num_thread=1,  # ARM CPU工作线程数
                 ):

        # 验证必要的参数格式
        assert nb_path is not None, \
            "Please make sure the model_nb_path has inputed!(now, nb_path is None.)"
        assert len(input_size) == 2, \
            "Please make sure the input_shape length is 2, but now its length is {0}".format(len(input_size))
        assert len(img_means) == 3, \
            "Please make sure the image_means shape is [3], but now get image_means' shape is [{0}]".format(
                len(img_means))
        assert len(img_stds) == 3, \
            "Please make sure the image_stds shape is [3], but now get image_stds' shape is [{0}]".format(len(img_stds))
        assert len([i for i in img_stds if i <= 0]) < 1, \
            "Please make sure the image_stds data is more than 0., but now get image_stds' data exists less than or equal 0."
        assert threshold > 0. and threshold < 1., \
            "Please make sure the threshold value > 0. and < 1., but now get its value is {0}".format(threshold)
        assert num_thread > 0 and num_thread <= 4, \
            "Please make sure the num_thread value > 1 and <= 4., but now get its value is {0}".format(num_thread)

        # 模型nb文件路径
        self.model_path = nb_path
        # ARM CPU工作线程数
        self.num_thread = num_thread

        # 预测显示阈值
        self.threshold = threshold
        # 预测输入图像大小
        self.input_size = input_size
        # 图片归一化参数
        # 均值
        self.img_means = img_means
        # 方差
        self.img_stds = img_stds

        # 预测类别list
        self.label_list = label_list
        # 预测类别数
        self.num_class = len(label_list) if (label_list is not None) and isinstance(label_list, list) else 1
        # 类别框颜色map
        self.box_color_map = self.random_colormap()

        # 记录模型加载参数的开始时间
        self.prepare_time = self.runtime()

        # 配置预测
        self.config = MobileConfig()
        # 设置模型路径
        self.config.set_model_from_file(nb_path)
        # 设置线程数
        self.config.set_threads(num_thread)
        # 构建预测器
        self.predictor = create_paddle_predictor(self.config)

        # 模型加载参数的总时间花销
        self.prepare_time = self.runtime() - self.prepare_time
        print("The Prepare Model Has Cost: {0:.4f} s".format(self.prepare_time))

    def get_input_img(self, input_img):
        '''输入预测图片
            input_img: 图片路径或者np.ndarray图像数据 - [h, w, c]
        '''
        assert isinstance(input_img, str) or isinstance(input_img, np.ndarray), \
            "Please enter input is Image Path or numpy.ndarray, but get ({0}) ".format(input_img)

        # 装载图像到预测器上的开始时间
        self.load_img_time = self.runtime()

        if isinstance(input_img, str):
            # 读取图片路径下的图像数据
            self.input_img = Image.open(input_img)
        elif isinstance(input_img, np.ndarray):
            # 读取ndarray数据下的图像数据
            self.input_img = Image.fromarray(input_img)

        # 获取图片原始高宽 ： h，w
        self.input_shape = np.asarray(self.input_img).shape[:-1]
        # 重置图片大小为指定的输入大小
        input_data = self.input_img.resize(self.input_size, Image.BILINEAR)
        # 转制图像shape为预测指定shape
        input_data = np.array(input_data).transpose(2, 0, 1).reshape([1, 3] + self.input_size).astype('float32')
        # 将图像数据进行归一化
        input_data = self.normlize(input_data)

        self.scale_factor = [1., 1.]  # [1., 1.]

        # 配置输入tensor

        # 输入[[shape, shape]]的图片大小
        self.input_tensor0 = self.predictor.get_input(0)
        self.input_tensor0.from_numpy(np.asarray([self.input_size], dtype=np.int32))

        # 输入[1, 3, shape, shape]的归一化后的图片数据
        self.input_tensor1 = self.predictor.get_input(1)
        self.input_tensor1.from_numpy(input_data)

        # 输入模型处理图像大小与实际图像大小的比例
        self.input_tensor2 = self.predictor.get_input(2)
        self.input_tensor2.from_numpy(np.asarray(self.scale_factor, dtype=np.int32))

        # 装载图像到预测器上的总时间花销
        self.load_img_time = self.runtime() - self.load_img_time
        # print("The Load Image Has Cost: {0:.4f} s".format(self.load_img_time))

    def get_output_img(self):
        '''获取输出标注图片
            num_bbox: 最大标注个数
        '''

        # 预测器开始预测的时间
        self.predict_time = self.runtime()

        # 根据get_input_img的图像进行预测
        self.predictor.run()
        # 获取输出预测bbox结果
        self.output_tensor = self.predictor.get_output(0)

        # 转化为numpy格式
        output_bboxes = self.output_tensor.numpy()
        # 根据阈值进行筛选，大于等于阈值的保留
        output_bboxes = output_bboxes[output_bboxes[:, 1] >= self.threshold]

        # 根据预测结果进行框绘制，返回绘制完成的图片
        # self.output_img = self.load_bbox(output_bboxes, num_bbox)

        # 预测器预测的总时间花销
        self.predict_time = self.runtime() - self.predict_time
        print("The Predict Image Has Cost: {0:.4f} s".format(self.predict_time))

        # return self.output_img
        return output_bboxes

    def normlize(self, input_img):
        '''数据归一化
            input_img: 图像数据--numpy.ndarray
        '''
        # 对RGB通道进行均值-方差的归一化
        input_img[0, 0] = (input_img[0, 0] / 255. - self.img_means[0]) / self.img_stds[0]
        input_img[0, 1] = (input_img[0, 1] / 255. - self.img_means[1]) / self.img_stds[1]
        input_img[0, 2] = (input_img[0, 2] / 255. - self.img_means[2]) / self.img_stds[2]

        return input_img

    def load_bbox(self, input_bboxs, num_bbox):
        '''根据预测框在原始图片上绘制框体，并标注
            input_bboxs: 预测框
            num_bbox: 允许的标注个数
        '''
        # 创建间绘图参数:[cls_id, score, x1, y1, x2, y2]
        self.draw_bboxs = [0] * 6
        # 绘图器 -- 根据get_input_img的输入图像
        draw = ImageDraw.Draw(self.input_img)
        # 根据最大标注个数进行实际标注个数的确定
        # input_bboxs.shape[0]： 表示预测到的有效框个数
        if len(input_bboxs) != 0:  # 存在有效框时
            num_bbox = input_bboxs.shape[0] if num_bbox > input_bboxs.shape[0] else num_bbox
        else:
            num_bbox = 0  # 没有有效框，直接不标注

        # 遍历框体，并进行标注
        for i in range(num_bbox):
            # 类别信息
            self.draw_bboxs[0] = input_bboxs[i][0]
            # 类别得分
            self.draw_bboxs[1] = input_bboxs[i][1]

            print(self.label_list[int(self.draw_bboxs[0])], '- score{', self.draw_bboxs[1], "} : ", input_bboxs[i][2],
                  input_bboxs[i][3], input_bboxs[i][4], input_bboxs[i][5])

            # 框体左上角坐标
            # max(min(input_bboxs[i][2] / self.input_size[0], 1.), 0.)：保证当前预测坐标始终在图像内(比例,0.-1.)
            # max(min(input_bboxs[i][2] / self.input_size[0], 1.), 0.) * self.input_shape[1]: 直接预测得到的坐标
            # min(max(min(input_bboxs[i][2] / self.input_size[0], 1.), 0.) * self.input_shape[1], self.input_shape[1])：保证坐标在图像内(h, w)
            self.draw_bboxs[2] = min(max(min(input_bboxs[i][2] / self.input_size[0], 1.), 0.) * self.input_shape[1],
                                     self.input_shape[1])
            self.draw_bboxs[3] = min(max(min(input_bboxs[i][3] / self.input_size[1], 1.), 0.) * self.input_shape[0],
                                     self.input_shape[0])
            # 框体右下角坐标
            self.draw_bboxs[4] = min(max(min(input_bboxs[i][4] / self.input_size[0], 1.), 0.) * self.input_shape[1],
                                     self.input_shape[1])
            self.draw_bboxs[5] = min(max(min(input_bboxs[i][5] / self.input_size[1], 1.), 0.) * self.input_shape[0],
                                     self.input_shape[0])

            # print(self.draw_bboxs[2], self.draw_bboxs[3], self.draw_bboxs[4], self.draw_bboxs[5])

            # 绘制框体
            # self.box_color_map[int(self.draw_bboxs[i][0])]: 对应类别的框颜色
            draw.rectangle(((self.draw_bboxs[2], self.draw_bboxs[3]),
                            (self.draw_bboxs[4], self.draw_bboxs[5])),
                           outline=tuple(self.box_color_map[int(self.draw_bboxs[0])]),
                           width=2)
            # 框体位置写上类别和得分信息
            draw.text((self.draw_bboxs[2], self.draw_bboxs[3] + 1),
                      "{0}:{1:.4f}".format(self.label_list[int(self.draw_bboxs[0])], self.draw_bboxs[1]),
                      tuple(self.box_color_map[int(self.draw_bboxs[0])]))

        # 返回标注好的图像数据
        return np.asarray(self.input_img)

    def random_colormap(self):
        '''获取与类别数量等量的color_map
        '''
        np.random.seed(2021)

        color_map = [[np.random.randint(20, 255),
                      np.random.randint(64, 200),
                      np.random.randint(128, 255)]
                     for i in range(self.num_class)]

        return color_map

    def runtime(self):
        '''返回当前计时
        '''
        return time()
