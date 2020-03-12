import torchvision.transforms as T
import torchvision.transforms.functional as F
from PIL import Image
import numpy as np
import cv2
import time
import os


class MyAugmentation():
    '''
        frame:a PIL img
        label:a numpy array
    '''

    def __init__(self):
        pass

    def pad2square(self,frame,labels,pad_value=0):
        '''

        :param frame:
        :param labels:
        :param pad_value:
        :return:
        '''
        '''
            images
        '''
        w,h = frame.size
        dim_diff = np.abs(w-h)
        pad1,pad2 = dim_diff//2,dim_diff-dim_diff//2
        #(left, top, right and bottom)
        # 这个pad是torchvision里面的，不是torch.nn.module.functional里面的
        # 两个pad的顺序不一样
        pad = (0, pad1,0, pad2) if h <= w else (pad1, 0,pad2, 0)
        new_img = F.pad(frame,pad,fill=pad_value)

        '''
            labels
        '''
        # get [cla, leftup_x, leftup_y, rightbot_x, rightbot_y ]
        parsed_labels = self.parse_label(frame,labels)
        if h<=w:
            parsed_labels[:,[2,4]]=parsed_labels[:,[2,4]]+pad1
        elif h>w:
            parsed_labels[:,[1,3]]=parsed_labels[:,[1,3]]+pad1

        new_labels = self.de_parse_label(new_img,parsed_labels)

        return new_img,new_labels

    def resize(self,frame,labels,size=1000):
        '''

        :param frame:
        :param labels:
        :param size:
        :return:
        '''
        new_frame = F.resize(frame,size)
        return new_frame,labels

    def horizontalFlip(self,frame,labels):
        '''
        水平翻转
        :param frame:
        :param labels:
        :return:
        '''
        new_img = F.hflip(frame)
        new_labels = labels.copy()
        new_labels[:,1] = 1-new_labels[:,1]
        return new_img,new_labels

    def verticalFlip(self,frame,labels):
        '''
        垂直翻转
        :param frame:
        :param labels:
        :return:
        '''
        new_img = F.vflip(frame)
        new_labels = labels.copy()
        new_labels[:, 2] = 1 - new_labels[:, 2]
        return new_img, new_labels

    def randomCrop(self, frame, labels,size=1000):

        '''
        :param frame: a PIL img
        :param labels: a numpy array,(-1,5) [cla,cx,cy,dw,dh]
        :param size:
        :return: a pIL img,a numpy array
        '''

        '''
            images
        '''
        # topy,leftx,height,width
        i, j, h, w = T.RandomCrop.get_params(frame,(size,size))
        new_img = F.crop(frame, i, j, h, w)
        print(new_img.size)


        '''
            labels
        '''
        parsed_label = self.parse_label(frame,labels)
        chosen_box = []
        for idx,bbox in enumerate(parsed_label):
            _,leftup_x, leftup_y, rightbot_x, rightbot_y = bbox
            if leftup_x>=j and leftup_y>=i and rightbot_x<=j+w and rightbot_y<=i+h:
                chosen_box.append(idx)

        chosen_labels = parsed_label[chosen_box]
        chosen_labels[:,1] = chosen_labels[:,1]-j
        chosen_labels[:,3] = chosen_labels[:,3]-j
        chosen_labels[:,2] = chosen_labels[:,2]-i
        chosen_labels[:,4] = chosen_labels[:,4]-i

        '''
            从[cla, leftup_x, leftup_y, rightbot_x, rightbot_y ]变成[cla, cx, cy, bw, bh]
        '''
        de_parsed_labes = self.de_parse_label(new_img,chosen_labels)

        return new_img,de_parsed_labes

    def de_parse_label(self,frame,labels):
        '''
        将[cla, leftup_x, leftup_y, rightbot_x, rightbot_y ]变成
        [cla, cx, cy, bw, bh]
        :param frame:
        :param labels:
        :return:
        '''
        w,h = frame.size
        de_parsed_labels = np.zeros_like(labels, dtype=np.float32)
        for i, bbox in enumerate(labels):
            cla, leftup_x, leftup_y, rightbot_x, rightbot_y = bbox
            cx = (leftup_x+rightbot_x)/(2*w)
            cy = (leftup_y+rightbot_y)/(2*h)
            dw = (rightbot_x-leftup_x)/w
            dh = (rightbot_y-leftup_y)/h

            de_parsed_labels[i, :] = [cla, cx,cy,dw,dh]

        return de_parsed_labels

    def parse_label(self,frame,labels):
        '''
        将[cla, cx, cy, bw, bh]变成
        [cla, leftup_x, leftup_y, rightbot_x, rightbot_y]
        :param frame:
        :param labels:
        :return:
        '''
        w,h = frame.size
        parsed_labels = np.zeros_like(labels,dtype=np.int32)
        for i,bbox in enumerate(labels):
            cla, cx, cy, bw, bh = bbox
            r_cx, r_bw = w * cx, w * bw
            r_cy, r_bh = h * cy, h * bh

            leftup_x = int(r_cx - r_bw / 2)
            leftup_y = int(r_cy - r_bh / 2)
            rightbot_x = int(r_cx + r_bw / 2)
            rightbot_y = int(r_cy + r_bh / 2)
            parsed_labels[i,:]=[cla,leftup_x,leftup_y,rightbot_x,rightbot_y]

        return parsed_labels

    def show(self, frame, labels,need_parse=True):
        '''

        :param frame:
        :param labels:
        :param need_parse: True表示label是[cla, cx, cy, bw, bh]这个格式；
                           False表示[cla, leftup_x, leftup_y, rightbot_x, rightbot_y]
        :return:
        '''

        if need_parse:
            parsed_labels = self.parse_label(frame,labels)
        else:
            print('here')
            parsed_labels = labels
        img = cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR)

        for bbox in parsed_labels:
            cla, leftup_x, leftup_y, rightbot_x, rightbot_y = bbox

            cv2.rectangle(img, (leftup_x, leftup_y), (rightbot_x, rightbot_y), (0, 0, 255),2)

        cv2.namedWindow('img',cv2.WINDOW_NORMAL)

        while True:
            cv2.imshow('img', img)
            if cv2.waitKey(1) == ord('q'):
                break

        cv2.destroyAllWindows()

    def save(self,image,labels,path,name=None):
        if name is None:
            name = str(int(time.time()*1000))
        pic_name = os.path.join(path,'{}.jpg'.format(name))
        image.save(pic_name)

        with open(os.path.join(path,'{}.txt'.format(name)),'w') as f:
            for line in labels:
                temp_line = [str(a) for a in line]
                f.write(' '.join(temp_line)+'\n')

def main():
    '''
    测试
    :return:
    '''
    path = '../output/'
    ma = MyAugmentation()
    image = Image.open('../data/detection/rebar/8ADCAE58.jpg')
    labels = np.loadtxt('../data/detection/rebar/8ADCAE58.txt').reshape(-1, 5)

    new_img, new_labels = ma.pad2square(image, labels)
    ma.save(new_img,new_labels,path)

    ma.show(new_img, new_labels)


if __name__ == '__main__':
    main()




