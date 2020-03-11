import motmetrics as mm
import numpy as np


def main(o_path,h_path):
    # 创建accumulator
    acc = mm.MOTAccumulator(auto_id=True)

    # 加载数据
    o_gt = np.loadtxt(o_path,delimiter=',')
    h_gt = np.loadtxt(h_path,delimiter=',')

    frame_max = o_gt[-1,0]
    frame_cnt = 1
    while frame_cnt!=frame_max:
        # 1、拿到当前frame的数据
        temp_o = o_gt[o_gt[:, 0] == frame_cnt]
        temp_h = h_gt[h_gt[:, 0] == frame_cnt]

        # 2、拿到gt的id和预测的id
        temp_o_id = temp_o[:, 1]
        temp_h_id = temp_h[:, 1]

        # 3、拿到当前frame对应的gt和预测框
        temp_o_rec = temp_o[:, [2, 3, 4, 5]]
        temp_h_rec = temp_h[:, [2, 3, 4, 5]]

        # 计算得到gt和预测框的距离
        temp_dis = mm.distances.iou_matrix(temp_o_rec, temp_h_rec, max_iou=0.5)
        # print('', temp_dis)

        # 4、更新accumulator
        acc.update(temp_o_id, temp_h_id, temp_dis)

        # 5、下一帧
        frame_cnt+=1
        print('frame:',frame_cnt)

    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=['num_frames', 'mota', 'motp'], name='acc')
    print(summary)




if __name__ == '__main__':
    path = '../../data/track/PETS09-S2L1/SORT_RES.txt'
    main(path,path)
