import numpy as np
import matplotlib.pyplot as plt

"""

Computes mean Average Precision to evaluate object detection

Inputs
conf: nested list of

computations in evaluate:
-iou

Current version uses the following method:
- takes the average across all classes
- uses monotomical decreasing AP curve

"""

def single_mAP(conf, hits):

    # convert to np arrays
    conf = np.array(conf)
    hits = np.array(hits)

    # sort the data by decreasing confidence
    # confidence should be NN output if detection was greater than IOU threshold else 0
    # hits should be 1 if TP or 0 if TN (we only consider TP and TN for a given category)
    idxs = np.argsort(conf)
    hits = hits[idxs]
    hits = np.flip(hits) # reverse the order

    # compute precision and recall going down the decreasing confidence order to get an AP curve
    precision = np.zeros(hits.size)
    recall = np.zeros(hits.size)
    running_tp = 0
    for i, h in enumerate(hits):
        if h == 1: running_tp += 1
        precision[i] = running_tp / (i+1)
        recall[i] = running_tp / hits.size

    # convert AP curve to a monotonically decreasing curve and find points where precision decreases wrt recall (js)
    r_flip = np.flip(recall)
    p_flip = np.flip(precision)
    js = [0] # include the last point
    m = p_flip[0] # current max value
    for j in range(len(p_flip)):
        p = p_flip[j] # precision value at a discrete point
        if p < m:
            p_flip[j] = m # set precission to max value to make this flipped array monotonically increasing
        elif p > m:
            m = p # reassign max value
            js.append(j) # find the point where a new rectangle is formed

    if len(p_flip)-1 not in js:
        js.append(len(p_flip)-1)# append ending index if not already included

    # if we have more than one data point
    if len(js) > 2:
        # numerically integate curve using a Riemann sum to get AUC
        AUC = r_flip[-1]*p_flip[-1] # first rectangle against the x and y axes of the AP curve
        for i in range(len(js)-1):
            x = r_flip[js[i]] - r_flip[js[i+1]]
            y = p_flip[js[i]]
            AUC += x*y

        # plt.scatter(recall, precision)
        # plt.xlim(-0.1,1.1)
        # plt.ylim(-0.1,1.1)
        # plt.show()
        # plt.scatter(recall, np.flip(p_flip))
        # plt.xlim(-0.1,1.1)
        # plt.ylim(-0.1,1.1)
        # plt.show()
        # print(recall, precision)
        # print(recall, np.flip(p_flip))
        # print(r_flip, p_flip)
        # print(js)

    # else AUC is either 0 or 1 for a single data point
    else:
        AUC = recall[0]*precision[0]

    return AUC

# compute mAP per class and take average
def mAP(class_confs, class_hits):
    class_mAPs = []
    for conf, hits in list(zip(class_confs, class_hits)):
        if len(conf) == 0: continue # skip a class if it is not present
        class_mAP = single_mAP(conf, hits)
        class_mAPs.append(class_mAP)
    return np.mean(class_mAPs), class_mAPs

# code inspired from https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
def iou(box1, box2):
    # x,y coords of the intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # area of the intersection
    inter_area = max(0, x2-x1+1) * max(0, y2-y1+1)

    # area of the union
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area

def nms(boxes, conf_thresh=0.1, iou_thresh=0.6):
    # a single box should be [label, conf, xmin, ymin, xmax, ymax]

    # filter by confidence
    cs = boxes[:, 1]
    idxs = np.where(cs > conf_thresh)[0]
    cs = cs[idxs]
    boxes = boxes[idxs, :]

    if len(cs) <= 1:
        return boxes

    # filter by IOU
    keep = []
    while len(cs) != 0:
        idx = np.argmax(cs)
        keep.append(boxes[idx, :])
        best_box = boxes[idx, -4:]
        cs = np.delete(cs, [idx])
        boxes = np.delete(boxes, [idx], axis=0)
        idxs = []
        for i in range(boxes.shape[0]):
            cmp_box = boxes[i, -4:]
            IOU = iou(best_box, cmp_box)
            if IOU > iou_thresh:
                idxs.append(i)
        cs = np.delete(cs, [idxs])
        boxes = np.delete(boxes, [idxs], axis=0)

    return np.array(keep)
