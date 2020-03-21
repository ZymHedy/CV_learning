##
# author:zym
# description:severl tools for showing
##
import cv2
import numpy as np


# 展示一张图片
def my_show(name, image):
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 展示角点特征点
def show_keyponts(imgA, kpsA, imgB, kpsB):
    dotA = np.zeros((imgA.shape[0],imgA.shape[1]))
    dotB = np.zeros((imgB.shape[0],imgB.shape[1]))
    dotA = cv2.drawKeypoints(image=imgA, keypoints=kpsA, outImage=dotA, color=(51, 163, 236),
                             flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
    dotB = cv2.drawKeypoints(image=imgB, keypoints=kpsB, outImage=dotB, color=(51, 163, 236),
                             flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
    my_show('A', dotA)
    my_show('B', dotB)


# 展示特征点匹配线
def drawMatches(imageA, imageB, kpsA, kpsB, matches, status):
    # initialize the output visualization image
    (hA, wA) = imageA.shape[:2]
    (hB, wB) = imageB.shape[:2]
    vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
    vis[0:hA, 0:wA] = imageA
    vis[0:hB, wA:] = imageB
    # loop over the matches
    for ((trainIdx, queryIdx), s) in zip(matches, status):
        # only process the match if the keypoint was successfully
        # matched
        if s == 1:
            # draw the match
            [xA, yA] = kpsA[trainIdx].pt
            ptA = (int(xA), int(yA))
            [xB, yB] = kpsB[queryIdx].pt
            ptB = (imageA.shape[1]+int(xB), int(yB))
            cv2.line(vis, ptA, ptB, (0, 0, 255), 1)
    # return the visualization
    my_show('vis',vis)
    return vis
