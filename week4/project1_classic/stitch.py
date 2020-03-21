##
# date:2020/02/21
# auther:zym
# description:image stitch
##
import cv2
import numpy as np
import imutils
import tools


# 用sift提取图像关键点+描述子
def sift_func(imgA, imgB):
    grayA = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY)

    sift = cv2.xfeatures2d.SIFT_create()
    kpsA, desA = sift.detectAndCompute(grayA, None)
    kpsB, desB = sift.detectAndCompute(grayB, None)

    return (kpsA, desA), (kpsB, desB)


# 特征匹配并完成ratio_test
def match(desA, desB, ratio=0.75):
    bf = cv2.BFMatcher()
    rawMatchs = bf.knnMatch(desA, desB, k=2)
    # print(type(rawMatchs))
    # 保存初次过滤后，匹配对儿的特征点下标，分别对应第一张图和第二张图
    matchs = []
    for m in rawMatchs:
        if len(m) == 2 and m[0].distance < ratio * m[1].distance:
            matchs.append((m[0].queryIdx, m[0].trainIdx))
    return matchs


# ransac去噪，并计算单应性矩阵
def transform(kpsA, kpsB, matches, reprojThreshold=4.0):
    ptsA = np.float32([kpsA[i].pt for (i, _) in matches])
    ptsB = np.float32([kpsB[i].pt for (_, i) in matches])
    # findHomegraphy返回值是tuple，其中包含一个单应性矩阵一个状态列表
    (H, status) = cv2.findHomography(ptsA, ptsB, method=cv2.RANSAC, ransacReprojThreshold=reprojThreshold)
    return (H, status)


# 对图一进行投影变换并拼接
def stitcher(imgA, imgB, H):
    hA, wA = imgA.shape[:2]
    hB, wB = imgB.shape[:2]
    shft = np.array([[1.0, 0, wA], [0, 1.0, 0], [0, 0, 1.0]])
    M = np.dot(shft, H)
    result = cv2.warpPerspective(imgA, M, (wA + wB, hA))
    # my_show('result0',result)
    result[0:hB, wA:2 * wA] = imgB
    return result


# 加载图片
def load_image(pathA, pathB):
    imgA = cv2.imread(pathA)
    imgB = cv2.imread(pathB)
    # imgA = imutils.resize(imgA, 600)
    # imgB = imutils.resize(imgB, 600)
    return imgA, imgB

# 载入图片
pathA = 'b1.jpg'
pathB = 'b2.jpg'
imgA, imgB = load_image(pathA, pathB)
tools.my_show('A', imgA)
tools.my_show('B', imgB)

# 提取关键点
(kpsA, desA), (kpsB, desB) = sift_func(imgA, imgB)
tools.show_keyponts(imgA,kpsA,imgB,kpsB)

# 匹配连线，生成转换矩阵
matches = match(desA, desB)
(H, status) = transform(kpsA, kpsB, matches)
tools.drawMatches(imgA, imgB, kpsA, kpsB, matches,status)

# 衔接融合两张图片
result = stitcher(imgA, imgB, H)
tools.my_show('result', result)
