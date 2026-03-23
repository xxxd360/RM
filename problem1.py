import cv2
import numpy as np

img_first = cv2.imread(r"/RM/asserts/example.png")
#图片去噪
#中值滤波
img_denoise1 = cv2.medianBlur(img_first,3)
#非局部均值去噪 (Non-Local Means, NLM)src: 输入图, h: 过滤强度(通常10), hColor: 颜色分量强度(通常10), templateWindowSize: 奇数(7), searchWindowSize: 奇数(21)
img_denoise2=cv2.fastNlMeansDenoisingColored(img_denoise1,None, 10, 10, 7, 21)
# #双边滤波(Bilateral Filtering)d: 直径, sigmaColor: 颜色差异阈值, sigmaSpace: 空间距离阈值
img_denoise3=cv2.bilateralFilter(img_denoise2, 9, 75, 75)
# cv2.imwrite("denoise_result.png",img_denoise3)
# cv2.imshow("denoise_result",img_denoise3)
#HSV检测
hsv = cv2.cvtColor(img_denoise2, cv2.COLOR_BGR2HSV)
lower_red = np.array([0,137,80])    # HSV 下限
upper_red = np.array([100, 225, 255])  # HSV 上限
mask = cv2.inRange(hsv, lower_red, upper_red)
img_hsv = cv2.bitwise_and(img_denoise2, img_denoise2, mask=mask)
# cv2.imshow("HSV_result",img_hsv)
#图形检测
imgContour = img_hsv.copy()
def getContours(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        # 找面积
        area = cv2.contourArea(cnt)
        if area > 400:
        # 找周长
           perimeter = cv2.arcLength(cnt, True)
        #确定近似轮廓
           approx = cv2.approxPolyDP(cnt, 0.03* perimeter, True)
           if len(approx)==4:
               points = approx.reshape(4, 2)
               cv2.polylines(imgContour, [points], True, (0, 255, 0), 2)
               for pt in points:
                   cv2.circle(imgContour, (pt[0], pt[1]), 4, (0, 0, 255), -1)

imgGray = cv2.cvtColor(img_hsv, cv2.COLOR_BGR2GRAY)
_, imgThresh = cv2.threshold(imgGray, 150, 255, cv2.THRESH_BINARY)
imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 0)
imgCanny = cv2.Canny(imgBlur, 50, 50)
getContours(imgCanny)
cv2.imshow("img_con",imgContour)
cv2.imwrite('results/problems1_result.jpg', imgContour)
cv2.waitKey(0)