import cv2
import numpy as np
def getContours(imgCanny,img):
    contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    idx = -1
    point = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 400:
           idx+=1
           perimeter = cv2.arcLength(cnt, True)
           approx = cv2.approxPolyDP(cnt, 0.03* perimeter, True)
           if len(approx)==4:
               center = [0,0]
               points = approx.reshape(4, 2)
               point.append(points)
               for pt in points:
                   center[0]+=pt[0]
                   center[1]+=pt[1]
               center[0] = int(center[0]/4)
               center[1] = int(center[1]/4)
               cv2.circle(img, center, 4, (0, 0, 255), -1)
    return img,point
def center_point(img):
    img_first = img
    hsv = cv2.cvtColor(img_first, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0,0,100])    # HSV 下限
    upper_red = np.array([117, 138, 255])  # HSV 上限
    mask = cv2.inRange(hsv, lower_red, upper_red)
    img_hsv = cv2.bitwise_and(img_first, img_first, mask=mask)
    imgGray = cv2.cvtColor(img_hsv, cv2.COLOR_BGR2GRAY)
    _, imgThresh = cv2.threshold(imgGray, 150, 255, cv2.THRESH_BINARY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 0)
    imgCanny = cv2.Canny(imgBlur, 50, 50)
    img_out,point=getContours(imgCanny,img)
    return img_out,point


def order_points_rect(points):
    # 计算 x+y 和 x-y
    rect = np.zeros((4, 2), dtype=np.float32)
    s = points.sum(axis=1)  # x+y
    diff = np.diff(points, axis=1)  # x-y
    rect[0] = points[np.argmin(s)]  # 左上：x+y 最小
    rect[2] = points[np.argmax(s)]  # 右下：x+y 最大
    rect[1] = points[np.argmin(diff)]  # 右上：x-y 最小
    rect[3] = points[np.argmax(diff)]  # 左下：x-y 最大
    return rect
def PNP(img):
    #准备3D-对应点
    obj_points = np.array([[55, -22.5, 60], [55, 22.5, 60],
                                  [55,-22.5,50],[55,22.5,50]],
                                  dtype=np.float32)  # 棋盘格角点世界坐标
    img_out,point = center_point(img)
    idx= len(point)
    #相机内参
    camera_matrix = np.array([[16854.0, 0.0, 686.0], [0.0, 16893.0, 511.0], [0.0, 0.0, 1.0]])
    dist_coeffs = np.array([3.2859, -337.662, 0.0, 0.0, 0.0])  # 假设无镜头畸变
    #求解PnP
    for i in range(idx):
        img_points = order_points_rect(point[i])
        success, rvec, tvec = cv2.solvePnP(obj_points, img_points, camera_matrix, dist_coeffs)
        if success:
            tx, ty, tz = tvec.flatten()
            tx = int(tx)
            ty = int(ty)
            tz = int(tz)
            cv2.putText(img_out, f"{i+1}number:{tx,ty,tz}", (1100,40+20*i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return img_out
def viedo_process(video_path,output_path=None):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not out.isOpened():
            print(f"错误：无法创建输出视频 {output_path}")
            return
        print(f"输出视频将保存到: {output_path}")
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"视频处理完成，共处理 {frame_count} 帧")
            break
        processed_frame = PNP(frame)
        cv2.imshow('Video Processing', processed_frame)
        if out:
            out.write(processed_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("用户中断处理")
            break
        frame_count += 1
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    input_video = "video.avi"
    viedo_process(input_video, output_path="results/out_video.avi")