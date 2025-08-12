import cv2
import numpy as np


def detect_aruco_and_edges(frame, aruco_dict, aruco_params):
    # 转换为灰度图像
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 检测 ArUco 标记
    corners, ids, _ = cv2.aruco.detectMarkers(
        gray, aruco_dict, parameters=aruco_params)

    # 创建掩码，初始化为与输入帧相同大小的全黑图像
    mask = np.zeros_like(frame, dtype=np.uint8)

    if ids is not None:
        for corner in corners:
            # 将 ArUco 标记的区域用绿色填充
            pts = np.int32(corner[0])  # 转换为整数点
            cv2.fillPoly(mask, [pts], (0, 255, 0))

    # 应用高斯模糊以减少噪声
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)

    # Canny 边缘检测
    edges = cv2.Canny(blurred, 100, 200)

    # 将掩码中的绿色区域覆盖到原始帧上
    frame_with_mask = cv2.addWeighted(frame, 1, mask, 1, 0)

    # 创建显示结果，将 ArUco 和 Canny 的结果合并
    combined_edges = cv2.bitwise_or(edges, cv2.cvtColor(
        mask[:, :, 1], cv2.COLOR_GRAY2BGR)[:, :, 1])

    return frame_with_mask, combined_edges


def main():
    # 打开摄像头或视频文件（0 表示默认摄像头）
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("无法打开摄像头")
        return

    # 加载预定义的 ArUco 字典
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters()
    while True:
        # 从摄像头读取帧
        ret, frame = cap.read()
        if not ret:
            print("无法读取帧，可能摄像头已断开")
            break

        # 检测 ArUco 标记和 Canny 边缘
        frame_with_mask, combined_edges = detect_aruco_and_edges(
            frame, aruco_dict, aruco_params)

        # 显示原始帧（带掩码）和边缘检测结果
        cv2.imshow('Original with ArUco Mask', frame_with_mask)
        cv2.imshow('Canny Edges with ArUco', combined_edges)

        # 按下 'q' 键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源并关闭窗口
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
