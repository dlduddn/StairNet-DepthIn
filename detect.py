# 실행 명령어
# python detect.py --width_factor 1.0 --weights logs/StairNet_DepthIn_1.0/best.pth --source 2 --input data/val/images --output inference/output --img-size 512 --conf-thres 0.5 --view-img True 
# 임포트 및 초기 설정
import argparse
import shutil
import glob
from pathlib import Path
from nets.NetV3 import StairNet_DepthIn # # StairNet V3 모델 정의
import torch
import os
import config 
import torch.backends.cudnn as cudnn
import cv2
import numpy as np
import time
from utils import Draw_results # # 결과 시각화 함수
from scipy.ndimage import gaussian_filter

img_formats = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.dng']

# RGB & Depth 이미지 불러오기
class LoadImages:  # for inference
    def __init__(self, RGB_path, Depth_path, img_size=512):
        # get RGB images
        RGB_path = str(Path(RGB_path))
        RGB_files = []
        if os.path.isdir(RGB_path):
            RGB_files = sorted(glob.glob(os.path.join(RGB_path, '*.*')))
        elif os.path.isfile(RGB_path):
            RGB_files = [RGB_path]
        RGB_images = [x for x in RGB_files if os.path.splitext(x)[-1].lower() in img_formats]
        # get Depth images
        Depth_path = str(Path(Depth_path))
        Depth_files = []
        if os.path.isdir(Depth_path):
            Depth_files = sorted(glob.glob(os.path.join(Depth_path, '*.*')))
        elif os.path.isfile(Depth_path):
            Depth_files = [Depth_path]
        Depth_images = [x for x in Depth_files if os.path.splitext(x)[-1].lower() in img_formats]

        self.img_size = img_size

        self.RGB_files = RGB_images
        self.nRGB = len(RGB_images)  # number of files
        self.Depth_files = Depth_images
        self.nDepth = len(Depth_images)

        assert self.nRGB > 0, 'No images or videos found in %s. Supported formats are:\nimages: %s' % \
                              (RGB_path, img_formats)

        assert self.nRGB == self.nDepth, 'The numbers of RGB and Depth images are different!'

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nRGB:
            raise StopIteration
        RGB_path = self.RGB_files[self.count]
        Depth_path = self.Depth_files[self.count]

        self.count += 1
        img0 = cv2.imread(RGB_path)
        img0_d = cv2.imread(Depth_path, cv2.IMREAD_UNCHANGED)
        assert img0 is not None, 'Image Not Found ' + RGB_path
        assert img0_d is not None, 'Image Not Found ' + Depth_path
        print('image %g/%g %s: ' % (self.count, self.nRGB, RGB_path), end='')

        # padded and resize
        padh = (img0.shape[1] - img0.shape[0]) // 2

        img = cv2.cvtColor(img0, cv2.COLOR_RGB2BGR)
        img = np.pad(img, ((padh, padh), (0, 0), (0, 0)), 'constant', constant_values=0)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = np.ascontiguousarray(img)

        img_d = np.pad(img0_d, ((padh, padh), (0, 0)), 'constant', constant_values=0)
        img_d = cv2.resize(img_d, (self.img_size, self.img_size))
        img_d = np.ascontiguousarray(img_d)

        return RGB_path, img, img_d, img0

    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.nframes = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.nRGB  # number of files


def detect():
    # 명령줄 인자로 받은 파라미터들을 unpack
    width_factor, inPath, out, source, weights, view_img, conf_thres, imgsz = \
        opt.width_factor, opt.input, opt.output, opt.source, opt.weights, opt.view_img, opt.conf_thres, opt.img_size
        
    # 모델 실행 장치 설정 (GPU 또는 CPU)
    device = config.device 
    half = device.type != 'cpu'  # half precision only supported on CUDA,  GPU 사용 시 FP16 연산 활성화 (속도 향상)
    
    # 이전 결과 디렉토리가 존재하면 삭제
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    
    # StairNet 모델 불러오기 및 설정
    model = StairNet_DepthIn(width=width_factor).to(device) # 모델 생성
    model.load_state_dict(torch.load(weights, map_location=device)) # 학습된 가중치 로드
    model.eval() # 평가 모드 설정 (dropout, batchnorm 비활성화)
    
    if half:
        model.half()  # 모델을 FP16으로 변환 (GPU 전용)
        
    # 입력 데이터 소스 설정
    if source == 0: # (미구현)
        cudnn.benchmark = True  # set True to speed up constant image size inference
        #    dataset = LoadStreams(source, img_size=imgsz)
    elif source == 1: # (미구현)
        cudnn.benchmark = True  # set True to speed up constant image size inference
        #    dataset = LoadStreams_depthcamera(source, img_size=imgsz)
    elif source == 2:
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadImages(inPath, inPath.replace('images', 'depthes'), img_size=imgsz)
    else:
        print("Please choose the correct index for source!") # # 잘못된 source 인덱스

    # 이미지 순회하면서 추론 수행
    for path, img, img_d, img0 in dataset:
        print(f"[DEBUG] Depth image dtype: {img_d.dtype}")
        # -----------------------------
        # RGB 이미지 전처리
        # -----------------------------
        input = (img / 255.0 - 0.5) / 0.5
        x = torch.Tensor(input)
        x = x.permute(2, 0, 1)
        x = torch.unsqueeze(x, dim=0)
        x = x.to(device)
        x = x.half() if half else x.float()
        
        # -----------------------------
        # Depth 이미지 전처리
        # -----------------------------
        input_d = (img_d / 65535.0 - 0.5) / 0.5
        x_d = torch.Tensor(input_d)
        x_d = torch.unsqueeze(x_d, dim=2)
        x_d = x_d.permute(2, 0, 1)
        x_d = torch.unsqueeze(x_d, dim=0)
        x_d = x_d.to(device)
        x_d = x_d.half() if half else x_d.float()
        
        # -----------------------------
        # 모델 추론 수행
        # -----------------------------
        # torch.cuda.synchronize()
        start = time.time()
        y = model(x, x_d)
        # torch.cuda.synchronize()
        endt1 = time.time()
        print("CNN inference time:" + format(endt1 - start, '.3f') + 's')
        
        # -----------------------------
        # 모델 출력 분리
        # -----------------------------
        fb1, fb2, fr1, fr2, y3 = y
        
        # -----------------------------
        # 모델 출력을 numpy로 변환
        # -----------------------------
        fb1, fb2, fr1, fr2 = fb1.cpu(), fb2.cpu(), fr1.cpu(), fr2.cpu()
        fb1 = torch.squeeze(fb1).detach().numpy()
        fb2 = torch.squeeze(fb2).detach().numpy()

        fr1 = torch.squeeze(fr1).detach().numpy()
        fr2 = torch.squeeze(fr2).detach().numpy()
        
        # -----------------------------
        # 시각화 수행 (라인 + 마스크)
        # -----------------------------
        img_result = Draw_results(img0, fb1, fb2, fr1, fr2, y3, conf=conf_thres)
        
        # -----------------------------
        # 결과 이미지 저장
        # -----------------------------
        if view_img:
            save_path = os.path.join(out, "masked_rgb")
            os.makedirs(save_path, exist_ok=True)
            cv2.imwrite(os.path.join(save_path, os.path.basename(path)), img_result)
        
        # mask prediction
        _, pred_mask = torch.max(y3, 1)
        pred_mask = pred_mask.squeeze().cpu().numpy()
        depth_np = img_d.copy()
        masked_depth = np.where((pred_mask == 1) | (pred_mask == 2), depth_np, 0)
        save_path = os.path.join(out, "masked_depth")
        os.makedirs(save_path, exist_ok=True)
        cv2.imwrite(os.path.join(save_path, os.path.basename(path)), masked_depth)  
        unique_labels, counts = np.unique(pred_mask, return_counts=True)
        print(f"[DEBUG] Label histogram: {dict(zip(unique_labels, counts))}")
  
        # -----------------------------
        # y3 기반 Mask → Pointcloud 추출 및 저장
        # -----------------------------
        with torch.no_grad():
            pred_mask = torch.argmax(y3.squeeze(0), dim=0).cpu().numpy()  # shape: (H, W)
            #depth_raw = img_d.astype(np.float32) * 255.0 / 2.0  # 역정규화
            #depth_raw = depth_raw.astype(np.uint16)  # depth 단위 복원(mm)
            # depth_raw = np.where((depth_raw > 300) & (depth_raw < 3500), depth_raw, 0)
            depth_raw = img_d.astype(np.uint16)  # 단위: mm 
            
            #depth_blur = gaussian_filter(depth_raw.astype(np.float32), sigma=1)
            #edge_diff = np.abs(depth_raw - depth_blur)
            #depth_raw[edge_diff > 300] = 0  # 30cm 이상 차이나면 제거
            
            
            fx, fy, cx, cy = 428.8284, 422.4822, 256, 239.7383  # RealSense D455 예시 intrinsic
            #fx, fy, cx, cy = 382.20, 382.20, 319.5, 239.5  # RealSense D435 예시 intrinsic
            point_list = []
            for v in range(depth_raw.shape[0]):
                for u in range(depth_raw.shape[1]):
                    z = depth_raw[v, u] * 0.001  # mm → m
                    if z == 0:
                        continue
                    x = (u - cx) * z / fx
                    y = (v - cy) * z / fy
                    label = pred_mask[v, u]
                    if label > 0:  # 계단 수직/수평면만 저장 (원하면 제거 가능)
                        point_list.append([x, y, z, label])

            pc_out = os.path.splitext(os.path.basename(path))[0] + '_pointcloud.txt'
            np.savetxt(os.path.join(out, pc_out), point_list, fmt="%.4f", delimiter=',', header='X,Y,Z,Label', comments='')
            print(f"Saved labeled pointcloud: {pc_out}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--width_factor", type=float, help="for scaling of models", default=0.5)
    parser.add_argument('--weights', type=str, default='logs/StairNet_DepthIn_0.5/best.pth', help='model.pth path')
    parser.add_argument('--source', type=int, default=2, help='source')  # 0 for webcam, 1 for realsense, 2 for files
    parser.add_argument('--input', type=str, default='data/val/images', help='output folder')  # input folder
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=512, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--view-img', type=bool, default=True, help='store results')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        detect()
