%%
clear all
close all
clc
%%
% 저장 폴더 경로 설정
rgbFolder   = 'C:\Users\SAI_001\SynologyDrive\Code\Application\Stair\StairNet-DepthIn\data\val\images';
depthFolder = 'C:\Users\SAI_001\SynologyDrive\Code\Application\Stair\StairNet-DepthIn\data\val\depthes';

% 기존 폴더가 있다면 삭제 후 재생성
if exist(rgbFolder, 'dir')
    rmdir(rgbFolder, 's');
end
mkdir(rgbFolder);

if exist(depthFolder, 'dir')
    rmdir(depthFolder, 's');
end
mkdir(depthFolder);

targetSize = [512, 512];

%% ROS bag 및 msg 불러오기
filePath = "D:\d455\20250716_103417.bag";
bag = rosbag(filePath);

depthTopic = '/device_0/sensor_0/Depth_0/image/data'; % 30 Hz
rgbTopic   = '/device_0/sensor_1/Color_0/image/data'; % 30 Hz
accelTopic = '/device_0/sensor_2/Accel_0/imu/data';   % 100 Hz

rgbMsgs   = readMessages(select(bag, 'Topic', rgbTopic));
depthMsgs = readMessages(select(bag, 'Topic', depthTopic));
accelMsgs = readMessages(select(bag, 'Topic', accelTopic));

for i = 1:length(rgbMsgs)
    rgb = readImage(rgbMsgs{i});
    depth = readImage(depthMsgs{i});

    % 크롭 또는 패딩 적용
    rgb_resized = resizeOrPad(rgb, targetSize);
    depth_resized = resizeOrPad(depth, targetSize);
    % rgb_resized = rgb;
    % depth_resized = depth;

    % 저장 파일 이름 지정
    rgbName   = fullfile(rgbFolder, sprintf('rgb_%04d.png', i));
    depthName = fullfile(depthFolder, sprintf('depth_%04d.png', i));

    % 저장
    imwrite(rgb_resized, rgbName);
    imwrite(uint16(depth_resized), depthName, 'BitDepth', 16);
end
