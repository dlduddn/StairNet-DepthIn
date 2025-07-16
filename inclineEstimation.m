%%
clear all
close all
clc

scale = 1; % 실제로는 scale = 1

%% [1] Load Masked Pointcloud from Segmentation Image
addpath('inference/output/');

% data = readmatrix('park170_l_pointcloud.txt'); % R_Align = eul2rotm([0, 20, 2] * pi/180);
% data = readmatrix('color_57_night_l_pointcloud.txt'); % R_Align = eul2rotm([0, 45, 15] * pi/180); 
% data = readmatrix('park108_r_pointcloud.txt'); % R_Align = eul2rotm([0, 40, -5] * pi/180); 
% data = readmatrix('color_67_night_r_pointcloud.txt'); % R_Align = eul2rotm([0, 40, 0] * pi/180);  
data = readmatrix('C:\Users\SAI_001\SynologyDrive\Code\Application\Stair\StairNet-DepthIn\inference\output\rgb_0020_pointcloud.txt');

% Transformation: Align to gravity & scale
R_Flu2Cam = eul2rotm([-90, 0, -90] * pi/180);  % camera to FLU
R_Align = eul2rotm([0, 0, 0] * pi/180);       % 실제로는 IMU 데이터를 이용하여 정렬
noAlign = eye(3);
pcNoAlign = (noAlign * R_Flu2Cam * data(:,1:3)')' * scale;
pc = (R_Align * R_Flu2Cam * data(:,1:3)')' * scale;
label = data(:,4); % 0=background, 1=vertical, 2=horizontal

%% [2] Separate Horizontal and Vertical Plane Pointclouds
pcH = pc(label == 2, :);  % 수평면
pcV = pc(label == 1, :);  % 수직면

ptCloudH = pointCloud(pcH);
ptCloudV = pointCloud(pcV);

% Voxel size (m 단위) 지정 – 예: 5cm 격자
voxelSize = 0.01;

% 다운샘플링 수행
ptCloudH_ds = pcdownsample(ptCloudH, 'gridAverage', voxelSize);
ptCloudV_ds = pcdownsample(ptCloudV, 'gridAverage', voxelSize);
figure; pcshow(ptCloudH_ds);

% 결과 출력 (point 좌표만)
pcH = ptCloudH_ds.Location;
pcV = ptCloudV_ds.Location;

%% [3] Vertical/Horizontal Plane Clustering
% 하이퍼파라미터
epsV = 0.05; minptsV = 100;
epsH = 0.01; minptsH = 100;

% 밀도기반 클러스터링
clusterV = dbscan(pcV(:,1:2), epsV, minptsV);  % 2D clustering on XY, XY 기준
clusterH = dbscan(pcH(:,3), epsH, minptsH);  % 1D DBSCAN, z축 기준

% 유효 클러스터
validVIdx = clusterV ~= -1;
validHIdx = clusterH ~= -1;
pcV_valid = pcV(validVIdx, :);
pcH_valid = pcH(validHIdx, :);
clusterV_valid = clusterV(validVIdx);
clusterH_valid = clusterH(validHIdx);
uniqueV = unique(clusterV_valid);
uniqueH = unique(clusterH_valid);
nV = numel(uniqueV);
nH = numel(uniqueH);
stairNum = min(nV, nH);

% 수직 단별 높이 및 대응 수평단 계산
riserHeights = zeros(nV, 1);
matchedH = zeros(stairNum, 1);
slopesDeg = zeros(stairNum, 1);
heights = zeros(nH, 1);
widths  = zeros(nH, 1);
depths  = zeros(nH, 1);
xyH_centers = zeros(nH, 2);

disp(uniqueH)
disp(uniqueV)
%% Fine Alignment : 수직면이 Y축과 평행이고, 수평면이 X축 방향으로 평평하며, Z축이 높이(상하)를 나타내도록 점군을 회전/정렬하고 싶다.
% Fine Alignment (Converge by rotation delta)
R_total = eye(3);  % 누적 회전 행렬
maxIter = 10;
rotDiffThresh = 1e-3;  % 수렴 기준 (Frobenius norm)
R_prev = eye(3);       % 초기 회전 행렬

for iter = 1:maxIter
    % (1) 수직면 정렬: X축과 정렬
    ptsV_sample = pcV_valid(clusterV_valid == mode(clusterV_valid), :);
    [~, ~, V] = svd(bsxfun(@minus, ptsV_sample, mean(ptsV_sample)));
    normalVec = V(:,3);
    v = cross(normalVec, [1; 0; 0]);
    s = norm(v); c = dot(normalVec, [1; 0; 0]);
    if s == 0
        R_align = eye(3);
    else
        vx = [0, -v(3), v(2); v(3), 0, -v(1); -v(2), v(1), 0];
        R_align = eye(3) + vx + vx^2 * ((1 - c) / s^2);
    end

    pcH_valid = (R_align * pcH_valid')';
    pcV_valid = (R_align * pcV_valid')';
    R_total = R_align * R_total;

    % (2) 수평면 정렬: Z축과 정렬
    ptsH_sample = pcH_valid(clusterH_valid == mode(clusterH_valid), :);
    [~, ~, V] = svd(bsxfun(@minus, ptsH_sample, mean(ptsH_sample)));
    normalVec = V(:,3);
    v = cross(normalVec, [0; 0; 1]);
    s = norm(v); c = dot(normalVec, [0; 0; 1]);
    if s == 0
        R_align = eye(3);
    else
        vx = [0, -v(3), v(2); v(3), 0, -v(1); -v(2), v(1), 0];
        R_align = eye(3) + vx + vx^2 * ((1 - c) / s^2);
    end

    pcH_valid = (R_align * pcH_valid')';
    pcV_valid = (R_align * pcV_valid')';
    R_total = R_align * R_total;

    % (3) 회전량 변화량으로 수렴 판단
    deltaR = norm(R_total - R_prev, 'fro');

    % --- 단일 Figure에 시각화 ---
    all = [pcV_valid; pcH_valid];
    scatter3(all(:,1), all(:,2), all(:,3), 5, [clusterV_valid; clusterH_valid], 'filled');
    view(45,30)
    title(sprintf('Iteration %d: ΔR = %.4e', iter, deltaR));
    xlabel('X'); ylabel('Y'); zlabel('Z');
    axis equal; grid on;

    % 
    xlim_ = xlim; ylim_ = ylim; zlim_ = zlim;
    minVal = min([xlim_, ylim_, zlim_]);
    maxVal = max([xlim_, ylim_, zlim_]);
    rangeMid = (minVal + maxVal) / 2;
    rangeSize = (maxVal - minVal) / 2;
    xlim([rangeMid - rangeSize, rangeMid + rangeSize]);
    ylim([rangeMid - rangeSize, rangeMid + rangeSize]);
    zlim([rangeMid - rangeSize, rangeMid + rangeSize]);
    drawnow;
    pause(1)

    if deltaR < rotDiffThresh
        disp(['회전 변화량 수렴 (iter = ' num2str(iter) ', ΔR = ' num2str(deltaR) ')']);
        break;
    end
    R_prev = R_total;
end

%% 수직면/수평면 정보 계산
for i = 1:nV  
    ptsV = pcV_valid(clusterV_valid == uniqueV(i), :);
    z_range = prctile(ptsV(:,3), [5, 90]);  % z축 5~95%
    riserHeights(i) = z_range(2) - z_range(1);
end

for i = 1:nH
    pts = pcH_valid(clusterH_valid == uniqueH(i), :);
    z_values = pts(:,3);
    heights(i) = abs(mean(z_values));  % 그대로 사용해도 무방

    x_range = prctile(pts(:,1), [5, 95]);  % x축 5~95%
    y_range = prctile(pts(:,2), [5, 95]);  % y축 5~95%
    widths(i) = abs(y_range(2) - y_range(1));
    depths(i) = abs(x_range(2) - x_range(1));

    xyH_centers(i,:) = mean(pts(:,1:2), 1);
end

%% [5] 계단 경사도 계산
for i = 1:stairNum
    ptsV = pcV_valid(clusterV_valid == uniqueV(i), :);
    z_range = prctile(ptsV(:,3), [5, 95]);
    riserHeights(i) = z_range(2) - z_range(1);

    matchedH(i) = i;  % i번째 수직 단 ↔ i번째 수평 단

    treadDepth = depths(i);  % 수평단의 depth 사용
    slopeRad = atan2(riserHeights(i), treadDepth);
    slopesDeg(i) = rad2deg(slopeRad);
end

%% [5] Table Summary
cm = 100;
% 수평 단 테이블
stepTable = table((1:nH)', heights * cm, widths* cm, depths* cm, ...
    'VariableNames', {'StepID', 'Height_Z_cm', 'Width_X_cm', 'Depth_Y_cm'});

% 수직 단 + 경사도 테이블
slopeTable = table((1:stairNum)', riserHeights(1:stairNum)* cm, depths(matchedH)* cm, slopesDeg, ...
    'VariableNames', {'RiserID', 'RiserHeight_cm', 'TreadDepth_cm', 'Slope_deg'});

%% [6] 출력
disp("=== 수평 단 정보 ===");
disp(stepTable);
disp("=== 수직 단 경사도 정보 ===");
disp(slopeTable);

% 통계 출력
fprintf("\n▶ 수평단 평균 (Z, X, Y): %.3f, %.3f, %.3f\n", mean(heights), mean(widths), mean(depths));
fprintf("▶ 표준편차 (Z, X, Y):     %.3f, %.3f, %.3f\n", std(heights), std(widths), std(depths));
fprintf("\n▶ 수직단 평균 높이: %.3f m, 평균 경사도: %.2f°\n", mean(riserHeights), mean(slopesDeg));

%% [7] 수평 단 클러스터 시각화
figure;
subplot(1,2,1)
scatter3(pcH_valid(:,1), pcH_valid(:,2), pcH_valid(:,3), 10, clusterH_valid, 'filled');
title('Horizontal Plane Clusters');
xlabel('X (cm)'); ylabel('Y (cm)'); zlabel('Z (cm)');
axis equal; grid on;

%% [8] 수직 단 클러스터 시각화
% figure;
subplot(1,2,2)
scatter3(pcV_valid(:,1), pcV_valid(:,2), pcV_valid(:,3), 10, clusterV_valid, 'filled');
title('Vertical Plane Clusters');
xlabel('X (cm)'); ylabel('Y (cm)'); zlabel('Z (cm)');
axis equal; grid on;

%%
figure;
barData = [stepTable.Height_Z_cm, stepTable.Width_X_cm, stepTable.Depth_Y_cm];
bar(stepTable.StepID, barData, 'grouped');
xlabel('Step ID');
ylabel('Length (cm)');
legend({'Height Z', 'Width X', 'Depth Y'}, 'Location', 'northeast');
title('Step Geometry per Step ID');
grid on;
%%
figure;
% yyaxis left
bar(slopeTable.RiserID - 0.15, slopeTable.RiserHeight_cm, 0.3, 'FaceColor', [0.2 0.4 0.8]);
hold on;
bar(slopeTable.RiserID + 0.15, slopeTable.TreadDepth_cm, 0.3, 'FaceColor', [0.2 0.8 0.4]);
ylabel('Length (cm)');

% yyaxis right
% plot(slopeTable.RiserID, slopeTable.Slope_deg, '-or', 'LineWidth', 2);
% ylabel('Slope (deg)');

xlabel('Stair ID');
title('Riser Height / Tread Depth');
legend({'Riser Height', 'Tread Depth'}, 'Location', 'northwest');
grid on;
