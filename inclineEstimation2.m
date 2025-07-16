%% [1] Load Masked Pointcloud from Segmentation Image
addpath('inference/output/');

% data = readmatrix('color_278_l_pointcloud.txt');
data = readmatrix('color_57_night_l_pointcloud.txt'); % R_Align = eul2rotm([0, 60, 15] * pi/180); 
% data = readmatrix('park108_r_pointcloud.txt'); % R_Align = eul2rotm([0, 20, 2] * pi/180);  

% Transformation: Align to gravity & scale
scale = 0.1; % 실제로는 scale = 1
R_Flu2Cam = eul2rotm([-90, 0, -90] * pi/180);  % camera to FLU
% R_Align = eul2rotm([0, 60, 15] * pi/180);       % 실제로는 IMU 데이터를 이용하여 정렬

% R_Align = eye(3);
pc = (R_Align * R_Flu2Cam * data(:,1:3)')' * scale;
label = data(:,4); % 0=background, 1=vertical, 2=horizontal

% Set colors: 0=gray, 1=red, 2=blue
colorMap = [0.5 0.5 0.5; 1 0 0; 0 0 1];
colors = colorMap(label+1, :);  % label이 0~2라면 index로 사용 가능

% Create and display point cloud
gridSize = 0.0;
figure
% ptCloud = pointCloud(pc);
ptCloud = pointCloud(pc, 'Color', uint8(colors * 255));
pcshow(ptCloud);
xlabel('X (m)', 'Color', 'w', 'FontWeight', 'bold');
ylabel('Y (m)', 'Color', 'w', 'FontWeight', 'bold');
zlabel('Z (m)', 'Color', 'w', 'FontWeight', 'bold');
axis equal
grid on
title('Stair Pointcloud', 'Color', 'w', 'FontWeight', 'bold');

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

%% Fine Alignment : 수직면이 Y축과 평행이고, 수평면이 X축 방향으로 평평하며, Z축이 높이(상하)를 나타내도록 점군을 회전/정렬하고 싶다.
% 수직면의 법선벡터가 X축과 평행하도록 정렬
ptsV_sample = pcV_valid(clusterV_valid == mode(clusterV_valid), :);
[~, ~, V] = svd(bsxfun(@minus, ptsV_sample, mean(ptsV_sample)));
normalVec = V(:,3);
xAxis = [1; 0; 0];
v = cross(normalVec, xAxis);
s = norm(v); c = dot(normalVec, xAxis);
if s == 0
    R_align_normal = eye(3);
else
    vx = [  0, -v(3), v(2);
          v(3),  0, -v(1);
         -v(2), v(1),  0 ];
    R_align_normal = eye(3) + vx + vx^2 * ((1 - c) / s^2);
end
pcH_valid = (R_align_normal * pcH_valid')';
pcV_valid = (R_align_normal * pcV_valid')';

% 수평면의 법선벡터가 Z축과 평행하도록 정렬 (수평면이 XY 평면과 평행하도록 회전)
ptsH_sample = pcH_valid(clusterH_valid == mode(clusterH_valid), :);
[~, ~, V] = svd(bsxfun(@minus, ptsH_sample, mean(ptsH_sample)));
normalVec = V(:,3);
zAxis = [0; 0; 1];
v = cross(normalVec, zAxis);
s = norm(v); c = dot(normalVec, zAxis);
if s == 0
    R_align_normal = eye(3);
else
    vx = [  0, -v(3), v(2);
          v(3),  0, -v(1);
         -v(2), v(1),  0 ];
    R_align_normal = eye(3) + vx + vx^2 * ((1 - c) / s^2);
end
pcH_valid = (R_align_normal * pcH_valid')';
pcV_valid = (R_align_normal * pcV_valid')';

%% 수직면/수평면 정보 계산
for i = 1:nV  
    ptsV = pcV_valid(clusterV_valid == uniqueV(i), :);
    z_range = prctile(ptsV(:,3), [5, 95]);  % z축 5~95%
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
%% 전체 시각화
figure
all = [pcV_valid;pcH_valid];
scatter3(all(:,1), all(:,2), all(:,3), 10, [clusterV_valid;clusterH_valid], 'filled');
title('정렬된 점군');
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
