%% Masked Pointcloud (NoAligned)
% Set colors: 0=gray, 1=red, 2=blue
colorMap = [0.5 0.5 0.5; 1 0 0; 0 0 1];
colors = colorMap(label+1, :);  % label이 0~2라면 index로 사용 가능

figure

% 실제 포인트클라우드 시각화
H = pointCloud(pcNoAlign(label == 2, :), 'Color', 'b');
V = pointCloud(pcNoAlign(label == 1, :), 'Color', 'r');
% B = pointCloud(pcNoAlign(label == 0, :), 'Color', 'g'); 
pcshow(H); hold on;
pcshow(V);
% pcshow(B);
% 더미 점 추가해서 legend용 핸들 만들기
hH = scatter3(nan, nan, nan, 36, 'b', 'filled'); % 수평면 (파랑)
hV = scatter3(nan, nan, nan, 36, 'r', 'filled'); % 수직면 (빨강)
ax = gca;
ax.GridAlpha = 1;                  % Grid 선의 투명도 (1 = 불투명)
ax.GridColor = [0.7 0.7 0.7];      % Grid 선의 색상 (연회색)
ax.LineWidth = 1.2;                % Grid 선 두께
ax.GridLineStyle = '-';           % 실선
% 레전드 생성
legend([hH, hV], {'Horizontal', 'Vertical'}, 'TextColor', 'w', 'Location', 'best');

% 축 라벨 및 제목
xlabel('X (m)', 'Color', 'w', 'FontWeight', 'bold');
ylabel('Y (m)', 'Color', 'w', 'FontWeight', 'bold');
zlabel('Z (m)', 'Color', 'w', 'FontWeight', 'bold');
title('Stair Pointcloud', 'Color', 'w', 'FontWeight', 'bold');

axis equal
grid on

figure
ptCloud = pointCloud(pcNoAlign(label == 2, :), 'Color', 'b');
pcshow(ptCloud);
xlabel('X (m)', 'Color', 'w', 'FontWeight', 'bold');
ylabel('Y (m)', 'Color', 'w', 'FontWeight', 'bold');
zlabel('Z (m)', 'Color', 'w', 'FontWeight', 'bold');
axis equal
grid on
title('Horizontal Plane Pointcloud', 'Color', 'w', 'FontWeight', 'bold');

figure
ptCloud = pointCloud(pcNoAlign(label == 1, :), 'Color', 'r');
pcshow(ptCloud);
xlabel('X (m)', 'Color', 'w', 'FontWeight', 'bold');
ylabel('Y (m)', 'Color', 'w', 'FontWeight', 'bold');
zlabel('Z (m)', 'Color', 'w', 'FontWeight', 'bold');
axis equal
grid on
title('Vertical Plane Pointcloud', 'Color', 'w', 'FontWeight', 'bold');

%% Masked Pointcloud (Aligned)
% Set colors: 0=gray, 1=red, 2=blue
colorMap = [0.5 0.5 0.5; 1 0 0; 0 0 1];
colors = colorMap(label+1, :);  % label이 0~2라면 index로 사용 가능

figure
% 실제 포인트클라우드 시각화
H = pointCloud(pc(label == 2, :), 'Color', 'b');
V = pointCloud(pc(label == 1, :), 'Color', 'r');
pcshow(H); hold on;
pcshow(V);

% 더미 점 추가해서 legend용 핸들 만들기
hH = scatter3(nan, nan, nan, 36, 'b', 'filled'); % 수평면 (파랑)
hV = scatter3(nan, nan, nan, 36, 'r', 'filled'); % 수직면 (빨강)
ax = gca;
ax.GridAlpha = 1;                  % Grid 선의 투명도 (1 = 불투명)
ax.GridColor = [0.7 0.7 0.7];      % Grid 선의 색상 (연회색)
ax.LineWidth = 1.2;                % Grid 선 두께
ax.GridLineStyle = '-';           % 실선
% 레전드 생성
legend([hH, hV], {'Horizontal', 'Vertical'}, 'TextColor', 'w', 'Location', 'best');

% 축 라벨 및 제목
xlabel('X (m)', 'Color', 'w', 'FontWeight', 'bold');
ylabel('Y (m)', 'Color', 'w', 'FontWeight', 'bold');
zlabel('Z (m)', 'Color', 'w', 'FontWeight', 'bold');
title('Stair Pointcloud', 'Color', 'w', 'FontWeight', 'bold');

axis equal
grid on

figure
ptCloud = pointCloud(pc(label == 2, :), 'Color', 'b');
pcshow(ptCloud);
xlabel('X (m)', 'Color', 'w', 'FontWeight', 'bold');
ylabel('Y (m)', 'Color', 'w', 'FontWeight', 'bold');
zlabel('Z (m)', 'Color', 'w', 'FontWeight', 'bold');
axis equal
grid on
title('Horizontal Plane Pointcloud', 'Color', 'w', 'FontWeight', 'bold');

figure
ptCloud = pointCloud(pc(label == 1, :), 'Color', 'r');
pcshow(ptCloud);
xlabel('X (m)', 'Color', 'w', 'FontWeight', 'bold');
ylabel('Y (m)', 'Color', 'w', 'FontWeight', 'bold');
zlabel('Z (m)', 'Color', 'w', 'FontWeight', 'bold');
axis equal
grid on
title('Vertical Plane Pointcloud', 'Color', 'w', 'FontWeight', 'bold');

%% dbscan 수행 대상 
% pcV의 XY 분포와 pcH의 Z축 값
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
figure;

% [1] 수직면 포인트들의 XY 분포 (Top View)
figure
scatter(pcV(:,1), pcV(:,2), 10, 'r', 'filled');
axis equal;
xlabel('X (m)');
ylabel('Y (m)');
title('Vertical Plane Pointcloud : XY Plane Distribution');
grid on;

% [2] 수평면 포인트들의 Z값 분포 (높이)
figure
plot(pcH(:,3), '.', 'Color', [0 0 1]);  % 파란색 점
xlabel('Point Index');
ylabel('Z ( m)');
title('Horizontal Plane Pointcloud : Z-axis Distribution');
grid on;

%% [7] 수평 단 / 수직 단 클러스터 시각화
figure;
subplot(1,2,1)
scatter3(pcH_valid(:,1), pcH_valid(:,2), pcH_valid(:,3), 10, clusterH_valid, 'filled');
title('Horizontal Plane Clusters');
xlabel('X (cm)'); ylabel('Y (cm)'); zlabel('Z (cm)');
axis equal; grid on;

% [8]  클러스터 시각화
% figure;
subplot(1,2,2)
scatter3(pcV_valid(:,1), pcV_valid(:,2), pcV_valid(:,3), 10, clusterV_valid, 'filled');
title('Vertical Plane Clusters');
xlabel('X (cm)'); ylabel('Y (cm)'); zlabel('Z (cm)');
axis equal; grid on;
