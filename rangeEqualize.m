%% 서브함수: x/y/z 범위 동기화
function rangeEqualize(xlim_, ylim_, zlim_)
    minVal = min([xlim_, ylim_, zlim_]);
    maxVal = max([xlim_, ylim_, zlim_]);
    rangeMid = (minVal + maxVal) / 2;
    rangeSize = (maxVal - minVal) / 2;
    xlim([rangeMid - rangeSize, rangeMid + rangeSize]);
    ylim([rangeMid - rangeSize, rangeMid + rangeSize]);
    zlim([rangeMid - rangeSize, rangeMid + rangeSize]);
end
