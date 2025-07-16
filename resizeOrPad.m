function out = resizeOrPad(img, targetSize)
    [h, w, c] = size(img);
    th = targetSize(1);
    tw = targetSize(2);

    % 크롭 (중앙 기준)
    if h > th
        yStart = floor((h - th)/2) + 1;
        img = img(yStart:yStart+th-1, :, :);
    end
    if w > tw
        xStart = floor((w - tw)/2) + 1;
        img = img(:, xStart:xStart+tw-1, :);
    end

    % 패딩 (중앙 기준)
    [h, w, c] = size(img);
    padTop = floor((th - h)/2);
    padBottom = ceil((th - h)/2);
    padLeft = floor((tw - w)/2);
    padRight = ceil((tw - w)/2);

    if isinteger(img)
        padVal = 0;
    else
        padVal = 0.0;
    end

    if c == 1
        out = padarray(img, [padTop, padLeft], padVal, 'pre');
        out = padarray(out, [padBottom, padRight], padVal, 'post');
    else
        out = zeros(th, tw, c, class(img));
        out(padTop+1:padTop+h, padLeft+1:padLeft+w, :) = img;
    end
end
