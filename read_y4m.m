mov = yuv4mpeg2mov('../football_cif.y4m');
[h,w,c] = size(mov(1).cdata);
offset = w - h;
for ind = 1:numel(mov)
    videoframe = mov(ind).cdata;
    videoframe = imresize(videoframe(:,1:end-offset,:),[192,192]);
    imwrite(videoframe, sprintf('./football/Frame%04d.png',ind));
end