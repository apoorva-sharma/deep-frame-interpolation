v = VideoReader('../football_cif.ogg');
offset = 128;
ind = 0;
while(hasFrame(v))
    ind = ind+1;
    videoframe = readFrame(v);
    videoframe = imresize(videoframe(:,1:end-offset,:),[384,384]);
    imwrite(videoframe, sprintf('./football/Frame%04d.png',ind));
end