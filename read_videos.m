v = VideoReader('IMG_4438.MOV');
offset = 420; % blaze it
ind = 0;
while(hasFrame(v))
    ind = ind+1;
    videoframe = readFrame(v);
    videoframe = imresize(videoframe(1:end-2*offset-1,:,:),[384,384]);
    imwrite(videoframe, sprintf('./SampleVid/Frame%04d.png',ind));
end