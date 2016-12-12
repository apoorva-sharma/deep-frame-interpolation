%%

close all

for i = 1:77
    
    vidframe = squeeze(frames_to_save(i,:,:,:));
    subplot(1,2,1)
    imagesc(vidframe)
    subplot(1,2,2)
    imagesc(vidframe)
    pause(1/25);
    
    vidframe = squeeze(outputs(i,:,:,:));
    baseline = 0.5*squeeze(frames_to_save(i,:,:,:)) +...
        0.5*squeeze(frames_to_save(i+1,:,:,:));
    subplot(1,2,1);
    imagesc(vidframe/255)
    subplot(1,2,2);
    imagesc(baseline)
    pause(1/25);
    
end

%%

close all

for i = 1:77
    
    vidframe = squeeze(saved_frames(i,:,:,:));
    subplot(1,2,1)
    imagesc(vidframe)
    
    median = squeeze(medians(i,:,:,:));
    subplot(1,2,2);
    imagesc(median)
    pause(1/25);
    
end