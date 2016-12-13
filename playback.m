%% Load Ground Truth
v = VideoReader('IMG_4441.MOV');
offset = 420; % blaze it
ind = 0;
i = 1;

groundtruth = zeros(77, 384, 384, 3);

while(hasFrame(v))
    ind = ind+1;
    videoframe = readFrame(v);
    videoframe = imresize(videoframe(1:end-2*offset-1,:,:),[384,384]);
    if mod(ind,2) == 0
        groundtruth(i,:,:,:) = videoframe;
        i = i + 1;
    end
end


%%
% % close all
hc = figure();
set(hc,'Units','Points');
imscaling = 0.8;
set(hc,'Position',[650,550,350*3.1*imscaling,300*imscaling]);



% i = 48, 63, 67
i = 95;
% for i = 1:77
%     
%     vidframe = squeeze(frames_to_save(i,:,:,:));
%     subplot(1,3,1)
%     imagesc(vidframe)
%     subplot(1,3,2)
%     imagesc(vidframe)
%     subplot(1,3,3);
%     imagesc(vidframe)
%     pause(1/25);

    vidframe = squeeze(outputs(i,:,:,:));
    baseline = 0.5*squeeze(frames_to_save(i,:,:,:)) +...
        0.5*squeeze(frames_to_save(i+1,:,:,:));
    grt_frame = squeeze(groundtruth(i,:,:,:));
    cnn_loss = mean(mean(mean( sqrt( (vidframe - grt_frame).^2) )));
    baseline_loss = mean(mean(mean( sqrt( (double(baseline) - grt_frame).^2) )));
    subplot(1,3,1);
    imagesc(double(baseline)/255)
    title(sprintf('Before-After Overlay (Loss Per Pixel: %0.2f)',baseline_loss));
    set(gca,'xtick',[])
    set(gca,'ytick',[])
    
    subplot(1,3,2);
    imagesc(grt_frame/255)
    title('Ground Truth')
    set(gca,'xtick',[])
    set(gca,'ytick',[])
    
    subplot(1,3,3);
    imagesc(vidframe/255)
    title(sprintf('CNN Output (Loss Per Pixel: %0.2f)',cnn_loss))
    set(gca,'xtick',[])
    set(gca,'ytick',[])
    pause(1/25);
    
% end


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


%% Optical Flow Est stuff
opticFlow = opticalFlowFarneback;
before_grayscale = rgb2gray(squeeze(frames_to_save(67,:,:,:)));
after_grayscale = rgb2gray(squeeze(frames_to_save(68,:,:,:)));

for i = 1:68
    gray_scale = rgb2gray(squeeze(frames_to_save(i,:,:,:)));
    forward_flow = estimateFlow(opticFlow, gray_scale);
end

opticalFlowBackward = opticalFlowFarneback;
for i = fliplr(67:78)
    gray_scale = rgb2gray(squeeze(frames_to_save(i,:,:,:)));
    backward_flow = estimateFlow(opticFlow, gray_scale);
end


img_width = 384;
[X,Y] = meshgrid(1:img_width,1:img_width);
Xp = X - forward_flow.Vx/2; 
Yp = Y - forward_flow.Vy/2;
Xp = round(min(max(Xp, ones(size(Xp))),img_width*ones(size(Xp))));
Yp = round(min(max(Yp, ones(size(Yp))),img_width*ones(size(Yp))));
interp_frame = zeros(size(before_grayscale));
for x=1:img_width
    for y=1:img_width
        interp_frame(Yp(y,x),Xp(y,x)) = before_grayscale(y,x);
    end
end

Xp = X + backward_flow.Vx/2; 
Yp = Y + backward_flow.Vy/2;
Xp = round(min(max(Xp, ones(size(Xp))),img_width*ones(size(Xp))));
Yp = round(min(max(Yp, ones(size(Yp))),img_width*ones(size(Yp))));
interp_frame2 = zeros(size(after_grayscale));
for x=1:img_width
    for y=1:img_width
        interp_frame2(Yp(y,x),Xp(y,x)) = after_grayscale(y,x);
    end
end

interp_frame = 0.5*(interp_frame + interp_frame2);