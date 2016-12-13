medians = repmat(reshape(median(reshape(downsampled(:,:,:,1:3),size(downsampled,1),[]),1),size(downsampled(1,:,:,:))),size(downsampled,1),1,1,1);

i = 48

figure(1)
subplot(2,3,1)
imagesc((0.5*squeeze(training_inputs(i,:,:,1:3))+0.5*squeeze(training_inputs(i,:,:,4:6)))/255);
title('Training Inputs');
subplot(2,3,2)
imagesc((squeeze(training_targets(i,:,:,:)))/255);
title('Training Target');
subplot(2,3,3)
imagesc((squeeze(train_outputs(i,:,:,:)))/255);
title('CNN Output');

subplot(2,3,4)
imagesc((0.5*squeeze(test_inputs(i,:,:,1:3))+0.5*squeeze(test_inputs(i,:,:,4:6)))/255);
title('Test Inputs');
subplot(2,3,5)
imagesc((squeeze(test_targets(i,:,:,:)))/255);
title('Test Target');
subplot(2,3,6)
imagesc((squeeze(test_outputs(i,:,:,:)))/255);
title('CNN Output');
