function [] = saveMNISTImages(images, n, k)
% saveMNISImages Saves the first every k-th image of the MNIST training
% data set up to n images.

    for i = 1: n
        imwrite(reshape(images(:,i*k), 28, 28), strcat('MNIST/', num2str(i*k), '.png'));
    end;
end

