%yuv4mpeg2mov creates a Matlab-Movie from a YUV4MPEG-file (.y4m extension).
%	yuv2mov('Filename') reads the specified file.
%
%   This function implments only a subset of the YUV4MPEG standard.
%   Specifically planar interpolation. The 420, 422 and 444 smapling
%   models. Only frames without parameters.
%   See http://wiki.multimedia.cx/index.php?title=YUV4MPEG2
%	
%	Filename --> Name of File (e.g. 'Test.yuv')
%
%   This is based on yuv2mov by Dima Pröfrock.
%   See https://www.mathworks.com/matlabcentral/fileexchange/11252-yuv-file-to-matlab-movie
%
%example: mov = yuv4mpeg2mov('Test.y4m');
%example: [mov, fields] = yuv4mpeg2mov('Test.y4m̈́');
%example: [mov, fields, accepted] = yuv4mpeg2mov('Test.y4m̈́');

function [mov, fields, accepted] = yuv4mpeg2mov(File)

    mov = struct('cdata', [], 'colormap', []);

    filep = dir(File); 
    fileBytes = filep.bytes; %Filesize
    clear filep;
    
	inFileId = fopen(File, 'r');
    
	[header, endOfHeaderPos] = textscan(inFileId, '%s', 1, 'delimiter', '\n');

    [fields, accepted] = readYuv4MpegHeader(header);
    fields.frameCount = 0;
    
	frameLength = fields.width * fields.height;
    fwidth = 0.5;
    fheight= 0.5;
    
    if strcmp(fields.colourSpace, 'C420')
		frameLength = (frameLength * 3) / 2;
        fwidth = 0.5;
        fheight= 0.5;
	elseif strcmp(fields.colourSpace, 'C422')
		frameLength = (frameLength * 2);
        fwidth = 0.5;
        fheight= 1;
	elseif strcmp(fields.colourSpace, 'C444')
		frameLength = (frameLength * 3);
        fwidth = 1;
        fheight= 1;
    end

    % compute number of frames, a frame starts with FRAME and then some
    % possible parameters and finally ending with the byte 0x0A.
    % Assume no parameters
    frameCount = (fileBytes - endOfHeaderPos)/(6 + frameLength);
    
    if mod(frameCount,1) ~= 0
        display('Error: wrong resolution, format or filesize');
        accepted = false;
    else
        fields.frameCount = frameCount;
        
        %h = waitbar(0,'Please wait ... ');
        
        %read YUV-Frames
        for framenumber = 1:frameCount
            %waitbar(framenumber/frameCount,h);
            
            fread(inFileId, 6, 'uchar');
            data = fread(inFileId, frameLength, 'uchar');
            
            YUV = readYUVFrame(data,fields.width, fields.height,...
                fheight, fwidth);
            RGB = ycbcr2rgb(YUV); %Convert YUV to RGB
            mov(framenumber).cdata = RGB;
            mov(framenumber).colormap = [];
        end
        
        %close(h);
    end
    
	fclose(inFileId);
end

function [fields, accepted] = readYuv4MpegHeader(header)
	colourSpace = 'C420';

	parts = strsplit(char(header{1}), ' ');

    accepted = strcmp(parts{1}, 'YUV4MPEG2');
	assert(accepted, 'file must start with YUV4MPEG2');

	width = textscan(parts{2}, 'W%n');
	height = textscan(parts{3}, 'H%n');

	fpsFraction = textscan(parts{4}, 'F%n:%n');
	fps = fpsFraction{1} / fpsFraction{2};

	interlacing = textscan(parts{5}, 'I%c');

	pixelAspectFraction = textscan(parts{6}, 'A%n:%n');
    pixelAspectRatio = pixelAspectFraction{1} / pixelAspectFraction{2};
    
    if size(parts,2) > 6 && strfind(parts{7}, 'C')
        colourSpace = parts{7};
    end
    
    fields = struct('width', width,...
        'height', height,...
        'fps', fps,...
        'interlacing',interlacing,...
        'pixelAspectRatio',pixelAspectRatio,...
        'colourSpace',colourSpace);
end

function YUV = readYUVFrame(data, width, height, Teil_h, Teil_b)

    % get size of U and V
    width_h = width * Teil_b;
    height_h = height * Teil_h;
      
    % create Y-Matrix
    YMatrix = data(1:width * height);
    YMatrix = int16(reshape(YMatrix, width, height)');
    offset = width * height+1;

    % create U- and V- Matrix
    if Teil_h == 0
        UMatrix = 0;
        VMatrix = 0;
    else
        UMatrix = data(offset:offset+(width_h * height_h)-1);
        UMatrix = int16(UMatrix);
        UMatrix = reshape(UMatrix, width_h, height_h).';
        offset = offset + (width_h * height_h);
        
        VMatrix = data(offset:offset+(width_h * height_h)-1);
        VMatrix = int16(VMatrix);
        VMatrix = reshape(VMatrix,width_h, height_h).';
        %offset = offset + (width_h * height_h) + 1;

    end
    % compose the YUV-matrix:
    YUV(1:height,1:width,1) = YMatrix;
    
    if Teil_h == 0
        YUV(:,:,2) = 127;
        YUV(:,:,3) = 127;
    end
    % consideration of the subsampling of U and V
    if Teil_b == 1
        UMatrix1(:,:) = UMatrix(:,:);
        VMatrix1(:,:) = VMatrix(:,:);
    
    elseif Teil_b == 0.5        
        UMatrix1(1:height_h,1:width) = int16(0);
        UMatrix1(1:height_h,1:2:end) = UMatrix(:,1:1:end);
        UMatrix1(1:height_h,2:2:end) = UMatrix(:,1:1:end);
 
        VMatrix1(1:height_h,1:width) = int16(0);
        VMatrix1(1:height_h,1:2:end) = VMatrix(:,1:1:end);
        VMatrix1(1:height_h,2:2:end) = VMatrix(:,1:1:end);
    
    elseif Teil_b == 0.25
        UMatrix1(1:height_h,1:width) = int16(0);
        UMatrix1(1:height_h,1:4:end) = UMatrix(:,1:1:end);
        UMatrix1(1:height_h,2:4:end) = UMatrix(:,1:1:end);
        UMatrix1(1:height_h,3:4:end) = UMatrix(:,1:1:end);
        UMatrix1(1:height_h,4:4:end) = UMatrix(:,1:1:end);
        
        VMatrix1(1:height_h,1:width) = int16(0);
        VMatrix1(1:height_h,1:4:end) = VMatrix(:,1:1:end);
        VMatrix1(1:height_h,2:4:end) = VMatrix(:,1:1:end);
        VMatrix1(1:height_h,3:4:end) = VMatrix(:,1:1:end);
        VMatrix1(1:height_h,4:4:end) = VMatrix(:,1:1:end);
    end
    
    if Teil_h == 1
        YUV(:,:,2) = UMatrix1(:,:);
        YUV(:,:,3) = VMatrix1(:,:);
        
    elseif Teil_h == 0.5        
        YUV(1:height,1:width,2) = int16(0);
        YUV(1:2:end,:,2) = UMatrix1(:,:);
        YUV(2:2:end,:,2) = UMatrix1(:,:);
        
        YUV(1:height,1:width,3) = int16(0);
        YUV(1:2:end,:,3) = VMatrix1(:,:);
        YUV(2:2:end,:,3) = VMatrix1(:,:);
        
    elseif Teil_h == 0.25
        YUV(1:height,1:width,2) = int16(0);
        YUV(1:4:end,:,2) = UMatrix1(:,:);

        YUV(2:4:end,:,2) = UMatrix1(:,:);
        YUV(3:4:end,:,2) = UMatrix1(:,:);
        YUV(4:4:end,:,2) = UMatrix1(:,:);
        
        YUV(1:height,1:width) = int16(0);
        YUV(1:4:end,:,3) = VMatrix1(:,:);
        YUV(2:4:end,:,3) = VMatrix1(:,:);
        YUV(3:4:end,:,3) = VMatrix1(:,:);
        YUV(4:4:end,:,3) = VMatrix1(:,:);
    end
    YUV = uint8(YUV);
end