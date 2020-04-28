function [outputImg] = localAffine_inv(inputImg,transform,transMap,varargin)
%LOCALAFFINE_INV    �ֲ�����任������ͼ�任������Ŀ��ͼ����ӳ�䵽����ͼ���꣬
%                   ������ͼ��ֵ����������Ŀ��ͼ����Ӧ��λ��
% ���룺
%   ��inputImg	���뵥ͨ��/��ͨ��ͼ�����
%   ��transform	Ԫ��������Ԫ����ÿ��Ԫ��Ϊÿ�������Ӧ�� 3 * 3 �������任����;
%                �任��ͼƬ����Ϊԭ��
%   ��transMap   Ŀ��ͼ��ÿ�����ص���Ҫ���еķ������任��0��ʾ�任������ĵ㣬
%                ����0������k��ʾ��������Ҫ����transform{k}�ķ���任
%   ��varargin   ���������ֶε�Ĭ�ϲ���
%       - 'dist_e'      ���ݾ������Ȩֵʱ��ָ����Ĭ��Ϊ1
%       - 'visualize'   �����Ͳ������Ƿ���п��ӻ���Ĭ��Ϊtrue
% �����
%   ��outputImg  �����ֲ�����任������ͼ�任����ͼ��

%% ��ʼ��
% �����������
p = inputParser;                                    % ������ʵ��
p.addRequired('img',@(x)ismatrix(x)||ndims(x)==3)   % ��ͨ��/��ͨ��ͼ�����
p.addRequired('transform',@(x)iscell(x))            % Ԫ������
p.addRequired('transMap',@(x)ismatrix(x))           % ��ά����
p.addParameter('dist_e',1,@(x)isscalar(x));         % ����
p.addParameter('visualize',true,@(x)islogical(x));  % ��������
p.parse(inputImg,transform,transMap,varargin{:});   % ����

% ��ȡͼ���С��ͨ����
[H,W,C] = size(inputImg);
% ����ռ�
outputImg = zeros(H,W,C);

% �������Ļ��������
coord = zeros([H,W,2]);     
coord(:,:,1) = repmat((1:H)'-H/2,[1,W]);
coord(:,:,2) = repmat((1:W)-W/2,[H,1]);
transMap = repmat(transMap,[1,1,2]);    % ��transMapά����coordͳһ���������

% ����任������������
numTrans = numel(transform);
regionCenter = zeros(numTrans,1,2);   
for t = 1:numTrans
    regionCenter(t,1,:) = mean(reshape(coord(transMap == t),[],2),1);
end

%% ����ֲ������ڵķ���任
for t = 1:numTrans
    originCoord = reshape(coord(transMap == t),[],2)';  % ��Ҫ��t�ֱ任��Ŀ��ͼ���ص�����
    numCoord = size(originCoord,2);
    transCoord = transform{t} * [originCoord;ones(1,numCoord)];     % ����任
    transCoord = transCoord(1:2,:);                     % ȥ�����һ�е�1
    for i = 1:numCoord
        dst = num2cell(originCoord(:,i) + [H/2;W/2]);   % ��Ҫ��������
        outputImg(dst{:},:) = linearInterp(inputImg,transCoord(:,i)+[H/2;W/2]); % ���
    end
end

%% ����ֲ�������ļ�Ȩ����任
originCoord = reshape(coord(transMap == 0),1,[],2);
numCoord = size(originCoord,2);
% ��������������ĵ�����������ĵ�֮��ľ���
dist_power = repmat(originCoord,[numTrans,1,1]) - repmat(regionCenter,[1,numCoord,1]);
dist_power = sum(dist_power.^2,3) .^ (1 / 2 * p.Results.dist_e);    % �������
dist_power = 1 ./ dist_power;                                       % ȡ����

% ���ݾ�������Ȩ����任
originCoord = reshape(originCoord,[],2)';
transCoord = zeros(2,numCoord);         % ��ʼ��
for t = 1:numTrans
    tmpCoord = transform{t} * [originCoord;ones(1,numCoord)];           % ����任
    weight = dist_power(t,:) ./ sum(dist_power,1);                      % ����Ȩ��
    transCoord = transCoord + tmpCoord(1:2,:) .* repmat(weight,[2,1]);  % ȥ�����һ�е�1����Ȩ�����
end

% ���ݷ���任�������������
for i = 1:numCoord
    dst = num2cell(originCoord(:,i) + [H/2;W/2]);                       % ��Ҫ��������
    outputImg(dst{:},:) = linearInterp(inputImg,transCoord(:,i)+[H/2;W/2]); % ���
end

%% ���ӻ�
if p.Results.visualize
    % ԭͼ
    subplot(1,2,1)
    imshow(inputImg)
    title('Original Image')
    
    % ���ڷ���ֲ�����任���ͼƬ
    subplot(1,2,2)
    imshow(outputImg)
    title('Transformed Image')
end

end

