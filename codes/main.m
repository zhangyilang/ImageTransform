clc;clear;
A = imread('D:\Documents\课程介绍\数据可视化\TeamProject1\zc.jpg');
A = double(A);
image(A)
left_eyeA = [384, 400, 568, 620];
right_eyeA = [366, 390, 665, 715];
noseA = [380,445,630,660,620,680];
mouthA = [460,490,600,700];

swl = imread('D:\Documents\课程介绍\数据可视化\TeamProject1\swl.jpg');
swl = double(swl);
image(swl)
left_eye_swl = [320, 350, 280, 330];
right_eye_swl = [300, 330, 370, 430];
nose_swl = [300,400,340,360,340,400];
mouth_swl = [410, 440, 320, 425];

shape_A = size(A);
shape_swl = size(swl);
transMap = form_transMap(shape_swl, left_eye_swl, right_eye_swl, nose_swl, mouth_swl);
transform{1} = form_transform(shape_A, shape_swl, left_eyeA, left_eye_swl);
transform{2} = form_transform(shape_A, shape_swl, right_eyeA, right_eye_swl);
transform{3} = form_transform(shape_A, shape_swl, noseA, nose_swl);
transform{4} = form_transform(shape_A, shape_swl, mouthA, mouth_swl);
localAffine_inv(A,transform,transMap)


