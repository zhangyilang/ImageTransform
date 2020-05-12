## 反向图像变换
在本例中，我们通过在(1)中实现的局部仿射形变方法进行图像变化，将原图像中的一部分变换成另一种图像的样子

图像变换的步骤如下：
1. 确定两张图片中待变换区域于控制点的坐标
2. 通过两张图片相对应控制点，计算仿射变换矩阵
3. 通过仿射变换矩阵，将原图像的点对应到相应的目标图像相应位置上，通过插值方法得到该点的三通道值
4. 输出仿射变换后的图像

计算仿射矩阵和待变换区域的程序如下：<br/>
**form_transMap.m**
```matlab
function [transMap] = form_transMap(shape, goal_control)
    % 这里生成transmap, 
	% goal_control :目标图控制点, 类型为cell
	% shape: 目标图大小
    transMap = zeros(shape(1:2));
    
    for i = 1:size(goal_control,2)
        if size(goal_control{i},2) == 4
            % 矩形
            transMap(goal_control{i}(1):goal_control{i}(2), goal_control{i}(3):goal_control{i}(4)) = i;
        end
        if size(goal_control{i},2) == 6
            % 梯形
            Trapezoid = goal_control{i};
            for j = Trapezoid(1):Trapezoid(2)
            transMap(j, fix(Trapezoid(3) - (Trapezoid(3)-Trapezoid(5)) / (Trapezoid(2)-Trapezoid(1)) * (j-Trapezoid(1))) : fix((Trapezoid(6)-Trapezoid(4)) / (Trapezoid(2)-Trapezoid(1)) * (j-Trapezoid(1)) + Trapezoid(4))) = 3;
            end
    end
end
```
**form_transform.m**
```matlab
function [transform] = form_transform(shape_A, shape_swl, origin, map)
if length(origin) == 4
    % 形成二维坐标（,3）——(x,y,1)，然后求逆，这里是正方形框，故有四个参数
    for i = 1:2
        %归一化
        origin(i) = origin(i) - shape_A(1)/2;
        map(i) = map(i) - shape_swl(1)/2;
    end
    for i = 3:4
        origin(i) = origin(i) - shape_A(2)/2;
        map(i) = map(i) - shape_swl(2)/2;    
    end
    % 形成矩阵然后求A
    origin_matrix = [origin(1),origin(3),1; origin(1),origin(4),1; origin(2),origin(3),1; origin(2),origin(4),1]';
    map_matrix = [map(1),map(3),1; map(1),map(4),1; map(2),map(3),1; map(2),map(4),1]';
    % origin = transform * transmap
    transform = origin_matrix * pinv(map_matrix);
end
if length(origin) == 6
    %同样的，这里是6个点（梯形）的情况
    for i = 1:2
        origin(i) = origin(i) - shape_A(1)/2;
        map(i) = map(i) - shape_swl(1)/2;
    end
    for i = 3:6
        origin(i) = origin(i) - shape_A(2)/2;
        map(i) = map(i) - shape_swl(2)/2;    
    end
    % 构建控制点
    origin_matrix = [origin(1),origin(3),1; origin(1),origin(4),1;origin(1),origin(5),1; origin(1),origin(6),1; origin(2),origin(3),1; origin(2),origin(4),1;origin(2),origin(5),1; origin(2),origin(6),1;]';
    map_matrix = [map(1),map(3),1; map(1),map(4),1;map(1),map(5),1; map(1),origin(6),1; map(2),map(3),1; map(2),map(4),1;map(2),map(5),1; map(2),map(6),1;]';
    % 求逆得到仿射矩阵
    transform = origin_matrix * pinv(map_matrix);
end
```

得到仿射矩阵和变换区域之后，便可以进行图像变换
```matlab
% 清理运行环境
clear
clc
close all
% 读取原本的图像和待改变的图像，并标记相应的区域
A = imread('../test_images/B.jpg');
A = A(1:545,:,:);
A = double(A) / 256;
A_control = {[150, 175, 205, 235], [150, 174, 245, 278], [175,198,232,245,225,255], [190,220,220,260]};

goal = imread('../test_images/dlrb.jpg');
goal = double(goal) / 256;
goal_control = {[220, 270, 160, 230], [195, 265, 260, 330], [270,320,240,260,220,290], [330, 365, 210, 295]};

% 计算相关transform矩阵
shape_A = size(A);
shape_goal = size(goal);
transMap = form_transMap(shape_goal, goal_control);
for i = 1:size(goal_control,2)
    transform{i} = form_transform(shape_A, shape_goal, A_control{i}, goal_control{i});
end
% 进行图像变换
localAffine_inv(A,transform,transMap);

```

