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