function [transMap] = form_transMap(shape, left_eye, right_eye, nose, mouth)
    %这里生成transmap， 左眼，右眼，鼻子，嘴分别为1，2，3，4
    transMap = zeros(shape(1:2));
    transMap(left_eye(1):left_eye(2), left_eye(3):left_eye(4)) = 1;
    transMap(right_eye(1):right_eye(2), right_eye(3):right_eye(4)) = 2;
    for j = nose(1):nose(2)
        % 这里鼻子是一个梯形
        transMap(j, fix(nose(3) - (nose(3)-nose(5)) / (nose(2)-nose(1)) * (j-nose(1))) : fix((nose(6)-nose(4)) / (nose(2)-nose(1)) * (j-nose(1)) + nose(4))) = 3;
    end
    transMap(mouth(1):mouth(2), mouth(3):mouth(4)) = 4;
end