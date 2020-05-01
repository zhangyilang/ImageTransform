function [transform] = form_transform(shape_A, shape_swl, origin, map)
if length(origin) == 4
    for i = 1:2
        origin(i) = origin(i) - shape_A(1)/2;
        map(i) = map(i) - shape_swl(1)/2;
    end
    for i = 3:4
        origin(i) = origin(i) - shape_A(2)/2;
        map(i) = map(i) - shape_swl(2)/2;    
    end
    
    origin_matrix = [origin(1),origin(3),1; origin(1),origin(4),1; origin(2),origin(3),1; origin(2),origin(4),1]';
    map_matrix = [map(1),map(3),1; map(1),map(4),1; map(2),map(3),1; map(2),map(4),1]';
    transform = origin_matrix * pinv(map_matrix);
end
if length(origin) == 6
    for i = 1:2
        origin(i) = origin(i) - shape_A(1)/2;
        map(i) = map(i) - shape_swl(1)/2;
    end
    for i = 3:6
        origin(i) = origin(i) - shape_A(2)/2;
        map(i) = map(i) - shape_swl(2)/2;    
    end
    origin_matrix = [origin(1),origin(3),1; origin(1),origin(4),1;origin(1),origin(5),1; origin(1),origin(6),1; origin(2),origin(3),1; origin(2),origin(4),1;origin(2),origin(5),1; origin(2),origin(6),1;]';
    map_matrix = [map(1),map(3),1; map(1),map(4),1;map(1),map(5),1; map(1),origin(6),1; map(2),map(3),1; map(2),map(4),1;map(2),map(5),1; map(2),map(6),1;]';
    transform = origin_matrix * pinv(map_matrix);
end