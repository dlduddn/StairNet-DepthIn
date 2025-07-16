function R = rodriguesRotation(normalVec, targetAxis)
    v = cross(normalVec, targetAxis);
    s = norm(v); c = dot(normalVec, targetAxis);
    if s == 0
        R = eye(3);
        return;
    end
    vx = [  0, -v(3), v(2);
          v(3),  0, -v(1);
         -v(2), v(1),  0 ];
    R = eye(3) + vx + vx^2 * ((1 - c) / s^2);
end
