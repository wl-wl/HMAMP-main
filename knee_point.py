sequence_index=[100, 237, 239, 401, 508, 552, 794, 797, 800, 877, 1119, 1133, 1601, 1745, 1911, 1954, 1990, 2218, 2461, 2552, 2671, 2761, 2866, 2993, 3104, 3284, 3374, 3648, 3733, 3772, 3778, 3985, 4057, 4081]
point_x=[-0.6409, -0.5052,  1.241,  -0.6903, -0.4413,  0.2783 ,-0.4643 ,-0.6628, -0.4473,
 -0.0787, -0.65 ,   0.7196 , 0.4482, -0.4642 ,-0.4328,  0.563 ,  0.0431, -0.385,
 -0.6343 ,-0.4072, -0.5008 , 0.04 ,  -0.5388 ,-0.4362,  0.9413 , 0.0713 , 0.1942,
  0.5399, -0.2906, -0.4305, -0.2128,  0.5815 , 0.6631, -0.3883]
point_y=[0.461 , 0.3686 ,0.1298, 0.515,  0.3086, 0.1854, 0.3446, 0.4744, 0.3381, 0.2315,
 0.4651 ,0.1349 ,0.1848, 0.3422, 0.2877, 0.1758 ,0.2186, 0.2558, 0.3872 ,0.2722,
 0.3453 ,0.2193 ,0.3769 ,0.2997, 0.1301, 0.2069 ,0.2025 ,0.18 ,  0.2504, 0.2798,
 0.2354 ,0.1641 ,0.1558 ,0.2657]
#KNEE [-0.6343, -0.5008, -0.4642, -0.4305, -0.385, 0.0713, 0.5815] [0.3872, 0.3453, 0.3422, 0.2798, 0.2558, 0.2069, 0.1641]
sorted_coords = sorted(zip(point_x, point_y,sequence_index))

# 拆分回两个列表
point_x, point_y ,sequence_index= zip(*sorted_coords)


maxx=max(point_x)
index_x=point_x.index(maxx)

maxy=max(point_y)
index_y=point_y.index(maxy)

first_point=[point_x[index_x],point_y[index_x]]
second_point=[point_x[index_y],point_y[index_y]]

x1=point_x[index_x]
y1=point_y[index_x]

def angle_between_three_points(P1, P2, P3):
    """
    计算点 P2 和点 P1、P3 之间的角度

    :param P1: tuple, 第一个点的坐标 (x1, y1)
    :param P2: tuple, 中间点的坐标 (x2, y2)
    :param P3: tuple, 第二个点的坐标 (x3, y3)
    :return: float, 角度的大小（单位：度）
    """
    x1, y1 = P1
    x2, y2 = P2
    x3, y3 = P3

    # 向量 u 和 v
    u = (x1 - x2, y1 - y2)
    v = (x3 - x2, y3 - y2)

    # 计算点积
    dot_product = u[0] * v[0] + u[1] * v[1]

    # 计算向量的模
    magnitude_u = math.sqrt(u[0] ** 2 + u[1] ** 2)
    magnitude_v = math.sqrt(v[0] ** 2 + v[1] ** 2)

    # 计算余弦值
    cos_theta = dot_product / (magnitude_u * magnitude_v)

    # 计算角度（弧度）
    angle_radians = math.acos(cos_theta)

    # 将角度转换为度
    angle_degrees = math.degrees(angle_radians)

    return angle_degrees

def is_point_left_of_line(P1, P2, P):
    """
    判断点 P 是否在由 P1 和 P2 确定的直线的左侧。

    :param P1: tuple, 点 P1 的坐标 (x1, y1)
    :param P2: tuple, 点 P2 的坐标 (x2, y2)
    :param P: tuple, 点 P 的坐标 (x, y)
    :return: bool, 如果点 P 在直线的左侧返回 True，否则返回 False
    """
    x1, y1 = P1
    x2, y2 = P2
    x, y = P

    # 计算向量 v1 和 v2
    v1 = (x2 - x1, y2 - y1)
    v2 = (x - x1, y - y1)

    # 计算叉积
    cross_product = v1[0] * v2[1] - v1[1] * v2[0]

    return cross_product
dis_list=[]
angle_list=[]
knee_x=[]
knee_y=[]
for i in range(len(point_x)):
 if i != 0 and i != len(point_x) - 1:
     point=[point_x[i],point_y[i]]
     dist=is_point_left_of_line([point_x[i-1],point_y[i-1]],[point_x[i+1],point_y[i+1]],point)
     angle=angle_between_three_points([point_x[i-1],point_y[i-1]],point,[point_x[i+1],point_y[i+1]])
     dis_list.append(dist)
     angle_list.append(angle)

 else:
     dis_list.append(0)
     angle_list.append(180)
print(dis_list)
print(angle_list)
sequence_list=[]
for i in range(len(dis_list)):
 if dis_list[i]<0 and angle_list[i]<160:
   knee_x.append(point_x[i])
   knee_y.append(point_y[i])
   sequence_list.append(sequence_index[i])

print(knee_x)
print(knee_y)
