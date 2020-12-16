# airsim_DDPG_Quadcopter_hovering
实现了利用DDPG算法来控制四旋翼悬停
视景仿真环境：Airsim
编译环境：python3.7.3 + torch1.7.0
state: x,y,z轴上的速度 + 3个姿态角 + 位置
action: continuous, 四个电机的电压值，范围[0.0, 1.0]
reward: 姿态角越小越好，离目标点距离越近越好
