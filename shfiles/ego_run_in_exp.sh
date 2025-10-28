sudo chmod 777 /dev/tty* & sleep 2; #电脑USB接口权限
roslaunch mavros px4.launch & sleep 10; #系统环境 mavros 是电脑与飞控连接的功能包  
rosrun mavros mavcmd long 511 105 2500 0 0 0 0 0 & sleep 2; #调整无人机发布频率的参数
rosrun mavros mavcmd long 511 31 2500 0 0 0 0 0 & sleep 2; #调整无人机发布频率的参数

roslaunch px4ctrl run_ctrl.launch & sleep 2;

roslaunch ego_planner run_in_exp.launch & sleep 2;
roslaunch ego_planner rviz.launch & sleep 10;

sh shfiles/takeoff.sh & sleep 1;

wait;

