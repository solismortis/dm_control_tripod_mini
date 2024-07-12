This is an example custom dm_control composer.Environment with 3D assets and video recording. When you make a 3D model yourself, make sure that pivot points of each part are where they attach to their parent parts and remember that mujoco's physics only works with convex shapes. Also, by default, mujoco uses degrees, but dm_control uses radians, so better use radians.

mujoco==2.3.5

dm-control==1.0.12
