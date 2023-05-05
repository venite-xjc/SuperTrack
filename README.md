My impletation of pipeline of paper: [SuperTrack: Motion Tracking for Physically Simulated Characters using Supervised Learning](https://theorangeduck.com/media/uploads/other_stuff/SuperTrack.pdf)


I use LAFAN1 as my database which you can get from [LAFAN1](https://github.com/ubisoft/ubisoft-laforge-animation-dataset).


## Paramaters presentataions in this repo
---
- world space positions $x^p\in \mathbb{R}^{3B}$
- velocities $\dot{x}^p\in \mathbb{R}^{3B}$
- accelerations $\ddot{x}^p\in \mathbb{R}^{3B}$
- rotations $x^r\in \mathbb{R}^{4B}$: in quaternions
- rotational velocities $\dot{x}^r\in \mathbb{R}^{3B}$: in an axis-angle vector and convert into rotational matrix by exp map.
- rotational accelerations $\ddot{x}^r\in \mathbb{R}^{3B}$: in an axis-angle vector to be integrated and just add to $\dot{x}^r$


## reference
---
There is other information that may help you reproduce this paper from scratch.

You can learn rotational velocities and rotational accelerations from Section 4.1 of [Orientation, Rotation, Velocity and Acceleration, and the SRM](https://www.sedris.org/wg8home/Documents/WG80485.pdf) and [Rigid Body Kinematics](https://ocw.mit.edu/courses/16-07-dynamics-fall-2009/419be4d742e628d70acfbc5496eab967_MIT16_07F09_Lec25.pdf).

[Motion Capture File Formats Explained](http://staffwww.dcs.shef.ac.uk/people/S.Maddock/publications/Motion%20Capture%20File%20Formats%20Explained.pdf) explains what is bvh file clearly.

[On the Continuity of Rotation Representations in Neural Networks](https://arxiv.org/pdf/1812.07035.pdf) explains why convert quaternions into the two-axis rotation matrix format.

---
Due to my uncertainty regarding how to check my code, there may be some errors here. Please use with caution.