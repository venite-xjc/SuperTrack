- world space positions $x^p\in \mathbb{R}^{3B}$
- velocities $\dot{x}^p\in \mathbb{R}^{3B}$
- accelerations $\ddot{x}^p\in \mathbb{R}^{3B}$
- rotations $x^r\in \mathbb{R}^{4B}$: in quaternions
- rotational velocities $\dot{x}^r\in \mathbb{R}^{3B}$: in an axis-angle vector and convert into rotational matrix by exp map.
- rotational accelerations $\ddot{x}^r\in \mathbb{R}^{3B}$: in an axis-angle vector to be integrated and just add to $\dot{x}^r$

You can learn rotational velocities and rotational accelerations from Section 4.1 of https://www.sedris.org/wg8home/Documents/WG80485.pdf

http://staffwww.dcs.shef.ac.uk/people/S.Maddock/publications/Motion%20Capture%20File%20Formats%20Explained.pdf explains what is bvh file clearly.

https://www.cs.cityu.edu.hk/~howard/Teaching/CS4185-5185-2007-SemA/Group12/BVH.html：Motion capture data is a two-dimensional representation of motion that takes place in a three-dimensional world. 为什么是二维表示