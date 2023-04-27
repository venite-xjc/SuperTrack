- world space positions $x^p\in \mathbb{R}^{3B}$
- velocities $\dot{x}^p\in \mathbb{R}^{3B}$
- accelerations $\ddot{x}^p\in \mathbb{R}^{3B}$
- rotations $x^r\in \mathbb{R}^{4B}$: in quaternions
- rotational velocities $\dot{x}^r\in \mathbb{R}^{3B}$: in an axis-angle vector and convert into rotational matrix by exp map.
- rotational accelerations $\ddot{x}^r\in \mathbb{R}^{3B}$: in an axis-angle vector to be integrated and just add to $\dot{x}^r$

You can learn rotational velocities and rotational accelerations from Section 4.1 of https://www.sedris.org/wg8home/Documents/WG80485.pdf and https://ocw.mit.edu/courses/16-07-dynamics-fall-2009/419be4d742e628d70acfbc5496eab967_MIT16_07F09_Lec25.pdf.

http://staffwww.dcs.shef.ac.uk/people/S.Maddock/publications/Motion%20Capture%20File%20Formats%20Explained.pdf explains what is bvh file clearly.

https://core.ac.uk/download/pdf/160483483.pdf explains why convert quaternions into the two-axis rotation matrix format.