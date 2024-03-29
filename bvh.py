
# reference: https://github.com/20tab/bvh-python

import numpy as np
import re
from transforms3d.euler import euler2mat, mat2euler
from transforms3d.axangles import axangle2mat, mat2axangle
from transforms3d.quaternions import mat2quat, quat2mat
from tqdm import tqdm
import pytorch3d.transforms as pyt
import torch

class BvhJoint:
    def __init__(self, name, parent):
        self.name = name
        self.parent = parent
        self.offset = np.zeros(3)
        self.channels = []
        self.children = []

    def add_child(self, child):
        self.children.append(child)

    def __repr__(self):
        return self.name

    def position_animated(self):
        return any([x.endswith('position') for x in self.channels])

    def rotation_animated(self):
        return any([x.endswith('rotation') for x in self.channels])


class Bvh:
    def __init__(self):
        self.joints = {}
        self.root = None
        self.keyframes = None
        self.frames = 0
        self.fps = 0
        self.world_space_p = None

    def _parse_hierarchy(self, text):
        lines = re.split('\\s*\\n+\\s*', text)

        joint_stack = []

        for line in lines:
            words = re.split('\\s+', line)
            instruction = words[0]

            if instruction == "JOINT" or instruction == "ROOT":
                parent = joint_stack[-1] if instruction == "JOINT" else None
                joint = BvhJoint(words[1], parent)
                self.joints[joint.name] = joint
                if parent:
                    parent.add_child(joint)
                joint_stack.append(joint)
                if instruction == "ROOT":
                    self.root = joint
            elif instruction == "CHANNELS":
                for i in range(2, len(words)):
                    joint_stack[-1].channels.append(words[i])
            elif instruction == "OFFSET":
                for i in range(1, len(words)):
                    joint_stack[-1].offset[i - 1] = float(words[i])
            elif instruction == "End":
                joint = BvhJoint(joint_stack[-1].name + "_end", joint_stack[-1])
                joint_stack[-1].add_child(joint)
                joint_stack.append(joint)
                self.joints[joint.name] = joint
            elif instruction == '}':
                joint_stack.pop()

    def _add_pose_recursive(self, joint, offset, poses):
        pose = joint.offset + offset
        poses.append(pose)

        for c in joint.children:
            self._add_pose_recursive(c, pose, poses)

    def plot_hierarchy(self):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import axes3d, Axes3D
        poses = []
        self._add_pose_recursive(self.root, np.zeros(3), poses)
        pos = np.array(poses)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(pos[:, 0], pos[:, 2], pos[:, 1])
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        plt.show()

    def parse_motion(self, text):
        lines = re.split('\\s*\\n+\\s*', text)

        frame = 0
        for line in lines:
            if line == '':
                continue
            words = re.split('\\s+', line)

            if line.startswith("Frame Time:"):
                self.fps = round(1 / float(words[2]))
                continue
            if line.startswith("Frames:"):
                self.frames = int(words[1])
                continue

            if self.keyframes is None:
                self.keyframes = np.empty((self.frames, len(words)), dtype=np.float32)

            for angle_index in range(len(words)):
                self.keyframes[frame, angle_index] = float(words[angle_index])

            frame += 1

    def parse_string(self, text):
        hierarchy, motion = text.split("MOTION")
        self._parse_hierarchy(hierarchy)
        self.parse_motion(motion)

    def _extract_rotation(self, frame_pose, index_offset, joint):
        local_rotation = np.zeros(3)
        for channel in joint.channels:
            if channel.endswith("position"):
                continue
            if channel == "Xrotation":
                local_rotation[0] = frame_pose[index_offset]
            elif channel == "Yrotation":
                local_rotation[1] = frame_pose[index_offset]
            elif channel == "Zrotation":
                local_rotation[2] = frame_pose[index_offset]
            else:
                raise Exception(f"Unknown channel {channel}")
            index_offset += 1

        local_rotation = np.deg2rad(local_rotation)
        M_rotation = np.eye(3)
        for channel in joint.channels:
            if channel.endswith("position"):
                continue

            if channel == "Xrotation":
                euler_rot = np.array([local_rotation[0], 0., 0.])
            elif channel == "Yrotation":
                euler_rot = np.array([0., local_rotation[1], 0.])
            elif channel == "Zrotation":
                euler_rot = np.array([0., 0., local_rotation[2]])
            else:
                raise Exception(f"Unknown channel {channel}")

            M_channel = euler2mat(*euler_rot)

            M_rotation = M_rotation.dot(M_channel)

        return M_rotation, index_offset

    def _extract_position(self, joint, frame_pose, index_offset):
        offset_position = np.zeros(3)
        for channel in joint.channels:
            if channel.endswith("rotation"):
                continue
            if channel == "Xposition":
                offset_position[0] = frame_pose[index_offset]
            elif channel == "Yposition":
                offset_position[1] = frame_pose[index_offset]
            elif channel == "Zposition":
                offset_position[2] = frame_pose[index_offset]
            else:
                raise Exception(f"Unknown channel {channel}")
            index_offset += 1

        return offset_position, index_offset

    def _recursive_apply_frame(self, joint, frame_pose, index_offset, p, r, lr, M_parent, p_parent):
        if joint.position_animated():
            offset_position, index_offset = self._extract_position(joint, frame_pose, index_offset)
        else:
            offset_position = np.zeros(3)

        if len(joint.channels) == 0:
            joint_index = list(self.joints.values()).index(joint)
            p[joint_index] = p_parent + M_parent.dot(joint.offset)
            # r[joint_index] = np.rad2deg(mat2euler(M_parent))
            r[joint_index] = mat2quat(M_parent)
            return index_offset

        if joint.rotation_animated():
            M_rotation, index_offset = self._extract_rotation(frame_pose, index_offset, joint)
        else:
            M_rotation = np.eye(3)

        M = M_parent.dot(M_rotation)
        position = p_parent + M_parent.dot(joint.offset) + offset_position

        # rotation = np.rad2deg(mat2euler(M))
        rotation = mat2quat(M)
        # local_rotation = np.rad2deg(mat2euler(M_rotation))
        local_rotation = mat2quat(M_rotation)
        joint_index = list(self.joints.values()).index(joint)
        p[joint_index] = position
        r[joint_index] = rotation
        lr[joint_index] = local_rotation

        for c in joint.children:
            index_offset = self._recursive_apply_frame(c, frame_pose, index_offset, p, r, lr, M, position)

        return index_offset

    def frame_pose(self, frame):
        p = np.empty((len(self.joints), 3))
        r = np.empty((len(self.joints), 4))
        lr = np.empty((len(self.joints), 4))

        frame_pose = self.keyframes[frame]
        M_parent = np.zeros((3, 3))
        M_parent[0, 0] = 1
        M_parent[1, 1] = 1
        M_parent[2, 2] = 1
        self._recursive_apply_frame(self.root, frame_pose, 0, p, r, lr, M_parent, np.zeros(3))

        return p, r, lr

    def all_frame_poses(self):
        p = np.empty((self.frames, len(self.joints), 3))
        r = np.empty((self.frames, len(self.joints), 4))
        lr = np.empty((self.frames, len(self.joints), 4))

        print("loading ", self.path, "...")
        for frame in range(len(self.keyframes)):
            p[frame], r[frame], lr[frame] = self.frame_pose(frame)

        self.world_space_p = torch.from_numpy(p) #position in world space of each body in each frame

        self.world_space_vp = np.empty((self.frames, len(self.joints), 3))#velocities in world space
        self.world_space_vp[:-1, :, :] = np.diff(p, axis = 0) 
        self.world_space_vp = self.world_space_vp/(1/self.fps)
        self.world_space_vp[-1, :, :] = self.world_space_vp[-2, :, :]
        self.world_space_vp = torch.from_numpy(self.world_space_vp)

        self.world_space_r = torch.from_numpy(r) #rotation in world space, represented in quaternion  

        world_space_r1 = self.world_space_r[:-1, :, :]
        world_space_r2 = self.world_space_r[1:, :, :]
        matrix1 = pyt.quaternion_to_matrix(world_space_r1)
        matrix2 = pyt.quaternion_to_matrix(world_space_r2)
        dmatrix = torch.matmul(matrix2, torch.transpose(matrix1, -1, -2))/(1/self.fps)
        self.world_space_vr = pyt.matrix_to_axis_angle(dmatrix)
        self.world_space_vr = torch.cat((self.world_space_vr, self.world_space_vr[-1:, :, :]), dim = 0)


        self.pd_r = torch.from_numpy(lr) #rotation of joints, represented in quaternion
        r1 = self.pd_r[:-1, :, :]
        r2 = self.pd_r[1:, :, :]
        matrix1 = pyt.quaternion_to_matrix(r1)
        matrix2 = pyt.quaternion_to_matrix(r2)
        dmatrix = torch.matmul(matrix2, torch.transpose(matrix1, -1, -2))/(1/self.fps)
        self.pd_vr = pyt.matrix_to_axis_angle(dmatrix)
        self.pd_vr = torch.cat((self.pd_vr, self.pd_vr[-1:, :, :]), dim = 0)


    def joint_names(self):
        return list(self.joints.keys())

    def parse_file(self, path):
        self.path = path
        with open(path, 'r') as f:
            self.parse_string(f.read())
    
    def animation_all_frames(self):
        import matplotlib.pyplot as plt
        import mpl_toolkits.mplot3d.axes3d as p3
        import matplotlib.animation as animation

        fig = plt.figure()
        ax = p3.Axes3D(fig)     

        p = self.world_space_p.numpy()
        x_range = [np.min(p[:, :, 0]), np.max(p[:, :, 0])]
        y_range = [np.min(p[:, :, 1]), np.max(p[:, :, 1])]
        z_range = [np.min(p[:, :, 2]), np.max(p[:, :, 2])]
        print(x_range, y_range, z_range)

        pairs = [[list(self.joints.values()).index(j.parent), list(self.joints.values()).index(j)] \
                 for j in list(self.joints.values()) if j.parent!=None]

        lines = [ax.plot(p[0, i, 0], p[0, i, 1], p[0, i, 2])[0] for i in pairs]

        def update(n, p, pairs, lines):
            for pair, line in zip(pairs, lines):
                line.set_data(p[n, pair, 0:2].T)
                line.set_3d_properties(p[n, pair, 2])
            return lines
        
        ax.set_xlim3d(list(x_range))
        ax.set_xlabel('X')
        ax.set_ylim3d(list(y_range))
        ax.set_ylabel('Y')
        ax.set_zlim3d(list(z_range))
        ax.set_zlabel('Z')

        ax.set_title('BVH Animation')
        line_ani = animation.FuncAnimation(fig, update, self.frames-2, fargs=(p, pairs, lines),
                              interval=100, blit=False)

        plt.show()

    def animation(self):
        import matplotlib.pyplot as plt
        import mpl_toolkits.mplot3d.axes3d as p3
        import matplotlib.animation as animation

        fig = plt.figure()
        ax = p3.Axes3D(fig)     

        p = np.load('a.npy')
        x_range = [np.min(p[:, :, 0]), np.max(p[:, :, 0])]
        y_range = [np.min(p[:, :, 1]), np.max(p[:, :, 1])]
        z_range = [np.min(p[:, :, 2]), np.max(p[:, :, 2])]
        
        pairs = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], [10, 11], [11, 12], [12, 13], [11, 14], [14, 15], [15, 16], [16, 17], [11, 18], [18, 19], [19, 20], [20, 21]]

        lines = [ax.plot(p[0, i, 0], p[0, i, 1], p[0, i, 2])[0] for i in pairs]

        def update(n, p, pairs, lines):
            for pair, line in zip(pairs, lines):
                line.set_data(p[n, pair, 0:2].T)
                line.set_3d_properties(p[n, pair, 2])
            return lines
        
        ax.set_xlim3d(list(x_range))
        ax.set_xlabel('X')
        ax.set_ylim3d(list(y_range))
        ax.set_ylabel('Y')
        ax.set_zlim3d(list(z_range))
        ax.set_zlabel('Z')

        ax.set_title('BVH Animation')
        line_ani = animation.FuncAnimation(fig, update, 98, fargs=(p, pairs, lines),
                              interval=100, blit=False)
        line_ani.save('output.gif',writer='imagemagick')
        plt.show()

    def __repr__(self):
        return f"BVH {len(self.joints.keys())} joints, {self.frames} frames"


if __name__ == '__main__':
    # create Bvh parser
    anim = Bvh()
    # parser file
    anim.parse_file("./BVH/jumps1_subject5.bvh")
    print(anim.joint_names())

    # # draw the skeleton in T-pose
    # anim.plot_hierarchy()


    # extract all poses: axis0=frame, axis1=joint, axis2=positionXYZ/rotationXYZ
    anim.all_frame_poses()

    anim.animation_all_frames()
