import torch
import numpy as np
import mmcv
import os
import math
from pathlib import Path

def write_ply(verts, colors, indices, output_file):
    if colors is None:
        colors = np.zeros_like(verts)
    if indices is None:
        indices = []

    file = open(output_file, "w")
    file.write("ply \n")
    file.write("format ascii 1.0\n")
    file.write("element vertex {:d}\n".format(len(verts)))
    file.write("property float x\n")
    file.write("property float y\n")
    file.write("property float z\n")
    file.write("property uchar red\n")
    file.write("property uchar green\n")
    file.write("property uchar blue\n")
    file.write("element face {:d}\n".format(len(indices)))
    file.write("property list uchar uint vertex_indices\n")
    file.write("end_header\n")
    for vert, color in zip(verts, colors):
        # print(vert)
        file.write(
            "{:f} {:f} {:f} {:d} {:d} {:d}\n".format(
                vert[0], vert[1], vert[2], int(color[0]), int(color[1]), int(color[2])
            )
        )
    for ind in indices:
        file.write("3 {:d} {:d} {:d}\n".format(ind[0], ind[1], ind[2]))
    file.close()

def create_cylinder_mesh(radius, p0, p1, stacks=10, slices=10):
    def compute_length_vec3(vec3):
        return math.sqrt(vec3[0] * vec3[0] + vec3[1] * vec3[1] + vec3[2] * vec3[2])

    def rotation(axis, angle):
        rot = np.eye(4)
        c = np.cos(-angle)
        s = np.sin(-angle)
        t = 1.0 - c
        axis /= compute_length_vec3(axis)
        x = axis[0]
        y = axis[1]
        z = axis[2]
        rot[0, 0] = 1 + t * (x * x - 1)
        rot[0, 1] = z * s + t * x * y
        rot[0, 2] = -y * s + t * x * z
        rot[1, 0] = -z * s + t * x * y
        rot[1, 1] = 1 + t * (y * y - 1)
        rot[1, 2] = x * s + t * y * z
        rot[2, 0] = y * s + t * x * z
        rot[2, 1] = -x * s + t * y * z
        rot[2, 2] = 1 + t * (z * z - 1)
        return rot

    verts = []
    indices = []
    diff = (p1 - p0).astype(np.float32)
    height = compute_length_vec3(diff)
    for i in range(stacks + 1):
        for i2 in range(slices):
            theta = i2 * 2.0 * math.pi / slices
            pos = np.array(
                [
                    radius * math.cos(theta),
                    radius * math.sin(theta),
                    height * i / stacks,
                ]
            )
            verts.append(pos)
    for i in range(stacks):
        for i2 in range(slices):
            i2p1 = math.fmod(i2 + 1, slices)
            indices.append(
                np.array(
                    [(i + 1) * slices + i2, i * slices + i2, i * slices + i2p1],
                    dtype=np.uint32,
                )
            )
            indices.append(
                np.array(
                    [(i + 1) * slices + i2, i * slices + i2p1, (i + 1) * slices + i2p1],
                    dtype=np.uint32,
                )
            )
    transform = np.eye(4)
    va = np.array([0, 0, 1], dtype=np.float32)
    vb = diff
    vb /= compute_length_vec3(vb)
    axis = np.cross(vb, va)
    angle = np.arccos(np.clip(np.dot(va, vb), -1, 1))
    if angle != 0:
        if compute_length_vec3(axis) == 0:
            dotx = va[0]
            if math.fabs(dotx) != 1.0:
                axis = np.array([1, 0, 0]) - dotx * va
            else:
                axis = np.array([0, 1, 0]) - va[1] * va
            axis /= compute_length_vec3(axis)
        transform = rotation(axis, -angle)
    transform[:3, 3] += p0
    verts = [np.dot(transform, np.array([v[0], v[1], v[2], 1.0])) for v in verts]
    verts = [np.array([v[0], v[1], v[2]]) / v[3] for v in verts]

    return verts, indices


def create_color_palette():
    return [
        (0, 0, 0),
        (174, 199, 232),  # wall
        (152, 223, 138),  # floor
        (31, 119, 180),  # cabinet
        (255, 187, 120),  # bed
        (188, 189, 34),  # chair
        (140, 86, 75),  # sofa
        (255, 152, 150),  # table
        (214, 39, 40),  # door
        (197, 176, 213),  # window
        (148, 103, 189),  # bookshelf
        (196, 156, 148),  # picture
        (23, 190, 207),  # counter
        (178, 76, 76),
        (247, 182, 210),  # desk
        (66, 188, 102),
        (219, 219, 141),  # curtain
        (140, 57, 197),
        (202, 185, 52),
        (51, 176, 203),
        (200, 54, 131),
        (92, 193, 61),
        (78, 71, 183),
        (172, 114, 82),
        (255, 127, 14),  # refrigerator
        (91, 163, 138),
        (153, 98, 156),
        (140, 153, 101),
        (158, 218, 229),  # shower curtain
        (100, 125, 154),
        (178, 127, 135),
        (120, 185, 128),
        (146, 111, 194),
        (44, 160, 44),  # toilet
        (112, 128, 144),  # sink
        (96, 207, 209),
        (227, 119, 194),  # bathtub
        (213, 92, 176),
        (94, 106, 211),
        (82, 84, 163),  # otherfurn
        (100, 85, 144),
    ]

def transform_bbox(bboxes, pred=False):
    cx = bboxes[..., 0:1]
    cy = bboxes[..., 1:2]
    cz = bboxes[..., 2:3]
    w = bboxes[..., 3:4]
    l = bboxes[..., 4:5]
    h = bboxes[..., 5:6]
    c = bboxes[..., 6:7]
    # This is gravity
    if pred:
        cz += h/2
    min_cx = cx - w / 2
    min_cy = cy - l / 2
    min_cz = cz - h / 2
    max_cx = cx + w / 2
    max_cy = cy + l / 2
    max_cz = cz + h / 2
    return np.concatenate([min_cx, min_cy, min_cz, max_cx, max_cy, max_cz, c], axis=-1)

def write_bbox(bbox, output_file=None):
    """
    bbox: np array (n, 7), last one is instance/label id
    output_file: string
    """

    def get_bbox_edges(bbox_min, bbox_max):
        def get_bbox_verts(bbox_min, bbox_max):
            verts = [
                np.array([bbox_min[0], bbox_min[1], bbox_min[2]]),
                np.array([bbox_max[0], bbox_min[1], bbox_min[2]]),
                np.array([bbox_max[0], bbox_max[1], bbox_min[2]]),
                np.array([bbox_min[0], bbox_max[1], bbox_min[2]]),
                np.array([bbox_min[0], bbox_min[1], bbox_max[2]]),
                np.array([bbox_max[0], bbox_min[1], bbox_max[2]]),
                np.array([bbox_max[0], bbox_max[1], bbox_max[2]]),
                np.array([bbox_min[0], bbox_max[1], bbox_max[2]]),
            ]
            return verts

        box_verts = get_bbox_verts(bbox_min, bbox_max)
        edges = [
            (box_verts[0], box_verts[1]),
            (box_verts[1], box_verts[2]),
            (box_verts[2], box_verts[3]),
            (box_verts[3], box_verts[0]),
            (box_verts[4], box_verts[5]),
            (box_verts[5], box_verts[6]),
            (box_verts[6], box_verts[7]),
            (box_verts[7], box_verts[4]),
            (box_verts[0], box_verts[4]),
            (box_verts[1], box_verts[5]),
            (box_verts[2], box_verts[6]),
            (box_verts[3], box_verts[7]),
        ]
        return edges

    radius = 0.03
    offset = [0, 0, 0]
    verts = []
    indices = []
    colors = []
    for box in bbox:
        box_min = np.array([box[0], box[1], box[2]])
        box_max = np.array([box[3], box[4], box[5]])
        r, g, b = create_color_palette()[int(box[6] % 41)]
        edges = get_bbox_edges(box_min, box_max)
        for k in range(len(edges)):
            cyl_verts, cyl_ind = create_cylinder_mesh(radius, edges[k][0], edges[k][1])
            cur_num_verts = len(verts)
            cyl_color = [[r, g, b] for _ in cyl_verts]
            cyl_verts = [x + offset for x in cyl_verts]
            cyl_ind = [x + cur_num_verts for x in cyl_ind]
            verts.extend(cyl_verts)
            indices.extend(cyl_ind)
            colors.extend(cyl_color)

    if output_file is None:
        return verts, colors, indices

    write_ply(verts, colors, indices, output_file)


bbox_read = mmcv.load('../bbox_predict.pkl')
annos = mmcv.load('../data/scannet/scannet_infos_val.pkl')
assert len(bbox_read) == len(annos)

save_path = 'vis_bbox'
for i in range(len(annos)):
    scene_id = annos[i]['pts_path'].split('.')[0].split('/')[-1]
    gt_bbox = annos[i]['annos']['gt_boxes_upright_depth']
    gt_class = annos[i]['annos']['class'].reshape(-1, 1)
    align_matrix = annos[i]['annos']['axis_align_matrix']
    pc_bboxes = np.concatenate([gt_bbox, gt_class], axis=1)
    bboxes = transform_bbox(pc_bboxes)
    bbox_path = os.path.join(save_path, 'gt_bbox')
    align_path = os.path.join(save_path, 'align_matrix')
    write_bbox(bboxes, os.path.join(bbox_path, scene_id+'.ply'))
    np.save(os.path.join(align_path, scene_id+'_align.npy'), align_matrix)

    pred_bbox = bbox_read[i]['boxes_3d'].tensor[:, :6]
    pred_class = bbox_read[i]['labels_3d'].view(-1, 1)
    pred_bboxes = torch.cat([pred_bbox, pred_class], dim=-1).numpy()
    pred_bboxes = transform_bbox(pred_bboxes, pred=True)
    pred_bbox_path = os.path.join(save_path, 'pred_bbox')
    write_bbox(pred_bboxes, os.path.join(pred_bbox_path, scene_id+'.ply'))
