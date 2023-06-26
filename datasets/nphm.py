"""Adapted from https://github.com/SimonGiebenhain/NPHM
"""
import os
import random
import torch
import trimesh
import traceback
import json
import numpy as np
import point_cloud_utils as pcu
from typing import Literal, Union, Dict, List, Optional
from torch.utils.data import Dataset


CODE_BASE = f'/home/jianjinx/data/My3D/thirdparty/NPHM'
ASSETS = f'{CODE_BASE}/assets/'
SUPERVISION_IDENTITY = f'{CODE_BASE}/dataset/point_cloud/identity'
SUPERVISION_DEFORMATION_OPEN = f'{CODE_BASE}/dataset/point_cloud/expression'
DATA_SINGLE_VIEW = f'{CODE_BASE}/dataset/NPHM_raw/single_view_synthetic_benchmark'
DATA = f'{CODE_BASE}/dataset/NPHM_raw'

DUMMY_DATA = f'{CODE_BASE}/dataset/dummy_data/dataset/'
DUMMY_single_view = f'{CODE_BASE}/dataset/dummy_data/single_view/'

EXPERIMENT_DIR = f'{CODE_BASE}/expr'
FITTING_DIR = f'{CODE_BASE}/expr/fitting'

ANCHOR_INDICES_PATH = ASSETS + 'lm_inds_39.npy'
ANCHOR_MEAN_PATH = ASSETS + 'anchors_39.npy'
FLAME_LM_INDICES_PATH = ASSETS + 'flame_up_lm_inds.npy'

NUM_SPLITS = 200
NUM_SPLITS_EXPR = 100

with open(CODE_BASE + '/dataset/neutrals_open.json') as f:
    _neutrals = json.load(f)
with open(CODE_BASE + '/dataset/neutrals_closed.json') as f:
    _neutrals_closed = json.load(f)
neutrals = {int(k): v for k,v in _neutrals.items()}
neutrals_closed = {int(k): v for k,v in _neutrals_closed.items()}
subjects_eval = [199, 286, 290, 291, 292, 293, 294, 295, 297, 298]
subjects_test = [99, 283, 143, 38, 241, 236, 276, 202, 98, 254, 204, 163, 267, 194, 20, 23, 209, 105, 186, 343, 341,  363, 350]
invalid_expressions_test = {
    143: [0, 1, 5],
    163: [6 ], # --> FLAME fitting failed to move in proper coordinate system
    38: [ 1, 5, 8, 9, 10, 11, 15, 16, 17, 18, 19], # hair changes --> maybe train or eval set
    236: [8, ],
    202: [24,],
    98: [0],
    254: [1, ],
    204: [16, ],
    267: [0, 7,  13, 22],
    194: [0, 1, 2, 3,  9, 11, 14, 18, 22 ],
    20: [17, 6, 11, 13, ],
    209: [7, 8, 9,  10, 15, 20, ],
    105: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    186: [7, 8, 9,  11, 21, ],
    343: [9, 11, ],
    363: [1, 11, 12, 14, ],
    350: [4, ],
}
for s in subjects_test:
    if s not in invalid_expressions_test.keys():
        invalid_expressions_test[s] = []
bad_scans = {
    261: [19],
    88: [19],
    79: [16, 17, 18, 19, 20],
    100:[0],
    125:[1, 4, 5],
    106:[20],
    362:[20],
    363:[1],
    345:[12],
    360:[6, 14],
    85:[2],
    292:[9],
    298:[23, 24, 25, 26],
}


class DataManager():
    def __init__(self, dummy_path = None):

        if dummy_path is not None:
            DATA = dummy_path + '/dataset/'
            DATA_SINGLE_VIEW = dummy_path + '/single_view/'

        self.lm_inds_upsampled = np.array([2212, 3060, 3485, 3384, 3386, 3389, 3418, 3395, 3414, 3598, 3637,
                                           3587, 3582, 3580, 3756, 2012, 730, 1984, 3157, 335, 3705, 3684,
                                           3851, 3863, 16, 2138, 571, 3553, 3561, 3501, 3526, 2748, 2792,
                                           3556, 1675, 1612, 2437, 2383, 2494, 3632, 2278, 2296, 3833, 1343,
                                           1034, 1175, 884, 829, 2715, 2813, 2774, 3543, 1657, 1696, 1579,
                                           1795, 1865, 3503, 2948, 2898, 2845, 2785, 3533, 1668, 1730, 1669,
                                           3509, 2786])

        self.anchor_indices = np.array([2712, 1579, 3485, 3756, 3430, 3659, 2711, 1575, 338, 27, 3631,
                                        3832, 2437, 1175, 3092, 2057, 3422, 3649, 3162, 2143, 617, 67,
                                        3172, 2160, 2966, 1888, 1470, 2607, 1896, 2981, 3332, 3231, 3494,
                                        3526, 3506, 3543, 3516, 3786, 3404])


    def get_all_subjects(self) -> List[int]:
        all_subjects = [int(pid) for pid in os.listdir(DATA) if pid.isdigit()]
        all_subjects.sort()
        return all_subjects


    def get_train_subjects(self,
                           neutral_type: Literal['open', 'closed'] = 'open',
                           exclude_missing_neutral : bool = True) -> List[int]:
        all_subjects = self.get_all_subjects()
        non_train = subjects_test + subjects_eval
        train_subjects =  [s for s in all_subjects if s not in non_train]
        if exclude_missing_neutral:
            train_subjects = [s for s in train_subjects if self.get_neutral_expression(s, neutral_type) is not None]
        return train_subjects


    def get_eval_subjects(self,
                          neutral_type: Literal['open', 'closed'] = 'open',
                          exclude_missing_neutral: bool = True) -> List[int]:
        eval_subjects = subjects_eval
        if exclude_missing_neutral:
            eval_subjects = [s for s in eval_subjects if self.get_neutral_expression(s, neutral_type) is not None]
        return eval_subjects


    def get_test_subjects(self) -> List[int]:
        return subjects_test


    def get_expressions(self,
                        subject : int,
                        testing : bool = False,
                        exclude_bad_scans : bool = True) -> List[int]:
        expressions = [int(f) for f in os.listdir(self.get_subject_dir(subject))]
        expressions.sort()
        if testing:
            expressions = [ex for ex in expressions if not (subject in invalid_expressions_test and
                                                            ex in invalid_expressions_test[subject])]
        if exclude_bad_scans:
            expressions = [ex for ex in expressions if not(subject in bad_scans and ex in bad_scans[subject])]
        return expressions


    def get_neutral_expression(self,
                               subject : int,
                               neutral_type : Literal['open', 'closed'] = 'open'
                               ) -> Optional[int]:
        if neutral_type == 'open':
            if subject not in neutrals:
                return None
            neutral_expression = neutrals[subject]
            if neutral_expression >= 0:
                return neutral_expression
            else:
                return None
        elif neutral_type == 'closed':
            if subject not in neutrals:
                return None
            neutral_expression = neutrals_closed[subject]
            if neutral_expression >= 0:
                return neutral_expression
            else:
                return None
        else:
            raise TypeError(f'Unknown neutral type {neutral_type} encountered! Expected on of [open, closed]!')


    def get_scan_dir(self,
                      subject : int,
                      expression : int) -> str:
        return f"{DATA}/{subject:03d}/{expression:03d}/"

    def get_subject_dir(self,
                      subject : int) -> str:
        return f"{DATA}/{subject:03d}/"


    def get_raw_path(self,
                     subject: int,
                     expression: int) -> str:
        return f"{self.get_scan_dir(subject, expression)}/scan.ply"


    def get_flame_path(self,
                     subject: int,
                     expression: int) -> str:
        return f"{self.get_scan_dir(subject, expression)}/flame.ply"

    def get_registration_path(self,
                     subject: int,
                     expression: int) -> str:
        return f"{self.get_scan_dir(subject, expression)}/registration.ply"

    def get_transform_from_metric(self,
                                  subject : int,
                                  expression : int) -> Dict[str, np.ndarray]:
        data_dir = self.get_scan_dir(subject, expression)
        s = np.load(f"{data_dir}/s.npy")
        R = np.load(f"{data_dir}/R.npy")
        t = np.load(f"{data_dir}/t.npy")
        return {"s": s, "R": R, "t": t}

    def get_raw_mesh(self,
                     subject : int,
                     expression : int,
                     coordinate_system : Literal['raw', 'flame', 'nphm'] = 'nphm',
                     mesh_type : Literal['trimesh', 'pcu'] = 'trimesh',
                     textured : bool = False # only relevant for mesh_type='pcu'
                     ) -> Union[trimesh.Trimesh, pcu.TriangleMesh]:

        raw_path = self.get_raw_path(subject, expression)
        if mesh_type == 'trimesh':
            m_raw = trimesh.load(raw_path, process=False)
        else:
            if textured:
                m_raw = pcu.load_triangle_mesh(raw_path)
            else:
                m_raw = pcu.TriangleMesh()
                v, f = pcu.load_mesh_vf(raw_path)
                m_raw.vertex_data.positions = v
                m_raw.face_data.vertex_ids = f

        if coordinate_system == 'flame':
            m_raw = self.transform_nphm_2_flame(m_raw)
        if coordinate_system == 'raw':
            m_raw = self.transform_nphm_2_raw(m_raw, subject, expression)

        return m_raw


    def get_flame_mesh(self,
                       subject : int,
                       expression : int,
                       coordinate_system : Literal['raw', 'flame', 'nphm'] = 'nphm',
                       mesh_type : Literal['trimesh', 'pcu'] = 'trimesh'
                       )-> Union[trimesh.Trimesh, pcu.TriangleMesh]:

        flame_path = self.get_flame_path(subject, expression)
        if mesh_type == 'trimesh':
            m_flame = trimesh.load(flame_path, process=False)
        else:
            m_flame = pcu.TriangleMesh()
            v, f = pcu.load_mesh_vf(flame_path)
            m_flame.vertex_data.positions = v
            m_flame.face_data.vertex_ids = f

        if coordinate_system == 'flame':
            m_flame = self.transform_nphm_2_flame(m_flame)
        if coordinate_system == 'raw':
            m_flame = self.transform_nphm_2_raw(m_flame, subject, expression)

        return m_flame

    def get_registration_mesh(self,
                              subject : int,
                              expression : int,
                              coordinate_system : Literal['raw', 'flame', 'nphm'] = 'nphm',
                              mesh_type: Literal['trimesh', 'pcu'] = 'trimesh'
                              ) -> Union[trimesh.Trimesh, pcu.TriangleMesh]:

        regi_path = self.get_registration_path(subject, expression)
        if mesh_type == 'trimesh':
            mesh = trimesh.load(regi_path, process=False)
        else:
            mesh = pcu.TriangleMesh()
            v, f = pcu.load_mesh_vf(regi_path)
            mesh.vertex_data.positions = v
            mesh.face_data.vertex_ids = f

        if coordinate_system == 'flame':
            mesh = self.transform_nphm_2_flame(mesh)
        if coordinate_system == 'raw':
            mesh = self.transform_nphm_2_raw(mesh, subject, expression)

        return mesh


    def get_landmarks(self,
                      subject : int,
                      expression : int,
                      coordinate_system : Literal['raw', 'flame', 'nphm'] = 'nphm') -> np.ndarray:
        fine_mesh = self.get_registration_mesh(subject, expression, coordinate_system)
        landmarks = fine_mesh.vertices[self.lm_inds_upsampled, :]
        return landmarks

    def get_facial_anchors(self,
                      subject : int,
                      expression : int,
                      coordinate_system : Literal['raw', 'flame', 'nphm'] = 'nphm') -> np.ndarray:
        fine_mesh = self.get_registration_mesh(subject, expression, coordinate_system)
        anchors = fine_mesh.vertices[self.anchor_indices, :]
        return np.array(anchors)



    def get_single_view_obs(self,
                            subject : int,
                            expression : int,
                            include_back : bool = True,
                            coordinate_system : Literal['raw', 'flame', 'nphm'] = 'nphm',
                            disable_cut_throat = False,
                            full_obs = False
                            ) -> np.ndarray:

        points = np.load(self.get_single_view_path(subject, expression, full_depth_map=full_obs))
        if include_back:
            back_path = self.get_single_view_path(subject, expression, full_depth_map=full_obs, is_back=True)
            if not os.path.exists(back_path):
                print('WARNING: observation from back not available!')
            else:
                points_back = np.load(back_path)
                points = np.concatenate([points, points_back], axis=0)

        if not disable_cut_throat:
            above = self.cut_throat(points, subject, expression)
            points = points[above, :]

        if coordinate_system == 'flame':
            points = self.transform_nphm_2_flame(points)
        if coordinate_system == 'raw':
            points = self.transform_nphm_2_raw(points, subject, expression)

        return points


    def cut_throat(self,
                   points : np.ndarray,
                   subject : int,
                   expression : int,
                   coordinate_system : Literal['raw', 'flame', 'nphm'] = 'nphm',
                   margin : float = 0) -> np.ndarray:

            template_mesh = self.get_flame_mesh(subject, expression, coordinate_system=coordinate_system)
            idv1 = 3276
            idv2 = 3207
            idv3 = 3310
            v1 = template_mesh.vertices[idv1, :]
            v2 = template_mesh.vertices[idv2, :]
            v3 = template_mesh.vertices[idv3, :]
            origin = v1
            line1 = v2 - v1
            line2 = v3 - v1
            normal = np.cross(line1, line2)

            direc = points - origin
            angle = np.sum(normal * direc, axis=-1)
            above = angle > margin
            return above


    ####################################################################################################################
    #### Transformations between cooridnate systems ####
    ####################################################################################################################

    def transform_nphm_2_flame(self,
                               object : Union[trimesh.Trimesh, np.ndarray, pcu.TriangleMesh]
                               ) -> Union[trimesh.Trimesh, np.ndarray, pcu.TriangleMesh]:

        if isinstance(object, np.ndarray):
            object /= 4
        elif isinstance(object, trimesh.Trimesh):
            object.vertices /= 4
        elif isinstance(object, pcu.TriangleMesh):
            object.vertex_data.positions /= 4
        else:
            raise TypeError(f'Unexpected type encountered in coordinate transormation. \n '
                            'Expected one of [trimesh.Trimesh, np.ndarray, pcu.TriangleMesh]. '
                            f'But found {type(object)}')

        return object


    def transform_nphm_2_raw(self,
                             object : Union[trimesh.Trimesh, np.ndarray, pcu.TriangleMesh],
                             subject : int,
                             expression : int
                             ) -> Union[trimesh.Trimesh, np.ndarray, pcu.TriangleMesh]:

        transform = self.get_transform_from_metric(subject, expression)

        if isinstance(object, np.ndarray):
            object = 1/transform['s'] * (object - transform['t']) @ transform['R']
        elif isinstance(object, trimesh.Trimesh):
            object.vertices = 1/transform['s'] * (object.vertices - transform['t']) @ transform['R']
        elif isinstance(object, pcu.TriangleMesh):
            object.vertex_data.positions = 1/transform['s'] * (object.vertex_data.positions - transform['t']) @ transform['R']
        else:
            raise TypeError(f'Unexpected type encountered in coordinate transormation. \n '
                            'Expected one of [trimesh.Trimesh, np.ndarray, pcu.TriangleMesh]. '
                            f'But found {type(object)}')

        return object


    ####################################################################################################################
    ######### get paths relevant for training ###########
    ####################################################################################################################


    def get_train_dir_identity(self,
                               subject : int) -> str:
        return f"{SUPERVISION_IDENTITY}/{subject:03d}/"


    def get_train_path_identity_face(self,
                                subject: int,
                                expression: int,
                                rnd_file: Optional[int] = None) -> str:
        if rnd_file is None:
            rnd_file = np.random.randint(0, NUM_SPLITS)
        return f"{self.get_train_dir_identity(subject)}/{expression}_{rnd_file}_face.npy"


    def get_train_path_identity_non_face(self,
                                subject: int,
                                expression: int,
                                rnd_file: Optional[int] = None
                                ) -> str:
        if rnd_file is None:
            rnd_file = np.random.randint(0, NUM_SPLITS)
        return f"{self.get_train_dir_identity(subject)}/{expression}_{rnd_file}_non_face.npy"


    def get_train_dir_deformation(self,
                                  subject : int,
                                  expression : int) -> str:
        return f"{SUPERVISION_DEFORMATION_OPEN}/{subject:03d}/{expression:03d}/"


    def get_train_path_deformation(self,
                                   subject : int,
                                   expression : int,
                                   rnd_file : Optional[int] = None) -> str:
        if rnd_file is None:
            rnd_file = np.random.randint(0, NUM_SPLITS_EXPR)
        return f"{self.get_train_dir_deformation(subject, expression)}/corresp_{rnd_file}.npy"


    def get_single_view_dir(self,
                            subject : int,
                            expression : int):
        return f"{DATA_SINGLE_VIEW}/{subject:03d}/{expression}"


    def get_single_view_path(self,
                             subject : int,
                             expression : int,
                             full_depth_map = False,
                             is_back = False):
        dir_name = self.get_single_view_dir(subject, expression)
        if not full_depth_map:
            if is_back:
                return f"{dir_name}/obs_back.npy"
            else:
                return f"{dir_name}/obs.npy"
        else:
            if is_back:
                return f"{dir_name}/full_obs_back.npy"
            else:
                return f"{dir_name}/full_obs.npy"


def uniform_ball(n_points, rad=1.0):
    angle1 = np.random.uniform(-1, 1, n_points)
    angle2 = np.random.uniform(0, 1, n_points)
    radius = np.random.uniform(0, rad, n_points)

    r = radius ** (1/3)
    theta = np.arccos(angle1) #np.pi * angle1
    phi = 2 * np.pi * angle2
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    return np.stack([x, y, z], axis=-1)


class ScannerData(Dataset):
    def __init__(self,
                 mode : Literal['train', 'val'],
                 n_supervision_points_face : int,
                 n_supervision_points_non_face : int,
                 batch_size : int = 16,
                 sigma_near : float = 0.01,
                 lm_inds : np.ndarray = None,
                 is_closed : bool = False):

        self.manager = DataManager()

        self.lm_inds = lm_inds
        self.mode = mode
        self.ROOT = SUPERVISION_IDENTITY

        if is_closed:
            self.neutral_expr_index = neutrals_closed
            self.neutral_type = 'closed'
        else:
            self.neutral_expr_index = neutrals
            self.neutral_type = 'open'


        if mode == 'train':
            self.subjects = self.manager.get_train_subjects(self.neutral_type)
        else:
            self.subjects = self.manager.get_eval_subjects(self.neutral_type)

        # obtain subjects and expression indices used for building batches
        self.subject_steps = []
        for s in self.subjects:
            self.subject_steps += [s]


        self.batch_size = batch_size
        self.n_supervision_points_face = n_supervision_points_face
        self.n_supervision_points_non_face = n_supervision_points_non_face
        print('Dataset has {} subjects'.format(len(self.subject_steps)))
        self.sigma_near = sigma_near

        # pre-fetch g.t. facial anchor points
        if self.lm_inds is not None:
            self.gt_anchors = {}

            for i, iden in enumerate(self.subject_steps):
                self.gt_anchors[iden] = self.manager.get_facial_anchors(subject=iden,
                                                                        expression=self.neutral_expr_index[iden])
        else:
            self.gt_anchors = np.zeros([39, 3])

    def set_n_points(self, n_points):
        self.n_supervision_points_face = n_points

    def __len__(self):
        return len(self.subject_steps)


    def __getitem__(self, idx):
        iden = self.subject_steps[idx]
        expr = self.neutral_expr_index[iden]

        if self.lm_inds is not None:
            gt_anchors = self.gt_anchors[iden]

        try:
            on_face = np.load(self.manager.get_train_path_identity_face(iden, expr))
            points = on_face[:, :3] / 3
            normals = on_face[:, 3:6]
            colors = on_face[:, 6:9] / 127.5 - 1
            non_face = np.load(self.manager.get_train_path_identity_non_face(iden, expr))
            points_outer = non_face[:, :3] / 3
            normals_non_face = non_face[:, 3:6]
            colors_non_face = non_face[:, 6:9]

            # subsample points for supervision
            indice = np.arange(0, points.shape[0])
            sup_idx = np.random.choice(indice, (self.n_supervision_points_face,), replace=False)
            sup_points = points[sup_idx, :]
            sup_normals = normals[sup_idx, :]
            sup_colors = colors[sup_idx, :]
            #sup_idx_non = np.random.randint(0, points_outer.shape[0], self.n_supervision_points_non_face//5)
            #sup_points_non = points_outer[sup_idx_non, :]
            #sup_normals_non = normals_non_face[sup_idx_non, :]
            #sup_colors_non = colors_non_face[sup_idx_non, :]

        except Exception as e:
            print('SUBJECT: {}'.format(iden))
            print('EXPRESSION: {}'.format(expr))
            print(traceback.format_exc())
            return self.__getitem__(np.random.randint(0, self.__len__()))



        # sample points for grad-constraint
        #sup_grad_far = uniform_ball(self.n_supervision_points_face // 8, rad=0.5)
        #sup_grad_near = np.concatenate([sup_points, sup_points_non], axis=0) + \
        #                np.random.randn(sup_points.shape[0]+sup_points_non.shape[0], 3) * self.sigma_near #0.01

        #print("points:", sup_points.shape, sup_points.min(), sup_points.max())
        #print("normals:", sup_normals.shape, sup_normals.min(), sup_normals.max())
        #print("colors:", sup_colors.shape, sup_colors.min(), sup_colors.max())
        train_points = np.concatenate([sup_points, sup_normals, sup_colors], 1)
        ret_dict = {
                    #'points_face': sup_points,
                    #'normals_face': sup_normals,
                    #'colors_face': sup_colors,
                    'train_points': train_points,
                    #'sup_grad_far': sup_grad_far,
                    #'sup_grad_near': sup_grad_near,
                    'idx': np.array([idx]),
                    #'points_non_face': sup_points_non,
                    #'normals_non_face': sup_normals_non,
                    #'colors_non_face': sup_colors_non
                    }

        if not self.lm_inds is None:
            ret_dict.update({'gt_anchors': np.array(gt_anchors)})

        return ret_dict

    def get_loader(self, shuffle=True):
        #random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        np.random.seed(0)
        return torch.utils.data.DataLoader(
            self, batch_size=self.batch_size, num_workers=8, shuffle=shuffle,
            worker_init_fn=self.worker_init_fn,
            pin_memory=True)

    def worker_init_fn(self, worker_id):
        random_data = os.urandom(4)
        base_seed = int.from_bytes(random_data, byteorder="big")
        np.random.seed(base_seed + worker_id)


class ScannerDeformationData(Dataset):
    def __init__(self,
                 mode : Literal['train', 'val'],
                 n_supervision_points : int,
                 batch_size : int,
                 lm_inds : np.ndarray
                 ):

        self.manager = DataManager()

        self.neutral_expr_index = neutrals

        self.mode = mode

        self.ROOT = SUPERVISION_DEFORMATION_OPEN
        self.lm_inds = lm_inds


        if mode == 'train':
            self.subjects = self.manager.get_train_subjects(neutral_type='open')
        else:
            self.subjects = self.manager.get_eval_subjects(neutral_type='open')

        print(f'Dataset has  {len(self.subjects)} Identities!')

        self.subject_steps = [] # stores subject id for each data point
        self.steps = [] # stores expression if for each data point
        self.subject_index = [] # defines order of subjects used in training, relevant for auto-decoder

        all_files = []
        for i, s in enumerate(self.subjects):
            expressions = self.manager.get_expressions(s)
            self.subject_steps += len(expressions) * [s, ]
            self.subject_index += len(expressions) * [i, ]
            self.steps += expressions
            all_files.append(expressions)

        self.batch_size = batch_size
        self.n_supervision_points = n_supervision_points

        # pre-fetch facial anchors for neutral expression
        self.anchors = {}
        for iden in self.subjects:
            self.anchors[iden] = self.manager.get_facial_anchors(subject=iden, expression=self.neutral_expr_index[iden])


    def __len__(self):
        return len(self.steps)

    def __getitem__(self, idx):

        expr = self.steps[idx]
        iden = self.subject_steps[idx]
        subj_ind = self.subject_index[idx]

        try:
            point_corresp = np.load(self.manager.get_train_path_deformation(iden, expr))
            valid = np.logical_not( np.any(np.isnan(point_corresp), axis=-1))
            point_corresp = point_corresp[valid, :].astype(np.float32)

        except Exception as e:
            print(iden)
            print(expr)
            print('FAILED')
            return self.__getitem__(0) # avoid crashing of training, dirty


        # subsample points for supervision
        sup_idx = np.random.randint(0, point_corresp.shape[0], self.n_supervision_points)
        sup_points_neutral = point_corresp[sup_idx, :3]
        sup_points_posed = point_corresp[sup_idx, 3:]

        neutral = sup_points_neutral
        posed = sup_points_posed

        gt_anchors = self.anchors[iden]

        return {'points_neutral': neutral,
                'points_posed': posed,
                'idx': np.array([idx]),
                'iden': np.array([self.subjects.index(iden)]),
                'expr': np.array([expr]),
                'subj_ind': np.array([subj_ind]),
                'gt_anchors': gt_anchors}

    def get_loader(self, shuffle=True):
        random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        np.random.seed(0)
        return torch.utils.data.DataLoader(
            self, batch_size=self.batch_size, num_workers=8, shuffle=shuffle,
            worker_init_fn=self.worker_init_fn,
            pin_memory=True)

    def worker_init_fn(self, worker_id):
        random_data = os.urandom(4)
        base_seed = int.from_bytes(random_data, byteorder="big")
        np.random.seed(base_seed + worker_id)

