import cv2

from utils.sbp_utils import SBPmAPCOCO


################################################################################################################
# Passenger Interaction System Pose Estimation Utils
################################################################################################################
class SBPmAPPIS(SBPmAPCOCO):
    def __init__(self, json_path, input_size, conf_threshold):
        super().__init__(json_path, input_size, conf_threshold)

    def update_state(self, target, y_pred):
        batch_size = y_pred.size(0)
        bbox = target['bbox']
        img_ids = target['image_id']
        cat_ids = target['category_id']
        
        for idx in range(batch_size):
            joints = self.decoder(y_pred[idx:idx+1]) # [num_keypoints, 3]
            
            # convert joints input_size scale to original image scale
            joints[..., :1] *= (bbox[idx][2] / self.input_size[1])
            joints[..., 1:2] *= (bbox[idx][3] / self.input_size[0])

            # convert joints to original image coordinate
            joints[..., :1] += bbox[idx][0]
            joints[..., 1:2] += bbox[idx][1]
            
            tmp_joints = []
            tmp_confs = []
            for (x, y, conf) in joints:
                if conf < 0:
                    tmp_joints.extend([0, 0, 0])
                    tmp_confs.append(0)
                    continue
                
                tmp_joints.extend([float(x), float(y), 1])
                tmp_confs.append(conf)
            tmp_joints.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            
            self.result_list.append({
                "image_id": int(img_ids[idx]),
                "category_id": int(cat_ids[idx]),
                "keypoints": tmp_joints,
                "score": float(sum(tmp_confs) / joints.size(0))
            })

def get_pis_tagged_img_sbp(img, joints):
    '''Return Tagged Image
    
    Arguments:
        img (Numpy): Image Array of Numpy
        joints (Tensor): joints '[num_keypoints, 3]', specified as [x, y, conf]
    
    Returns:
        img (Numpy): Tagged Image Array of Numpy
    '''
    tagged_img = img.copy()
    h, w, _ = tagged_img.shape
    
    limb_colors = [
        (0, 102, 102), # right face
        (102, 0, 102), # left face
        (0, 204, 0), # right arm
        (204, 0, 0), # left arm
        (0, 102, 0), # right leg
        (102, 0, 0), # left leg
        (0, 0, 0) # others
    ]

    # [joint_idx, joint_idx, limb_color_idx]
    joint_limbs = [
        [0, 1, 1], 
        [0, 2, 0], 
        [1, 3, 1], 
        [2, 4, 0], 
        [5, 7, 3], 
        [6, 8, 2], 
        [7, 9, 3], 
        [8, 10, 2],  
        [5, 6, 6] 
    ]
    
    # Draw keypoints limbs
    for limb in joint_limbs:
        tmp_joints = joints[limb[:2]]
        joint1 = tmp_joints[0]
        joint2 = tmp_joints[1]
        if joint1[-1] < 0 or joint2[-1] < 0:
            continue
        x1, y1 = int(joint1[0]), int(joint1[1])
        x2, y2 = int(joint2[0]), int(joint2[1])
        cv2.line(tagged_img, (x1, y1),  (x2, y2), limb_colors[limb[-1]], 4)
        
    # Draw keypoints joints   
    for (x, y, conf) in joints:
        if conf < 0:
            continue
        x, y = int(x), int(y)
        cv2.circle(tagged_img, (x, y), 4, (0, 0, 255), -1)    
    
    return tagged_img

class HandleGrip:
    '''PIS Handle Grip Utils
    
    Arguments:
        handle_roi (Tuple): ((x1, y1), (x2, y2)), Handle roi for handle grip. 2 points on Image.
    '''
    def __init__(self, handle_roi):
        self.handle_roi = handle_roi
    
    def get_handle_grip_result(self, point):
        '''
        Arguments:
            point (Tensor): joint point, (x, y)
        Returns:
            result (Bool): True - Handle Grip, False - No Handle Grip
        '''
        gradient = (self.handle_roi[0][1] - self.handle_roi[1][1]) / (self.handle_roi[0][0] - self.handle_roi[1][0])
        y_intercept = self.handle_roi[0][1] - (gradient * self.handle_roi[0][0])
        
        intersection_x = int((point[1] - y_intercept)/gradient)
        
        return point[0] > intersection_x

class FallingDown:
    '''PIS Falling Down Utils
    
    Arguments:
        neg_max (int): Max Negative Gradient Value
        pos_min (int): Min Positive Gradient Value
    '''
    def __init__(self, neg_max, pos_min):
        self.neg_max = neg_max
        self.pos_min = pos_min
        
    def get_falling_down_result(self, point1, point2):
        '''
        Arguments:
            point1 (Tensor): joint point, (x, y)
            point2 (Tensor): joint point, (x, y)
        Returns:
            result (Bool): True - Normal, False - Falling Down
        '''
        gradient = (point1[1] - point2[1]) / (point1[0] - point2[0] + 1e-6)
        
        return gradient < self.neg_max or self.pos_min < gradient