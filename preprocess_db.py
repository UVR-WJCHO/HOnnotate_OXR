import os
import cv2
import mediapipe as mp
import copy, time

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


class RecordSet():
    def __init__(self):
        self.db_path = '/home/uvr-1080ti/projects/HOnnotate/sequence/xrstudio/'

        self.prev_bbox = [0, 0, 640, 480]

    def read_record(self, idx):
        img_rgb_path = os.path.join(self.db_path, 'rgb', 'mas_%d.png' % idx)
        img = cv2.imread(img_rgb_path)
        img_depth_path = os.path.join(self.db_path, 'depth', 'mas_%d.png' % idx)
        depth = cv2.imread(img_depth_path, cv2.IMREAD_ANYDEPTH)

        return img, depth

    def getItem_idx(self, idx):
        img, depth = self.read_record(idx)

        img_cv = copy.deepcopy(img)
        cv2.imshow("original", img_cv)
        cv2.waitKey(1)
        image_rows, image_cols, _ = img.shape

        #### extract image bounding box ####
        with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.3) as hands:
            image = cv2.flip(img, 1)
            # image = np.copy(img)
            # Convert the BGR image to RGB before processing.
            t1 = time.time()
            results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            print("t : ", time.time() - t1)

            idx_to_coord_0 = None
            idx_to_coord_1 = None

            hand_idx = 0
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    idx_to_coordinates = {}
                    for idx_land, landmark in enumerate(hand_landmarks.landmark):
                        landmark_px = mp_drawing._normalized_to_pixel_coordinates(landmark.x, landmark.y,
                                                                       image_cols, image_rows)
                        if landmark_px:
                            # landmark_px has fliped x axis
                            orig_x = image_cols - landmark_px[0]
                            idx_to_coordinates[idx_land] = [orig_x, landmark_px[1]]
                    if hand_idx == 0:
                        idx_to_coord_0 = idx_to_coordinates
                        hand_idx += 1
                    else:
                        idx_to_coord_1 = idx_to_coordinates

            ### Draw the hand annotations on the image.
            image.flags.writeable = True
            # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())


        bbox, img_crop, depth_crop = self.extract_roi_single(img, depth, idx_to_coord_0, image_rows, image_cols)

        img_name = "./samples/221115_bowl_18_00/crop/mas_{}.png".format(idx)
        cv2.imwrite(img_name, img_crop)

        img_name = "./samples/221115_bowl_18_00/crop_depth/mas_{}.png".format(idx)
        cv2.imwrite(img_name, depth_crop)

        return None

    def extract_roi_single(self, img, depth, idx_to_coord_0, image_rows, image_cols):

        # if tracking fails, use the previous bbox
        if idx_to_coord_0 is None:
            bbox = self.prev_bbox
        else:
            bbox = self.extract_bbox(idx_to_coord_0, image_rows, image_cols)

        # img_crop, img2bb_trans, bb2img_trans, _, _, = augmentation_real(img, bbox, flip=False)
        img_crop = img[int(bbox[1]):int(bbox[1] + bbox[3]), int(bbox[0]):int(bbox[0] + bbox[2])]
        depth_crop = depth[int(bbox[1]):int(bbox[1] + bbox[3]), int(bbox[0]):int(bbox[0] + bbox[2])]

        cv2.imshow('crop single', img_crop / 255.)
        cv2.waitKey(1)

        self.prev_bbox = copy.deepcopy(bbox)

        return bbox, img_crop, depth_crop

    def extract_bbox(self, idx_to_coord, image_rows, image_cols):
        x_min = min(idx_to_coord.values(), key=lambda x: x[0])[0]
        x_max = max(idx_to_coord.values(), key=lambda x: x[0])[0]
        y_min = min(idx_to_coord.values(), key=lambda x: x[1])[1]
        y_max = max(idx_to_coord.values(), key=lambda x: x[1])[1]

        x_avg = (x_min + x_max) / 2
        y_avg = (y_min + y_max) / 2

        x_min = max(0, x_avg - 320)
        y_min = max(0, y_avg - 240)

        if (x_min + 640) > image_cols:
            x_min = image_cols - 640
        if (y_min + 480) > image_rows:
            y_min = image_rows - 480

        bbox = [x_min, y_min, 640, 480]
        return bbox

if __name__ == '__main__':
    db = RecordSet()

    for i in range(200):
        db.getItem_idx(i)




