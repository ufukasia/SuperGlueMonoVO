import numpy as np
import cv2
import torch
from models.matching import Matching
from models.utils import frame2tensor

STAGE_FIRST_FRAME = 0
STAGE_SECOND_FRAME = 1
STAGE_DEFAULT_FRAME = 2
kMinNumFeature = 1500

# SuperGlue yapılandırması
# Burada outdoor ağırlıkları örnek olarak seçildi, ihtiyaca göre değiştirebilirsiniz.
superglue_conf = {
    'superpoint': {
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': 500
    },
    'superglue': {
        'weights': 'outdoor',
        'sinkhorn_iterations': 15,
        'match_threshold': 0.15,
    }
}


class PinholeCamera:
    def __init__(self, width, height, fx, fy, cx, cy,
                 k1=0.0, k2=0.0, p1=0.0, p2=0.0, k3=0.0):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.distortion = (abs(k1) > 0.0000001)
        self.d = [k1, k2, p1, p2, k3]


class VisualOdometry:
    def __init__(self, cam, annotations):
        self.frame_stage = STAGE_FIRST_FRAME
        self.cam = cam
        self.new_frame = None
        self.last_frame = None
        self.cur_R = np.eye(3)
        self.cur_t = np.zeros((3, 1))
        self.px_ref = None
        self.px_cur = None
        self.focal = cam.fx
        self.pp = (cam.cx, cam.cy)
        self.trueX, self.trueY, self.trueZ = 0, 0, 0

        # Cihaz tanıma
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Matching objesi oluşturma
        self.matching = Matching(superglue_conf).eval().to(self.device)

        with open(annotations) as f:
            self.annotations = f.readlines()

    def getAbsoluteScale(self, frame_id):
        ss = self.annotations[frame_id - 1].strip().split()
        x_prev = float(ss[3])
        y_prev = float(ss[7])
        z_prev = float(ss[11])
        ss = self.annotations[frame_id].strip().split()
        x = float(ss[3])
        y = float(ss[7])
        z = float(ss[11])
        self.trueX, self.trueY, self.trueZ = x, y, z
        return np.sqrt((x - x_prev) ** 2 + (y - y_prev) ** 2 + (z - z_prev) ** 2)

    def processFirstFrame(self):
        # İlk karede sadece superpoint ile özellik çıkarımı yapmak için aynı görüntüyü iki kere matching'e veriyoruz.
        # Bu sayede keypoints0'ı elde ederiz. Eşleşme olmayacağından px_ref'i keypoints0 olarak alacağız.
        inp0 = frame2tensor(self.new_frame, self.device)
        pred = self.matching({'image0': inp0, 'image1': inp0})
        keypoints0 = pred['keypoints0'][0].cpu().numpy()

        # İlk framede sadece px_ref keypointleri saklarız (asıl referans noktalar)
        self.px_ref = keypoints0
        self.frame_stage = STAGE_SECOND_FRAME

    def processSecondFrame(self):
        # İkinci karede artık last_frame ile new_frame arasında gerçek eşleşmeleri bulabiliriz.
        inp0 = frame2tensor(self.last_frame, self.device)
        inp1 = frame2tensor(self.new_frame, self.device)
        pred = self.matching({'image0': inp0, 'image1': inp1})

        keypoints0 = pred['keypoints0'][0].cpu().numpy()
        keypoints1 = pred['keypoints1'][0].cpu().numpy()
        matches = pred['matches0'][0].cpu().numpy()

        valid = matches > -1
        mkpts0 = keypoints0[valid]
        mkpts1 = keypoints1[matches[valid]]

        # mkpts0: önceki frameden match edilen noktalar
        # mkpts1: yeni frameden match edilen noktalar
        # Essential mat hesabı için:
        E, mask = cv2.findEssentialMat(mkpts1, mkpts0, focal=self.focal, pp=self.pp, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        _, self.cur_R, self.cur_t, mask = cv2.recoverPose(E, mkpts1, mkpts0, focal=self.focal, pp=self.pp)
        
        # px_ref'i güncelle: artık bu frame'in noktaları referans
        self.px_ref = mkpts1
        self.frame_stage = STAGE_DEFAULT_FRAME

    def processFrame(self, frame_id):
        # STAGE_DEFAULT_FRAME için benzer şekilde eşleşmeleri hesapla
        inp0 = frame2tensor(self.last_frame, self.device)
        inp1 = frame2tensor(self.new_frame, self.device)
        pred = self.matching({'image0': inp0, 'image1': inp1})

        keypoints0 = pred['keypoints0'][0].cpu().numpy()
        keypoints1 = pred['keypoints1'][0].cpu().numpy()
        matches = pred['matches0'][0].cpu().numpy()

        valid = matches > -1
        mkpts0 = keypoints0[valid]
        mkpts1 = keypoints1[matches[valid]]

        E, mask = cv2.findEssentialMat(mkpts1, mkpts0, focal=self.focal, pp=self.pp, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        _, R, t, mask = cv2.recoverPose(E, mkpts1, mkpts0, focal=self.focal, pp=self.pp)

        absolute_scale = self.getAbsoluteScale(frame_id)
        if absolute_scale > 0.1:
            self.cur_t += absolute_scale * self.cur_R.dot(t)
            self.cur_R = R.dot(self.cur_R)

        # Eğer eşleşen özellik sayısı çok az ise (mkpts1 < kMinNumFeature), 
        # yeniden özellik çıkarımı yapılabilir ancak SuperGlue otomatik ayıklama yaptığı için
        # bu aşamada ek bir işlem gerekmez.
        # Yine de çok az nokta varsa, ileride yeniden ayarlar yapılabilir.

        self.px_ref = mkpts1

    def update(self, img, frame_id):
        assert img.ndim == 2 and img.shape[0] == self.cam.height and img.shape[1] == self.cam.width, \
            "Frame: sağlanan görüntü kamera modeliyle aynı boyutta değil veya gri ölçekli değil"
        self.new_frame = img

        if self.frame_stage == STAGE_DEFAULT_FRAME:
            self.processFrame(frame_id)
        elif self.frame_stage == STAGE_SECOND_FRAME:
            self.processSecondFrame()
        elif self.frame_stage == STAGE_FIRST_FRAME:
            self.processFirstFrame()

        self.last_frame = self.new_frame
