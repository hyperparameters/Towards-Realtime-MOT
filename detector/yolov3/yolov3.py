from .model import Darknet, non_max_suppression, scale_coords
import torch

class YOLOv3:
    def __init__(self,opt):
        self.opt = opt
        self.model = Darknet(opt.cfg)
        self.model.load_state_dict(torch.load(opt.weights, map_location='cpu')['model'], strict=False)
        self.model.cuda().eval()

    def process(self, img):
        pred = self.model(img)
        return pred

    def postprocess(self, pred, img):
        pred = pred[pred[:, :, 4] > self.opt.conf_thres]
        # pred now has lesser number of proposals. Proposals rejected on basis of object confidence score
        if len(pred) > 0:
            dets = non_max_suppression(pred.unsqueeze(0), self.opt.conf_thres, self.opt.nms_thres)[0].cpu()
            # Final proposals are obtained in dets. Information of bounding box and embeddings also included
            # Next step changes the detection scales
            scale_coords(self.opt.img_size, dets[:, :4], img.shape).round()
            '''Detections is list of (x1, y1, x2, y2, object_conf, class_score, class_pred)'''
            # class_pred is the embeddings.
            return dets[:5], dets[5:]
        else:
            return [],[]