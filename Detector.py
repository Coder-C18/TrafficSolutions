import torch

model = torch.hub.load('yolov5', 'custom', path='yolov5m.pt', force_reload=True, source='local')
model.eval()
model.conf = 0.35
model.classes = [1, 2, 3, 5, 7, 13]
model.iou_threshold =0.7
