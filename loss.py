import torch
import torch.nn as nn
from utils import iou

class YoLoss(nn.Module):
    def __init__(self, grid_size=7, n_boxes=2, n_classes=20, lambda_coord=5, lambda_noobj=0.5):
        super(YoLoss, self).__init__()
        self.mse          = nn.MSELoss(reduction='sum')
        self.grid_size    = grid_size
        self.n_boxes      = n_boxes
        self.n_classes    = n_classes
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def forward(self, preds, target):
        preds = torch.reshape(preds, (-1, self.grid_size, self.grid_size, self.n_classes + self.n_boxes * 5))
        iou_1 = iou(preds[..., 21:25], target[..., 21:25]).unsqueeze(0)
        iou_2 = iou(preds[..., 26:30], target[..., 21:25]).unsqueeze(0)
        ious = torch.cat([iou_1, iou_2], dim=0)
        best_box = torch.argmax(ious, dim=0)
        exists_box = target[..., 20].unsqueeze(3)

        # Box coordinates

        box_preds = exists_box * ((1 - best_box) * preds[..., 21:25] + best_box * preds[..., 26:30])
        
        box_targets = exists_box * target[..., 21:25]

        # width and height
        # torch.abs pra nao tirar raiz negativa 
        # somar 1e-8 pra evitar gradiente de 0 (infinito)
        # torch.sign pra voltar o sinal do gradiente
        box_preds[..., 2:4] = torch.sign(box_preds[..., 2:4]) * torch.sqrt(torch.abs(box_preds[..., 2:4] + 1e-8))
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        box_loss = self.mse(torch.flatten(box_preds, end_dim=-2), torch.flatten(box_targets, end_dim=-2))

        # object loss
        pred_box = (best_box *  preds[..., 25:26] + (1 - best_box) * preds[..., 20:21])
        
        object_loss = self.mse(torch.flatten(exists_box * pred_box), torch.flatten(exists_box * target[..., 20:21]))

        # no object loss
        no_object_loss = self.mse(torch.flatten((1 - exists_box) * preds[..., 20:21], start_dim=1), torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1))

        # class loss
        class_loss = self.mse(torch.flatten(exists_box * preds[..., :20], end_dim=-2), torch.flatten(exists_box * target[..., :20], end_dim=-2))

        loss = (self.lambda_coord * box_loss) + object_loss + (self.lambda_noobj * no_object_loss) + class_loss

        return loss
