import torch
import torch.nn.functional as F

class FocalLoss(torch.nn.Module):
    """
    Focal Loss implementation for multi-class classification.
    
    Args:
        alpha (float or tensor): Weighting factor for rare class (default: 1.0)
        gamma (float): Focusing parameter (default: 2.0)
        reduction (str): Specifies the reduction to apply to the output
    """
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean', ignore_index: int=-100):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index
        # print("created focal loss")
    
    def forward(self, inputs, targets):
        # Compute cross entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=self.ignore_index)
        # print("passing through the focal loss")
        # Compute probabilities
        pt = torch.exp(-ce_loss)
        
        # Compute focal loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
        
# Alternative implementation using cross entropy directly
class FocalLossV2(torch.nn.Module):
    """
    Alternative stable implementation using cross entropy loss directly.
    """
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean', ignore_index=-100):
        super(FocalLossV2, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index
    
    def forward(self, inputs, targets):
        # Input validation
        assert not torch.isnan(inputs).any(), "Input contains NaN"
        # assert not torch.isinf(inputs).any(), "Input contains Inf"
        
        # Compute cross entropy loss
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', 
                                ignore_index=self.ignore_index)
        
        # Compute probabilities using softmax
        with torch.no_grad():
            p = F.softmax(inputs, dim=-1)
            p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)
            
            # Handle ignore_index
            if self.ignore_index >= 0:
                mask = targets != self.ignore_index
                p_t = p_t * mask.float() + (1 - mask.float())  # Set ignored to 1 to avoid (1-1)^gamma = 0
        
        # Clamp probabilities to avoid numerical issues
        p_t = torch.clamp(p_t, min=1e-8, max=1.0 - 1e-8)
        
        # Compute focal weight
        focal_weight = (1 - p_t) ** self.gamma
        
        # Compute focal loss
        focal_loss = self.alpha * focal_weight * ce_loss
        
        # Handle ignore_index for reduction
        if self.ignore_index >= 0:
            mask = targets != self.ignore_index
            focal_loss = focal_loss * mask.float()
            
            if mask.sum() == 0:
                return torch.tensor(0.0, device=inputs.device, requires_grad=True)
        
        # Apply reduction
        if self.reduction == 'mean':
            if self.ignore_index >= 0:
                return focal_loss.sum() / mask.sum().clamp(min=1)
            else:
                return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss