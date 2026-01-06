import torch
from torchvision.models.detection import (
    retinanet_resnet50_fpn_v2,
    RetinaNet_ResNet50_FPN_V2_Weights
)

class RetinaCloakEngine:
    def __init__(self, device="cpu"):
        self.device = device
        weights = RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT
        self.model = retinanet_resnet50_fpn_v2(
            weights=weights,
            box_score_thresh=0.3
        ).to(self.device).eval()
        
        # Freeze all model parameters
        for p in self.model.parameters():
            p.requires_grad = False
        
        # Cache for intermediate features
        self.features = {}
        self._register_hooks()
    
    def _register_hooks(self):
        """
        Register forward hook to capture pre-sigmoid classification logits.
        Shape is approximately [N, A*C, H, W] where:
        N is batch size, A is anchors per location, C is number of classes
        """
        def hook(_, __, output):
            self.features["cls_logits"] = output
        
        self.model.head.classification_head.register_forward_hook(hook)
    
    def _cloak_loss(self):
        """
        Compute adversarial loss targeting pre-NMS classification logits.
        Uses two components:
        1. Global confidence suppression via logsumexp
        2. Entropy penalty to prevent anchor blooming
        """
        cls_logits = self.features.get("cls_logits")
        if cls_logits is None:
            return None
        
        # Suppress overall confidence across all anchors and classes
        loss_logits = torch.logsumexp(cls_logits, dim=1).mean()
        
        # Add entropy regularization to prevent spreading low-confidence detections
        prob = torch.sigmoid(cls_logits)
        loss_entropy = (prob * torch.log(prob + 1e-6)).mean()
        
        return loss_logits + 0.2 * loss_entropy
    
    def apply_cloak(self, image_tensor, steps=40, alpha=0.002, epsilon=0.03):
        """
        Apply PGD-based adversarial perturbation to evade object detection.
        Uses L-infinity constraint to keep perturbations imperceptible.
        
        Args:
            image_tensor: Input image tensor [1, 3, H, W] normalized to [0, 1]
            steps: Number of PGD iterations
            alpha: Step size per iteration
            epsilon: Maximum L-infinity perturbation bound
        
        Returns:
            Perturbed image tensor with same shape as input
        """
        image = image_tensor.clone().detach().to(self.device)
        
        # Initialize perturbation with small random noise
        delta = torch.empty_like(image).uniform_(
            -epsilon * 0.1, epsilon * 0.1
        )
        delta.requires_grad_(True)
        
        for step in range(steps):
            # Clear feature cache from previous iteration
            self.features.clear()
            
            # Create perturbed image and run forward pass
            perturbed = torch.clamp(image + delta, 0, 1)
            _ = self.model(perturbed)
            
            # Print debug info at first and last step
            if step == 0 or step == steps - 1:
                mean_conf = torch.sigmoid(
                    self.features["cls_logits"]
                ).mean().item()
                print(f"[step {step}] mean classification probability: {mean_conf:.6f}")
            
            # Compute loss
            loss = self._cloak_loss()
            if loss is None:
                break
            
            # Backward pass
            loss.backward()
            
            # Update perturbation
            with torch.no_grad():
                grad = delta.grad
                
                # Handle vanishing gradient with random noise injection
                if grad is None or grad.abs().sum() == 0:
                    delta += torch.randn_like(delta) * alpha * 0.1
                else:
                    # PGD update: move in direction of gradient sign
                    delta += alpha * grad.sign()
                
                # Project perturbation to L-infinity ball
                delta.clamp_(-epsilon, epsilon)
                
                # Ensure final image stays in valid range [0, 1]
                delta.copy_(torch.clamp(image + delta, 0, 1) - image)
            
            # Zero out gradients for next iteration
            delta.grad.zero_()
        
        # Return final perturbed image
        return torch.clamp(image + delta, 0, 1).detach()
    
    @torch.no_grad()
    def run_inference(self, image_tensor):
        """
        Run object detection inference on input image.
        
        Args:
            image_tensor: Input image tensor [1, 3, H, W]
        
        Returns:
            Dictionary with keys 'boxes', 'scores', 'labels'
        """
        result = self.model(image_tensor.to(self.device))
        return result[0] if result else {
            "boxes": [],
            "scores": [],
            "labels": []
        }