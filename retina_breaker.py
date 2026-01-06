import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision.models.detection import (
    retinanet_resnet50_fpn_v2,
    RetinaNet_ResNet50_FPN_V2_Weights,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
class RetinaCloakEngine:
    def __init__(self, device=device):
        self.device = device
        weights = RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT
        self.model = retinanet_resnet50_fpn_v2(weights=weights)
        self.model.to(self.device)

        for p in self.model.parameters():
            p.requires_grad = False

        self.model.eval()
        self._cls_logits = None

        self._register_hook()
        print("RetinaCloakEngine initialized on device:", self.device)

    def _register_hook(self):
        def hook(module, inputs, output):
            self._cls_logits = output

        self.model.head.classification_head.register_forward_hook(hook)

    def calculate_loss(self, cls_logits, score_threshold=0.45, margin=0.45):
        probs = torch.sigmoid(cls_logits)
        max_scores, labels = probs.max(dim=-1)

        active_mask = max_scores > score_threshold
        if active_mask.sum() == 0:
            return max_scores.sum() * 0.0

        active_scores = max_scores[active_mask]
        active_labels = labels[active_mask]

        loss = F.relu(active_scores - margin).sum()

        person_mask = active_labels == 1
        if person_mask.any():
            loss = loss + F.relu(active_scores[person_mask] - margin).sum()

        print("Anchor + person logit suppression loss:", loss.item())
        return loss

    def apply_cloak(self, image_tensor, steps=40, alpha=0.002, epsilon=0.03):
        image_tensor = image_tensor.clone().detach().to(self.device)

        delta = torch.zeros_like(image_tensor, requires_grad=True)
        delta.data.uniform_(-epsilon * 0.05, epsilon * 0.05)

        optimizer = optim.Adam([delta], lr=alpha)

        print("Starting attack")
        print("Steps:", steps, "Alpha:", alpha, "Epsilon:", epsilon)

        for step in range(steps):
            self._cls_logits = None
            perturbed = torch.clamp(image_tensor + delta, 0, 1)

            self.model(perturbed)

            cls_logits = self._cls_logits
            cls_logits = cls_logits.reshape(-1, cls_logits.shape[-1])

            loss = self.calculate_loss(cls_logits)

            optimizer.zero_grad()
            loss.backward()

            delta.data = delta.data - alpha * delta.grad.sign()
            delta.data = torch.clamp(delta.data, -epsilon, epsilon)
            delta.data = torch.clamp(image_tensor + delta.data, 0, 1) - image_tensor

            if step % 5 == 0:
                print(
                    "Step",
                    step,
                    "loss:",
                    loss.item(),
                    "delta max:",
                    delta.abs().max().item(),
                )

        final_image = torch.clamp(image_tensor + delta, 0, 1).detach()
        print("Attack finished")
        return final_image

    @torch.no_grad()
    def run_inference(self, image_tensor):
        results = self.model(image_tensor)
        scores = results[0]["scores"] if results else []
        active = (scores > 0.45).sum().item() if len(scores) else 0
        print("Inference complete, detections:", active)
        return results[0] if results else {"boxes": [], "scores": [], "labels": []}
