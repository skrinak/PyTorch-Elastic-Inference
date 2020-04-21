import torch, torchvision
import subprocess

# Toggle inference mode
model = torchvision.models.densenet121(pretrained=True).eval()
cv_input = torch.rand(1,3,224,224)
model = torch.jit.trace(model,cv_input)
torch.jit.save(model, 'model.pt')
subprocess.call(['tar', '-czvf', 'densenet121_traced.tar.gz', 'model.pt'])
