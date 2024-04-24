import math
import numpy as np
import torch 

def set_timesteps_patched(self, num_inference_steps: int, device = None):
    self.num_inference_steps = num_inference_steps
    
    ramp = np.linspace(0, 1, self.num_inference_steps)
    sigmas = torch.linspace(math.log(self.config.sigma_min), math.log(self.config.sigma_max), len(ramp)).exp().flip(0)
    
    sigmas = (sigmas).to(dtype=torch.float32, device=device)
    self.timesteps = self.precondition_noise(sigmas)
    
    self.sigmas = torch.cat([sigmas, torch.zeros(1, device=sigmas.device)])
    self._step_index = None
    self._begin_index = None
    self.sigmas = self.sigmas.to("cpu")  # to avoid too much CPU/GPU communication
