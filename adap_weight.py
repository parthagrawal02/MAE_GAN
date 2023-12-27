import numpy as np
import torch


def aw_loss(L_mae, L_adv, Gen_opt, Gen_net):
    # resetting gradient back to zero
    Gen_opt.zero_grad()

    # computing real batch gradient 
    L_mae.backward(retain_graph=True)
    # tensor with real gradients
    grad_real_tensor = [param.grad.clone() for _, param in Gen_net.named_parameters() if param.grad is not None]
    grad_real_list = torch.cat([grad.reshape(-1) for grad in grad_real_tensor], dim=0)
    # calculating the norm of the real gradient
    rdotr = torch.dot(grad_real_list, grad_real_list).item() 
    mae_norm = np.sqrt(rdotr)
    # resetting gradient back to zero
    Gen_opt.zero_grad()

    # computing fake batch gradient 
    L_adv.backward(retain_graph = True)#(retain_graph=True)
    # tensor with real gradients
    grad_fake_tensor = [param.grad.clone() for _, param in Gen_net.named_parameters() if param.grad is not None]
    grad_fake_list = torch.cat([grad.reshape(-1) for grad in grad_fake_tensor], dim=0)
    # calculating the norm of the fake gradient
    fdotf = torch.dot(grad_fake_list, grad_fake_list).item() + 1e-6 # 1e-4 added to avoid division by zero
    adv_norm = np.sqrt(fdotf)
    
    # resetting gradient back to zero
    Gen_opt.zero_grad()

    # dot product between real and fake gradients
    adaptive_weight = mae_norm/adv_norm
    # print(adaptive_weight)
    # print(L_mae)
    # print(L_adv)
    # calculating aw_loss
    aw_loss = L_mae + adaptive_weight * L_adv

    # updating gradient, i.e. getting aw_loss gradient
    for index, (_, param) in enumerate(Gen_net.named_parameters()):
        # print(grad_real_tensor[index])
        # print(grad_fake_tensor[index])
        if param.grad is not None:
            param.grad = grad_real_tensor[index] + adaptive_weight * grad_fake_tensor[index]

    return aw_loss