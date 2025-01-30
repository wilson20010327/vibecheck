import torch
import torch.nn as nn

# Soft update function
def soft_update_target_network(target_network, main_network, tau):
    for target_param, main_param in zip(target_network.parameters(), main_network.parameters()):
        target_param.data.copy_(
            tau * main_param.data + (1.0 - tau) * target_param.data
        )

# Hard update function
def hard_update_target_network(target_network, main_network):
    target_network.load_state_dict(main_network.state_dict())