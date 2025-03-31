import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd



def get_student_and_teacher_output(student_input, teacher_input,student_weights, teacher_weights, si_weights, teacher_selection_strategy = "weighted_activation_min"):
    x_student = nn.functional.conv2d
