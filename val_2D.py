import torch
from medpy import metric


def calculate_metric_percase(pred, gt):
    pred[pred > 0.6] = 1
    pred[pred <= 0.6] = 0
    dice = metric.binary.dc(pred, gt)
    return dice


def test_single_volume(image, label, net, mod):
    label = label.squeeze(0).cpu().detach().numpy()
    net.eval()
    with torch.no_grad():
        if mod == "ctc" or mod == '' or mod == 'transunet':
            _, out = net(image)
        else:
            out = net(image)  #_,
        out = out.squeeze(0).cpu().detach().numpy()
    metric_list = calculate_metric_percase(out, label)
    return metric_list