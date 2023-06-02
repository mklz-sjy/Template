from tensorboardX import SummaryWriter
import os
import torch


class Recoder:
    def __init__(self):
        self.metrics = {}

    def record(self, name, value):
        if name in self.metrics.keys():
            self.metrics[name].append(value)
        else:
            self.metrics[name] = [value]

    def summary(self):
        kvs = {}
        for key in self.metrics.keys():
            kvs[key] = sum(self.metrics[key]) / len(self.metrics[key])
            del self.metrics[key][:]
            self.metrics[key] = []
        return kvs


class Logger:
    def __init__(self, args):
        self.writer = SummaryWriter(os.path.join(args.model_dir, args.exp_name,'logs'))
        self.recoder = Recoder()
        self.config = args
        self.model_dir = os.path.join(args.model_dir, args.exp_name)

    def tensor2img(self, tensor):
        # implement according to your data, for example call viz.py
        return tensor.cpu().detach().numpy()

    def record_scalar(self, name, value):
        self.recoder.record(name, value)

    def save_curves(self, epoch, fd):
        kvs = self.recoder.summary()
        fd.write('EPOCH_{} '.format(epoch))
        for key in kvs.keys():
            self.writer.add_scalar(key, kvs[key], epoch)
            # write metrics to result dir,
            # you can also use pandas or other methods for better stats
            fd.write("{}:{m:06f}    ".format(key, m=kvs[key].item()))

    def save_imgs(self, names2imgs, epoch):
        for name in names2imgs.keys():
            self.writer.add_image(name, self.tensor2img(names2imgs[name]), epoch)

    def save_check_point(self, model, epoch, metrics):
        metrics_text = ''
        for key in metrics.keys():
            metrics_text+='{}_{:06f}'.format(key, metrics[key])
        model_name = '{epoch:02d}_{m}.pth'.format(epoch=epoch, m= metrics_text)
        path = os.path.join(self.model_dir, "checkpoints", model_name)
        # don't save model, which depends on python path
        # save model state dict
        torch.save(model.state_dict(), path)
