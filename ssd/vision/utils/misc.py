import time
import torch


def str2bool(s):
    return s.lower() in ('true', '1')


class Timer:
    def __init__(self):
        self.clock = {}

    def start(self, key="default"):
        self.clock[key] = time.time()

    def end(self, key="default"):
        if key not in self.clock:
            raise Exception(f"{key} is not in the clock.")
        interval = time.time() - self.clock[key]
        del self.clock[key]
        return interval
        

def save_checkpoint(epoch, net_state_dict, optimizer_state_dict, best_score, checkpoint_path, model_path):
    torch.save({
        'epoch': epoch,
        'model': net_state_dict,
        'optimizer': optimizer_state_dict,
        'best_score': best_score
    }, checkpoint_path)
    torch.save(net_state_dict, model_path)
        
        
def load_checkpoint(checkpoint_path):
    return torch.load(checkpoint_path)


def freeze_net_layers(net):
    for param in net.parameters():
        param.requires_grad = False


def store_labels(path, labels):
    with open(path, "w") as f:
        f.write("\n".join(labels))

def get_annotations_from_string(anns_string, nr_objs):
    anns_string = anns_string.replace("'",'')
    anns_string = anns_string.replace('[','')
    anns_string = anns_string.replace(']','')
    anns_string = anns_string.replace('\n','')

    if nr_objs > 1:
        splitted_objs = anns_string.split(',')
    else:
        splitted_objs = [anns_string]

    objects = []
                
    for object in splitted_objs:
        obj = []

        values = object.split(' ')

        for val in values:
            if val != '':
                obj.append(float(val))

        objects.append(obj)
    
    return objects