# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""
import errno
import json
import pickle as pkl
import os
import os.path as osp
import yaml

try:
    from PIL import Image, ImageFile
except ImportError:
    Image = None
    ImageFile = None

try:
    from easydict import EasyDict as edict
except ImportError:
    class edict(dict):
        def __getattr__(self, name):
            try:
                return self[name]
            except KeyError as error:
                raise AttributeError(name) from error

        def __setattr__(self, name, value):
            self[name] = value

        def __delattr__(self, name):
            try:
                del self[name]
            except KeyError as error:
                raise AttributeError(name) from error

if ImageFile is not None:
    ImageFile.LOAD_TRUNCATED_IMAGES = True


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    if Image is None:
        raise ImportError("Pillow is required to read images")
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def check_isfile(path):
    isfile = osp.isfile(path)
    if not isfile:
        print("=> Warning: no file found at '{}' (ignored)".format(path))
    return isfile


def read_json(fpath):
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj


def write_json(obj, fpath):
    mkdir_if_missing(osp.dirname(fpath))
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))


def get_text_embedding(path, length):
    with open(path, 'rb') as f:
        word_frequency = pkl.load(f)


def save_train_configs(path, args):
    if not os.path.exists(path):
        os.makedirs(path)
    config_dict = vars(args)
    for filename in ("config.yaml", "configs.yaml"):
        with open(osp.join(path, filename), 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)

def load_train_configs(path):
    with open(path, 'r') as f:
        args = yaml.load(f, Loader=yaml.FullLoader)
    return edict(args)
