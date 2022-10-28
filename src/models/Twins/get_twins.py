
from .twins import alt_gvt_small

def small_twins_model(args):
    return alt_gvt_small(img_size = args.image_size,  num_classes = args.num_classes, drop_path_rate = args.drop_path_rate)
