from easydict import EasyDict as edict
import yaml
import argparse


def load_config():
    parser = argparse.ArgumentParser(description="CardNN experiment protocol.")
    parser.add_argument(
        "--cfg",
        "--config",
        dest="cfg_file",
        help="path to the configuration file",
        default=None,
        type=str,
    )
    args = parser.parse_args()
    if args.cfg_file is None:
        raise ValueError("Please specify path to the configuration file!")
    return cfg_from_file(args.cfg_file)


def load_config_egnpb_facility_location():
    parser = argparse.ArgumentParser(description="CardNN experiment protocol.")
    parser.add_argument(
        "--cfg",
        "--config",
        dest="cfg_file",
        help="path to the configuration file",
        default=None,
        type=str,
    )
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--reg", type=float, default=0.1)
    parser.add_argument("--timestamp", type=str, default="")
    args = parser.parse_args()
    if args.cfg_file is None:
        raise ValueError("Please specify path to the configuration file!")
    return args, cfg_from_file(args.cfg_file)

def load_config_egnpb_facility_location_aligned():
    parser = argparse.ArgumentParser(description="CardNN experiment protocol.")
    parser.add_argument(
        "--cfg",
        "--config",
        dest="cfg_file",
        help="path to the configuration file",
        default=None,
        type=str,
    )    
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--reg", type=float, default=0.1)
    parser.add_argument("--timestamp", type=str, default="")
    
    # additional arguments for the "aligned" version
    parser.add_argument("--temp", type=float, default=1.0)  # softmax temperature
    # temperature scheduling: temp = temp * {tempdecay} every {tempfreq} epochs
    parser.add_argument("--tempdecay", type=float, default=1.0)
    parser.add_argument("--tempfreq", type=int, default=1)
    
    parser.add_argument("--regratio", type=float, default=1.0)
    # regratio scheduling: regratio = regratio * {regratiodecay} every {regratiofreq} epochs
    parser.add_argument("--regratiodecay", type=float, default=1.0)
    parser.add_argument("--regratiofreq", type=int, default=1)
        
    parser.add_argument("--shift", action="store_true")  # shift or assign
    parser.add_argument("--signsigmoid", type=float, default=-1)  # sigmoid scaling for sign (<0 = no sigmoid)
    
    parser.add_argument("--gumbel", action="store_true")  # gumbel softmax or softmax
    parser.add_argument("--gbonehot", action="store_true")  # gumbel uses onehot (straight-through) or not    
    
    args = parser.parse_args()
    if args.cfg_file is None:
        raise ValueError("Please specify path to the configuration file!")
    return args, cfg_from_file(args.cfg_file)


def load_config_egnpb_facility_location_aligned_greedy():
    parser = argparse.ArgumentParser(description="CardNN experiment protocol.")
    parser.add_argument(
        "--cfg",
        "--config",
        dest="cfg_file",
        help="path to the configuration file",
        default=None,
        type=str,
    )
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--reg", type=float, default=0.1)
    parser.add_argument("--timestamp", type=str, default="")
    
    # additional arguments for the "aligned" version
    
    parser.add_argument("--temp1", type=float, default=1.0)  # softmax temperature
    # temperature scheduling: temp = temp * {tempdecay} every {tempfreq} epochs
    parser.add_argument("--tempdecay1", type=float, default=1.0)
    parser.add_argument("--tempfreq1", type=int, default=1)
    
    parser.add_argument("--temp2", type=float, default=1.0)  # softmax temperature
    # temperature scheduling: temp = temp * {tempdecay} every {tempfreq} epochs
    parser.add_argument("--tempdecay2", type=float, default=1.0)
    parser.add_argument("--tempfreq2", type=int, default=1)
    
    parser.add_argument("--regratio", type=float, default=1.0)
    # regratio scheduling: regratio = regratio * {regratiodecay} every {regratiofreq} epochs
    parser.add_argument("--regratiodecay", type=float, default=1.0)
    parser.add_argument("--regratiofreq", type=int, default=1)
        
    parser.add_argument("--shift", action="store_true")  # shift or assign
    parser.add_argument("--signsigmoid", type=float, default=-1)  # sigmoid scaling for sign (<0 = no sigmoid)
    
    parser.add_argument("--gumbel1", action="store_true")  # gumbel softmax or softmax
    parser.add_argument("--gbonehot1", action="store_true")  # gumbel uses onehot (straight-through) or not
    
    parser.add_argument("--gumbel2", action="store_true")  # gumbel softmax or softmax
    parser.add_argument("--gbonehot2", action="store_true")  # gumbel uses onehot (straight-through) or not
    
    parser.add_argument("--aligniter", type=int, default=1)  # number of alignment iterations

    args = parser.parse_args()
    if args.cfg_file is None:
        raise ValueError("Please specify path to the configuration file!")
    return args, cfg_from_file(args.cfg_file)

def load_config_egnpb_facility_location_aligned_combined():
    parser = argparse.ArgumentParser(description="CardNN experiment protocol.")
    parser.add_argument(
        "--cfg",
        "--config",
        dest="cfg_file",
        help="path to the configuration file",
        default=None,
        type=str,
    )    
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--reg", type=float, default=0.1)
    parser.add_argument("--timestamp", type=str, default="")
    
    # additional arguments for the "aligned" version
    
    parser.add_argument("--temp1", type=float, default=1.0)  # softmax temperature
    # temperature scheduling: temp = temp * {tempdecay} every {tempfreq} epochs
    parser.add_argument("--tempdecay1", type=float, default=1.0)
    parser.add_argument("--tempfreq1", type=int, default=1)        
    
    parser.add_argument("--regratio", type=float, default=1.0)
    # regratio scheduling: regratio = regratio * {regratiodecay} every {regratiofreq} epochs
    parser.add_argument("--regratiodecay", type=float, default=1.0)
    parser.add_argument("--regratiofreq", type=int, default=1)
        
    parser.add_argument("--signsigmoid", type=float, default=-1)  # sigmoid scaling for sign (<0 = no sigmoid)
    
    parser.add_argument("--gumbel1", action="store_true")  # gumbel softmax or softmax
    parser.add_argument("--gbonehot1", action="store_true")  # gumbel uses onehot (straight-through) or not        

    parser.add_argument("--no_align", action="store_true")  # no alignment
    parser.add_argument("--simultaneous", action="store_true")  # using simultaneous alignment or not
    parser.add_argument("--before_and_after", action="store_true")  # use both before and after alignment
    parser.add_argument("--gradient_match", action="store_true")  # TODO: gradient matching for before and after alignment
    parser.add_argument("--accumulate_grad", action="store_true")  # accumulate gradient for all the training data or not
    parser.add_argument("--aligniter", type=int, default=1)  # number of alignment iterations
    parser.add_argument("--warmup_iter", type=int, default=0)  # number of warmup iterations
    parser.add_argument("--scr", type=int, default=10)
    parser.add_argument("--sct", type=int, default=0)
    
    args = parser.parse_args()
    if args.cfg_file is None:
        raise ValueError("Please specify path to the configuration file!")
    return args, cfg_from_file(args.cfg_file)

def load_config_egnpb_facility_location_aligned_greedy_flatten():
    parser = argparse.ArgumentParser(description="CardNN experiment protocol.")
    parser.add_argument(
        "--cfg",
        "--config",
        dest="cfg_file",
        help="path to the configuration file",
        default=None,
        type=str,
    )    
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--reg", type=float, default=0.1)
    parser.add_argument("--timestamp", type=str, default="")
    
    # additional arguments for the "aligned" version
    
    parser.add_argument("--temp1", type=float, default=1.0)  # softmax temperature
    # temperature scheduling: temp = temp * {tempdecay} every {tempfreq} epochs
    parser.add_argument("--tempdecay1", type=float, default=1.0)
    parser.add_argument("--tempfreq1", type=int, default=1)        
    
    parser.add_argument("--regratio", type=float, default=1.0)
    # regratio scheduling: regratio = regratio * {regratiodecay} every {regratiofreq} epochs
    parser.add_argument("--regratiodecay", type=float, default=1.0)
    parser.add_argument("--regratiofreq", type=int, default=1)
        
    parser.add_argument("--shift", action="store_true")  # shift or assign
    parser.add_argument("--signsigmoid", type=float, default=-1)  # sigmoid scaling for sign (<0 = no sigmoid)
    
    parser.add_argument("--gumbel1", action="store_true")  # gumbel softmax or softmax
    parser.add_argument("--gbonehot1", action="store_true")  # gumbel uses onehot (straight-through) or not        
    
    parser.add_argument("--aligniter", type=int, default=1)  # number of alignment iterations
    
    args = parser.parse_args()
    if args.cfg_file is None:
        raise ValueError("Please specify path to the configuration file!")
    return args, cfg_from_file(args.cfg_file)


def load_config_egnpb_max_covering():
    parser = argparse.ArgumentParser(description="CardNN experiment protocol.")
    parser.add_argument(
        "--cfg",
        "--config",
        dest="cfg_file",
        help="path to the configuration file",
        default=None,
        type=str,
    )
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--reg", type=float, default=500)
    parser.add_argument("--timestamp", type=str, default="")   
    
    args = parser.parse_args()
    if args.cfg_file is None:
        raise ValueError("Please specify path to the configuration file!")
    return args, cfg_from_file(args.cfg_file)


def load_config_egnpb_max_covering_aligned():
    parser = argparse.ArgumentParser(description="CardNN experiment protocol.")
    parser.add_argument(
        "--cfg",
        "--config",
        dest="cfg_file",
        help="path to the configuration file",
        default=None,
        type=str,
    )
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--reg", type=float, default=500)
    parser.add_argument("--timestamp", type=str, default="")
    
        # additional arguments for the "aligned" version
    parser.add_argument("--temp", type=float, default=1.0)  # softmax temperature
    # temperature scheduling: temp = temp * {tempdecay} every {tempfreq} epochs
    parser.add_argument("--tempdecay", type=float, default=1.0)
    parser.add_argument("--tempfreq", type=int, default=1)
    
    parser.add_argument("--regratio", type=float, default=1.0)
    # regratio scheduling: regratio = regratio * {regratiodecay} every {regratiofreq} epochs
    parser.add_argument("--regratiodecay", type=float, default=1.0)
    parser.add_argument("--regratiofreq", type=int, default=1)
        
    parser.add_argument("--shift", action="store_true")  # shift or assign
    parser.add_argument("--signsigmoid", type=float, default=-1)  # sigmoid scaling for sign (<0 = no sigmoid)
    
    parser.add_argument("--gumbel", action="store_true")  # gumbel softmax or softmax
    parser.add_argument("--gbonehot", action="store_true")  # gumbel uses onehot (straight-through) or not
    
    args = parser.parse_args()
    if args.cfg_file is None:
        raise ValueError("Please specify path to the configuration file!")
    return args, cfg_from_file(args.cfg_file)

def load_config_egnpb_max_covering_aligned_greedy():
    parser = argparse.ArgumentParser(description="CardNN experiment protocol.")
    parser.add_argument(
        "--cfg",
        "--config",
        dest="cfg_file",
        help="path to the configuration file",
        default=None,
        type=str,
    )
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--reg", type=float, default=500)
    parser.add_argument("--timestamp", type=str, default="")
    
        # additional arguments for the "aligned" version
    parser.add_argument("--temp1", type=float, default=1.0)  # softmax temperature
    # temperature scheduling: temp = temp * {tempdecay} every {tempfreq} epochs
    parser.add_argument("--tempdecay1", type=float, default=1.0)
    parser.add_argument("--tempfreq1", type=int, default=1)
    
    parser.add_argument("--temp2", type=float, default=1.0)  # softmax temperature
    # temperature scheduling: temp = temp * {tempdecay} every {tempfreq} epochs
    parser.add_argument("--tempdecay2", type=float, default=1.0)
    parser.add_argument("--tempfreq2", type=int, default=1)
    
    parser.add_argument("--regratio", type=float, default=1.0)
    # regratio scheduling: regratio = regratio * {regratiodecay} every {regratiofreq} epochs
    parser.add_argument("--regratiodecay", type=float, default=1.0)
    parser.add_argument("--regratiofreq", type=int, default=1)
        
    parser.add_argument("--shift", action="store_true")  # shift or assign
    parser.add_argument("--signsigmoid", type=float, default=-1)  # sigmoid scaling for sign (<0 = no sigmoid)
    
    parser.add_argument("--gumbel1", action="store_true")  # gumbel softmax or softmax
    parser.add_argument("--gbonehot1", action="store_true")  # gumbel uses onehot (straight-through) or not
    
    parser.add_argument("--gumbel2", action="store_true")  # gumbel softmax or softmax
    parser.add_argument("--gbonehot2", action="store_true")  # gumbel uses onehot (straight-through) or not
    
    parser.add_argument("--aligniter", type=int, default=1)  # number of alignment iterations
    
    args = parser.parse_args()
    if args.cfg_file is None:
        raise ValueError("Please specify path to the configuration file!")
    return args, cfg_from_file(args.cfg_file)

def load_config_egnpb_max_covering_aligned_greedy_flatten():
    parser = argparse.ArgumentParser(description="CardNN experiment protocol.")
    parser.add_argument(
        "--cfg",
        "--config",
        dest="cfg_file",
        help="path to the configuration file",
        default=None,
        type=str,
    )
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--reg", type=float, default=500)
    parser.add_argument("--timestamp", type=str, default="")
    
        # additional arguments for the "aligned" version
    parser.add_argument("--temp1", type=float, default=1.0)  # softmax temperature
    # temperature scheduling: temp = temp * {tempdecay} every {tempfreq} epochs
    parser.add_argument("--tempdecay1", type=float, default=1.0)
    parser.add_argument("--tempfreq1", type=int, default=1)        
    
    parser.add_argument("--regratio", type=float, default=1.0)
    # regratio scheduling: regratio = regratio * {regratiodecay} every {regratiofreq} epochs
    parser.add_argument("--regratiodecay", type=float, default=1.0)
    parser.add_argument("--regratiofreq", type=int, default=1)
        
    parser.add_argument("--shift", action="store_true")  # shift or assign
    parser.add_argument("--signsigmoid", type=float, default=-1)  # sigmoid scaling for sign (<0 = no sigmoid)
    
    parser.add_argument("--gumbel1", action="store_true")  # gumbel softmax or softmax
    parser.add_argument("--gbonehot1", action="store_true")  # gumbel uses onehot (straight-through) or not    
    
    parser.add_argument("--aligniter", type=int, default=1)  # number of alignment iterations
    
    args = parser.parse_args()
    if args.cfg_file is None:
        raise ValueError("Please specify path to the configuration file!")
    return args, cfg_from_file(args.cfg_file)

def load_config_egnpb_max_covering_aligned_combined():
    parser = argparse.ArgumentParser(description="CardNN experiment protocol.")
    parser.add_argument(
        "--cfg",
        "--config",
        dest="cfg_file",
        help="path to the configuration file",
        default=None,
        type=str,
    )
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--reg", type=float, default=500)
    parser.add_argument("--timestamp", type=str, default="")
    
    # additional arguments for the "aligned" version
    
    parser.add_argument("--temp1", type=float, default=1.0)  # softmax temperature
    # temperature scheduling: temp = temp * {tempdecay} every {tempfreq} epochs
    parser.add_argument("--tempdecay1", type=float, default=1.0)
    parser.add_argument("--tempfreq1", type=int, default=1)        
    
    parser.add_argument("--regratio", type=float, default=1.0)
    # regratio scheduling: regratio = regratio * {regratiodecay} every {regratiofreq} epochs
    parser.add_argument("--regratiodecay", type=float, default=1.0)
    parser.add_argument("--regratiofreq", type=int, default=1)
        
    parser.add_argument("--signsigmoid", type=float, default=-1)  # sigmoid scaling for sign (<0 = no sigmoid)
    
    parser.add_argument("--gumbel1", action="store_true")  # gumbel softmax or softmax
    parser.add_argument("--gbonehot1", action="store_true")  # gumbel uses onehot (straight-through) or not        

    parser.add_argument("--no_align", action="store_true")  # no alignment
    parser.add_argument("--simultaneous", action="store_true")  # using simultaneous alignment or not
    parser.add_argument("--before_and_after", action="store_true")  # use both before and after alignment
    parser.add_argument("--gradient_match", action="store_true")  # TODO: gradient matching for before and after alignment
    parser.add_argument("--accumulate_grad", action="store_true")  # accumulate gradient for all the training data or not
    parser.add_argument("--aligniter", type=int, default=1)  # number of alignment iterations
    parser.add_argument("--warmup_iter", type=int, default=0)  # number of warmup iterations
    
    args = parser.parse_args()
    if args.cfg_file is None:
        raise ValueError("Please specify path to the configuration file!")
    return args, cfg_from_file(args.cfg_file)


def load_config_egnpb_robust_coloring():
    parser = argparse.ArgumentParser(description="CardNN experiment protocol.")
    parser.add_argument(
        "--cfg",
        "--config",
        dest="cfg_file",
        help="path to the configuration file",
        default=None,
        type=str,
    )
    parser.add_argument("--lr", type=float)
    parser.add_argument("--color", type=int)
    parser.add_argument("--regtrain", type=float)
    parser.add_argument("--regtest", type=float)
    args = parser.parse_args()
    if args.cfg_file is None:
        raise ValueError("Please specify path to the configuration file!")
    return args, cfg_from_file(args.cfg_file)


def cfg_from_file(filename, cfg=None):
    """Load a config file and merge it into the default options."""
    with open(filename, "r") as f:
        yaml_cfg = edict(yaml.full_load(f))

    if cfg is None:
        cfg = edict()

    _merge_a_into_b(yaml_cfg, cfg)
    return cfg


def _merge_a_into_b(a, b, strict=False):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.items():
        if strict:
            # a must specify keys that are in b
            if k not in b:
                raise KeyError("{} is not a valid config key".format(k))

            # the types must match, too
            if type(b[k]) is not type(v):
                if type(b[k]) is float and type(v) is int:
                    v = float(v)
                else:
                    if not k in ["CLASS"]:
                        raise ValueError(
                            "Type mismatch ({} vs. {}) for config key: {}".format(
                                type(b[k]), type(v), k
                            )
                        )

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print("Error under config key: {}".format(k))
                raise
        else:
            b[k] = v
