from .cma import build as few_build2
from .self_cma import build as few_build4
from .base_cma import build as base_build

def build_model(args):
    return build(args)

def few_build_model(args):
    return few_build2(args)


def base_build_model(args):
    return base_build(args)

def self_few_build(args):
    return few_build4(args)