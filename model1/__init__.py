from .referformer import build
from .few_referformer import build as few_build


def build_model(args):
    return build(args)


def few_build_model(args):
    return few_build(args)