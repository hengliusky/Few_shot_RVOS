from .referformer import build
from .few_referformer import build as few_build
from .few_referformer1 import build as few_build1
from .few_referformer2 import build as few_build2
from .few_referformer3 import build as few_build3
from .few_referformer4 import build as few_build4
from .base_few_referformer import build as base_build

def build_model(args):
    return build(args)

def few_build_model(args):
    return few_build(args)

def few_build_model1(args):
    return few_build1(args)

def few_build_model2(args):
    return few_build2(args)

def few_build_model3(args):
    return few_build3(args)

def base_build_model(args):
    return base_build(args)

def few_build_model4(args):
    return few_build4(args)