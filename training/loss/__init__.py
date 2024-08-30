from utils.registry import LOSSFUNC
from .cross_entropy_loss import CrossEntropyLoss
from .consistency_loss import ConsistencyCos
from .capsule_loss import CapsuleLoss
from .bce_loss import BCELoss
from .am_softmax import AMSoftmaxLoss
from .am_softmax import AMSoftmax_OHEM
from .contrastive_regularization import ContrastiveLoss
from .l1_loss import L1Loss
from .id_loss import IDLoss
from .vgg_loss import VGGLoss
from .js_loss import JS_Loss