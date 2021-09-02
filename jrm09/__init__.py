__version__ = '0.0.0'

from . import Globals
from ._ReadCoeffs import _ReadCoeffs
from ._CoeffGrids import _CoeffGrids
from ._Schmidt import _Schmidt
from ._SphHarm import _SphHarm
from ._Legendre import _Legendre
from .Model import Model
from .ModelCart import ModelCart
from .Test import Test,TestCompFunc


#this code will be removed after verification
from .CompFunc import CompFunc,CompFuncCart
