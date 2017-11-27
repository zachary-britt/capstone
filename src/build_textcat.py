# coding: utf8
from __future__ import unicode_literals

import numpy
import numpy as np
from thinc.v2v import Model, Maxout, Softmax, Affine, ReLu
from thinc.i2v import HashEmbed, StaticVectors
from thinc.t2t import ExtractWindow, ParametricAttention
from thinc.t2v import Pooling, sum_pool
from thinc.misc import Residual
from thinc.misc import LayerNorm as LN
from thinc.api import add, layerize, chain, clone, concatenate, with_flatten
from thinc.api import FeatureExtracter, with_getitem, flatten_add_lengths
from thinc.api import uniqued, wrap, noop
from thinc.linear.linear import LinearModel
from thinc.neural.ops import NumpyOps, CupyOps
from thinc.neural.util import get_array_module, copy_array
from thinc.neural._lsuv import svd_orthonormal
from thinc.neural.optimizers import Adam

from thinc import describe
from thinc.describe import Dimension, Synapses, Biases, Gradient
from thinc.neural._classes.affine import _set_dimensions_if_needed
import thinc.extra.load_nlp

from spacy.attrs import ID, ORTH, LOWER, NORM, PREFIX, SUFFIX, SHAPE
from spacy import util


''' Adapted from https://github.com/explosion/spaCy/blob/master/spacy/_ml.py '''

from spacy._ml import *

@layerize
def preprocess_doc(docs, drop=0.):
    keys = [doc.to_array([LOWER]) for doc in docs]
    ops = Model.ops
    lengths = ops.asarray([arr.shape[0] for arr in keys])
    keys = ops.xp.concatenate(keys)
    vals = ops.allocate(keys.shape[0]) + 1
    return (keys, vals, lengths), None

@layerize
def _preprocess_doc(docs, drop=0.):
    keys = [doc.to_array(LOWER) for doc in docs]
    ops = Model.ops
    # The dtype here matches what thinc is expecting -- which differs per
    # platform (by int definition). This should be fixed once the problem
    # is fixed on Thinc's side.
    lengths = ops.asarray([arr.shape[0] for arr in keys], dtype=numpy.int_)
    keys = ops.xp.concatenate(keys)
    vals = ops.allocate(keys.shape) + 1.
    return (keys, vals, lengths), None



def build_text_classifier(nr_class, width=64, **cfg):
    nr_vector = cfg.get('nr_vector', 5000)
    pretrained_dims = cfg.get('pretrained_dims', 0)
    with Model.define_operators({'>>': chain, '+': add, '|': concatenate,
                                 '**': clone}):
        if cfg.get('low_data') and pretrained_dims:
            model = (
                SpacyVectors
                >> flatten_add_lengths
                >> with_getitem(0, Affine(width, pretrained_dims))
                >> ParametricAttention(width)
                >> Pooling(sum_pool)
                >> Residual(ReLu(width, width)) ** 2
                >> zero_init(Affine(nr_class, width, drop_factor=0.0))
                >> logistic
            )
            return model

        lower = HashEmbed(width, nr_vector, column=1)
        prefix = HashEmbed(width//2, nr_vector, column=2)
        suffix = HashEmbed(width//2, nr_vector, column=3)
        shape = HashEmbed(width//2, nr_vector, column=4)

        trained_vectors = (
            FeatureExtracter([ORTH, LOWER, PREFIX, SUFFIX, SHAPE, ID])
            >> with_flatten(
                uniqued(
                    (lower | prefix | suffix | shape)
                    >> LN(Maxout(width, width+(width//2)*3)),
                    column=0
                )
            )
        )

        if pretrained_dims:
            static_vectors = (
                SpacyVectors
                >> with_flatten(Affine(width, pretrained_dims))
            )
            # TODO Make concatenate support lists
            vectors = concatenate_lists(trained_vectors, static_vectors)
            vectors_width = width*2
        else:
            vectors = trained_vectors
            vectors_width = width
            static_vectors = None
        cnn_model = (
            vectors
            >> with_flatten(
                LN(Maxout(width, vectors_width))
                >> Residual(
                    (ExtractWindow(nW=1) >> LN(Maxout(width, width*3)))
                ) ** 2, pad=2
            )
            >> flatten_add_lengths
            >> ParametricAttention(width)
            >> Pooling(sum_pool)
            >> Residual(zero_init(Maxout(width, width)))
            >> zero_init(Affine(nr_class, width, drop_factor=0.0))
        )

        linear_model = (
            _preprocess_doc
            >> LinearModel(nr_class)
        )
        #model = linear_model >> logistic

        model = (
            (linear_model | cnn_model)
            >> zero_init(Affine(nr_class, nr_class*2, drop_factor=0.0))
            >> logistic
        )
    model.nO = nr_class
    model.lsuv = False
    return model




if __name__ == '__main__':
    cfg = {'pretrained_dims':True}
    textcat=build_text_classifier(1, **cfg)
