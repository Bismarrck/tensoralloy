#!coding=utf-8
"""
The partition and stitch functions.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf

from collections import Counter
from typing import List, Dict

from tensoralloy.utils import get_elements_from_kbody_term, ModeKeys

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def dynamic_partition(dists_and_masks: dict,
                      elements: List[str],
                      kbody_terms_for_element: Dict[str, List[str]],
                      mode: ModeKeys,
                      angular=False,
                      merge_symmetric=True):
    """
    Split the descriptors of type `Dict[element, (tensor, mask)]` to `Np`
    partitions where `Np` is the total number of unique k-body terms.

    * If `angular` is False:
        * If `merge_symmetric` is False, `Np` is equal to `N**2`.
        * If `merge_symmetric` is True, `Np` will be `N * (N + 1) / 2`.
    * If `angular` is True,
        * `Np` is equal to `N * N * (N + 1) / 2`

    Here N denotes the total number of elements.

    Parameters
    ----------
    dists_and_masks : Dict[str, Tuple[tf.Tensor, tf.Tensor]]
        A dict. The keys are elements and values are tuples of (dists, masks)
        where `dists` represents the interatomic distances and displacements.
            * If `mode` is TRAIN or EVAL, both should be 4D tensors of shape
              `[N_in, batch_size, max_n_terms, max_n_element, ndim]`.
            * If `mode` is PREDICT, `value` should be a 4D tensor of shape
              `[N_in, max_n_terms, max_n_element, nnl]` and `mask` should be
              a 3D tensor of shape `[max_n_terms, max_n_element, ndim]`.
        The size of the first axis, `N_in`, should be 4 or 12 (angular is True):
            * rij  = 0
            * rijx = 1
            * rijy = 2
            * rijz = 3
        where `rij = sqrt(rijx**2 + rijy**2 + rijz**2)`
        If angular is True,
            * rik  = 4
            * rikx = 5
            * riky = 6
            * rikz = 7
            * rjk  = 8
            * rjkx = 9
            * rjky = 10
            * rjkz = 11
    elements : List[str]
        A list of str as the sorted elements.
    kbody_terms_for_element : Dict[str, List[str]]
        A dict. The keys are elements and values are their corresponding k-body
        terms.
    mode : ModeKeys
        Specifies if this is training, evaluation or prediction.
    angular : bool
        If True, the input
    merge_symmetric : bool
        A bool.

    Returns
    -------
    partitions : Dict[str, Tuple[tf.Tensor, tf.Tensor]]
        A dict. The keys are unique kbody terms and values are tuples of
        (dists, masks) where `dists` represents the interatomic displacements.
        Both `value` and `mask` are 4D tensors of shape
        `[batch_size, Nt, max_n_element, nnl]`.
    max_occurs : Counter
        The maximum occurance of each type of element.

    """
    partitions = dict()
    max_occurs = {}
    if angular:
        kmax = 3
        if merge_symmetric:
            raise ValueError(
                "`merge_symmetric` is not supported for angular interactions")
    else:
        kmax = 2
    if merge_symmetric:
        name_scope = "Partition/Symmetric"
    else:
        name_scope = "Partition"

    with tf.name_scope(name_scope):
        for element in elements:
            with tf.name_scope(f"{element}"):
                kbody_terms = [x for x in kbody_terms_for_element[element]
                               if len(get_elements_from_kbody_term(x)) == kmax]
                dists, masks = dists_and_masks[element]
                dists = tf.convert_to_tensor(dists, name='dists')
                masks = tf.convert_to_tensor(masks, name='masks')

                if mode == ModeKeys.PREDICT:
                    assert dists.shape.ndims == 5
                    dists = tf.expand_dims(dists, axis=1)
                    masks = tf.expand_dims(masks, axis=0)
                    max_occurs[element] = tf.shape(dists)[3]
                else:
                    assert dists.shape.ndims == 6
                    max_occurs[element] = dists.shape[3].value

                if angular:
                    num = len([x for x in kbody_terms
                               if len(get_elements_from_kbody_term(x)) == 3])
                else:
                    num = len([x for x in kbody_terms
                               if len(get_elements_from_kbody_term(x)) == 2])

                dlists = tf.split(
                    dists, num_or_size_splits=num, axis=2, name='dlist')
                mlists = tf.split(
                    masks, num_or_size_splits=num, axis=1, name='mlist')
                for i, (dists, masks) in enumerate(zip(dlists, mlists)):
                    kbody_term = kbody_terms[i]
                    if merge_symmetric:
                        kbody_term = ''.join(
                            sorted(get_elements_from_kbody_term(
                                kbody_term)))
                        if kbody_term in partitions:
                            dists = tf.concat(
                                (partitions[kbody_term][0], dists), axis=3)
                            masks = tf.concat(
                                (partitions[kbody_term][1], masks), axis=2)
                    partitions[kbody_term] = (dists, masks)
        return partitions, Counter(max_occurs)
