#!coding=utf-8
"""
The Rose Equation of State constraint for training a `BasicNN`.
"""
import tensorflow as tf

from ase import Atoms
from typing import List
from collections import Counter
from tensoralloy.nn.utils import is_first_replica
from tensoralloy.neighbor import find_neighbor_size_of_atoms, NeighborSize
from tensoralloy.nn.dataclasses import ExtraDBConstraintOptions
from tensoralloy.io.db import connect
from tensoralloy.utils import ModeKeys, GraphKeys
from tensoralloy.nn.dataclasses import LossParameters
from tensoralloy.transformer import BatchUniversalTransformer


def get_batch_transformer(original_clf: BatchUniversalTransformer,
                          trajectory: List[Atoms],
                          sizes: List[NeighborSize],
                          max_occurs: Counter):
    """
    Return a `BatchDescriptorTransformer` for the trajectory.

    Parameters
    ----------
    original_clf : BatchDescriptorTransformer
        The original batch descriptor transformer.
    trajectory : List[Atoms]
        A list of scaled `Atoms` object.
    sizes : List[NeighborSize]
        The corresponding `NeighborSize` for each `Atoms`.
    max_occurs : Counter
        The max occurances of the elements.

    Returns
    -------
    clf : BatchUniversalTransformer
        The newly created transformer for this trajectory.

    """
    configs = original_clf.as_dict()
    cls = configs.pop('class')

    if cls == "BatchUniversalTransformer":
        nij_max = max(map(lambda x: x.nij, sizes))
        nnl_max = max(map(lambda x: x.nnl, sizes))
        if configs["angular"]:
            ij2k_max = max(map(lambda x: x.ij2k, sizes))
            nijk_max = max(map(lambda x: x.nijk, sizes))
        else:
            ij2k_max = 0
            nijk_max = 0
        configs['nij_max'] = nij_max
        configs['nnl_max'] = nnl_max
        configs['ij2k_max'] = ij2k_max
        configs['nijk_max'] = nijk_max
    else:
        raise ValueError(f"Unsupported batch transformer: {cls}")

    # Make sure every element appears in this dict.
    for element in original_clf.elements:
        max_occurs[element] = max(max_occurs[element], 1)

    configs['max_occurs'] = max_occurs
    configs['batch_size'] = len(trajectory)

    return original_clf.__class__(**configs)


def get_extra_db_constraint_loss(base_nn,
                                 options: ExtraDBConstraintOptions = None,
                                 max_train_steps=None,
                                 verbose=True) -> tf.Tensor:
    """
    Create a constraint using the given database. 

    Parameters
    ----------
    base_nn : BasicNN
        A `BasicNN`. Its variables will be reused.
    options : ExtraDBConstraintOptions
        The options for this loss tensor.
    max_train_steps : int
        The maximum number of training steps.
    verbose : bool
        If True, key tensors will be logged.

    Returns
    -------
    loss : tf.Tensor
        The total loss of the constraint.

    """
    configs = base_nn.as_dict()
    configs.pop('class')

    print(options)

    configs['minimize_properties'] = options.minimize
    configs['export_properties'] = options.minimize

    if options is None:
        return None
    
    with tf.name_scope("Extra"):
        nn = base_nn.__class__(**configs)
        base_clf = base_nn.transformer
        rc = base_clf.rc
        angular = base_clf.angular

        trajectory = []
        sizes = []
        max_occurs = Counter()

        db = connect(options.filename)
        for i in range(1, len(db) + 1):
            atoms = db.get_atoms(id=i, add_additional_information=True)
            symbols = atoms.get_chemical_symbols()
            for el, n in Counter(symbols).items():
                max_occurs[el] = max(max_occurs[el], n)
            trajectory.append(atoms)
            sizes.append(
                find_neighbor_size_of_atoms(atoms, rc, angular))

        batch_clf = get_batch_transformer(
            base_clf, trajectory, sizes, max_occurs)
        nn.attach_transformer(batch_clf)

        # Initialize the fixed batch input pipeline
        with tf.name_scope("Pipeline"):
            fixed_batch = dict()
            decoded_list = []
            for i, atoms in enumerate(trajectory):
                with tf.name_scope(f"{i}"):
                    protobuf = tf.convert_to_tensor(
                        batch_clf.encode(atoms).SerializeToString())
                    decoded_list.append(
                        batch_clf.decode_protobuf(protobuf))
            keys = decoded_list[0].keys()
            for key in keys:
                fixed_batch[key] = tf.stack(
                    [decoded[key] for decoded in decoded_list],
                    name=f'{key}/batch')

        predictions = nn.build(
            features=fixed_batch,
            mode=ModeKeys.TRAIN,
            verbose=verbose)
        labels = {}
        for key in options.minimize:
            labels[key] = fixed_batch.pop(key)
        
        loss_parameters = LossParameters(**{
            "energy": {
                "weight": options.weight, 
                "method": "rmse", 
                "per_atom_loss": True
            }, 
            "forces": {
                "weight": options.weight, 
                "method": "rmse", 
            },
            "eentropy": {
                "weight": options.weight, 
                "method": "rmse", 
                "per_atom_loss": True
            }, 
            "free_energy": {
                "weight": options.weight, 
                "method": "rmse", 
                "per_atom_loss": True
            }, 
            "stress": {
                "weight": options.weight, 
                "method": "rmse", 
            }, 
            "l2": {
                "weight": 0.0,
                "decayed": False,
            }
        })
        n_atoms = fixed_batch.pop("n_atoms_vap")
        atom_masks = fixed_batch.pop("atom_masks")
        loss = nn.get_total_loss(predictions, labels, n_atoms, atom_masks, 
                                 loss_parameters, 
                                 max_train_steps=max_train_steps,
                                 mode=ModeKeys.TRAIN)[0]
        if is_first_replica():
            tf.add_to_collection(GraphKeys.TRAIN_METRICS, loss)
            tf.add_to_collection(GraphKeys.EVAL_METRICS, loss)
        return loss


def main():
    from tensoralloy.nn.eam import EamAlloyNN
    from tensoralloy.transformer import BatchUniversalTransformer

    clf = BatchUniversalTransformer(Counter({"Ni": 108}), 6.0, 
        nij_max=1000, nnl_max=100, use_forces=True, use_stress=True)
    nn = EamAlloyNN(["Ni"], "zjw04", 
                    minimize_properties=('energy', 'forces', 'stress'), 
                    export_properties=('energy', 'forces', 'stress'))
    nn.attach_transformer(clf)

    options = ExtraDBConstraintOptions(**{
        "weight": 1.0, 
        "filename": "test.db", 
        "minimize": ('energy', 'forces', 'stress')})
    
    loss = get_extra_db_constraint_loss(nn, options)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        print(sess.run(loss))


if __name__ == "__main__":
    main()
