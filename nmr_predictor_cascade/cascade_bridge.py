"""Run a CASCADE-2.0 13C prediction inside the CASCADE Python environment.

This script is NOT imported by MoleditPy. The plugin starts it as a separate
process with the interpreter the user configured — a Python 3.10 environment
with TensorFlow 2.11, KGCNN 2.2.1 and RDKit, as CASCADE-2.0 requires — because
none of that can be installed next to MoleditPy itself.

    python cascade_bridge.py --model-dir DIR --input in.json --output out.json
    python cascade_bridge.py --model-dir DIR --check --output out.json

``DIR`` is one of the CASCADE-2.0 model folders, ``models/Predict_SMILES_FF_GPR``
(preferred: it also gives an uncertainty) or ``models/Predict_SMILES_FF``.

Input JSON::

    {"molecules": [{"molblock": "<V2000 with explicit H and 3D coords>",
                    "atom_index": [0, 1, ...]}, ...]}

Output JSON (always written, also on failure)::

    {"ok": true, "model": "gpr" | "plain",
     "results": [{"shift": [...], "std": [...] | null}, ...],
     "unknown_elements": [...]}
    {"ok": false, "error": "..."}

``shift`` is in ppm; ``std`` is the GPR standard deviation in ppm. The
preprocessing and scaling follow the ``predictions_SMILES.ipynb`` notebooks
of https://github.com/asbhd/CASCADE-2.0 (MIT licence).

Only the standard library is imported at module level so that ``--check``
can report a missing package as a readable message.
"""

import argparse
import json
import os
import sys
import traceback

#: Inverse of the target scaling used in training (see the notebooks).
SHIFT_SCALE = 50.484337
SHIFT_OFFSET = 99.798111

GPR_WEIGHTS = "best_model_val_mae.h5"
PLAIN_WEIGHTS = "best_model.h5"
PREPROCESSOR = "preprocessor_orig.p"


def atomic_number_tokenizer(atom):
    """Referenced by name from the pickled preprocessor (as ``__main__.``)."""
    return atom.GetAtomicNum()


def model_kind(model_dir):
    """'gpr', 'plain' or None for a folder that holds no usable model."""
    if not os.path.isfile(os.path.join(model_dir, PREPROCESSOR)):
        return None
    if os.path.isfile(os.path.join(model_dir, GPR_WEIGHTS)) and os.path.isfile(
        os.path.join(model_dir, "model.py")
    ):
        return "gpr"
    if os.path.isfile(os.path.join(model_dir, PLAIN_WEIGHTS)):
        return "plain"
    return None


def _write(path, payload):
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle)


def _prepare_environment(model_dir):
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    # model.py loads its inducing points by relative path, and the pickled
    # preprocessor imports ``nfp`` from the bundled ``modules`` folder.
    os.chdir(model_dir)
    for entry in (model_dir, os.path.join(model_dir, "modules")):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    # The pickle was written from a notebook, so it looks the tokenizer up
    # on __main__; make that work when this file is imported as well.
    main = sys.modules.get("__main__")
    if main is not None and not hasattr(main, "atomic_number_tokenizer"):
        main.atomic_number_tokenizer = atomic_number_tokenizer


def _load_model(kind):
    import tensorflow as tf

    tf.get_logger().setLevel("ERROR")
    if kind == "gpr":
        tf.keras.backend.set_floatx("float64")
        from model import make_model  # CASCADE-2.0's model.py

        model = make_model()
        model.load_weights(GPR_WEIGHTS)
        return model

    # The plain model is a saved Keras model; importing these registers the
    # custom kgcnn layers it was serialised with.
    from keras.models import load_model
    from kgcnn.layers.casting import ChangeTensorType  # noqa: F401
    from kgcnn.layers.conv.painn_conv import (  # noqa: F401
        EquivariantInitialize,
        PAiNNconv,
        PAiNNUpdate,
    )
    from kgcnn.layers.geom import (  # noqa: F401
        CosCutOffEnvelope,
        EdgeDirectionNormalized,
        NodeDistanceEuclidean,
        NodePosition,
        ShiftPeriodicLattice,
    )
    from kgcnn.layers.mlp import MLP, GraphMLP  # noqa: F401
    from kgcnn.layers.modules import LazyAdd, OptionalInputEmbedding  # noqa: F401
    from kgcnn.layers.norm import (  # noqa: F401
        GraphBatchNormalization,
        GraphLayerNormalization,
    )
    from modules.bessel_basis import BesselBasisLayer  # noqa: F401
    from modules.pooling import PoolingNodes  # noqa: F401

    return load_model(PLAIN_WEIGHTS)


def _sequence_class():
    import numpy as np
    import tensorflow as tf
    from nfp.preprocessing import GraphSequence

    def stacked_offsets(sizes, repeats):
        return np.repeat(np.cumsum(np.hstack([0, sizes[:-1]])), repeats)

    def ragged(arr):
        return tf.ragged.constant(np.expand_dims(arr, axis=0), ragged_rank=1)

    class RBFSequence(GraphSequence):
        def process_data(self, batch_data):
            offset = stacked_offsets(batch_data["n_pro"], batch_data["n_atom"])
            offset = np.where(batch_data["atom_index"] >= 0, offset, 0)
            batch_data["atom_index"] += offset
            for feature in (
                "node_attributes",
                "node_coordinates",
                "edge_indices",
                "atom_index",
                "n_pro",
            ):
                batch_data[feature] = ragged(batch_data[feature])
            for key in ("n_atom", "n_bond", "distance", "bond", "node_graph_indices"):
                del batch_data[key]
            return batch_data

    return RBFSequence


def predict(model_dir, molecules, batch_size=32):
    """Predict 13C shifts for already-embedded molecules.

    ``molecules`` is the ``molecules`` list of the input JSON. Returns the
    payload of the output JSON.
    """
    import pickle

    import numpy as np
    from rdkit import Chem

    kind = model_kind(model_dir)
    if kind is None:
        raise FileNotFoundError(
            f"No CASCADE-2.0 model in {model_dir} (expected {PREPROCESSOR} plus "
            f"{GPR_WEIGHTS} + model.py, or {PLAIN_WEIGHTS})."
        )
    _prepare_environment(model_dir)

    mols, indices = [], []
    for i, entry in enumerate(molecules):
        mol = Chem.MolFromMolBlock(entry["molblock"], removeHs=False, sanitize=True)
        if mol is None:
            raise ValueError(f"Conformer {i} could not be read back by RDKit.")
        if mol.GetNumConformers() == 0:
            raise ValueError(f"Conformer {i} has no coordinates.")
        idx = np.array([int(x) for x in entry["atom_index"]], dtype=int)
        if idx.size == 0:
            raise ValueError(f"Conformer {i} has no carbon atoms to predict.")
        mols.append(mol)
        indices.append(idx)

    with open(PREPROCESSOR, "rb") as handle:
        preprocessor = pickle.load(handle)["preprocessor"]
    tokenizer = getattr(preprocessor, "atom_tokenizer", None)
    if tokenizer is not None and hasattr(tokenizer, "unknown"):
        tokenizer.unknown = []

    inputs = preprocessor.predict(zip(mols, indices))
    sequence = _sequence_class()(inputs, batch_size=batch_size, shuffle=False)
    model = _load_model(kind)

    shifts, stds = [], []
    for batch in sequence:
        out = model(batch)
        if kind == "gpr":
            shifts.extend(out.mean().numpy().flatten().tolist())
            stds.extend(out.stddev().numpy().flatten().tolist())
        else:
            shifts.extend(np.asarray(out).flatten().tolist())

    expected = sum(len(i) for i in indices)
    if len(shifts) != expected:
        raise RuntimeError(f"Model returned {len(shifts)} values for {expected} carbons.")

    results, pos = [], 0
    for idx in indices:
        n = len(idx)
        result = {
            "shift": [v * SHIFT_SCALE + SHIFT_OFFSET for v in shifts[pos : pos + n]],
            "std": [v * SHIFT_SCALE for v in stds[pos : pos + n]] if stds else None,
        }
        results.append(result)
        pos += n

    unknown = sorted({int(z) for z in getattr(tokenizer, "unknown", []) if z != "unk"})
    return {"ok": True, "model": kind, "results": results, "unknown_elements": unknown}


def check(model_dir):
    """Import everything and load the model once, without predicting."""
    kind = model_kind(model_dir)
    if kind is None:
        raise FileNotFoundError(f"No CASCADE-2.0 model in {model_dir}.")
    _prepare_environment(model_dir)
    import rdkit
    import tensorflow as tf

    _load_model(kind)
    return {
        "ok": True,
        "model": kind,
        "python": sys.version.split()[0],
        "tensorflow": tf.__version__,
        "rdkit": rdkit.__version__,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--input")
    parser.add_argument("--output", required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    model_dir = os.path.abspath(args.model_dir)
    try:
        if args.check:
            payload = check(model_dir)
        else:
            if not args.input:
                raise ValueError("--input is required unless --check is given.")
            with open(args.input, encoding="utf-8") as handle:
                request = json.load(handle)
            payload = predict(model_dir, request.get("molecules", []))
    except Exception as exc:  # reported back to the plugin, never raised
        _write(
            args.output,
            {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=8),
            },
        )
        return 1
    _write(args.output, payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
