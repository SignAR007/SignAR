import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

os.environ["TF_USE_LEGACY_KERAS"] = "1"
from train_model import build_model, normalize_tfjs_input_layer


class TensorFlowJsInputLayerTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.model_path = Path(self.temp_dir.name) / "model.json"

    def tearDown(self):
        self.temp_dir.cleanup()

    def write_model(self, input_config):
        self.model_path.write_text(
            json.dumps({
                "modelTopology": {
                    "model_config": {
                        "config": {
                            "layers": [{
                                "class_name": "InputLayer",
                                "config": input_config,
                            }]
                        }
                    }
                }
            }),
            encoding="utf-8",
        )

    def test_translates_keras_batch_shape(self):
        self.write_model({"batch_shape": [None, 30, 456], "name": "landmark_sequence"})

        normalize_tfjs_input_layer(self.model_path)

        config = json.loads(self.model_path.read_text(encoding="utf-8"))["modelTopology"]["model_config"]["config"]["layers"][0]["config"]
        self.assertEqual(config["batchInputShape"], [None, 30, 456])
        self.assertNotIn("batch_shape", config)

    def test_translates_legacy_batch_input_shape(self):
        self.write_model({"batch_input_shape": [None, 30, 456], "name": "landmark_sequence"})

        normalize_tfjs_input_layer(self.model_path)

        config = json.loads(self.model_path.read_text(encoding="utf-8"))["modelTopology"]["model_config"]["config"]["layers"][0]["config"]
        self.assertEqual(config["batchInputShape"], [None, 30, 456])
        self.assertNotIn("batch_input_shape", config)

    def test_rejects_input_layer_without_shape(self):
        self.write_model({"name": "landmark_sequence"})

        with self.assertRaisesRegex(ValueError, "batchInputShape or inputShape"):
            normalize_tfjs_input_layer(self.model_path)

    def test_trainer_uses_a_compact_clipped_temporal_cnn(self):
        model = build_model(2)

        self.assertEqual(model.get_layer("temporal_conv_1").filters, 48)
        self.assertEqual(model.get_layer("temporal_conv_2").filters, 64)
        self.assertEqual(model.get_layer("temporal_order_flatten").output_shape, (None, 30 * 64))
        self.assertEqual(model.optimizer.clipnorm, 1.0)
        self.assertEqual(model.get_layer("temporal_norm_1").__class__.__name__, "BatchNormalization")
        self.assertGreater(model.get_layer("temporal_conv_1").kernel_regularizer.l2, 0)
        self.assertGreaterEqual(
            sum(layer.__class__.__name__ == "Dropout" for layer in model.layers), 3
        )

    def test_trained_layers_convert_to_native_tensorflowjs_layers(self):
        import tensorflow as tf

        if sys.platform == "win32":
            try:
                import tensorflow_decision_forests  # noqa: F401
            except ModuleNotFoundError as error:
                if error.name != "tensorflow_decision_forests":
                    raise
                sys.modules["tensorflow_decision_forests"] = types.ModuleType(
                    "tensorflow_decision_forests"
                )
        import tensorflowjs as tfjs

        model = build_model(3)
        with tempfile.TemporaryDirectory() as output_directory:
            tfjs.converters.save_keras_model(model, output_directory)
            model_path = Path(output_directory) / "model.json"
            normalize_tfjs_input_layer(model_path)
            document = json.loads(model_path.read_text(encoding="utf-8"))
        layers = document["modelTopology"]["model_config"]["config"]["layers"]
        supported_layers = {
            "InputLayer", "Dense", "GlobalAveragePooling1D", "Conv1D",
            "BatchNormalization", "Dropout", "Flatten", "Concatenate",
        }
        self.assertTrue({layer["class_name"] for layer in layers} <= supported_layers)
        input_layer = next(layer for layer in layers if layer["class_name"] == "InputLayer")
        self.assertEqual(input_layer["config"]["batchInputShape"], [None, 30, 456])
        self.assertFalse(any(
            key.endswith("_regularizer")
            for layer in layers
            for key in layer.get("config", {})
        ), "training-only regularizers must not be deserialized by the browser interpreter")
        tf.keras.backend.clear_session()

    def test_checked_in_models_use_tensorflow_js_supported_reset_mode(self):
        toolkit_dir = Path(__file__).resolve().parents[1]
        repo_dir = toolkit_dir.parent
        model_paths = [toolkit_dir / "trained-model" / "model.json"]
        interpreter_model_dir = repo_dir / "SignLanguageAR" / "model"
        interpreter_manifest = interpreter_model_dir / "vocabulary.json"
        if interpreter_manifest.is_file():
            manifest = json.loads(interpreter_manifest.read_text(encoding="utf-8"))
            interpreter_model_path = interpreter_model_dir / Path(manifest["modelPath"]).name
            if interpreter_model_path.is_file():
                model_paths.append(interpreter_model_path)

        for model_path in model_paths:
            with self.subTest(model=model_path):
                document = json.loads(model_path.read_text(encoding="utf-8"))
                layers = document["modelTopology"]["model_config"]["config"]["layers"]
                self.assertFalse(any(
                    key.endswith("_regularizer")
                    for layer in layers
                    for key in layer.get("config", {})
                ), "checked-in TF.js models must not require Python regularizer deserialization")
                gru_layers = [layer for layer in layers if layer.get("class_name") == "GRU"]
                if gru_layers:
                    self.assertTrue(all(layer["config"].get("reset_after") is False for layer in gru_layers))
                    gru_biases = [
                        weight for weight in document["weightsManifest"][0]["weights"]
                        if "gru_cell/bias" in weight["name"]
                    ]
                    self.assertEqual(len(gru_biases), len(gru_layers))
                    self.assertTrue(all(len(weight["shape"]) == 1 for weight in gru_biases))
                else:
                    supported_layers = {
                        "InputLayer", "Dense", "GlobalAveragePooling1D", "Conv1D",
                        "BatchNormalization", "Dropout", "Flatten", "Concatenate",
                    }
                    self.assertTrue({layer["class_name"] for layer in layers} <= supported_layers)


if __name__ == "__main__":
    unittest.main()
