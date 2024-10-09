import logging
import tensorflow as tf
from transformers import AutoModelForCausalLM, AutoTokenizer
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
from typing import Optional, List, Tuple, Dict, Callable, Any
from functools import lru_cache
import asyncio
import time
import nest_asyncio  # Allow nested async loops

# Set up logging
logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

# Apply the workaround for nested event loops
nest_asyncio.apply()

class ModelInitializationError(Exception):
    """Raised when model initialization fails."""

def log_execution_time(func: Callable) -> Callable:
    """
    A decorator that logs the execution time of the function.
    
    :param func: The function to be wrapped.
    :return: Wrapped function with execution time logging.
    """
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        end_time = time.time()
        logging.info(f"Executed {func.__name__} in {end_time - start_time:.2f} seconds.")
        return result
    return wrapper

class AGISystemSTEM:
    def __init__(self, model_loader: Optional[Callable] = None) -> None:
        """
        Initializes the AGI system.

        :param model_loader: Optional function for loading models, useful for dependency injection.
        """
        self.memory: Dict[str, Any] = {}
        self.models: Dict[str, Any] = {}
        self.device = self._detect_device()
        self.model_loader = model_loader if model_loader else self._default_model_loader
        self.model_cache = {}

    @lru_cache(maxsize=5)
    @log_execution_time
    async def load_model(self, model_name: str, model_class: str) -> None:
        """
        Asynchronously loads a model.

        :param model_name: Name of the model to load.
        :param model_class: Class of the model (e.g., nlp_v1).
        """
        if model_class in self.models:
            logging.info(f"{model_class} already loaded.")
            return
        try:
            start_time = time.time()
            self.models[model_class] = await asyncio.to_thread(
                self.model_loader, model_name, model_class
            )
            logging.info(f"{model_class} loaded successfully in {time.time() - start_time:.2f} seconds.")
        except ModelInitializationError as e:
            logging.error(f"Model load failed: {str(e)}")
        except Exception as e:
            logging.error(f"Unexpected error while loading model: {str(e)}")

    def _create_dense_model(
        self, input_shape: Tuple[int], model_name: str, num_classes: int
    ) -> Optional[Model]:
        """
        Creates a dense neural network model.

        :param input_shape: Input shape of the model.
        :param model_name: Name of the model.
        :param num_classes: Number of classes for the model.
        :return: The created model or None if creation fails.
        """
        try:
            with tf.device(self.device):
                model = tf.keras.Sequential(
                    [
                        tf.keras.layers.Dense(128, activation="relu", input_shape=input_shape),
                        tf.keras.layers.Dense(64, activation="relu"),
                        tf.keras.layers.Dense(num_classes, activation="softmax" if num_classes > 1 else "sigmoid"),
                    ]
                )
                loss = "categorical_crossentropy" if num_classes > 1 else "binary_crossentropy"
                model.compile(
                    optimizer="adam",
                    loss=loss,
                    metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
                )
                logging.info(f"{model_name} model created successfully.")
                return model
        except tf.errors.ResourceExhaustedError:
            logging.error(f"Model creation failed due to insufficient memory on {self.device}.")
            return None
        except Exception as e:
            logging.error(f"Error creating {model_name} model: {str(e)}")
            return None

    def _detect_device(self) -> str:
        """Detects the available device (GPU or CPU)."""
        return "/GPU:0" if tf.config.list_physical_devices('GPU') else "/CPU:0"

    def _default_model_loader(self, model_name: str, model_class: str) -> Any:
        """Default model loading function."""
        try:
            if model_class == "nlp_v1":
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                tokenizer.pad_token = tokenizer.eos_token  # Add pad token
                model = AutoModelForCausalLM.from_pretrained(model_name)
                return {"tokenizer": tokenizer, "model": model}
            else:
                raise ModelInitializationError(f"Unknown model class: {model_class}")
        except Exception as e:
            logging.error(f"Error loading model {model_name}: {str(e)}")
            raise ModelInitializationError(f"Failed to load model {model_name}")

    def save_model(self, model: Model, model_name: str) -> None:
        """
        Saves a TensorFlow model.

        :param model: The Keras model to save.
        :param model_name: The file name for saving the model.
        """
        try:
            model.save(f'{model_name}.h5')
            logging.info(f"Model {model_name} saved successfully.")
        except Exception as e:
            logging.error(f"Failed to save model {model_name}: {str(e)}")

    def load_trained_model(self, model_name: str) -> Optional[Model]:
        """
        Loads a previously saved TensorFlow model.

        :param model_name: The name of the model file to load.
        :return: The loaded model.
        """
        try:
            model = tf.keras.models.load_model(f'{model_name}.h5')
            logging.info(f"Model {model_name} loaded successfully.")
            return model
        except Exception as e:
            logging.error(f"Failed to load model {model_name}: {str(e)}")
            return None

# Example usage
async def main():
    agi_system = AGISystemSTEM()
    await agi_system.load_model("Salesforce/codegen-350M-multi", "nlp_v1")

# Run the example
if __name__ == "__main__":
    asyncio.run(main())
