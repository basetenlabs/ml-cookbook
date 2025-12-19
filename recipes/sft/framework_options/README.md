# Choose Your Adventure

In the Choose Your Adventure set of examples, we demonstrate the flexibility of the Baseten training platform. In this recipe, you'll
see implementations of training runs at three different layers: 
* Raw pytorch: Here, every detail is written out explicitly, allowing for customization and flexibility at every layer
    * [Classifier](../../../examples/qwen3-0.6b-classifier-pytorch/training)
    * [Next Token Prediction](../../../examples/qwen3-0.6b-pytorch/training)
* Huggingface Trainer: Here, we leverage huggingface's abstractions around models and training to illusrate a form factor that abstracts away many of details, but still leaves plenty of knobs to turn
    * [Classifier](../../../examples/qwen3-0.6b-classifier-hf-trainer/training)
* Axolotl: Lastly, we demonstrate training with axolotl using config-driven methods. This allows for a more point-and-click / plug-and-play feel - just bring your models and you data.
    * [Next Token Prediction](../../../examples/qwen3-0.6b-axolotl/training)