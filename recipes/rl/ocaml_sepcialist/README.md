# OCaml Specialist 

The recipe here provides a way to train a model code OCaml. The reward function is simple: it returns a 1 for code that compiles, and a 0 for code that doesn't. While 
this won't produce the most prolific OCaml coder, it serves to demonstrate the power and fleixbility of Baseten's training platform, allowing you to instrument
environments with dependencies in a manner that gives you all of the control. 

## Training Configuration Details 

The configuration has special properties that lead to better rewards in the training run:
* Both runs implement (Truncated Importance Sampling)[https://fengyao.notion.site/off-policy-rl]
* The LoRA run is a faithful adaptation of Lora-without-regret using VeRL, specifically with rank = 8. Existing dependencies don't allow for rank = 1.


## Examples 

Below are examples of the model from before it was trained, and after

<TODO>

## Experiments

* Full finetune:
  * DAPO? 
  * GSPO? 

* LoRA
  * How small can we deploy on
  * If I install HF from source, can I get the lora rank to 1? 