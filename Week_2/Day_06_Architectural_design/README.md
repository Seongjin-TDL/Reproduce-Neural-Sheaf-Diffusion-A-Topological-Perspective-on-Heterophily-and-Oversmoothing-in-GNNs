# Day 6: NSD Model Architecture – Learning Log


## Purpose
Today's goal was to translate the NSD paper's theory into a concrete PyTorch architectural plan. 
This process is crucial for ensuring the final implementation is modular, testable, and correctly structured.

## What I Learned Today

- Understood the modular structure of NSD model consists of (1) compute_sheaf_laplacian (2) SheafDiffusionLayer using sheaf Laplacian (3) NSD model by stacking multiple SheafDiffusionLayers.
- Deepened my understanding of Python package, module, class, object, instance, function, method.
- Learned why every PyTorch model should inherit from nn.Module and call super().__init__().
- Practiced breaking down complex code into smaller, understandable parts.
- Practiced the class torch.nn.Module [Docs > torch.nn > Module](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module).

## What I Didn’t Finish

- I did not yet fully implement or understand the components of torch.nn.Module class.
- I did not yet fully implement or understand the full NSDModel class that stacks multiple SheafDiffusionLayer blocks.

## Next Steps

- Review how nn.ModuleList works for stacking layers.
- Study and implement the NSDModel class.
- Try running the full model on dummy data.
