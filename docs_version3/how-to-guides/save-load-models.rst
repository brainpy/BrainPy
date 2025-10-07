How to Save and Load Models
============================

This guide shows you how to save and load BrainPy models for checkpointing, resuming training, and deployment.

.. contents:: Table of Contents
   :local:
   :depth: 2

Quick Start
-----------

**Save a trained model:**

.. code-block:: python

   import brainpy as bp
   import brainstate
   import pickle

   # After training...
   state_dict = {
       'params': net.states(brainstate.ParamState),
       'epoch': current_epoch,
   }

   with open('model.pkl', 'wb') as f:
       pickle.dump(state_dict, f)

**Load a model:**

.. code-block:: python

   # Create model with same architecture
   net = MyNetwork()
   brainstate.nn.init_all_states(net)

   # Load saved state
   with open('model.pkl', 'rb') as f:
       state_dict = pickle.load(f)

   # Restore parameters
   for name, state in state_dict['params'].items():
       net.states(brainstate.ParamState)[name].value = state.value

Understanding What to Save
---------------------------

State Types
~~~~~~~~~~~

BrainPy has three state types with different persistence requirements:

**ParamState (Always save)**
   - Learnable weights and biases
   - Required to restore trained model
   - Examples: synaptic weights, neural biases

**LongTermState (Usually save)**
   - Persistent statistics and counters
   - Not updated by gradients
   - Examples: running averages, spike counts

**ShortTermState (Never save)**
   - Temporary dynamics that reset each trial
   - Will be re-initialized anyway
   - Examples: membrane potentials, synaptic conductances

Recommended Approach
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def save_checkpoint(net, optimizer, epoch, filepath):
       """Save model checkpoint."""
       state_dict = {
           # Required: model parameters
           'params': net.states(brainstate.ParamState),

           # Optional but recommended: long-term states
           'long_term': net.states(brainstate.LongTermState),

           # Training metadata
           'epoch': epoch,
           'optimizer_state': optimizer.state_dict(),  # If continuing training

           # Model configuration (helpful for loading)
           'config': {
               'n_input': net.n_input,
               'n_hidden': net.n_hidden,
               'n_output': net.n_output,
               # ... other hyperparameters
           }
       }

       with open(filepath, 'wb') as f:
           pickle.dump(state_dict, f)

       print(f"✅ Saved checkpoint to {filepath}")

Basic Save/Load
---------------

Using Pickle (Simple)
~~~~~~~~~~~~~~~~~~~~~

**Advantages:**
- Simple and straightforward
- Works with any Python object
- Good for quick prototyping

**Disadvantages:**
- Python-specific format
- Version compatibility issues
- Not human-readable

.. code-block:: python

   import pickle
   import brainpy as bp
   import brainstate

   # Define your model
   class SimpleNet(brainstate.nn.Module):
       def __init__(self, n_neurons=100):
           super().__init__()
           self.lif = bp.LIF(n_neurons, V_rest=-65*u.mV, V_th=-50*u.mV, tau=10*u.ms)
           self.fc = brainstate.nn.Linear(n_neurons, 10)

       def update(self, x):
           self.lif(x)
           return self.fc(self.lif.get_spike())

   # Train model
   net = SimpleNet()
   brainstate.nn.init_all_states(net)
   # ... training code ...

   # Save
   params = net.states(brainstate.ParamState)
   with open('simple_net.pkl', 'wb') as f:
       pickle.dump(params, f)

   # Load
   net_new = SimpleNet()
   brainstate.nn.init_all_states(net_new)

   with open('simple_net.pkl', 'rb') as f:
       loaded_params = pickle.load(f)

   # Restore parameters
   for name, state in loaded_params.items():
       net_new.states(brainstate.ParamState)[name].value = state.value

Using NumPy (Arrays Only)
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Advantages:**
- Language-agnostic
- Efficient storage
- Widely supported

**Disadvantages:**
- Only saves arrays (not structure)
- Need to manually track parameter names

.. code-block:: python

   import numpy as np

   # Save parameters as .npz
   params = net.states(brainstate.ParamState)
   param_dict = {name: np.array(state.value) for name, state in params.items()}
   np.savez('model_params.npz', **param_dict)

   # Load parameters
   loaded = np.load('model_params.npz')
   for name, array in loaded.items():
       net.states(brainstate.ParamState)[name].value = jnp.array(array)

Checkpointing During Training
------------------------------

Periodic Checkpoints
~~~~~~~~~~~~~~~~~~~~

Save at regular intervals during training.

.. code-block:: python

   import braintools

   # Training setup
   net = MyNetwork()
   optimizer = braintools.optim.Adam(lr=1e-3)
   optimizer.register_trainable_weights(net.states(brainstate.ParamState))

   save_interval = 5  # Save every 5 epochs
   checkpoint_dir = './checkpoints'
   import os
   os.makedirs(checkpoint_dir, exist_ok=True)

   # Training loop
   for epoch in range(num_epochs):
       # Training step
       for batch in train_loader:
           loss = train_step(net, optimizer, batch)

       # Periodic save
       if (epoch + 1) % save_interval == 0:
           checkpoint_path = f'{checkpoint_dir}/epoch_{epoch+1}.pkl'
           save_checkpoint(net, optimizer, epoch, checkpoint_path)

           print(f"Epoch {epoch+1}: Loss={loss:.4f}, Checkpoint saved")

Best Model Checkpoint
~~~~~~~~~~~~~~~~~~~~~

Save only when validation performance improves.

.. code-block:: python

   best_val_loss = float('inf')
   best_model_path = 'best_model.pkl'

   for epoch in range(num_epochs):
       # Training
       train_loss = train_epoch(net, optimizer, train_loader)

       # Validation
       val_loss = validate(net, val_loader)

       # Save if best
       if val_loss < best_val_loss:
           best_val_loss = val_loss
           save_checkpoint(net, optimizer, epoch, best_model_path)
           print(f"✅ New best model! Val loss: {val_loss:.4f}")

       print(f"Epoch {epoch+1}: Train={train_loss:.4f}, Val={val_loss:.4f}")

Resuming Training
~~~~~~~~~~~~~~~~~

Continue training from a checkpoint.

.. code-block:: python

   def load_checkpoint(filepath, net, optimizer=None):
       """Load checkpoint and restore state."""
       with open(filepath, 'rb') as f:
           state_dict = pickle.load(f)

       # Restore model parameters
       params = net.states(brainstate.ParamState)
       for name, state in state_dict['params'].items():
           if name in params:
               params[name].value = state.value

       # Restore long-term states
       if 'long_term' in state_dict:
           long_term = net.states(brainstate.LongTermState)
           for name, state in state_dict['long_term'].items():
               if name in long_term:
                   long_term[name].value = state.value

       # Restore optimizer state
       if optimizer is not None and 'optimizer_state' in state_dict:
           optimizer.load_state_dict(state_dict['optimizer_state'])

       start_epoch = state_dict.get('epoch', 0) + 1
       return start_epoch

   # Resume training
   net = MyNetwork()
   brainstate.nn.init_all_states(net)
   optimizer = braintools.optim.Adam(lr=1e-3)
   optimizer.register_trainable_weights(net.states(brainstate.ParamState))

   # Load checkpoint
   start_epoch = load_checkpoint('checkpoint_epoch_50.pkl', net, optimizer)

   # Continue training from where we left off
   for epoch in range(start_epoch, num_epochs):
       train_step(net, optimizer, train_loader)

Advanced Saving Strategies
---------------------------

Versioned Checkpoints
~~~~~~~~~~~~~~~~~~~~~

Keep multiple checkpoints without overwriting.

.. code-block:: python

   from datetime import datetime

   def save_versioned_checkpoint(net, epoch, base_dir='checkpoints'):
       """Save checkpoint with timestamp."""
       timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
       filename = f'model_epoch{epoch}_{timestamp}.pkl'
       filepath = os.path.join(base_dir, filename)

       state_dict = {
           'params': net.states(brainstate.ParamState),
           'epoch': epoch,
           'timestamp': timestamp,
       }

       with open(filepath, 'wb') as f:
           pickle.dump(state_dict, f)

       return filepath

Keep Last N Checkpoints
~~~~~~~~~~~~~~~~~~~~~~~~

Automatically delete old checkpoints to save disk space.

.. code-block:: python

   import glob

   def save_with_cleanup(net, epoch, checkpoint_dir='checkpoints', keep_last=5):
       """Save checkpoint and keep only last N."""

       # Save new checkpoint
       filepath = f'{checkpoint_dir}/epoch_{epoch:04d}.pkl'
       save_checkpoint(net, None, epoch, filepath)

       # Get all checkpoints
       checkpoints = sorted(glob.glob(f'{checkpoint_dir}/epoch_*.pkl'))

       # Delete old ones
       if len(checkpoints) > keep_last:
           for old_checkpoint in checkpoints[:-keep_last]:
               os.remove(old_checkpoint)
               print(f"Removed old checkpoint: {old_checkpoint}")

Conditional Saving
~~~~~~~~~~~~~~~~~~

Save based on custom criteria.

.. code-block:: python

   class CheckpointManager:
       """Manage model checkpoints with custom logic."""

       def __init__(self, checkpoint_dir, keep_best=True, keep_last=3):
           self.checkpoint_dir = checkpoint_dir
           self.keep_best = keep_best
           self.keep_last = keep_last
           self.best_metric = float('inf')
           os.makedirs(checkpoint_dir, exist_ok=True)

       def save(self, net, epoch, metric, is_better=None):
           """Save checkpoint based on metric.

           Args:
               net: Network to save
               epoch: Current epoch
               metric: Validation metric
               is_better: Function to compare metrics (default: lower is better)
           """
           if is_better is None:
               is_better = lambda new, old: new < old

           # Save if best
           if self.keep_best and is_better(metric, self.best_metric):
               self.best_metric = metric
               filepath = f'{self.checkpoint_dir}/best_model.pkl'
               save_checkpoint(net, None, epoch, filepath)
               print(f"💾 Saved best model (metric: {metric:.4f})")

           # Save periodic
           filepath = f'{self.checkpoint_dir}/epoch_{epoch:04d}.pkl'
           save_checkpoint(net, None, epoch, filepath)

           # Cleanup old checkpoints
           self._cleanup()

       def _cleanup(self):
           """Keep only last N checkpoints."""
           checkpoints = sorted(glob.glob(f'{self.checkpoint_dir}/epoch_*.pkl'))
           if len(checkpoints) > self.keep_last:
               for old in checkpoints[:-self.keep_last]:
                   os.remove(old)

   # Usage
   manager = CheckpointManager('./checkpoints', keep_best=True, keep_last=3)

   for epoch in range(num_epochs):
       train_loss = train_epoch(net, optimizer, train_loader)
       val_loss = validate(net, val_loader)

       manager.save(net, epoch, metric=val_loss)

Model Export for Deployment
----------------------------

Minimal Model File
~~~~~~~~~~~~~~~~~~

Save only what's needed for inference.

.. code-block:: python

   def export_for_inference(net, filepath, metadata=None):
       """Export minimal model for inference."""

       export_dict = {
           'params': net.states(brainstate.ParamState),
           'config': {
               # Only architecture info, no training state
               'model_type': net.__class__.__name__,
               # ... architecture hyperparameters
           }
       }

       if metadata:
           export_dict['metadata'] = metadata

       with open(filepath, 'wb') as f:
           pickle.dump(export_dict, f)

       # Report size
       size_mb = os.path.getsize(filepath) / (1024 * 1024)
       print(f"📦 Exported model: {size_mb:.2f} MB")

   # Export trained model
   export_for_inference(
       net,
       'deployed_model.pkl',
       metadata={
           'description': 'LIF network for digit classification',
           'accuracy': 0.95,
           'date': datetime.now().isoformat()
       }
   )

Loading for Inference
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def load_for_inference(filepath, model_class):
       """Load model for inference only."""

       with open(filepath, 'rb') as f:
           export_dict = pickle.load(f)

       # Create model from config
       config = export_dict['config']
       net = model_class(**config)  # Must match saved config
       brainstate.nn.init_all_states(net)

       # Load parameters
       params = net.states(brainstate.ParamState)
       for name, state in export_dict['params'].items():
           params[name].value = state.value

       return net, export_dict.get('metadata')

   # Load and use
   net, metadata = load_for_inference('deployed_model.pkl', MyNetwork)
   print(f"Loaded model: {metadata['description']}")

   # Run inference
   brainstate.nn.init_all_states(net)
   output = net(input_data)

Saving Model Architecture
--------------------------

Configuration-Based
~~~~~~~~~~~~~~~~~~~

Save hyperparameters to recreate model.

.. code-block:: python

   class ConfigurableNetwork(brainstate.nn.Module):
       """Network that can be created from config."""

       def __init__(self, config):
           super().__init__()
           self.config = config

           # Build from config
           self.input_layer = brainstate.nn.Linear(
               config['n_input'],
               config['n_hidden']
           )
           self.hidden = bp.LIF(
               config['n_hidden'],
               V_rest=config['V_rest'],
               V_th=config['V_th'],
               tau=config['tau']
           )
           # ... more layers

       @classmethod
       def from_config(cls, config):
           """Create model from config dict."""
           return cls(config)

       def get_config(self):
           """Get configuration dict."""
           return self.config.copy()

   # Save with config
   config = {
       'n_input': 784,
       'n_hidden': 128,
       'n_output': 10,
       'V_rest': -65.0,
       'V_th': -50.0,
       'tau': 10.0
   }

   net = ConfigurableNetwork(config)
   # ... train ...

   # Save both params and config
   checkpoint = {
       'config': net.get_config(),
       'params': net.states(brainstate.ParamState)
   }

   with open('model_with_config.pkl', 'wb') as f:
       pickle.dump(checkpoint, f)

   # Load from config
   with open('model_with_config.pkl', 'rb') as f:
       checkpoint = pickle.load(f)

   net_new = ConfigurableNetwork.from_config(checkpoint['config'])
   brainstate.nn.init_all_states(net_new)

   for name, state in checkpoint['params'].items():
       net_new.states(brainstate.ParamState)[name].value = state.value

Handling Model Updates
----------------------

Version Compatibility
~~~~~~~~~~~~~~~~~~~~~

Handle changes in model architecture.

.. code-block:: python

   VERSION = '2.0'

   def save_with_version(net, filepath):
       """Save model with version info."""
       checkpoint = {
           'version': VERSION,
           'params': net.states(brainstate.ParamState),
           'config': net.get_config()
       }

       with open(filepath, 'wb') as f:
           pickle.dump(checkpoint, f)

   def load_with_migration(filepath, model_class):
       """Load model with version migration."""
       with open(filepath, 'rb') as f:
           checkpoint = pickle.load(f)

       version = checkpoint.get('version', '1.0')

       # Migrate old versions
       if version == '1.0':
           print("Migrating from v1.0 to v2.0...")
           checkpoint = migrate_v1_to_v2(checkpoint)

       # Create model
       net = model_class.from_config(checkpoint['config'])
       brainstate.nn.init_all_states(net)

       # Load parameters
       for name, state in checkpoint['params'].items():
           if name in net.states(brainstate.ParamState):
               net.states(brainstate.ParamState)[name].value = state.value
           else:
               print(f"⚠️  Skipping unknown parameter: {name}")

       return net

   def migrate_v1_to_v2(checkpoint):
       """Migrate checkpoint from v1.0 to v2.0."""
       # Example: rename parameter
       if 'old_param_name' in checkpoint['params']:
           checkpoint['params']['new_param_name'] = checkpoint['params'].pop('old_param_name')

       checkpoint['version'] = '2.0'
       return checkpoint

Partial Loading
~~~~~~~~~~~~~~~

Load only some parameters (e.g., for transfer learning).

.. code-block:: python

   def load_partial(filepath, net, param_filter=None):
       """Load only specified parameters.

       Args:
           filepath: Checkpoint file
           net: Network to load into
           param_filter: Function that takes param name and returns True to load
       """
       with open(filepath, 'rb') as f:
           checkpoint = pickle.load(f)

       if param_filter is None:
           param_filter = lambda name: True

       loaded_count = 0
       skipped_count = 0

       for name, state in checkpoint['params'].items():
           if param_filter(name):
               if name in net.states(brainstate.ParamState):
                   net.states(brainstate.ParamState)[name].value = state.value
                   loaded_count += 1
               else:
                   print(f"⚠️  Parameter not found in model: {name}")
                   skipped_count += 1
           else:
               skipped_count += 1

       print(f"✅ Loaded {loaded_count} parameters, skipped {skipped_count}")

   # Example: Load only encoder parameters
   load_partial(
       'pretrained.pkl',
       net,
       param_filter=lambda name: name.startswith('encoder.')
   )

Common Patterns
---------------

Pattern 1: Training Session Manager
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class TrainingSession:
       """Manage full training session with checkpointing."""

       def __init__(self, net, optimizer, checkpoint_dir='./checkpoints'):
           self.net = net
           self.optimizer = optimizer
           self.checkpoint_dir = checkpoint_dir
           self.epoch = 0
           self.best_metric = float('inf')

           os.makedirs(checkpoint_dir, exist_ok=True)

       def save(self, metric=None):
           """Save current state."""
           checkpoint = {
               'params': self.net.states(brainstate.ParamState),
               'optimizer': self.optimizer.state_dict(),
               'epoch': self.epoch,
               'best_metric': self.best_metric
           }

           # Regular checkpoint
           filepath = f'{self.checkpoint_dir}/checkpoint_latest.pkl'
           with open(filepath, 'wb') as f:
               pickle.dump(checkpoint, f)

           # Best checkpoint
           if metric is not None and metric < self.best_metric:
               self.best_metric = metric
               best_path = f'{self.checkpoint_dir}/checkpoint_best.pkl'
               with open(best_path, 'wb') as f:
                   pickle.dump(checkpoint, f)

       def restore(self, filepath=None):
           """Restore from checkpoint."""
           if filepath is None:
               filepath = f'{self.checkpoint_dir}/checkpoint_latest.pkl'

           with open(filepath, 'rb') as f:
               checkpoint = pickle.load(f)

           # Restore state
           for name, state in checkpoint['params'].items():
               self.net.states(brainstate.ParamState)[name].value = state.value

           self.optimizer.load_state_dict(checkpoint['optimizer'])
           self.epoch = checkpoint['epoch']
           self.best_metric = checkpoint['best_metric']

           print(f"✅ Restored from epoch {self.epoch}")

Pattern 2: Model Zoo
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class ModelZoo:
       """Collection of pre-trained models."""

       def __init__(self, zoo_dir='./model_zoo'):
           self.zoo_dir = zoo_dir
           os.makedirs(zoo_dir, exist_ok=True)

       def save_model(self, net, name, metadata=None):
           """Add model to zoo."""
           model_path = f'{self.zoo_dir}/{name}.pkl'
           export_dict = {
               'params': net.states(brainstate.ParamState),
               'config': net.get_config(),
               'metadata': metadata or {}
           }

           with open(model_path, 'wb') as f:
               pickle.dump(export_dict, f)

           print(f"📦 Added {name} to model zoo")

       def load_model(self, name, model_class):
           """Load model from zoo."""
           model_path = f'{self.zoo_dir}/{name}.pkl'

           with open(model_path, 'rb') as f:
               export_dict = pickle.load(f)

           net = model_class.from_config(export_dict['config'])
           brainstate.nn.init_all_states(net)

           for param_name, state in export_dict['params'].items():
               net.states(brainstate.ParamState)[param_name].value = state.value

           return net, export_dict['metadata']

       def list_models(self):
           """List available models."""
           models = glob.glob(f'{self.zoo_dir}/*.pkl')
           return [os.path.basename(m).replace('.pkl', '') for m in models]

   # Usage
   zoo = ModelZoo()

   # Save trained models
   zoo.save_model(net1, 'mnist_classifier', {'accuracy': 0.98})
   zoo.save_model(net2, 'fashion_classifier', {'accuracy': 0.92})

   # List and load
   print("Available models:", zoo.list_models())
   net, metadata = zoo.load_model('mnist_classifier', MyNetwork)
   print(f"Loaded model with accuracy: {metadata['accuracy']}")

Troubleshooting
---------------

Issue: Pickle version mismatch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptom:** `AttributeError` or `ModuleNotFoundError` when loading

**Solution:** Use protocol version 4 or lower for compatibility

.. code-block:: python

   # Save with specific protocol
   with open('model.pkl', 'wb') as f:
       pickle.dump(state_dict, f, protocol=4)

Issue: JAX array serialization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptom:** Can't pickle JAX arrays directly

**Solution:** Convert to NumPy before saving

.. code-block:: python

   import numpy as np

   # Convert to NumPy for saving
   params_np = {
       name: np.array(state.value)
       for name, state in net.states(brainstate.ParamState).items()
   }

   with open('model.pkl', 'wb') as f:
       pickle.dump(params_np, f)

   # Convert back when loading
   for name, array in params_np.items():
       net.states(brainstate.ParamState)[name].value = jnp.array(array)

Issue: Model architecture changed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptom:** Parameter names don't match

**Solution:** Use partial loading with error handling

.. code-block:: python

   def safe_load(checkpoint_path, net):
       """Load with error handling."""
       with open(checkpoint_path, 'rb') as f:
           checkpoint = pickle.load(f)

       current_params = net.states(brainstate.ParamState)
       loaded_params = checkpoint['params']

       # Check compatibility
       missing = set(current_params.keys()) - set(loaded_params.keys())
       unexpected = set(loaded_params.keys()) - set(current_params.keys())

       if missing:
           print(f"⚠️  Missing parameters: {missing}")
       if unexpected:
           print(f"⚠️  Unexpected parameters: {unexpected}")

       # Load matching parameters
       for name in current_params.keys() & loaded_params.keys():
           current_params[name].value = loaded_params[name].value

       print(f"✅ Loaded {len(current_params.keys() & loaded_params.keys())} parameters")

Best Practices
--------------

✅ **Always save configuration** - Include hyperparameters for reproducibility

✅ **Version your checkpoints** - Track model version for compatibility

✅ **Save metadata** - Include training metrics, date, description

✅ **Regular backups** - Save periodically during long training

✅ **Keep best model** - Separate best and latest checkpoints

✅ **Test loading** - Verify checkpoint can be loaded before continuing

✅ **Use relative paths** - Make checkpoints portable

✅ **Document format** - Comment what's in your checkpoint files

❌ **Don't save ShortTermState** - It resets anyway

❌ **Don't save everything** - Minimize checkpoint size

❌ **Don't overwrite** - Keep multiple checkpoints for safety

Summary
-------

**Quick reference:**

.. code-block:: python

   # Save
   checkpoint = {
       'params': net.states(brainstate.ParamState),
       'epoch': epoch,
       'config': net.get_config()
   }
   with open('checkpoint.pkl', 'wb') as f:
       pickle.dump(checkpoint, f)

   # Load
   with open('checkpoint.pkl', 'rb') as f:
       checkpoint = pickle.load(f)

   net = MyNetwork.from_config(checkpoint['config'])
   brainstate.nn.init_all_states(net)

   for name, state in checkpoint['params'].items():
       net.states(brainstate.ParamState)[name].value = state.value

See Also
--------

- :doc:`../core-concepts/state-management` - Understanding states
- :doc:`../tutorials/advanced/05-snn-training` - Training models
- :doc:`gpu-tpu-usage` - Accelerated training
