from __future__ import annotations

from typing import TYPE_CHECKING, Callable, List, Tuple

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
from jaxtyping import Array, Float
from loguru import logger

if TYPE_CHECKING:
    from pyroki._robot import Robot
    from ._geometry import CollGeom
from ._robot_collision import RobotCollisionSpherized
from ._geometry import CollGeom
import jaxlie
from typing import cast

# First 50 prime numbers for Halton sequence bases
_HALTON_PRIMES = jnp.array([
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
    73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151,
    157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229
])


def _halton_single(index: int, base: int) -> float:
    """Generate a single Halton number using the radical inverse function."""
    result = 0.0
    f = 1.0 / base
    i = index
    while i > 0:
        result += f * (i % base)
        i //= base
        f /= base
    return result


def _halton_sequence(num_samples: int, dim: int, skip: int = 100) -> jax.Array:
    """
    Generate a Halton sequence for quasi-random sampling.
    
    The Halton sequence provides better coverage of the sample space compared to
    uniform random sampling, which is beneficial for diverse sample collection.
    
    Args:
        num_samples: Number of samples to generate.
        dim: Dimensionality of each sample.
        skip: Number of initial samples to skip (improves uniformity).
    
    Returns:
        JAX array of shape (num_samples, dim) with values in [0, 1].
    """
    if dim > len(_HALTON_PRIMES):
        raise ValueError(f"Halton sequence dimension {dim} exceeds available primes ({len(_HALTON_PRIMES)})")
    
    # Generate samples using vectorized operations where possible
    indices = jnp.arange(skip, skip + num_samples)
    bases = _HALTON_PRIMES[:dim]
    
    # Compute maximum number of digits needed for the largest index
    max_index = skip + num_samples
    max_digits = int(jnp.ceil(jnp.log(max_index + 1) / jnp.log(2))) + 1
    
    def halton_for_base(base: int) -> jax.Array:
        """Vectorized Halton sequence for a single base."""
        # Compute radical inverse for all indices at once
        result = jnp.zeros(num_samples)
        f = 1.0 / base
        current = indices.astype(jnp.float32)
        
        for _ in range(max_digits):
            digit = jnp.mod(current, base)
            result = result + f * digit
            current = jnp.floor(current / base)
            f = f / base
        
        return result
    
    # Stack results for all dimensions
    samples = jnp.stack([halton_for_base(int(b)) for b in bases], axis=1)
    
    return samples




@jdc.pytree_dataclass
class NeuralRobotCollisionSpherized(RobotCollisionSpherized):
    """
    A subclass of RobotCollisionSpherized that uses a neural network to approximate
    collision distances for a specific static environment (set of obstacles).
    
    The network is trained to overfit to a specific scene, mapping robot link poses
    directly to collision distances between robot links and the static obstacles.
    
    Input: Flattened link poses (N links × 7 pose params = N*7 dimensions)
    Output: Flattened distance matrix (N links × M obstacles = N*M dimensions)
    """
    
    # Neural network parameters (weights and biases for each layer)
    # We store them as a list of arrays.
    nn_params: List[Tuple[Float[Array, "fan_in fan_out"], Float[Array, "fan_out"]]] = jdc.field(default_factory=list)
    
    # Metadata about the training - these must be static for use in JIT conditionals
    is_trained: jdc.Static[bool] = False
    
    # We keep track of the number of obstacles this network was trained for (M)
    trained_num_obstacles: jdc.Static[int] = 0
    
    # Input normalization parameters (computed during training)
    input_mean: jax.Array = jdc.field(default_factory=lambda: jnp.zeros(1))
    input_std: jax.Array = jdc.field(default_factory=lambda: jnp.ones(1))

    @staticmethod
    def from_existing(
        original: RobotCollisionSpherized,
        layer_sizes: List[int] = None,
        key: jax.Array = None
    ) -> "NeuralRobotCollisionSpherized":
        """
        Creates a NeuralRobotCollisionSpherized instance from an existing RobotCollisionSpherized object.
        Initializes the neural network with random weights.
        
        Args:
            original: The original collision model.
            layer_sizes: List of hidden layer sizes. The input size is determined by robot DOF,
                         and output size by num_links * num_obstacles (determined at training time).
                         For initialization, we just set up the structure.
            key: JAX PRNG key for initialization.
        """
        if layer_sizes is None:
            layer_sizes = [256, 256, 256]
            
        if key is None:
            key = jax.random.PRNGKey(0)

        # We can't fully initialize the network structure until we know the output dimension (N*M),
        # which depends on the number of obstacles M. 
        # For now, we just copy the fields and return an untrained instance.
        # The actual weights will be initialized/shaped during the training setup or first call.
        
        return NeuralRobotCollisionSpherized(
            num_links=original.num_links,
            link_names=original.link_names,
            coll=original.coll,
            active_idx_i=original.active_idx_i,
            active_idx_j=original.active_idx_j,
            nn_params=[],
            is_trained=False,
            trained_num_obstacles=0
        )

    def _forward_nn(self, x: Float[Array, "input_dim"]) -> Float[Array, "output_dim"]:
        """
        Forward pass of the MLP.
        """
        # Simple MLP with ReLU activations
        for i, (w, b) in enumerate(self.nn_params):
            x = x @ w + b
            if i < len(self.nn_params) - 1:
                x = jax.nn.relu(x)
        return x

    @jdc.jit
    def at_config(
        self, robot: "Robot", cfg: Float[Array, "*batch actuated_count"]
    ) -> "CollGeom":
        """
        Returns the collision geometry transformed to the given robot configuration.

        This override fixes the shape mismatch in the parent class by extracting
        the transform for each specific link before applying it.

        Args:
            robot: The Robot instance containing kinematics information.
            cfg: The robot configuration (actuated joints).

        Returns:
            The collision geometry (CollGeom) transformed to the world frame
            according to the provided configuration.
        """
        assert self.link_names == robot.links.names, (
            "Link name mismatch between RobotCollision and Robot kinematics."
        )
        
        Ts_link_world_wxyz_xyz = robot.forward_kinematics(cfg)
        
        coll_transformed = []
        for link in range(len(self.coll)):
            # Extract transform for this specific link: shape (*batch, 7)
            Ts_this_link = jaxlie.SE3(Ts_link_world_wxyz_xyz[..., link, :])
            coll_transformed.append(self.coll[link].transform(Ts_this_link))
        coll_transformed = cast(CollGeom, jax.tree.map(lambda *args: jnp.stack(args), *coll_transformed))
        
        return coll_transformed

    @jdc.jit
    def compute_world_collision_distance(
        self,
        robot: "Robot",
        cfg: Float[Array, "*batch_cfg actuated_count"],
        world_geom: "CollGeom",  # Shape: (*batch_world, M, ...)
    ) -> Float[Array, "*batch_combined N M"]:
        """
        Overrides the compute_world_collision_distance to use the trained neural network.
        
        This assumes that world_geom represents the SAME static obstacles that the network
        was trained on. The network uses link poses (from forward kinematics) as input
        and predicts distances based on those poses.
        """
        if not self.is_trained:
            # Fallback to the original exact computation if not trained
            return super().compute_world_collision_distance(robot, cfg, world_geom)

        # Determine batch shapes
        batch_cfg_shape = cfg.shape[:-1]
        
        # Check world geom shape to ensure consistency with training (M)
        world_axes = world_geom.get_batch_axes()
        if len(world_axes) == 0:
            M = 1
            batch_world_shape = ()
        else:
            M = world_axes[-1]
            batch_world_shape = world_axes[:-1]
            
        if M != self.trained_num_obstacles:
            logger.warning(
                f"Neural network was trained for {self.trained_num_obstacles} obstacles, "
                f"but current world_geom has {M}. Falling back to exact computation."
            )
            return super().compute_world_collision_distance(robot, cfg, world_geom)

        # Compute link poses via forward kinematics
        # Shape: (*batch_cfg, num_links, 7) where 7 = wxyz (4) + xyz (3)
        link_poses = robot.forward_kinematics(cfg)
        N = self.num_links
        
        # Flatten link poses to use as network input
        # Shape: (*batch_cfg, num_links * 7)
        link_poses_flat = link_poses.reshape(*batch_cfg_shape, N * 7)
        
        # Apply input normalization (using stored mean/std from training)
        link_poses_normalized = (link_poses_flat - self.input_mean) / self.input_std
        
        # Flatten batch for inference
        input_flat = link_poses_normalized.reshape(-1, N * 7)
        
        # Run inference
        predict_fn = jax.vmap(self._forward_nn)
        dists_flat = predict_fn(input_flat)  # Shape: (batch_size, N * M)
        
        # Reshape output to (*batch_cfg, N, M)
        dists = dists_flat.reshape(*batch_cfg_shape, N, M)
        
        # Handle broadcasting with world batch shape if necessary.
        if batch_world_shape:
             expected_batch_combined = jnp.broadcast_shapes(batch_cfg_shape, batch_world_shape)
             dists = jnp.broadcast_to(dists, (*expected_batch_combined, N, M))

        return dists

    def train(
        self,
        robot: "Robot",
        world_geom: "CollGeom",
        num_samples: int = 10000,
        batch_size: int = 1000,
        epochs: int = 50,
        learning_rate: float = 1e-3,
        key: jax.Array = None,
        layer_sizes: List[int] = [256, 256, 256, 256]
    ) -> "NeuralRobotCollisionSpherized":
        """
        Trains the neural network to approximate the collision distances for the given world_geom.
        Returns a new instance with trained weights.

        The network maps from link poses (N*7 dimensions) to distances (N*M dimensions).
        Using full SE3 poses (quaternion + position) since link orientation affects
        where collision spheres end up in world space.
        """
        logger.info("Starting neural collision training...")
        
        if key is None:
            key = jax.random.PRNGKey(0)

        key_samples, key_init, key_train = jax.random.split(key, 3)

        N = self.num_links
        world_axes = world_geom.get_batch_axes()
        M = world_axes[-1] if len(world_axes) > 0 else 1

        # 1. Generate training data with collision-aware sampling
        logger.info(f"Generating {num_samples} samples with collision-aware sampling...")

        # Sample configurations using Halton sequence for better space coverage
        dof = robot.joints.num_actuated_joints
        lower_limits = robot.joints.lower_limits
        upper_limits = robot.joints.upper_limits
        
        # Generate initial Halton sequence samples (2x to have pool for filtering)
        initial_pool_size = num_samples * 2
        halton_samples = _halton_sequence(initial_pool_size, dof)
        q_pool = lower_limits + halton_samples * (upper_limits - lower_limits)
        
        # Compute distances for the pool to identify collision samples
        logger.info("Computing distances to identify collision samples...")
        
        def compute_min_dist(q):
            dists = super(NeuralRobotCollisionSpherized, self).compute_world_collision_distance(
                robot, q, world_geom
            )
            return jnp.min(dists)
        
        compute_all_min_dists = jax.vmap(compute_min_dist)
        min_dists = compute_all_min_dists(q_pool)  # Shape: (initial_pool_size,)
        
        # Separate samples into collision (dist <= 0) and near-collision (0 < dist < threshold)
        collision_threshold = 0.1  # Samples within 10cm of collision
        
        is_in_collision = min_dists <= 0
        is_near_collision = (min_dists > 0) & (min_dists < collision_threshold)
        is_free_space = min_dists >= collision_threshold
        
        collision_samples = q_pool[is_in_collision]
        near_collision_samples = q_pool[is_near_collision]
        free_space_samples = q_pool[is_free_space]
        
        num_collision = collision_samples.shape[0]
        num_near_collision = near_collision_samples.shape[0]
        num_free = free_space_samples.shape[0]
        
        logger.info(f"Sample distribution from pool: collision={num_collision}, near-collision={num_near_collision}, free-space={num_free}")
        
        # Target distribution: 80% collision, 15% near-collision, 5% free space
        target_collision = int(num_samples * 0.8)
        target_near = int(num_samples * 0.15)
        target_free = num_samples - target_collision - target_near
        
        # Augment collision samples by perturbing existing ones and verifying they're still in collision
        key_augment = key_samples
        max_augment_iterations = 10  # Limit iterations to avoid infinite loops
        
        if num_collision < target_collision and num_collision > 0:
            logger.info(f"Augmenting collision samples from {num_collision} to {target_collision}...")
            
            samples_needed = target_collision - num_collision
            augmented_list = []
            iteration = 0
            
            while len(augmented_list) < samples_needed and iteration < max_augment_iterations:
                iteration += 1
                # Generate candidate perturbations
                batch_size_aug = min(samples_needed * 2, 5000)  # Generate extras to account for filtering
                key_augment, subk1, subk2 = jax.random.split(key_augment, 3)
                indices = jax.random.randint(subk1, (batch_size_aug,), 0, num_collision)
                base_samples = collision_samples[indices]
                
                # Add small perturbations (within 5% of joint range)
                perturbation_scale = 0.05 * (upper_limits - lower_limits)
                perturbations = jax.random.uniform(subk2, (batch_size_aug, dof), minval=-1, maxval=1) * perturbation_scale
                candidates = jnp.clip(base_samples + perturbations, lower_limits, upper_limits)
                
                # Verify candidates are still in collision
                candidate_dists = compute_all_min_dists(candidates)
                valid_mask = candidate_dists <= 0
                valid_candidates = candidates[valid_mask]
                
                if valid_candidates.shape[0] > 0:
                    augmented_list.append(valid_candidates)
                    
                logger.debug(f"  Iteration {iteration}: {valid_candidates.shape[0]} valid collision samples generated")
            
            if augmented_list:
                all_augmented = jnp.concatenate(augmented_list, axis=0)
                # Take only what we need
                all_augmented = all_augmented[:samples_needed]
                collision_samples = jnp.concatenate([collision_samples, all_augmented], axis=0)
                num_collision = collision_samples.shape[0]
                logger.info(f"  Final collision sample count: {num_collision}")
        
        # Similarly augment near-collision samples with verification
        if num_near_collision < target_near and num_near_collision > 0:
            logger.info(f"Augmenting near-collision samples from {num_near_collision} to {target_near}...")
            
            samples_needed = target_near - num_near_collision
            augmented_list = []
            iteration = 0
            
            while len(augmented_list) < samples_needed and iteration < max_augment_iterations:
                iteration += 1
                batch_size_aug = min(samples_needed * 2, 5000)
                key_augment, subk1, subk2 = jax.random.split(key_augment, 3)
                indices = jax.random.randint(subk1, (batch_size_aug,), 0, num_near_collision)
                base_samples = near_collision_samples[indices]
                
                perturbation_scale = 0.03 * (upper_limits - lower_limits)
                perturbations = jax.random.uniform(subk2, (batch_size_aug, dof), minval=-1, maxval=1) * perturbation_scale
                candidates = jnp.clip(base_samples + perturbations, lower_limits, upper_limits)
                
                # Verify candidates are in near-collision range
                candidate_dists = compute_all_min_dists(candidates)
                valid_mask = (candidate_dists > 0) & (candidate_dists < collision_threshold)
                valid_candidates = candidates[valid_mask]
                
                if valid_candidates.shape[0] > 0:
                    augmented_list.append(valid_candidates)
            
            if augmented_list:
                all_augmented = jnp.concatenate(augmented_list, axis=0)
                all_augmented = all_augmented[:samples_needed]
                near_collision_samples = jnp.concatenate([near_collision_samples, all_augmented], axis=0)
                num_near_collision = near_collision_samples.shape[0]
                logger.info(f"  Final near-collision sample count: {num_near_collision}")
        
        # Construct final training set
        actual_collision = min(num_collision, target_collision)
        actual_near = min(num_near_collision, target_near)
        actual_free = max(0, num_samples - actual_collision - actual_near)
        actual_free = min(actual_free, num_free)  # Can't use more free samples than available
        
        logger.info(f"Assembling training set: collision={actual_collision}, near={actual_near}, free={actual_free}")
        
        # Select samples from each category
        key_augment, subk = jax.random.split(key_augment)
        
        selected_collision = collision_samples[:actual_collision] if actual_collision > 0 else jnp.empty((0, dof))
        selected_near = near_collision_samples[:actual_near] if actual_near > 0 else jnp.empty((0, dof))
        
        if actual_free > 0 and num_free > 0:
            free_indices = jax.random.choice(subk, num_free, shape=(actual_free,), replace=False)
            selected_free = free_space_samples[free_indices]
        else:
            selected_free = jnp.empty((0, dof))
        
        # Combine all samples
        parts = [p for p in [selected_collision, selected_near, selected_free] if p.shape[0] > 0]
        q_train = jnp.concatenate(parts, axis=0) if parts else jnp.empty((0, dof))
        
        # If we still need more samples, fill with random samples from the pool
        if q_train.shape[0] < num_samples:
            shortfall = num_samples - q_train.shape[0]
            logger.info(f"Filling shortfall of {shortfall} samples from original pool...")
            key_augment, subk = jax.random.split(key_augment)
            extra_indices = jax.random.choice(subk, initial_pool_size, shape=(shortfall,), replace=True)
            extra_samples = q_pool[extra_indices]
            q_train = jnp.concatenate([q_train, extra_samples], axis=0)
        
        # Shuffle the training data
        key_augment, subk = jax.random.split(key_augment)
        shuffle_perm = jax.random.permutation(subk, q_train.shape[0])
        q_train = q_train[shuffle_perm][:num_samples]  # Truncate to exact size
        
        logger.info(f"Final training set: {q_train.shape[0]} samples")

        # Compute link poses for all configurations via forward kinematics
        # Shape: (num_samples, num_links, 7) where 7 = wxyz (4) + xyz (3)
        logger.info("Computing link poses via forward kinematics...")
        link_poses_all = robot.forward_kinematics(q_train)
        
        # Flatten link poses to (num_samples, num_links * 7)
        X_train_raw = link_poses_all.reshape(num_samples, N * 7)
        
        # Normalize inputs: compute mean and std for better training
        X_mean = jnp.mean(X_train_raw, axis=0, keepdims=True)
        X_std = jnp.std(X_train_raw, axis=0, keepdims=True) + 1e-8
        X_train = (X_train_raw - X_mean) / X_std

        # 2. Compute ground truth labels using vmap for acceleration
        logger.info("Computing ground truth distances (vectorized)...")
        
        # Use vmap to compute distances for all configurations in parallel
        def compute_single_dist(q):
            dists = super(NeuralRobotCollisionSpherized, self).compute_world_collision_distance(
                robot, q, world_geom
            )
            return dists.reshape(-1)  # Flatten to (N*M,)
        
        # Vectorize over all training samples
        compute_all_dists = jax.vmap(compute_single_dist)
        Y_train = compute_all_dists(q_train)  # Shape: (num_samples, N*M)
        
        # Compute sample weights based on minimum distance
        # Give higher weight to collision and near-collision samples
        Y_min_per_sample = jnp.min(Y_train, axis=1)  # Shape: (num_samples,)
        
        # Weight function: higher weight for collision (dist <= 0) and near-collision
        # collision: weight = 3.0, near-collision: weight = 2.0, free: weight = 1.0
        sample_weights = jnp.where(
            Y_min_per_sample <= 0,
            3.0,  # Collision samples get 3x weight
            jnp.where(
                Y_min_per_sample < collision_threshold,
                2.0,  # Near-collision samples get 2x weight
                1.0   # Free space samples get normal weight
            )
        )
        # Normalize weights so they sum to num_samples (to maintain loss scale)
        sample_weights = sample_weights * (num_samples / jnp.sum(sample_weights))
        
        logger.info(f"Sample weights - collision (3x): {jnp.sum(Y_min_per_sample <= 0)}, near-collision (2x): {jnp.sum((Y_min_per_sample > 0) & (Y_min_per_sample < collision_threshold))}")

        # 3. Initialize Network
        input_dim = N * 7  # num_links * 7 (wxyz_xyz pose representation)
        output_dim = N * M  # num_links * num_obstacles

        sizes = [input_dim] + layer_sizes + [output_dim]
        params = []
        k = key_init
        for i in range(len(sizes) - 1):
            k, subk = jax.random.split(k)
            fan_in, fan_out = sizes[i], sizes[i + 1]
            w = jax.random.normal(subk, (fan_in, fan_out)) * jnp.sqrt(2.0 / fan_in)
            b = jnp.zeros((fan_out,))
            params.append((w, b))

        logger.info(
            f"Training neural network (Input: {input_dim} [link positions], Output: {output_dim} [distances])..."
        )

        # 4. Define JIT-compiled training step
        @jax.jit
        def forward_pass(params, x):
            """Forward pass through the network."""
            for i, (w, b) in enumerate(params):
                x = x @ w + b
                if i < len(params) - 1:
                    x = jax.nn.relu(x)
            return x

        @jax.jit
        def loss_fn(params, x, y, weights):
            """Compute weighted MSE loss."""
            pred = forward_pass(params, x)
            # Per-sample MSE, then weight and average
            sample_mse = jnp.mean((pred - y) ** 2, axis=1)  # (batch_size,)
            return jnp.mean(sample_mse * weights)

        @jax.jit
        def train_step(params, opt_state, x_batch, y_batch, w_batch, t):
            """Single training step with Adam optimizer."""
            m, v = opt_state
            beta1, beta2, epsilon = 0.9, 0.999, 1e-8
            
            # Compute gradients
            loss_val, grads = jax.value_and_grad(loss_fn)(params, x_batch, y_batch, w_batch)
            
            # Adam update
            new_params = []
            new_m = []
            new_v = []
            
            for i in range(len(params)):
                w, b = params[i]
                dw, db = grads[i]
                mw, mb = m[i]
                vw, vb = v[i]
                
                # Update biased first moment estimate
                mw = beta1 * mw + (1.0 - beta1) * dw
                mb = beta1 * mb + (1.0 - beta1) * db
                
                # Update biased second moment estimate
                vw = beta2 * vw + (1.0 - beta2) * (dw ** 2)
                vb = beta2 * vb + (1.0 - beta2) * (db ** 2)
                
                # Bias correction
                m_hat_w = mw / (1.0 - beta1 ** t)
                m_hat_b = mb / (1.0 - beta1 ** t)
                v_hat_w = vw / (1.0 - beta2 ** t)
                v_hat_b = vb / (1.0 - beta2 ** t)
                
                # Update parameters
                w_new = w - learning_rate * m_hat_w / (jnp.sqrt(v_hat_w) + epsilon)
                b_new = b - learning_rate * m_hat_b / (jnp.sqrt(v_hat_b) + epsilon)
                
                new_params.append((w_new, b_new))
                new_m.append((mw, mb))
                new_v.append((vw, vb))
            
            return new_params, (new_m, new_v), loss_val

        # Initialize Adam state
        m = [(jnp.zeros_like(w), jnp.zeros_like(b)) for w, b in params]
        v = [(jnp.zeros_like(w), jnp.zeros_like(b)) for w, b in params]
        opt_state = (m, v)
        
        params_state = params
        t = 0
        num_batches = num_samples // batch_size

        # 5. Training loop
        for epoch in range(epochs):
            key_train, subk = jax.random.split(key_train)
            perm = jax.random.permutation(subk, num_samples)
            X_shuffled = X_train[perm]
            Y_shuffled = Y_train[perm]
            W_shuffled = sample_weights[perm]  # Shuffle weights along with data

            epoch_loss = 0.0

            for b_idx in range(num_batches):
                start = b_idx * batch_size
                end = start + batch_size
                x_batch = X_shuffled[start:end]
                y_batch = Y_shuffled[start:end]
                w_batch = W_shuffled[start:end]

                t += 1
                params_state, opt_state, loss_val = train_step(
                    params_state, opt_state, x_batch, y_batch, w_batch, t
                )
                epoch_loss += loss_val

            if epoch % 10 == 0:
                logger.info(
                    f"Epoch {epoch}: Loss = {epoch_loss / num_batches:.6f}"
                )

        logger.info("Training complete.")

        return jdc.replace(
            self,
            nn_params=params_state,
            is_trained=True,
            trained_num_obstacles=M,
            input_mean=X_mean.squeeze(0),
            input_std=X_std.squeeze(0),
        )