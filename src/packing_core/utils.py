import copy
import json
import os
import torch
import torch.nn.functional as F
import numpy as np
import random
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')  # non-interactive backend — plt.show() becomes a no-op
import matplotlib.pyplot as plt
from collections import deque
from torch_geometric.data import Batch
from .drl_env import DRLBinPackingEnv
from .models import ActorGNN, CriticGNN, SharedGNNBackbone
from scu_manifest_generator import manifest_to_item_list, SCU_DEFINITIONS

STOP_FILE = "STOP_TRAINING"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_ships_from_json(path='ships_cargo_grids.json'):
    """Load ship configurations from JSON for training."""
    try:
        with open(path, 'r') as f:
            data = json.load(f)
            return [[(tuple(g['dimensions']), g['name']) for g in ship['grids']] for ship in data['ships']]
    except Exception as e:
        print(f"Failed to load ships from {path}: {e}")
        return [[((10, 10, 10), "Main")]]


def _strip_custom_attrs(data):
    """Create a copy of a PyG Data object with only standard graph fields for batching."""
    from torch_geometric.data import Data
    return Data(x=data.x, edge_index=data.edge_index, node_type=data.node_type, pos=data.pos)


def batched_critic_values(critic, states, batch_size=32):
    """Compute critic values for a list of PyG Data states using batched inference."""
    values = []
    for i in range(0, len(states), batch_size):
        batch_states = [_strip_custom_attrs(s) for s in states[i:i + batch_size]]
        batched = Batch.from_data_list(batch_states)
        with torch.no_grad():
            batch_values = critic(batched).squeeze(-1)
        values.extend(batch_values.cpu().tolist())
    return values


def soft_update(target_net, source_net, tau=0.01):
    """Soft-update target network weights: θ_target = τ*θ_source + (1-τ)*θ_target"""
    for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
        target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)


def compute_gae(rewards, values, dones, gamma=0.95, gae_lambda=0.95):
    """
    Compute Generalized Advantage Estimation.

    Args:
        rewards: list of float rewards per step
        values: list of float V(s) per step (len = len(rewards))
        dones: list of bool done flags per step
        gamma: discount factor
        gae_lambda: GAE lambda for bias-variance tradeoff

    Returns:
        advantages: torch.Tensor of GAE advantages
        returns: torch.Tensor of discounted returns (advantages + values)
    """
    n = len(rewards)
    advantages = torch.zeros(n, device=device)
    gae = 0.0

    for t in reversed(range(n)):
        if t == n - 1 or dones[t]:
            next_value = 0.0
        else:
            next_value = values[t + 1]

        delta = rewards[t] + gamma * next_value * (1.0 - float(dones[t])) - values[t]
        gae = delta + gamma * gae_lambda * (1.0 - float(dones[t])) * gae
        advantages[t] = gae

    returns = advantages + torch.tensor(values, dtype=torch.float, device=device)
    return advantages, returns


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = []
        self.capacity = capacity
        self.position = 0

    def push(self, state, action, reward, next_state, done, log_prob=None, feasibility_mask=None):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done, log_prob, feasibility_mask)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        return list(zip(*batch))

    def __len__(self):
        return len(self.buffer)


def train_agent(num_episodes=1000, gamma=0.95, gae_lambda=0.95,
                lr_actor=3e-4, lr_critic=3e-4,
                print_interval=100, save_interval=10, checkpoint_path="multi_gnn_model_checkpoint.pt",
                batch_size=32, replay_buffer_size=10000, weight_decay=1e-4,
                ppo_epochs=4, ppo_clip=0.2, entropy_coeff=0.01,
                target_critic_tau=0.01, hidden_dim=128, possible_ships=None,
                resume=False):
    """
    Train the GNN-based DRL agent with PPO, GAE, entropy bonus, and target critic.

    Args:
        num_episodes: Number of episodes to train
        gamma: Discount factor
        gae_lambda: GAE lambda for advantage estimation
        lr_actor: Learning rate for actor
        lr_critic: Learning rate for critic
        print_interval: Print frequency
        save_interval: Checkpoint save frequency
        checkpoint_path: Path for model checkpoints
        batch_size: Batch size for PPO updates
        replay_buffer_size: Experience replay capacity
        weight_decay: L2 regularization
        ppo_epochs: Number of PPO optimization passes per episode
        ppo_clip: PPO surrogate clipping epsilon
        entropy_coeff: Entropy bonus coefficient (prevents premature convergence)
        target_critic_tau: Soft update rate for target critic network
        possible_ships: List of ship grid configurations for training
    """
    if possible_ships is None:
        possible_ships = load_ships_from_json()

    # Initialize networks
    temp_env = DRLBinPackingEnv(grids_list=possible_ships[0])
    test_state = temp_env.reset()
    node_feature_dim = test_state.x.size(1)
    del temp_env

    # Shared backbone — actor and critic learn the same state representation
    shared_backbone = SharedGNNBackbone(node_feature_dim=node_feature_dim, hidden_dim=hidden_dim).to(device)
    actor = ActorGNN(node_feature_dim=node_feature_dim, hidden_dim=hidden_dim, backbone=shared_backbone).to(device)
    critic = CriticGNN(node_feature_dim=node_feature_dim, hidden_dim=hidden_dim, backbone=shared_backbone).to(device)

    # #7: Target critic network (soft-updated copy for stable bootstrapping)
    target_critic = copy.deepcopy(critic).to(device)
    target_critic.eval()

    optimizer_actor = optim.Adam(actor.parameters(), lr=lr_actor, weight_decay=weight_decay)
    optimizer_critic = optim.Adam(critic.parameters(), lr=lr_critic, weight_decay=weight_decay)

    scheduler_actor = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_actor, mode='max', factor=0.5, patience=50)
    scheduler_critic = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_critic, mode='max', factor=0.5, patience=50)

    # Training metrics
    episode_rewards = []
    avg_rewards = deque(maxlen=100)
    actor_losses = []
    critic_losses = []
    success_rates = deque(maxlen=100)

    # #3: Running reward statistics for normalization (prevents critic divergence on large ships)
    reward_running_mean = 0.0
    reward_running_var = 1.0
    reward_count = 0

    # #7: Per-ship success tracking for weighted sampling
    ship_success_history = {i: deque(maxlen=20) for i in range(len(possible_ships))}
    difficulties = ["very-easy", "easy", "medium-low", "medium-high", "hard", "very-hard"]

    start_episode = 0
    if resume and os.path.exists(checkpoint_path):
        print(f"[resume] Loading checkpoint from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        actor.load_state_dict(ckpt['actor_state_dict'], strict=False)
        critic.load_state_dict(ckpt['critic_state_dict'], strict=False)
        if 'target_critic_state_dict' in ckpt:
            target_critic.load_state_dict(ckpt['target_critic_state_dict'], strict=False)
        if 'optimizer_actor_state_dict' in ckpt:
            optimizer_actor.load_state_dict(ckpt['optimizer_actor_state_dict'])
        if 'optimizer_critic_state_dict' in ckpt:
            optimizer_critic.load_state_dict(ckpt['optimizer_critic_state_dict'])
        if 'scheduler_actor_state_dict' in ckpt:
            scheduler_actor.load_state_dict(ckpt['scheduler_actor_state_dict'])
        if 'scheduler_critic_state_dict' in ckpt:
            scheduler_critic.load_state_dict(ckpt['scheduler_critic_state_dict'])
        if 'episode_rewards' in ckpt:
            episode_rewards = list(ckpt['episode_rewards'])
        if 'actor_losses' in ckpt:
            actor_losses = list(ckpt['actor_losses'])
        if 'critic_losses' in ckpt:
            critic_losses = list(ckpt['critic_losses'])
        if 'reward_running_mean' in ckpt:
            reward_running_mean = ckpt['reward_running_mean']
            reward_running_var = ckpt['reward_running_var']
            reward_count = ckpt['reward_count']
        start_episode = ckpt.get('episode', -1) + 1
        print(f"[resume] Resuming at episode {start_episode}/{num_episodes}")
        if start_episode >= num_episodes:
            print(f"[resume] Already complete. Skipping.")
            return actor, critic, {'episode_rewards': episode_rewards,
                                   'actor_losses': actor_losses,
                                   'critic_losses': critic_losses}

    for episode in range(start_episode, num_episodes):
        # #7: Performance-weighted ship sampling — train more on ships we're bad at
        if possible_ships:
            ship_weights = []
            for i in range(len(possible_ships)):
                history = ship_success_history[i]
                if len(history) < 3:
                    ship_weights.append(1.0)  # unexplored ships get base weight
                else:
                    avg_sr = np.mean(list(history))
                    # Soft weighting: weak ships ~2x more likely, not 10x
                    ship_weights.append(max(1.0 - 0.5 * avg_sr, 0.3))
            ship_idx = random.choices(range(len(possible_ships)), weights=ship_weights, k=1)[0]
            selected_ship = possible_ships[ship_idx]
            env = DRLBinPackingEnv(grids_list=selected_ship)
        else:
            ship_idx = 0
            env = DRLBinPackingEnv()

        # Smooth curriculum — generous easy/medium phase, gradual ramp
        progress = episode / max(num_episodes - 1, 1)
        if progress < 0.20:
            difficulty = "very-easy"
        elif progress < 0.30:
            difficulty = "easy"
        elif progress < 0.45:
            difficulty = "medium-low"
        elif progress < 0.60:
            difficulty = "medium-high"
        elif progress < 0.80:
            difficulty = "hard"
        else:
            difficulty = "very-hard"

        state = env.reset(difficulty=difficulty)

        episode_reward = 0
        trajectory = []

        # --- #11: Set actor to eval mode for rollout (deterministic BN/no dropout) ---
        actor.eval()

        # --- Episode rollout ---
        while True:
            feasibility_mask = env.get_feasibility_mask()

            if not feasibility_mask.any():
                # step(0) may still place via interior-anchor search even when
                # the mask reports no feasible action (e.g. when interior
                # search hits its probe cap and bails out of the mask, but a
                # specific position still works at placement time). Keep the
                # returned state so MER count stays in sync with the env.
                next_state, reward, done, _ = env.step(0)
                state = next_state
                episode_reward += reward
                if done:
                    break
                continue

            with torch.no_grad():
                action_dist = actor(state, feasibility_mask)
                action = action_dist.sample()
                log_prob = action_dist.log_prob(action)

            next_state, reward, done, _ = env.step(action.item())
            episode_reward += reward

            trajectory.append({
                'state': state,
                'action': action,
                'reward': reward,
                'done': done,
                'log_prob': log_prob,
                'feasibility_mask': feasibility_mask,
            })

            state = next_state
            if done:
                break

        # Skip update if episode was too short
        if len(trajectory) < 2:
            episode_rewards.append(episode_reward)
            avg_rewards.append(episode_reward)
            actor_losses.append(0)
            critic_losses.append(0)
            success_rate = env.successful_placements / env.total_items if env.total_items > 0 else 0
            success_rates.append(success_rate)
            ship_success_history[ship_idx].append(success_rate)
            continue

        # #3: Update running reward stats and normalize rewards for critic stability
        raw_rewards = [t['reward'] for t in trajectory]
        for r in raw_rewards:
            reward_count += 1
            delta = r - reward_running_mean
            reward_running_mean += delta / reward_count
            delta2 = r - reward_running_mean
            reward_running_var += (delta * delta2 - reward_running_var) / max(reward_count, 2)
        reward_std = max(reward_running_var ** 0.5, 1e-4)
        normalized_rewards = [(r - reward_running_mean) / reward_std for r in raw_rewards]

        # Compute GAE advantages using target critic with normalized rewards
        with torch.no_grad():
            values = batched_critic_values(target_critic, [t['state'] for t in trajectory])

        dones_list = [t['done'] for t in trajectory]
        advantages, returns = compute_gae(normalized_rewards, values, dones_list, gamma, gae_lambda)

        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Store old values for value clipping (#4)
        old_values = torch.tensor(values, dtype=torch.float, device=device)

        # --- #11: Set networks to train mode for PPO updates ---
        actor.train()
        critic.train()

        # --- #5: Minibatch PPO updates (accumulate gradients over trajectory) ---
        actor_loss_total = 0
        critic_loss_total = 0
        num_updates = 0

        for _ppo_epoch in range(ppo_epochs):
            # #5: Shuffle trajectory indices for each epoch
            indices = list(range(len(trajectory)))
            random.shuffle(indices)

            # Process in minibatches
            for mb_start in range(0, len(indices), batch_size):
                mb_indices = indices[mb_start:mb_start + batch_size]

                # Accumulate losses over minibatch
                mb_actor_loss = torch.tensor(0.0, device=device)
                mb_critic_loss = torch.tensor(0.0, device=device)

                for i in mb_indices:
                    trans = trajectory[i]

                    # #6: Re-evaluate with correct feasibility mask (dropout active in train mode)
                    action_dist = actor(trans['state'], trans['feasibility_mask'])
                    new_log_prob = action_dist.log_prob(trans['action'])

                    # PPO importance ratio
                    ratio = torch.exp(new_log_prob - trans['log_prob'])

                    # Clipped surrogate objective
                    adv = advantages[i].detach()
                    surr1 = ratio * adv
                    surr2 = torch.clamp(ratio, 1.0 - ppo_clip, 1.0 + ppo_clip) * adv
                    policy_loss = -torch.min(surr1, surr2)

                    # Entropy bonus
                    entropy = action_dist.entropy()
                    mb_actor_loss = mb_actor_loss + policy_loss - entropy_coeff * entropy

                    # #4: Clipped value loss (prevents critic from making huge jumps)
                    value = critic(trans['state']).squeeze()
                    value_clipped = old_values[i] + torch.clamp(
                        value - old_values[i], -ppo_clip, ppo_clip)
                    vf_loss1 = F.mse_loss(value, returns[i])
                    vf_loss2 = F.mse_loss(value_clipped, returns[i])
                    mb_critic_loss = mb_critic_loss + torch.max(vf_loss1, vf_loss2)

                # Average over minibatch
                mb_size = len(mb_indices)
                mb_actor_loss = mb_actor_loss / mb_size
                mb_critic_loss = mb_critic_loss / mb_size

                # Combined backward through shared backbone (single pass)
                total_loss = mb_actor_loss + mb_critic_loss
                optimizer_actor.zero_grad()
                optimizer_critic.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=0.5)
                torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=0.5)
                optimizer_actor.step()
                optimizer_critic.step()

                actor_loss_total += mb_actor_loss.item()
                critic_loss_total += mb_critic_loss.item()
                num_updates += 1

        # Soft-update target critic
        soft_update(target_critic, critic, tau=target_critic_tau)

        # Average losses
        actor_loss_avg = actor_loss_total / max(num_updates, 1)
        critic_loss_avg = critic_loss_total / max(num_updates, 1)

        # Track metrics
        if isinstance(episode_reward, torch.Tensor):
            episode_reward = episode_reward.detach().cpu().item()
        episode_rewards.append(episode_reward)
        avg_rewards.append(episode_reward)
        actor_losses.append(actor_loss_avg)
        critic_losses.append(critic_loss_avg)

        success_rate = env.successful_placements / env.total_items if env.total_items > 0 else 0
        success_rates.append(success_rate)
        ship_success_history[ship_idx].append(success_rate)

        # Schedule LR based on success rate (not reward, which drops with curriculum)
        if episode % 10 == 0:
            mean_success = np.mean(list(success_rates))
            scheduler_actor.step(mean_success)
            scheduler_critic.step(mean_success)

        # Print progress
        if (episode + 1) % print_interval == 0:
            avg_reward = np.mean(list(avg_rewards))
            avg_success = np.mean(list(success_rates))
            print(f"Episode {episode + 1}/{num_episodes} | "
                  f"Ship Grids: {len(env.grids)} | "
                  f"Difficulty: {difficulty} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Success Rate: {avg_success:.2f} | "
                  f"Actor Loss: {actor_loss_avg:.4f} | "
                  f"Critic Loss: {critic_loss_avg:.4f}")

        # Save checkpoint
        should_stop = os.path.exists(STOP_FILE)
        if (episode + 1) % save_interval == 0 or should_stop:
            torch.save({
                'episode': episode,
                'actor_state_dict': actor.state_dict(),
                'critic_state_dict': critic.state_dict(),
                'target_critic_state_dict': target_critic.state_dict(),
                'optimizer_actor_state_dict': optimizer_actor.state_dict(),
                'optimizer_critic_state_dict': optimizer_critic.state_dict(),
                'scheduler_actor_state_dict': scheduler_actor.state_dict(),
                'scheduler_critic_state_dict': scheduler_critic.state_dict(),
                'episode_rewards': episode_rewards,
                'actor_losses': actor_losses,
                'critic_losses': critic_losses,
                'reward_running_mean': reward_running_mean,
                'reward_running_var': reward_running_var,
                'reward_count': reward_count,
            }, checkpoint_path)
        if should_stop:
            print(f"[pause] STOP_TRAINING file detected. Saved checkpoint at episode {episode}. Exiting.")
            return actor, critic, {'episode_rewards': episode_rewards,
                                   'actor_losses': actor_losses,
                                   'critic_losses': critic_losses,
                                   'stopped': True}

    # Plot training curves
    plot_training_curves(episode_rewards, actor_losses, critic_losses)

    return actor, critic, {
        'episode_rewards': episode_rewards,
        'actor_losses': actor_losses,
        'critic_losses': critic_losses
    }

def plot_training_curves(rewards, actor_losses, critic_losses):
            """Plot training metrics"""
            plt.figure(figsize=(15, 5))

            plt.subplot(1, 3, 1)
            plt.plot(rewards)
            plt.title('Episode Rewards')
            plt.xlabel('Episode')
            plt.ylabel('Reward')

            plt.subplot(1, 3, 2)
            plt.plot(actor_losses)
            plt.title('Actor Loss')
            plt.xlabel('Episode')
            plt.ylabel('Loss')

            plt.subplot(1, 3, 3)
            plt.plot(critic_losses)
            plt.title('Critic Loss')
            plt.xlabel('Episode')
            plt.ylabel('Loss')

            plt.tight_layout()
            plt.savefig('enhanced_gnn_training_curves.png')
            plt.close('all')

def evaluate_agent(env, actor, num_episodes=10, visualize=False):
            """
            Evaluate the trained GNN agent

            Args:
                env: DRLBinPackingEnv instance
                actor: Trained actor network
                num_episodes: Number of episodes to evaluate
                visualize: Whether to visualize the final packing

            Returns:
                Average reward and success rate
            """
            episode_rewards = []
            success_rates = []
            volume_utilizations = []

            for episode in range(num_episodes):
                state = env.reset(difficulty="hard")  # Evaluate on hard difficulty
                episode_reward = 0
                total_item_volume = 0
                container_volume = sum(torch.prod(g['dims']).item() for g in env.grids)

                # Episode loop
                while True:
                    # Get feasibility mask
                    feasibility_mask = env.get_feasibility_mask()

                    # Check if there are any feasible actions
                    if not feasibility_mask.any():
                        # No feasible placements, move to next item
                        _, reward, done, _ = env.step(0)  # Dummy action
                        episode_reward += reward

                        if done:
                            break

                        # Continue with next item
                        continue

                    # Get action from actor (greedy)
                    with torch.no_grad():
                        action_dist = actor(state, feasibility_mask)
                        # Select action with highest probability
                        action = torch.argmax(action_dist.probs)

                    # Take step in environment
                    next_state, reward, done, _ = env.step(action.item())
                    episode_reward += reward

                    # If successful placement, add volume
                    if env.successful_placements > 0 and env.current_item_idx > 0:
                        prev_item = env.cargo_manifest[env.current_item_idx - 1]
                        w, l, h, _, _ = prev_item
                        total_item_volume += w * l * h

                    # Update state
                    state = next_state

                    # Break if episode is done
                    if done:
                        break

                # Calculate metrics
                success_rate = env.successful_placements / env.total_items if env.total_items > 0 else 0
                volume_utilization = total_item_volume / container_volume

                episode_rewards.append(episode_reward)
                success_rates.append(success_rate)
                volume_utilizations.append(volume_utilization)

                # Optional: Visualize final packing
                if visualize and episode == 0:  # Visualize first episode
                    visualize_packing(env)

            avg_reward = np.mean(episode_rewards)
            avg_success_rate = np.mean(success_rates)
            avg_volume_utilization = np.mean(volume_utilizations)

            print(f"Evaluation over {num_episodes} episodes:")
            print(f"Average Reward: {avg_reward:.2f}")
            print(f"Average Success Rate: {avg_success_rate:.2f}")
            print(f"Average Volume Utilization: {avg_volume_utilization:.2f}")

            return avg_reward, avg_success_rate, avg_volume_utilization

def visualize_packing(env):
    """
    Visualize the 3D bin packing result
    Requires matplotlib and possibly mplot3d
    """
    try:
        from mpl_toolkits.mplot3d import Axes3D
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        import matplotlib.pyplot as plt
        import numpy as np

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot container boundaries
        container_width, container_length, container_height = [x.item() for x in env.grids[0]['dims']]

        # Define the vertices of the container
        vertices = [
            [0, 0, 0],
            [container_width, 0, 0],
            [container_width, container_length, 0],
            [0, container_length, 0],
            [0, 0, container_height],
            [container_width, 0, container_height],
            [container_width, container_length, container_height],
            [0, container_length, container_height]
        ]

        # Define the faces of the container
        faces = [
            [vertices[0], vertices[1], vertices[2], vertices[3]],  # Bottom
            [vertices[4], vertices[5], vertices[6], vertices[7]],  # Top
            [vertices[0], vertices[1], vertices[5], vertices[4]],  # Front
            [vertices[2], vertices[3], vertices[7], vertices[6]],  # Back
            [vertices[0], vertices[3], vertices[7], vertices[4]],  # Left
            [vertices[1], vertices[2], vertices[6], vertices[5]]  # Right
        ]

        # Plot container as wireframe
        for face in faces:
            face_array = np.array(face)
            ax.plot3D(face_array[:, 0], face_array[:, 1], face_array[:, 2], 'k-', alpha=0.2)

        # Plot placed items
        for i, placed in enumerate(env.placed_items):
            box = placed['box']
            priority = placed['priority']

            # Define the vertices of the box (Box3D fields are floats)
            x, y, z = box.x1, box.y1, box.z1
            w, l, h = box.width, box.length, box.height

            box_vertices = [
                [x, y, z],
                [x + w, y, z],
                [x + w, y + l, z],
                [x, y + l, z],
                [x, y, z + h],
                [x + w, y, z + h],
                [x + w, y + l, z + h],
                [x, y + l, z + h]
            ]

        # Define the faces of the box
            box_faces = [
                [box_vertices[0], box_vertices[1], box_vertices[2], box_vertices[3]],  # Bottom
                [box_vertices[4], box_vertices[5], box_vertices[6], box_vertices[7]],  # Top
                [box_vertices[0], box_vertices[1], box_vertices[5], box_vertices[4]],  # Front
                [box_vertices[2], box_vertices[3], box_vertices[7], box_vertices[6]],  # Back
                [box_vertices[0], box_vertices[3], box_vertices[7], box_vertices[4]],  # Left
                [box_vertices[1], box_vertices[2], box_vertices[6], box_vertices[5]]  # Right
            ]

        # Color based on priority (1-5)
            import matplotlib
            cmap = matplotlib.colormaps.get_cmap('viridis')
            color = cmap((priority - 1) / 4)  # normalize to [0, 1]

            # Create a 3D collection of polygons
            box_collection = Poly3DCollection(box_faces, alpha=0.7)
            box_collection.set_facecolor(color)
            box_collection.set_edgecolor('black')

            # Add the collection to the plot
            ax.add_collection3d(box_collection)

            # Add text label in the center of the box
            box_center = [x + w / 2, y + l / 2, z + h / 2]
            ax.text(box_center[0], box_center[1], box_center[2], str(i + 1),
                    color='white', ha='center', va='center', fontsize=8)

        # Set labels and title
        ax.set_xlabel('Width (X)')
        ax.set_ylabel('Length (Y)')
        ax.set_zlabel('Height (Z)')
        ax.set_title('3D Bin Packing Result')

        # Set axis limits
        ax.set_xlim([0, container_width])
        ax.set_ylim([0, container_length])
        ax.set_zlim([0, container_height])

        # Add a colorbar to show priority mapping
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(1, 5))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.6, aspect=10)
        cbar.set_label('Unload Priority')

        # Add volume utilization information
        total_volume = 0
        container_volume = float(sum(torch.prod(g['dims']).item() for g in env.grids))
        for placed in env.placed_items:
            box = placed['box']
            total_volume += float(box.volume)

        utilization = total_volume / container_volume * 100
        plt.figtext(0.02, 0.02, f"Volume Utilization: {utilization:.1f}%", fontsize=10)
        plt.figtext(0.02, 0.05, f"Success Rate: {env.successful_placements}/{env.total_items}", fontsize=10)

        plt.tight_layout()
        plt.savefig('enhanced_3d_packing_visualization.png')
        plt.close('all')

    except ImportError as e:
        print(f"Visualization requires additional libraries: {e}")

# Also create a function to load a saved model for inference
def load_trained_model(checkpoint_path="enhanced_gnn_model_checkpoint.pt", device=None):
    """
    Load a trained model from checkpoint
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Initialize actor network
    # We need to determine the feature dimension - use a small test environment
    temp_env = DRLBinPackingEnv(grids_list=[((4, 4, 4), "Main")])
    test_state = temp_env.reset()
    node_feature_dim = test_state.x.size(1)
    del temp_env

    # Detect hidden_dim from checkpoint weights
    actor_weights = checkpoint['actor_state_dict']
    # The score_head final layer output is always 2; its input reveals hidden_dim
    hidden_dim = 128  # default
    for key, val in actor_weights.items():
        if 'score_head.0.weight' in key:
            hidden_dim = val.shape[1]
            break

    actor = ActorGNN(node_feature_dim=node_feature_dim, hidden_dim=hidden_dim).to(device)
    actor.load_state_dict(checkpoint['actor_state_dict'], strict=False)
    actor.eval()

    print(f"Model loaded from {checkpoint_path}")
    print(f"Trained for {checkpoint['episode']} episodes")

    return actor, checkpoint

# Function for single inference (for API use)
def _brute_force_find_placement(env, item_tuple):
    """Diagnostic: integer-grid scan for any feasible placement of item_tuple.
    Returns (grid_idx, x, y, z, rot) or None."""
    w, l, h, weight, priority = item_tuple
    rotations = [(w, l), (l, w)] if w != l else [(w, l)]
    for grid_idx, grid in enumerate(env.grids):
        gw, gl, gh = [int(d.item()) for d in grid['dims']]
        grid_dims = grid['dims']
        for rot, (iw, il) in enumerate(rotations):
            iw_i, il_i, ih_i = int(iw), int(il), int(h)
            if iw_i > gw or il_i > gl or ih_i > gh:
                continue
            for x in range(gw - iw_i + 1):
                for y in range(gl - il_i + 1):
                    for z in range(gh - ih_i + 1):
                        pos = torch.tensor([float(x), float(y), float(z)])
                        dims = torch.tensor([float(iw_i), float(il_i), float(ih_i)])
                        if env._check_additional_constraints(pos, dims, weight, priority, grid_idx, grid_dims):
                            return (grid_idx, x, y, z, rot, iw_i, il_i, ih_i)
    return None


def pack_single_manifest(actor, grids_list, manifest, device=None, diagnose=False):
    """
    Pack a single manifest across multiple grids using the trained model.
    If diagnose=True, when the env skips an item, brute-force scan for any
    feasible integer-grid placement and record it in result['diagnostics'].
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create environment
    env = DRLBinPackingEnv(grids_list=grids_list)

    # Convert manifest to item list
    cargo_manifest = manifest_to_item_list(manifest)

    # Reset environment with manifest
    state = env.reset(cargo_manifest=cargo_manifest)

    diagnostics = {"missed_placements": [], "skipped_items": []}
    skipped_indices = []  # actual manifest indices that failed to place

    # Run packing
    steps = 0
    max_steps = 500

    while steps < max_steps and env.current_item is not None:
        steps += 1

        # Get feasibility mask
        feasibility_mask = env.get_feasibility_mask()
        pre_step_idx = env.current_item_idx
        pre_step_placed = env.successful_placements

        # Check if there are any feasible actions
        if not feasibility_mask.any():
            if diagnose:
                idx = env.current_item_idx
                item_tuple = cargo_manifest[idx]
                skipped_info = {
                    "item_idx": idx,
                    "dims": list(item_tuple[:3]),
                    "weight": item_tuple[3],
                    "priority": item_tuple[4],
                }
                found = _brute_force_find_placement(env, item_tuple)
                if found is not None:
                    gidx, x, y, z, rot, iw, il, ih = found
                    miss = {
                        **skipped_info,
                        "grid_idx": gidx,
                        "grid_name": env.grids[gidx]['name'],
                        "feasible_position": [x, y, z],
                        "rotated_dims": [iw, il, ih],
                        "rotation": rot,
                    }
                    diagnostics["missed_placements"].append(miss)
                    print(f"[diagnose] MER MISS: item#{idx} dims={skipped_info['dims']} P{skipped_info['priority']} "
                          f"— brute-force found ({x},{y},{z}) on grid '{env.grids[gidx]['name']}' rot={rot}",
                          flush=True)
                else:
                    diagnostics["skipped_items"].append(skipped_info)
                    print(f"[diagnose] NO-FIT (given current layout): item#{idx} dims={skipped_info['dims']} P{skipped_info['priority']}",
                          flush=True)
            skipped_indices.append(pre_step_idx)
            env._move_to_next_item()
            if env.current_item is None:
                break
            state = env._build_graph_state()
            continue

        # Get action from actor (greedy)
        with torch.no_grad():
            action_dist = actor(state, feasibility_mask)
            action = torch.argmax(action_dist.probs)

        # Take step
        next_state, reward, done, info = env.step(action.item())
        state = next_state

        # Detect skip-via-step-failure: item idx advanced but placements didn't
        if env.successful_placements == pre_step_placed and env.current_item_idx > pre_step_idx:
            skipped_indices.append(pre_step_idx)

        if done:
            break

    # Prepare results
    placements = []
    for i, placed in enumerate(env.placed_items):
        box = placed['box']

        # Find which SCU type this corresponds to (compare sorted dims since rotation changes order)
        dims = [box.width, box.length, box.height]
        dims_sorted = sorted(dims, reverse=True)
        scu_type = None
        for scu, scu_info in SCU_DEFINITIONS.items():
            if dims_sorted == sorted(scu_info["dimensions"], reverse=True):
                scu_type = scu
                break

        placement = {
            "item_id": f"{scu_type}_{i + 1:03d}",
            "scu_type": scu_type or "Unknown",
            "position": [box.x1, box.y1, box.z1],
            "dimensions": dims,
            "priority": placed['priority'],
            "grid_idx": placed['grid_idx'],
            "grid_name": env.grids[placed['grid_idx']]['name']
        }
        placements.append(placement)

    # Calculate metrics
    success_rate = env.successful_placements / env.total_items if env.total_items > 0 else 0
    total_volume = sum(p['box'].volume for p in env.placed_items)
    container_volume = sum(torch.prod(g['dims']).item() for g in env.grids)
    utilization = total_volume / container_volume

    # Identify unplaced items: the ACTUAL skipped indices, plus any manifest tail
    # the loop never reached (shouldn't normally happen, but guard against it).
    unplaced_items = []
    reached_idx = env.current_item_idx  # one past the last item processed
    leftover = [i for i in range(reached_idx, len(cargo_manifest))]
    for i in skipped_indices + leftover:
        if i >= len(cargo_manifest):
            continue
        item = cargo_manifest[i]
        item_dims_sorted = sorted(item[:3], reverse=True)
        for scu, scu_info in SCU_DEFINITIONS.items():
            if item_dims_sorted == sorted(scu_info["dimensions"], reverse=True):
                unplaced_items.append({
                    "scu_type": scu,
                    "priority": item[4],
                    "item_idx": i,
                })
                break

    result = {
        "success": True,
        "ship_grids": grids_list,
        "placements": placements,
        "unplaced_items": unplaced_items,
        "metrics": {
            "success_rate": float(success_rate),
            "volume_utilization": float(utilization),
            "items_placed": env.successful_placements,
            "total_items": env.total_items
        }
    }

    if diagnose:
        result["diagnostics"] = diagnostics

    return result

        # Example usage
