import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from collections import deque
from cargo_manifest_generator import (
    parse_provided_data,
    sample_cargo_manifest,
    generate_cargo_manifest,
    CargoItem
)
# Parse the provided cargo item data
RAW_CARGO_DATA = """
1,1 SCU,1,1,1,1,Standard 1 SCU cargo container
3,4 SCU,2,2,1,4,Standard 4 SCU cargo container
5,16 SCU,2,4,2,16,Standard 16 SCU cargo container
6,32 SCU,2,8,2,32,Standard 32 SCU cargo container
4,8 SCU,2,2,2,8,Standard 8 SCU cargo container
2,2 SCU,2,1,1,2,Standard 2 SCU cargo container

72,1,main,Main,2,4,3,2
73,16,main,Main,6,6,3,32
74,17,auxiliary,Left,1,2,2,2
75,17,main,Main,5,8,3,32
76,17,auxiliary,Right,1,2,2,2
77,18,main,Main,4,8,3,32
78,2,main,Main,4,6,3,32
79,3,main,Main,4,9,3,32
80,4,auxiliary,Cargo L1,3,2,3,2
81,4,auxiliary,Cargo L2,3,3,3,2
82,4,auxiliary,Cargo L3,3,3,3,2
83,4,auxiliary,Cargo L4,3,2,3,2
84,4,auxiliary,Cargo R1,3,2,3,2
85,4,auxiliary,Cargo R2,3,3,3,2
86,4,auxiliary,Cargo R3,3,3,3,2
87,4,auxiliary,Cargo R4,3,2,3,2
88,4,auxiliary,Salvage L1,5,7,2,32
89,4,auxiliary,Salvage L2,5,5,2,32
90,4,auxiliary,Salvage R1,5,7,2,32
91,4,auxiliary,Salvage R2,5,5,2,32
92,19,main,Main,2,4,1,2
93,20,secondary,Aft,8,15,4,32
94,20,auxiliary,For,6,9,4,32
95,5,auxiliary,Left,2,8,2,2
96,5,auxiliary,Right,2,8,2,2
97,6,auxiliary,1,2,4,2,2
98,6,auxiliary,2,2,4,2,2
99,6,auxiliary,3,2,4,2,2
100,6,auxiliary,4,2,4,2,2
101,21,main,Main,8,12,4,32
102,7,secondary,Aft,2,3,1,2
103,7,main,Main,4,14,3,32
104,22,secondary,Aft,3,7,2,32
105,22,main,Main,3,6,2,32
106,23,main,Main,4,5,2,32
107,24,auxiliary,1,4,4,6,32
108,24,auxiliary,2,4,4,6,32
109,24,auxiliary,3,4,4,6,32
110,24,auxiliary,4,4,4,6,32
111,24,auxiliary,5,6,5,2,32
112,8,main,Main,3,7,2,32
113,25,main,Main,4,5,2,32
114,9,main,Main,1,1,2,2
115,26,main,Main,4,4,2,2
116,10,main,Main,8,6,2,32
117,27,secondary,Aft,8,15,3,32
118,27,auxiliary,For,6,9,3,32
119,28,auxiliary,1,4,4,4,32
120,28,auxiliary,2,4,4,4,32
121,28,auxiliary,3,4,4,4,32
122,28,auxiliary,4,4,4,4,32
123,28,auxiliary,5,4,4,4,32
124,28,auxiliary,6,4,4,4,32
125,28,auxiliary,7,4,3,2,2
126,28,auxiliary,8,4,3,2,2
127,28,auxiliary,9,4,3,2,2
128,11,main,Main,5,7,5,32
129,29,main,Main,4,10,2,32
130,30,main,Main,4,8,3,32
131,12,auxiliary,1,12,6,8,32
132,12,auxiliary,2,12,6,8,32
133,12,auxiliary,3,12,6,8,32
134,12,auxiliary,4,12,6,8,32
135,12,auxiliary,5,12,6,8,32
136,12,auxiliary,6,12,6,8,32
137,12,auxiliary,7,12,6,8,32
138,12,auxiliary,8,12,6,8,32
139,13,main,Main,2,9,3,32
140,14,main,Main,2,4,1,2
141,31,main,Main,5,6,1,2
142,15,main,Main,6,18,2,32
"""
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import random

@dataclass
class CargoItem:
    width: float
    length: float
    height: float
    weight: float
    unload_priority: int
    item_id: str
    item_type: str = "standard"
    location: str = "main"

    def __str__(self):
        return f"ID: {self.item_id}, Type: {self.item_type}, Dims: {self.width}x{self.length}x{self.height}, Weight: {self.weight}, Priority: {self.unload_priority}"

    def to_tuple(self):
        """Convert to the format expected by the DRL environment"""
        return (self.width, self.length, self.height, self.weight, self.unload_priority)

def parse_cargo_data(data_lines):
    """Parse the cargo data from the provided lines"""
    standard_containers = {}
    cargo_items = {}

    # Parse standard containers
    for i, line in enumerate(data_lines):
        if i < 6:  # First 6 lines are standard containers
            parts = line.strip().split(',')
            if len(parts) >= 6:
                container_id = parts[0]
                container_name = parts[1]
                width = float(parts[2])
                length = float(parts[3])
                height = float(parts[4])
                weight = float(parts[5])
                standard_containers[container_name] = (width, length, height, weight)
        elif line.strip():  # Skip empty lines
            # Parse cargo items
            parts = line.strip().split(',')
            if len(parts) >= 8:
                item_id = parts[0]
                ship_id = parts[1]
                location_type = parts[2]
                location_name = parts[3]
                width = float(parts[4])
                length = float(parts[5])
                height = float(parts[6])
                weight = float(parts[7])

                # Create a unique identifier
                unique_id = f"{item_id}_{location_type}_{location_name}"

                cargo_items[unique_id] = CargoItem(
                    width=width,
                    length=length,
                    height=height,
                    weight=weight,
                    unload_priority=random.randint(1, 5),  # Assign random unload priority for now
                    item_id=unique_id,
                    item_type="cargo",
                    location=f"{location_type}_{location_name}"
                )

    return standard_containers, cargo_items

def sample_cargo_manifest(
        min_items: int = 5,
        max_items: int = 15,
        standard_containers: Optional[Dict] = None,
        cargo_items: Optional[Dict] = None,
        seed: Optional[int] = None
) -> List[Tuple]:
    """
    Sample a cargo manifest for training the DRL model

    Args:
        min_items: Minimum number of items in the manifest
        max_items: Maximum number of items in the manifest
        standard_containers: Dictionary of standard container definitions
        cargo_items: Dictionary of custom cargo items
        seed: Random seed

    Returns:
        List of tuples in the format (width, length, height, weight, unload_priority)
    """
    if seed is not None:
        random.seed(seed)

    if standard_containers is None:
        standard_containers = STANDARD_CONTAINERS

    # Generate a random number of items
    num_items = random.randint(min_items, max_items)

    # Create manifest
    manifest = []

    # If we have custom cargo items, use a mix of custom and standard
    if cargo_items:
        # Determine how many custom items to use (up to 70% of total)
        num_custom = min(len(cargo_items), int(num_items * 0.7))
        if num_custom > 0:
            selected_custom_keys = random.sample(list(cargo_items.keys()), num_custom)

            # Add custom items to manifest
            for key in selected_custom_keys:
                manifest.append(cargo_items[key].to_tuple())

        # Fill remaining slots with standard containers
        remaining = num_items - len(manifest)
        for _ in range(remaining):
            container_type = random.choice(list(standard_containers.keys()))
            w, l, h, weight = standard_containers[container_type]

            # Create tuple with random unload priority
            manifest.append((w, l, h, weight, random.randint(1, 5)))
    else:
        # Only use standard containers
        for _ in range(num_items):
            container_type = random.choice(list(standard_containers.keys()))
            w, l, h, weight = standard_containers[container_type]

            # Create tuple with random unload priority
            manifest.append((w, l, h, weight, random.randint(1, 5)))

    # Shuffle the manifest for random order
    random.shuffle(manifest)

    return manifest

# Initialize cargo data
STANDARD_CONTAINERS, CARGO_ITEMS = parse_cargo_data(RAW_CARGO_DATA.strip().split('\n'))
# --- 1. Define a custom 3D bin packing environment ---
class DRLBinPackingEnv:
    def __init__(self, container_dims=(10, 10, 10), grid_resolution=1.0, max_stack_weight=100.0):
        self.container_dims = container_dims
        self.grid_resolution = grid_resolution
        self.grid_width = int(container_dims[0] / grid_resolution)
        self.grid_length = int(container_dims[1] / grid_resolution)
        self.action_dim = self.grid_width * self.grid_length
        self.state_dim = self.grid_width * self.grid_length
        self.placements = []
        self.max_stack_weight = max_stack_weight
        self.current_item = None
        self.door_y = 0.0
        self.cargo_manifest = []
        self.current_item_idx = 0
        self.current_episode_items = 0
        self.successful_placements = 0

    def reset(self, cargo_manifest=None):
        """Reset the environment with an optional cargo manifest"""
        self.placements = []

        # Either use provided manifest or generate a new one
        if cargo_manifest is not None:
            self.cargo_manifest = cargo_manifest
        else:
            self.cargo_manifest = sample_cargo_manifest(
                min_items=5,
                max_items=15,
                standard_containers=STANDARD_CONTAINERS,
                cargo_items=CARGO_ITEMS
            )

        self.current_item_idx = 0
        self.current_episode_items = len(self.cargo_manifest)
        self.successful_placements = 0

        # Set the first item as current
        if self.cargo_manifest:
            self.current_item = self.cargo_manifest[0]
        else:
            self.current_item = None

        self.state = np.zeros((self.grid_width, self.grid_length), dtype=np.float32)
        return self.state

    def step(self, action):
        """Take a step in the environment by placing the current item"""
        if self.current_item is None:
            raise ValueError("No current item set!")

        grid_x = action % self.grid_width
        grid_y = action // self.grid_width
        possible_x = grid_x * self.grid_resolution
        possible_y = grid_y * self.grid_resolution
        possible_z = self.state[grid_x, grid_y]

        # Unpack current item
        w, l, h, weight, unload_priority = self.current_item
        possible = (possible_x, possible_y, possible_z, w, l, h, unload_priority, weight)

        # Check if placement is feasible
        feasible = (self.check_container_bounds(possible) and
                    self.check_no_overlap(possible) and
                    self.check_access_path(possible) and
                    self.check_route_constraint(possible) and
                    self.check_stacking_weights(possible))

        if not feasible:
            reward = -10.0
            done = False  # Don't end episode, move to next item
            info = {"feasible": False}

            # Move to next item if available
            self.current_item_idx += 1
            if self.current_item_idx < len(self.cargo_manifest):
                self.current_item = self.cargo_manifest[self.current_item_idx]
            else:
                # No more items, end episode
                self.current_item = None
                done = True

            return self.state, reward, done, info

        # If placement is feasible, update the environment
        self.placements.append(possible)
        self.state[grid_x, grid_y] = possible_z + h
        self.successful_placements += 1

        # Calculate reward - can be modified to include more sophisticated metrics
        # like packing density, stability, etc.
        reward = 10.0  # Base reward for successful placement

        # Bonus for efficient space utilization (items placed in corners or edges)
        if grid_x == 0 or grid_x == self.grid_width - 1 or grid_y == 0 or grid_y == self.grid_length - 1:
            reward += 2.0

        # Bonus for placing items with similar unload priorities near each other
        priority_bonus = 0
        for p in self.placements[:-1]:  # Check against all previous placements
            _, _, _, _, _, _, p_priority, _ = p
            if abs(p_priority - unload_priority) <= 1:  # Similar priority
                priority_bonus += 1.0
        reward += min(priority_bonus, 5.0)  # Cap the bonus

        # Move to next item if available
        self.current_item_idx += 1
        if self.current_item_idx < len(self.cargo_manifest):
            self.current_item = self.cargo_manifest[self.current_item_idx]
        else:
            # No more items, end episode
            self.current_item = None
            # Bonus for completing the entire manifest
            if self.successful_placements == self.current_episode_items:
                reward += 50.0

            done = True
            info = {"feasible": True, "completed": True,
                    "success_rate": self.successful_placements / self.current_episode_items}
            return self.state, reward, done, info

        done = False
        info = {"feasible": True}
        return self.state, reward, done, info

    # Add additional methods for feasibility checks (unchanged from original)
    def get_feasibility_mask(self):
        """
        Compute a binary mask over the action space.
        For each possible grid cell, simulate the placement of the current item,
        and set mask = 1 if all constraints are met, otherwise 0.
        """
        mask = np.zeros(self.action_dim, dtype=np.float32)
        if self.current_item is None:
            return mask

        for action in range(self.action_dim):
            grid_x = action % self.grid_width
            grid_y = action // self.grid_width
            possible_x = grid_x * self.grid_resolution
            possible_y = grid_y * self.grid_resolution
            possible_z = self.state[grid_x, grid_y]

            w, l, h, weight, unload_priority = self.current_item
            possible = (possible_x, possible_y, possible_z, w, l, h, unload_priority, weight)

            if (self.check_container_bounds(possible) and
                    self.check_no_overlap(possible) and
                    self.check_access_path(possible) and
                    self.check_route_constraint(possible) and
                    self.check_stacking_weights(possible)):
                mask[action] = 1.0
            else:
                mask[action] = 0.0

        return mask

    # Implement all the constraint check methods
    # These would be identical to those in your original code
    def check_container_bounds(self, possible):
        """Ensure the box fits in the container."""
        x, y, z, w, l, h, _, _ = possible
        W, L, H_container = self.container_dims
        if x + w > W or y + l > L or z + h > H_container:
            return False
        return True

    def check_no_overlap(self, possible):
        """Ensure possible does not overlap any placed box."""
        cx, cy, cz, cw, cl, ch, _, _ = possible
        for p in self.placements:
            px, py, pz, pw, pl, ph, _, _ = p
            # Two boxes do NOT overlap if one is completely to the left/right,
            # front/behind, or above/below the other.
            if not ((cx + cw <= px) or (px + pw <= cx) or
                    (cy + cl <= py) or (py + pl <= cy) or
                    (cz + ch <= pz) or (pz + ph <= cz)):
                return False
        return True

    def check_access_path(self, possible):
        """
        Checks that the possible placement does not block the access path to the door.
        For example, ensure that if a box is placed, there remains a clear vertical corridor near y=0.
        """
        _, y, _, _, cl, _, unload_priority, _ = possible
        # For simplicity, assume that if the possible's y is too low (close to door)
        # and its length cl is large, it might block the access path.
        if y < self.door_y + 1.0 and cl > 0.5:  # threshold values as example
            # Also, check if a box with a lower unload_priority is already placed in front.
            for p in self.placements:
                _, py, _, _, p_cl, _, p_priority, _ = p
                if p_priority < unload_priority and py < y:
                    return False
        return True

    def check_route_constraint(self, possible):
        """
        Enforce that boxes with lower unload_priority (unload first) are placed
        in front (lower y) than boxes with higher unload_priority.
        """
        _, y, _, _, _, _, unload_priority, _ = possible
        for p in self.placements:
            _, py, _, _, _, _, p_priority, _ = p
            # If the possible should unload before a placed box,
            # it should be positioned further front (lower y value).
            if unload_priority < p_priority and y >= py:
                return False
            # Conversely, if possible unloads later, it should be behind (higher y)
            if unload_priority > p_priority and y <= py:
                return False
        return True

    def check_stacking_weights(self, possible):
        """
        Ensure that the weight of boxes stacked on top of a possible placement
        does not exceed a maximum threshold.
        """
        cx, cy, cz, cw, cl, ch, _, weight = possible
        total_weight_above = 0.0
        for p in self.placements:
            px, py, pz, pw, pl, ph, _, p_weight = p
            # Check if placed box is above possible and overlaps in x and y.
            if pz >= cz + ch:
                # Check horizontal overlap (simple intersection test)
                if (px < cx + cw and px + pw > cx and
                        py < cy + cl and py + pl > cy):
                    total_weight_above += p_weight
        if total_weight_above > self.max_stack_weight:
            return False
        return True

# Example usage to create training samples for the DRL model
def create_training_data(num_samples=100, min_items=5, max_items=15, seed=None):
    """
    Create a set of cargo manifests for training

    Args:
        num_samples: Number of manifests to create
        min_items: Minimum items per manifest
        max_items: Maximum items per manifest
        seed: Random seed for reproducibility

    Returns:
        List of cargo manifests
    """
    if seed is not None:
        np.random.seed(seed)

    manifests = []
    for i in range(num_samples):
        # Create a random seed for each manifest to ensure diversity
        manifest_seed = None if seed is None else seed + i

        # Generate number of items
        num_items = np.random.randint(min_items, max_items + 1)

        # Generate manifest
        manifest = sample_cargo_manifest(
            min_items=num_items,
            max_items=num_items,
            standard_containers=STANDARD_CONTAINERS,
            cargo_items=CARGO_ITEMS,
            seed=manifest_seed
        )

        manifests.append(manifest)

    return manifests
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, state, feas_mask=None):
        logits = self.fc(state)
        if feas_mask is not None:
            # Convert feasibility mask to tensor and ensure it is on the same device
            feas_mask_tensor = torch.FloatTensor(feas_mask).to(logits.device)
            # Replace logits for infeasible actions with a very small value
            logits = logits + torch.log(feas_mask_tensor + 1e-10)
        return Categorical(logits=logits)

class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state):
        return self.fc(state)
def train_agent(env, num_episodes=1000, gamma=0.99, lr_actor=1e-4, lr_critic=1e-3,
                print_interval=10, save_interval=100, checkpoint_path="drl_model_checkpoint.pt"):
    """
    Main training function for the DRL bin packing agent

    Args:
        env: The bin packing environment
        num_episodes: Total number of episodes to train for
        gamma: Discount factor
        lr_actor: Learning rate for the actor network
        lr_critic: Learning rate for the critic network
        print_interval: How often to print training statistics
        save_interval: How often to save model checkpoints
        checkpoint_path: Where to save model checkpoints

    Returns:
        trained actor and critic networks, and training statistics
    """
    # Initialize networks
    state_dim = env.grid_width * env.grid_length
    action_dim = env.action_dim

    actor = Actor(state_dim=state_dim, action_dim=action_dim)
    critic = Critic(state_dim=state_dim)

    optimizer_actor = optim.Adam(actor.parameters(), lr=lr_actor)
    optimizer_critic = optim.Adam(critic.parameters(), lr=lr_critic)

    # For tracking progress
    episode_rewards = []
    avg_rewards = deque(maxlen=100)
    actor_losses = []
    critic_losses = []
    success_rates = deque(maxlen=100)

    # Training loop
    for episode in range(num_episodes):
        # Get a new cargo manifest for this episode
        cargo_manifest = sample_cargo_manifest()
        state = env.reset()

        # Process each item in the manifest
        episode_reward = 0
        episode_transitions = []
        success_count = 0
        total_items = len(cargo_manifest)

        for item in cargo_manifest:
            env.current_item = item  # Set the current item to be placed

            # Get feasibility mask for valid actions
            feas_mask = env.get_feasibility_mask()

            # Check if there are any feasible actions
            if np.sum(feas_mask) == 0:
                # No feasible placement for this item
                continue

            # Convert state to tensor and flatten
            state_tensor = torch.FloatTensor(state.flatten()).unsqueeze(0)

            # Get action distribution from actor
            dist = actor(state_tensor, feas_mask)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            # Step the environment
            next_state, reward, done, info = env.step(action.item())
            episode_reward += reward

            # Store transition
            episode_transitions.append((
                state.flatten(),
                action,
                reward,
                next_state.flatten() if not done else None,
                log_prob,
                done
            ))

            # Update state
            state = next_state

            # Track successful placements
            if info.get("feasible", False):
                success_count += 1

            if done:
                break

        # Calculate returns and advantages
        returns = []
        G = 0
        for _, _, reward, _, _, _ in reversed(episode_transitions):
            G = reward + gamma * G
            returns.insert(0, G)

        returns = torch.FloatTensor(returns)

        # Collect states for batch processing
        states = torch.FloatTensor(np.vstack([t[0] for t in episode_transitions]))
        actions = torch.cat([t[1] for t in episode_transitions])
        log_probs = torch.cat([t[4] for t in episode_transitions])

        # Get values from critic
        values = critic(states).squeeze()

        # Calculate advantages
        advantages = returns - values.detach()

        # Actor loss
        actor_loss = -(log_probs * advantages).mean()

        # Critic loss (MSE)
        critic_loss = nn.MSELoss()(values, returns)

        # Update actor
        optimizer_actor.zero_grad()
        actor_loss.backward()
        # Optional: clip gradients to prevent exploding gradients
        nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
        optimizer_actor.step()

        # Update critic
        optimizer_critic.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1.0)
        optimizer_critic.step()

        # Track metrics
        episode_rewards.append(episode_reward)
        avg_rewards.append(episode_reward)
        actor_losses.append(actor_loss.item())
        critic_losses.append(critic_loss.item())
        success_rate = success_count / total_items if total_items > 0 else 0
        success_rates.append(success_rate)

        # Print progress
        if (episode + 1) % print_interval == 0:
            avg_reward = np.mean(list(avg_rewards))
            avg_success = np.mean(list(success_rates))
            print(f"Episode {episode+1}/{num_episodes} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Success Rate: {avg_success:.2f} | "
                  f"Actor Loss: {actor_loss.item():.4f} | "
                  f"Critic Loss: {critic_loss.item():.4f}")

        # Save checkpoint
        if (episode + 1) % save_interval == 0:
            torch.save({
                'episode': episode,
                'actor_state_dict': actor.state_dict(),
                'critic_state_dict': critic.state_dict(),
                'optimizer_actor_state_dict': optimizer_actor.state_dict(),
                'optimizer_critic_state_dict': optimizer_critic.state_dict(),
                'episode_rewards': episode_rewards,
                'actor_losses': actor_losses,
                'critic_losses': critic_losses
            }, checkpoint_path)

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
    plt.savefig('training_curves.png')
    plt.show()

def evaluate_agent(env, actor, num_episodes=10):
    """
    Evaluate the trained agent's performance

    Args:
        env: The bin packing environment
        actor: The trained actor network
        num_episodes: Number of episodes to evaluate

    Returns:
        Average reward and success rate
    """
    episode_rewards = []
    success_rates = []

    for episode in range(num_episodes):
        cargo_manifest = sample_cargo_manifest(
            standard_containers=STANDARD_CONTAINERS,
            cargo_items=CARGO_ITEMS
        )
        state = env.reset()

        episode_reward = 0
        success_count = 0
        total_items = len(cargo_manifest)

        for item in cargo_manifest:
            env.current_item = item
            feas_mask = env.get_feasibility_mask()

            if np.sum(feas_mask) == 0:
                continue

            state_tensor = torch.FloatTensor(state.flatten()).unsqueeze(0)

            # For evaluation, we use the action with highest probability
            with torch.no_grad():
                dist = actor(state_tensor, feas_mask)
                action = dist.probs.argmax().item()

            next_state, reward, done, info = env.step(action)
            episode_reward += reward

            if info.get("feasible", False):
                success_count += 1

            state = next_state

            if done:
                break

        success_rate = success_count / total_items if total_items > 0 else 0
        episode_rewards.append(episode_reward)
        success_rates.append(success_rate)

    avg_reward = np.mean(episode_rewards)
    avg_success_rate = np.mean(success_rates)

    print(f"Evaluation over {num_episodes} episodes:")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Average Success Rate: {avg_success_rate:.2f}")

    return avg_reward, avg_success_rate

# Example usage:
if __name__ == "__main__":
    # Create environment
    env = DRLBinPackingEnv(container_dims=(10, 10, 10))

    # Train agent
    actor, critic, stats = train_agent(
        env=env,
        num_episodes=1000,
        gamma=0.99,
        lr_actor=1e-4,
        lr_critic=1e-3,
        print_interval=10,
        save_interval=100
    )

    # Evaluate trained agent
    evaluate_agent(env, actor, num_episodes=10)
