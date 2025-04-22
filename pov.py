import numpy as np
import pandas as pd
from collections import defaultdict
import random
import time
from tqdm import tqdm


class CuckooHashTable:
    """Implementation of a collisionless embedding table using Cuckoo Hashing."""
    
    def __init__(self, embedding_dim=8, max_attempts=100):
        self.table0 = {}  # First hash table
        self.table1 = {}  # Second hash table
        self.embedding_dim = embedding_dim
        self.max_attempts = max_attempts
        self.touched_keys = set()  # For tracking updates for parameter sync
        self.last_access_time = {}  # For expiration mechanism
    
    def _hash0(self, key):
        """First hash function."""
        return hash(key) % (10**9 + 7)
    
    def _hash1(self, key):
        """Second hash function."""
        return hash(key) % (10**9 + 9)
    
    def insert(self, key, embedding=None):
        """Insert a key with an embedding."""
        if key in self.table0 or key in self.table1:
            return True  # Key already exists
            
        if embedding is None:
            embedding = np.random.normal(0, 0.1, self.embedding_dim)
            
        # Try to insert in table0
        return self._insert_with_cuckoo(key, embedding, 0, 0)
    
    def _insert_with_cuckoo(self, key, embedding, table_idx, attempt):
        """Implementation of cuckoo hashing insertion algorithm."""
        if attempt >= self.max_attempts:
            # If too many attempts, need to rehash (simplified by just failing)
            print(f"Failed to insert key {key} after {attempt} attempts")
            return False
            
        if table_idx == 0:
            hash_key = self._hash0(key)
            if hash_key not in self.table0:
                self.table0[hash_key] = (key, embedding)
                self.last_access_time[key] = time.time()
                return True
            else:
                # Collision in table0, evict and try to insert in table1
                evicted_key, evicted_emb = self.table0[hash_key]
                self.table0[hash_key] = (key, embedding)
                self.last_access_time[key] = time.time()
                return self._insert_with_cuckoo(evicted_key, evicted_emb, 1, attempt + 1)
        else:
            hash_key = self._hash1(key)
            if hash_key not in self.table1:
                self.table1[hash_key] = (key, embedding)
                self.last_access_time[key] = time.time()
                return True
            else:
                # Collision in table1, evict and try to insert in table0
                evicted_key, evicted_emb = self.table1[hash_key]
                self.table1[hash_key] = (key, embedding)
                self.last_access_time[key] = time.time()
                return self._insert_with_cuckoo(evicted_key, evicted_emb, 0, attempt + 1)
    
    def lookup(self, key):
        """Lookup a key in the hash table and return its embedding."""
        hash0 = self._hash0(key)
        if hash0 in self.table0 and self.table0[hash0][0] == key:
            self.last_access_time[key] = time.time()
            return self.table0[hash0][1]
            
        hash1 = self._hash1(key)
        if hash1 in self.table1 and self.table1[hash1][0] == key:
            self.last_access_time[key] = time.time()
            return self.table1[hash1][1]
            
        # Key not found, create new embedding
        embedding = np.random.normal(0, 0.1, self.embedding_dim)
        self.insert(key, embedding)
        return embedding
    
    def update(self, key, gradient, learning_rate=0.01):
        """Update the embedding of a key with a gradient."""
        hash0 = self._hash0(key)
        if hash0 in self.table0 and self.table0[hash0][0] == key:
            k, emb = self.table0[hash0]
            updated_emb = emb - learning_rate * gradient
            self.table0[hash0] = (k, updated_emb)
            self.touched_keys.add(key)
            self.last_access_time[key] = time.time()
            return
            
        hash1 = self._hash1(key)
        if hash1 in self.table1 and self.table1[hash1][0] == key:
            k, emb = self.table1[hash1]
            updated_emb = emb - learning_rate * gradient
            self.table1[hash1] = (k, updated_emb)
            self.touched_keys.add(key)
            self.last_access_time[key] = time.time()
            return
        
        # Key not found, insert a new embedding and update it
        embedding = np.random.normal(0, 0.1, self.embedding_dim)
        self.insert(key, embedding)
        self.update(key, gradient, learning_rate)
    
    def get_updated_embeddings(self):
        """Get embeddings that were updated since last sync."""
        updated_embeddings = {}
        
        for hash_key, (key, emb) in self.table0.items():
            if key in self.touched_keys:
                updated_embeddings[key] = emb
                
        for hash_key, (key, emb) in self.table1.items():
            if key in self.touched_keys:
                updated_embeddings[key] = emb
        
        # Reset touched keys after getting updates
        self.touched_keys = set()
        return updated_embeddings
    
    def expire_old_keys(self, max_age_seconds=3600):
        """Remove embeddings that haven't been accessed recently."""
        current_time = time.time()
        keys_to_remove = []
        
        for key, last_time in self.last_access_time.items():
            if current_time - last_time > max_age_seconds:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            hash0 = self._hash0(key)
            if hash0 in self.table0 and self.table0[hash0][0] == key:
                del self.table0[hash0]
                
            hash1 = self._hash1(key)
            if hash1 in self.table1 and self.table1[hash1][0] == key:
                del self.table1[hash1]
            
            del self.last_access_time[key]
            
        return len(keys_to_remove)


class SimpleHashTable:
    """Implementation of a hash table with collisions."""
    
    def __init__(self, embedding_dim=8, table_size=1000):
        self.table_size = table_size
        self.embedding_dim = embedding_dim
        self.embeddings = {}
    
    def _hash(self, key):
        """Hash function that will create collisions."""
        return hash(key) % self.table_size
    
    def lookup(self, key):
        """Lookup a key in the hash table, may have collisions."""
        hash_key = self._hash(key)
        
        if hash_key not in self.embeddings:
            # Create a new embedding for this hash bucket
            self.embeddings[hash_key] = np.random.normal(0, 0.1, self.embedding_dim)
            
        return self.embeddings[hash_key]
    
    def update(self, key, gradient, learning_rate=0.01):
        """Update the embedding of a hash bucket."""
        hash_key = self._hash(key)
        
        if hash_key not in self.embeddings:
            self.embeddings[hash_key] = np.random.normal(0, 0.1, self.embedding_dim)
            
        self.embeddings[hash_key] -= learning_rate * gradient


class DeepFM:
    """Simplified DeepFM model for recommendation."""
    
    def __init__(self, embedding_dim=8, user_hash_table=None, item_hash_table=None):
        self.embedding_dim = embedding_dim
        self.user_hash_table = user_hash_table or CuckooHashTable(embedding_dim=embedding_dim)
        self.item_hash_table = item_hash_table or CuckooHashTable(embedding_dim=embedding_dim)
        
        # Dense parameters
        self.W = np.random.normal(0, 0.1, (embedding_dim * 2, 1))
        self.b = np.random.normal(0, 0.1, 1)
    
    def forward(self, user_id, item_id):
        """Forward pass of the model."""
        # Get embeddings
        user_emb = self.user_hash_table.lookup(user_id)
        item_emb = self.item_hash_table.lookup(item_id)
        
        # Concatenate embeddings
        concat_emb = np.concatenate([user_emb, item_emb])
        
        # FM part (simplified)
        fm_term = np.dot(user_emb, item_emb)
        
        # DNN part (simplified)
        dnn_term = np.dot(concat_emb, self.W) + self.b
        
        # Final prediction
        logit = fm_term + dnn_term
        pred = 1 / (1 + np.exp(-logit))
        
        return pred.item(), (user_emb, item_emb, concat_emb)
    
    def backward(self, user_id, item_id, label, pred, embeddings, learning_rate=0.01):
        """Backward pass of the model."""
        user_emb, item_emb, concat_emb = embeddings
        
        # Compute gradient
        dloss = pred - label
        
        # Update dense parameters
        dW = dloss * concat_emb.reshape(-1, 1)
        db = dloss
        
        self.W -= learning_rate * dW
        self.b -= learning_rate * db
        
        # Compute gradients for embeddings
        d_user_emb = dloss * item_emb + dloss * self.W[:self.embedding_dim].flatten()
        d_item_emb = dloss * user_emb + dloss * self.W[self.embedding_dim:].flatten()
        
        # Update embeddings
        self.user_hash_table.update(user_id, d_user_emb, learning_rate)
        self.item_hash_table.update(item_id, d_item_emb, learning_rate)


class ModelServer:
    """Server that handles prediction requests."""
    
    def __init__(self, model):
        self.model = model
    
    def predict(self, user_id, item_id):
        """Predict the probability of a user interacting with an item."""
        pred, _ = self.model.forward(user_id, item_id)
        return pred
    
    def update_model(self, new_model_params):
        """Update the model parameters."""
        # Update user embeddings
        if 'user_embeddings' in new_model_params:
            for key, emb in new_model_params['user_embeddings'].items():
                self.model.user_hash_table.insert(key, emb)
                
        # Update item embeddings
        if 'item_embeddings' in new_model_params:
            for key, emb in new_model_params['item_embeddings'].items():
                self.model.item_hash_table.insert(key, emb)
                
        # Update dense parameters (less frequently)
        if 'W' in new_model_params:
            self.model.W = new_model_params['W']
            
        if 'b' in new_model_params:
            self.model.b = new_model_params['b']


class TrainingWorker:
    """Worker that handles training."""
    
    def __init__(self, model):
        self.model = model
        
    def train_batch(self, batch_data, learning_rate=0.01):
        """Train on a batch of data."""
        losses = []
        
        for user_id, item_id, label in batch_data:
            # Forward pass
            pred, embeddings = self.model.forward(user_id, item_id)
            
            # Compute loss
            loss = -label * np.log(pred) - (1 - label) * np.log(1 - pred)
            losses.append(loss)
            
            # Backward pass
            self.model.backward(user_id, item_id, label, pred, embeddings, learning_rate)
        
        return np.mean(losses)
    
    def get_params_for_sync(self):
        """Get parameters to sync to serving model."""
        params = {}
        
        # Get updated user embeddings
        params['user_embeddings'] = self.model.user_hash_table.get_updated_embeddings()
        
        # Get updated item embeddings
        params['item_embeddings'] = self.model.item_hash_table.get_updated_embeddings()
        
        # Add dense parameters (less frequently)
        # In a real system, we would do this on a different schedule
        params['W'] = self.model.W
        params['b'] = self.model.b
        
        return params


def generate_synthetic_data(n_users=100, n_items=1000, n_samples=10000):
    """Generate synthetic recommendation data."""
    users = [f"user_{i}" for i in range(n_users)]
    items = [f"item_{i}" for i in range(n_items)]
    
    # Assign random preferences to users
    user_prefs = {}
    for user in users:
        user_prefs[user] = np.random.normal(0, 1, 5)  # 5 latent factors
        
    # Assign random attributes to items
    item_attrs = {}
    for item in items:
        item_attrs[item] = np.random.normal(0, 1, 5)  # 5 latent factors
    
    # Generate interactions
    data = []
    for _ in range(n_samples):
        user = random.choice(users)
        item = random.choice(items)
        
        # Compute probability of interaction based on dot product
        p = 1 / (1 + np.exp(-np.dot(user_prefs[user], item_attrs[item])))
        label = np.random.binomial(1, p)
        
        data.append((user, item, label))
    
    return data


def run_experiment_batch_vs_online_training():
    """Compare batch training with online training."""
    # Generate data
    print("Generating synthetic data...")
    batch_data = generate_synthetic_data(n_users=100, n_items=1000, n_samples=5000)
    online_data = generate_synthetic_data(n_users=100, n_items=1000, n_samples=5000)
    test_data = generate_synthetic_data(n_users=100, n_items=1000, n_samples=1000)
    
    # Create models
    batch_model = DeepFM(embedding_dim=8)
    online_model = DeepFM(embedding_dim=8)
    
    # Create training workers
    batch_worker = TrainingWorker(batch_model)
    online_worker = TrainingWorker(online_model)
    
    # Create servers
    batch_server = ModelServer(DeepFM(embedding_dim=8))
    online_server = ModelServer(DeepFM(embedding_dim=8))
    
    # Batch training
    print("Performing batch training...")
    batch_size = 64
    n_batches = len(batch_data) // batch_size
    
    for i in tqdm(range(n_batches)):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        mini_batch = batch_data[start_idx:end_idx]
        batch_worker.train_batch(mini_batch)
    
    # Sync batch model to server
    batch_server.update_model(batch_worker.get_params_for_sync())
    
    # Evaluate batch model
    batch_preds = []
    batch_labels = []
    
    for user_id, item_id, label in test_data:
        pred = batch_server.predict(user_id, item_id)
        batch_preds.append(pred)
        batch_labels.append(label)
    
    batch_auc = compute_auc(batch_labels, batch_preds)
    print(f"Batch training AUC: {batch_auc:.4f}")
    
    # Online training with frequent syncing
    print("Performing online training with frequent syncing...")
    sync_interval = 100  # Sync every 100 examples
    
    for i in tqdm(range(0, len(online_data), sync_interval)):
        # Train on a batch of online data
        online_batch = online_data[i:i+sync_interval]
        online_worker.train_batch(online_batch)
        
        # Sync parameters
        online_server.update_model(online_worker.get_params_for_sync())
        
        # Expire old embeddings (in a real system, this would be on a different schedule)
        if i % 1000 == 0:
            n_expired = online_model.user_hash_table.expire_old_keys()
            print(f"Expired {n_expired} old user embeddings")
    
    # Evaluate online model
    online_preds = []
    online_labels = []
    
    for user_id, item_id, label in test_data:
        pred = online_server.predict(user_id, item_id)
        online_preds.append(pred)
        online_labels.append(label)
    
    online_auc = compute_auc(online_labels, online_preds)
    print(f"Online training AUC: {online_auc:.4f}")
    
    return batch_auc, online_auc


def run_experiment_collision_vs_collisionless():
    """Compare hash table with collisions vs collisionless hash table."""
    # Generate data
    print("Generating synthetic data...")
    train_data = generate_synthetic_data(n_users=100, n_items=1000, n_samples=5000)
    test_data = generate_synthetic_data(n_users=100, n_items=1000, n_samples=1000)
    
    # Create models
    collision_hash_table = SimpleHashTable(embedding_dim=8, table_size=50)  # Small table to force collisions
    collision_model = DeepFM(
        embedding_dim=8,
        user_hash_table=collision_hash_table,
        item_hash_table=SimpleHashTable(embedding_dim=8, table_size=200)  # More collisions for users than items
    )
    
    collisionless_model = DeepFM(embedding_dim=8)  # Uses CuckooHashTable by default
    
    # Create training workers
    collision_worker = TrainingWorker(collision_model)
    collisionless_worker = TrainingWorker(collisionless_model)
    
    # Batch training for both models
    print("Training both models...")
    batch_size = 64
    n_batches = len(train_data) // batch_size
    
    for i in tqdm(range(n_batches)):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        mini_batch = train_data[start_idx:end_idx]
        
        # Train both models
        collision_worker.train_batch(mini_batch)
        collisionless_worker.train_batch(mini_batch)
    
    # Evaluate collision model
    collision_preds = []
    collision_labels = []
    
    for user_id, item_id, label in test_data:
        pred, _ = collision_model.forward(user_id, item_id)
        collision_preds.append(pred)
        collision_labels.append(label)
    
    collision_auc = compute_auc(collision_labels, collision_preds)
    print(f"Model with hash collisions AUC: {collision_auc:.4f}")
    
    # Evaluate collisionless model
    collisionless_preds = []
    collisionless_labels = []
    
    for user_id, item_id, label in test_data:
        pred, _ = collisionless_model.forward(user_id, item_id)
        collisionless_preds.append(pred)
        collisionless_labels.append(label)
    
    collisionless_auc = compute_auc(collisionless_labels, collisionless_preds)
    print(f"Collisionless model AUC: {collisionless_auc:.4f}")
    
    return collision_auc, collisionless_auc


def compute_auc(labels, preds):
    """Compute Area Under the ROC Curve."""
    # Sort predictions and corresponding labels
    sorted_indices = np.argsort(preds)[::-1]
    sorted_labels = np.array(labels)[sorted_indices]
    
    # Count positive and negative examples
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    
    if n_pos == 0 or n_neg == 0:
        return 0.5  # Random guess
    
    # Compute AUC
    pos_rank_sum = 0
    for i, label in enumerate(sorted_labels):
        if label == 1:
            pos_rank_sum += i + 1
    
    auc = (pos_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return auc


if __name__ == "__main__":
    print("==== Monolith Recommendation System - Minimal POC ====")
    print("\n1. Testing collision vs collisionless hash tables")
    collision_auc, collisionless_auc = run_experiment_collision_vs_collisionless()
    
    print("\n2. Testing batch vs online training")
    batch_auc, online_auc = run_experiment_batch_vs_online_training()
    
    print("\n==== Summary of Results ====")
    print(f"Hash collision vs collisionless: {collision_auc:.4f} vs {collisionless_auc:.4f}")
    print(f"Batch vs online training: {batch_auc:.4f} vs {online_auc:.4f}")