# monolith-recomm-system


### Why

You want to build simple AI web applications using LLMs and other AI models for TikTok-like use cases, particularly focusing on real-time recommendation systems with collisionless embedding tables as described in the Monolith paper.

### What

**Concepts to Learn:**

- Collisionless embedding tables (using Cuckoo Hashing)
- Real-time recommendation systems
- Online training vs batch training
- Parameter synchronization between training and serving models

**Facts to Learn:**

- Embedding tables with hash collisions negatively affect model quality
- Sparse features require special handling in recommendation systems
- User behavior changes rapidly (concept drift), requiring real-time model updates
- Dense parameters move slower than sparse embeddings

**Procedures to Learn:**

- Implementing a collisionless hash table
- Setting up online training with parameter synchronization
- Filtering feature IDs by frequency
- Implementing periodic parameter updates

### How

**Benchmarking:**
For a minimal viable proof of concept, we'll create a simplified version of the Monolith system focusing on its key innovations:

**Emphasis/Exclude:**
This minimal POC implements the core innovations from the Monolith paper while simplifying several aspects to make it more approachable:

1. **Implemented key concepts:**
    - Collisionless embedding table using Cuckoo Hashing
    - Comparison between collisionless and collision-based embedding tables
    - Online training with parameter synchronization
    - Expirable embeddings for memory efficiency
2. **Simplified aspects:**
    - Used a simplified DeepFM model instead of the full multi-tower architecture
    - Generated synthetic data instead of using real user logs
    - Implemented a basic synchronization mechanism rather than the more complex Kafka-based streaming pipeline
    - Simplified the negative sampling and parameter server architecture

## Running the POC

To run this POC, you'll need Python with NumPy, Pandas, and tqdm. The code demonstrates two key experiments:

1. **Collision vs. Collisionless Embedding Tables**: This experiment compares the performance of a model using hash tables with collisions against one using collisionless hash tables, showing how collisions can negatively impact model quality.
2. **Batch vs. Online Training**: This experiment compares traditional batch training with online training that includes frequent parameter synchronization, demonstrating how online training can improve model performance for time-sensitive applications.

## Next Steps

After testing this minimal POC, you could:

1. Scale it up to handle larger datasets
2. Implement a more sophisticated online joiner for real user data
3. Add the Kafka-based streaming architecture for true real-time updates
4. Integrate it with a web application for live recommendations

This POC provides a foundation for understanding how real-time recommendation systems like TikTok's work, particularly focusing on the collisionless embedding tables and online training aspects that make them effective at handling rapidly changing user preferences.
