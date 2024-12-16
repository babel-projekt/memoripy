from loguru import logger
import numpy as np
import time
import uuid
import os
import torch
import json
import redis
from typing import Tuple, List, Dict, Any
from model import HumanMessage, SystemMessage
from sentence_transformers import SentenceTransformer

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'


class RedisStorage:
    """
    Redis-based storage implementation for memory management
    """
    def __init__(self, host='localhost', port=6379, db=0):
        self.redis_client = redis.Redis(
            host=host,
            port=port,
            db=db,
            decode_responses=True
        )

    def _get_user_session_keys(self, user_id: str, session_id: str) -> tuple[str, str]:
        """Generate Redis keys for a specific user's session"""
        return f"memory:{user_id}:{session_id}:short_term", f"memory:{user_id}:{session_id}:long_term"

    def save_memory_to_history(self, memory_store, user_id: str, session_id: str):
        """Save current memory state to Redis for a specific user's session"""
        short_term_key, long_term_key = self._get_user_session_keys(user_id, session_id)
        
        short_term = []
        for interaction in memory_store.short_term_memory:
            interaction_copy = interaction.copy()
            if isinstance(interaction_copy['embedding'], np.ndarray):
                interaction_copy['embedding'] = interaction_copy['embedding'].tolist()
            short_term.append(interaction_copy)

        self.redis_client.set(
            short_term_key,
            json.dumps(short_term)
        )
        # Set 60-minute TTL (3600 seconds)
        self.redis_client.expire(short_term_key, 3600)

    def load_history(self, user_id: str, session_id: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Load memory history from Redis for a specific user's session"""
        short_term_key, _ = self._get_user_session_keys(user_id, session_id)
        short_term_data = self.redis_client.get(short_term_key)
        short_term = json.loads(short_term_data) if short_term_data else []
        return short_term, []

    def list_user_sessions(self, user_id: str) -> List[str]:
        """List all sessions for a specific user"""
        pattern = f"memory:{user_id}:*:short_term"
        keys = self.redis_client.keys(pattern)
        return [key.split(":")[2] for key in keys]

    def delete_user_session(self, user_id: str, session_id: str):
        """Delete a specific session for a user"""
        short_term_key, long_term_key = self._get_user_session_keys(user_id, session_id)
        self.redis_client.delete(short_term_key, long_term_key)

    def delete_user_data(self, user_id: str):
        """Delete all data for a specific user"""
        pattern = f"memory:{user_id}:*"
        keys = self.redis_client.keys(pattern)
        if keys:
            self.redis_client.delete(*keys)

class MemoryStore:
    """Memory store implementation with clustering capabilities"""
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.short_term_memory = []
        self.long_term_memory = []
        
    def add_interaction(self, interaction: Dict[str, Any]):
        self.short_term_memory.append(interaction)
        
    def cluster_interactions(self):
        # Implement clustering logic if needed
        pass
        
    def retrieve(self, query_embedding: np.ndarray, query_concepts: List[str], 
                similarity_threshold: float, exclude_last_n: int = 0) -> List[Dict[str, Any]]:
        results = []
        
        # Skip the last n interactions if specified
        search_memory = self.short_term_memory[:-exclude_last_n] if exclude_last_n > 0 else self.short_term_memory
        
        for interaction in search_memory:
            # Calculate embedding similarity
            memory_embedding = np.array(interaction['embedding'])
            similarity = np.dot(query_embedding, memory_embedding.T)[0][0]
            
            # Calculate concept overlap
            memory_concepts = set(interaction['concepts'])
            concept_overlap = len(set(query_concepts) & memory_concepts)
            
            if similarity * 100 >= similarity_threshold or concept_overlap > 0:
                results.append({
                    **interaction,
                    'similarity_score': similarity * 100,
                    'concept_overlap': concept_overlap
                })
        
        # Sort by similarity score and concept overlap
        results.sort(key=lambda x: (x['concept_overlap'], x['similarity_score']), reverse=True)
        return results

class MemoryManager:
    """
    Manages the memory store with user and session support
    """
    def __init__(self, chat_model, redis_host='localhost', redis_port=6379, redis_db=0):
        device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        self.embedding_model = SentenceTransformer(
            'sentence-transformers/all-mpnet-base-v2',
            trust_remote_code=True,
            device=device
        )
        self.chat_model = chat_model
        self.device = device
        self.dimension = 768
        self.memory_store = MemoryStore(dimension=self.dimension)
        self.storage = RedisStorage(
            host=redis_host,
            port=redis_port,
            db=redis_db
        )

    def __del__(self):
        """Cleanup when the MemoryManager is destroyed"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def standardize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """Standardize embedding to the target dimension"""
        current_dim = len(embedding)
        if current_dim == self.dimension:
            return embedding
        elif current_dim < self.dimension:
            return np.pad(embedding, (0, self.dimension - current_dim), 'constant')
        else:
            return embedding[:self.dimension]

    def load_history(self, user_id: str, session_id: str):
        """Load history for a specific user's session"""
        short_term, _ = self.storage.load_history(user_id, session_id)
        
        # Convert loaded embeddings back to numpy arrays
        for interaction in short_term:
            if 'embedding' in interaction:
                interaction['embedding'] = np.array(interaction['embedding'])
        
        return short_term, []

    def add_interaction(self, prompt: str, output: str, embedding: np.ndarray, 
                       concepts: list[str], user_id: str, session_id: str):
        """Add an interaction to a specific user's session with 60-minute TTL"""
        timestamp = time.time()
        interaction_id = str(uuid.uuid4())
        interaction = {
            "id": interaction_id,
            "user_id": user_id,
            "session_id": session_id,
            "prompt": prompt,
            "output": output,
            "embedding": embedding.tolist(),
            "timestamp": timestamp,
            "access_count": 1,
            "concepts": list(concepts),
            "decay_factor": 1.0,
        }

        # Get existing interactions for this user's session
        short_term_key, _ = self.storage._get_user_session_keys(user_id, session_id)
        existing_data = self.storage.redis_client.get(short_term_key)
        existing_interactions = json.loads(existing_data) if existing_data else []
        
        # Add new interaction
        existing_interactions.append(interaction)
        
        # Save to Redis with 60-minute TTL
        self.storage.redis_client.set(
            short_term_key,
            json.dumps(existing_interactions)
        )
        self.storage.redis_client.expire(short_term_key, 3600)

    def get_embedding(self, text: str, max_tokens: int = 768, stride: int = 192) -> np.ndarray:
        """Get embedding using sliding window approach"""
        print(f"Generating embedding for the provided text...")
        try:
            words = text.split()
            embeddings = []
            
            if len(words) > max_tokens:
                for i in range(0, len(words) - max_tokens + 1, stride):
                    window = ' '.join(words[i:i + max_tokens])
                    print(f"Processing window {len(embeddings)+1}: tokens {i} to {i + max_tokens}")
                    
                    with torch.no_grad():
                        window_embedding = self.embedding_model.encode(
                            window,
                            convert_to_numpy=True,
                            show_progress_bar=False,
                            batch_size=1
                        )
                    embeddings.append(window_embedding)
                    
                    if self.device in ["cuda", "mps"]:
                        torch.cuda.empty_cache() if self.device == "cuda" else torch.mps.empty_cache()
                
                embedding = np.mean(embeddings, axis=0)
            else:
                with torch.no_grad():
                    embedding = self.embedding_model.encode(
                        text,
                        convert_to_numpy=True,
                        show_progress_bar=False,
                        batch_size=1
                    )
            
            if embedding is None:
                raise ValueError("Failed to generate embedding.")
                
            standardized_embedding = self.standardize_embedding(embedding)
            return np.array(standardized_embedding).reshape(1, -1)
            
        except Exception as e:
            print(f"Error in get_embedding: {e}")
            raise
        finally:
            if self.device in ["cuda", "mps"]:
                torch.cuda.empty_cache() if self.device == "cuda" else torch.mps.empty_cache()

    def extract_concepts(self, text: str) -> list[str]:
        """Extract key concepts from text"""
        print("Extracting key concepts from the provided text...")
        return self.chat_model.extract_concepts(text)

    def initialize_memory(self):
        """No longer needed in session-based implementation"""
        pass

    def retrieve_relevant_interactions(self, query: str, user_id: str, session_id: str,
                                    similarity_threshold=40, exclude_last_n=0) -> list:
        """Retrieve relevant interactions for a specific user's session"""
        query_embedding = self.get_embedding(query)
        query_concepts = self.extract_concepts(query)
        
        # Load session-specific interactions
        session_interactions, _ = self.load_history(user_id, session_id)
        results = []
        
        search_memory = session_interactions[:-exclude_last_n] if exclude_last_n > 0 else session_interactions
        
        for interaction in search_memory:
            memory_embedding = np.array(interaction['embedding'])
            similarity = np.dot(query_embedding, memory_embedding.T)[0][0]
            
            memory_concepts = set(interaction['concepts'])
            concept_overlap = len(set(query_concepts) & memory_concepts)
            
            if similarity * 100 >= similarity_threshold or concept_overlap > 0:
                results.append({
                    **interaction,
                    'similarity_score': similarity * 100,
                    'concept_overlap': concept_overlap
                })
        
        results.sort(key=lambda x: (x['concept_overlap'], x['similarity_score']), reverse=True)
        return results

    def generate_response(self, prompt: str, last_interactions: list, retrievals: list, context_window=3) -> str:
        """Generate response with user's session-specific context"""
        context = ""
        if last_interactions:
            context_interactions = last_interactions[-context_window:]
            context += "\n".join([f"Previous prompt: {r['prompt']}\nPrevious output: {r['output']}" 
                                for r in context_interactions])
            print(f"Using the following last interactions as context for response generation:\n{context}")
        else:
            context = "No previous interactions available."
            print(context)

        if retrievals:
            retrieved_context_interactions = retrievals[:context_window]
            retrieved_context = "\n".join([f"Relevant prompt: {r['prompt']}\nRelevant output: {r['output']}" 
                                         for r in retrieved_context_interactions])
            print(f"Using the following retrieved interactions as context for response generation:\n{retrieved_context}")
            context += "\n" + retrieved_context

        messages = [
            SystemMessage(content="You're a helpful assistant."),
            HumanMessage(content=f"{context}\nCurrent prompt: {prompt}")
        ]

        logger.debug(f"Messages: {messages}")
        return self.chat_model.invoke(messages)

    # Utility methods for user/session management
    def list_user_sessions(self, user_id: str) -> List[str]:
        """List all sessions for a user"""
        return self.storage.list_user_sessions(user_id)

    def delete_user_session(self, user_id: str, session_id: str):
        """Delete a specific session for a user"""
        self.storage.delete_user_session(user_id, session_id)

    def delete_user_data(self, user_id: str):
        """Delete all data for a user"""
        self.storage.delete_user_data(user_id)