import numpy as np

def cosine_similarity(a, b):
    """
    Compute cosine similarity between two vectors.
    
    Parameters:
        a (numpy.ndarray): First vector
        b (numpy.ndarray): Second vector
        
    Returns:
        float: Cosine similarity between vectors a and b
    """
    # Calculate dot product
    dot_product = np.dot(a, b)
    
    # Calculate magnitudes
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    # Compute cosine similarity
    similarity = dot_product / (norm_a * norm_b)
    
    return similarity
