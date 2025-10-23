"""
Enhanced Pinecone Search Manager with Pure Vector and Hybrid Search Support
"""

import logging
from typing import List, Dict, Any, Optional
from pinecone import Pinecone
from config import Config

logger = logging.getLogger(__name__)

class EnhancedPineconeSearchManager:
    """Enhanced Pinecone search manager supporting both pure vector and hybrid search."""
    
    def __init__(self, api_key: str = None, index_name: str = None):
        self.pc = Pinecone(api_key=api_key or Config.PINECONE_API_KEY)
        self.index_name = index_name or Config.PINECONE_INDEX_NAME
        self.index = None
    
    def connect_to_index(self):
        """Connect to existing Pinecone index."""
        try:
            if self.index_name in self.pc.list_indexes().names():
                self.index = self.pc.Index(self.index_name)
                logger.info(f"Connected to Pinecone index: {self.index_name}")
                return True
            else:
                logger.error(f"Pinecone index '{self.index_name}' not found")
                return False
        except Exception as e:
            logger.error(f"Error connecting to Pinecone index: {e}")
            return False
    
    def pure_vector_search(self, query_embedding: List[float], top_k: int = None) -> List[Dict[str, Any]]:
        """
        Pure vector search - no metadata filtering.
        Best for semantic similarity without constraints.
        """
        if not self.index:
            raise Exception("Pinecone index not connected")
        
        try:
            top_k = top_k or Config.TOP_K_RESULTS
            
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            
            matches = []
            for match in results.matches:
                matches.append({
                    'id': match.id,
                    'score': match.score,
                    'metadata': match.metadata
                })
            
            logger.info(f"Pure vector search found {len(matches)} matches")
            return matches
            
        except Exception as e:
            logger.error(f"Error in pure vector search: {e}")
            raise
    
    def hybrid_search(self, query_embedding: List[float], metadata_filter: Dict[str, Any] = None, 
                     top_k: int = None) -> List[Dict[str, Any]]:
        """
        Hybrid search - vector similarity + metadata filtering.
        Uses Pinecone's native filtering capabilities.
        """
        if not self.index:
            raise Exception("Pinecone index not connected")
        
        try:
            top_k = top_k or Config.TOP_K_RESULTS
            
            # Build Pinecone filter expression
            filter_expression = self._build_filter_expression(metadata_filter)
            
            query_params = {
                'vector': query_embedding,
                'top_k': top_k,
                'include_metadata': True
            }
            
            # Add filter if provided
            if filter_expression:
                query_params['filter'] = filter_expression
            
            results = self.index.query(**query_params)
            
            matches = []
            for match in results.matches:
                matches.append({
                    'id': match.id,
                    'score': match.score,
                    'metadata': match.metadata
                })
            
            logger.info(f"Hybrid search found {len(matches)} matches with filter: {metadata_filter}")
            return matches
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            raise
    
    def _build_filter_expression(self, metadata_filter: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Build Pinecone filter expression from metadata filter.
        
        Pinecone supports various filter operators:
        - $eq: equals
        - $ne: not equals  
        - $in: in list
        - $nin: not in list
        - $gt: greater than
        - $gte: greater than or equal
        - $lt: less than
        - $lte: less than or equal
        """
        if not metadata_filter:
            return None
        
        filter_conditions = []
        
        for field, value in metadata_filter.items():
            if isinstance(value, list):
                # Multiple values - use $in operator
                filter_conditions.append({field: {"$in": value}})
            elif isinstance(value, str):
                # Single string value - use $eq operator
                filter_conditions.append({field: {"$eq": value}})
            elif isinstance(value, dict):
                # Complex filter - use as-is
                filter_conditions.append({field: value})
            else:
                # Other types - use $eq operator
                filter_conditions.append({field: {"$eq": value}})
        
        if len(filter_conditions) == 1:
            return filter_conditions[0]
        else:
            # Multiple conditions - combine with $and
            return {"$and": filter_conditions}
    
    def post_process_filter(self, matches: List[Dict[str, Any]], 
                          metadata_filter: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Post-process filtering for complex logic not supported by Pinecone.
        Use this when Pinecone's native filtering is insufficient.
        """
        if not metadata_filter:
            return matches
        
        filtered_matches = []
        
        for match in matches:
            metadata = match.get('metadata', {})
            passes_filter = True
            
            for field, expected_value in metadata_filter.items():
                actual_value = metadata.get(field)
                
                if isinstance(expected_value, list):
                    if actual_value not in expected_value:
                        passes_filter = False
                        break
                elif isinstance(expected_value, str):
                    if actual_value != expected_value:
                        passes_filter = False
                        break
                elif callable(expected_value):
                    # Custom filter function
                    if not expected_value(actual_value):
                        passes_filter = False
                        break
            
            if passes_filter:
                filtered_matches.append(match)
        
        logger.info(f"Post-process filtering: {len(matches)} -> {len(filtered_matches)} matches")
        return filtered_matches
    
    def search_with_fallback(self, query_embedding: List[float], metadata_filter: Dict[str, Any] = None,
                           top_k: int = None, use_native_filter: bool = True) -> List[Dict[str, Any]]:
        """
        Search with fallback strategy:
        1. Try hybrid search with native filtering
        2. If no results, fallback to pure vector search
        3. Apply post-processing if needed
        """
        try:
            if use_native_filter and metadata_filter:
                # Try hybrid search first
                matches = self.hybrid_search(query_embedding, metadata_filter, top_k)
                
                if matches:
                    logger.info("Hybrid search successful")
                    return matches
                else:
                    logger.info("Hybrid search returned no results, trying pure vector search")
            
            # Fallback to pure vector search
            matches = self.pure_vector_search(query_embedding, top_k)
            
            # Apply post-processing if needed
            if metadata_filter and not use_native_filter:
                matches = self.post_process_filter(matches, metadata_filter)
            
            return matches
            
        except Exception as e:
            logger.error(f"Error in search with fallback: {e}")
            # Final fallback - return empty results
            return []
    
    def get_metadata_statistics(self, top_k: int = 1000) -> Dict[str, Any]:
        """
        Get statistics about metadata distribution in the index.
        Useful for understanding available filter values.
        """
        if not self.index:
            raise Exception("Pinecone index not connected")
        
        try:
            # Get sample of vectors to analyze metadata
            results = self.index.query(
                vector=[0.0] * Config.EMBEDDING_DIMENSION,  # Dummy vector
                top_k=min(top_k, 1000),
                include_metadata=True
            )
            
            stats = {
                'total_vectors_analyzed': len(results.matches),
                'metadata_fields': set(),
                'field_values': {}
            }
            
            for match in results.matches:
                metadata = match.metadata
                stats['metadata_fields'].update(metadata.keys())
                
                for field, value in metadata.items():
                    if field not in stats['field_values']:
                        stats['field_values'][field] = set()
                    stats['field_values'][field].add(str(value))
            
            # Convert sets to lists for JSON serialization
            stats['metadata_fields'] = list(stats['metadata_fields'])
            for field in stats['field_values']:
                stats['field_values'][field] = list(stats['field_values'][field])
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting metadata statistics: {e}")
            return {}

# Example usage and test functions
def test_search_methods():
    """Test different search methods."""
    
    search_manager = EnhancedPineconeSearchManager()
    
    if not search_manager.connect_to_index():
        print("❌ Failed to connect to Pinecone index")
        return
    
    # Sample query embedding (replace with actual embedding)
    sample_embedding = [0.1] * Config.EMBEDDING_DIMENSION
    
    print("🧪 Testing Search Methods")
    print("=" * 50)
    
    # Test 1: Pure vector search
    print("\n1. Pure Vector Search:")
    try:
        results = search_manager.pure_vector_search(sample_embedding, top_k=5)
        print(f"   Found {len(results)} results")
        if results:
            print(f"   Top score: {results[0]['score']:.3f}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test 2: Hybrid search with filter
    print("\n2. Hybrid Search with Filter:")
    try:
        filter_dict = {"regulator": "Reserve Bank of India"}
        results = search_manager.hybrid_search(sample_embedding, filter_dict, top_k=5)
        print(f"   Found {len(results)} results with filter: {filter_dict}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test 3: Search with fallback
    print("\n3. Search with Fallback:")
    try:
        filter_dict = {"industry": "Financial Services"}
        results = search_manager.search_with_fallback(sample_embedding, filter_dict, top_k=5)
        print(f"   Found {len(results)} results with fallback strategy")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test 4: Metadata statistics
    print("\n4. Metadata Statistics:")
    try:
        stats = search_manager.get_metadata_statistics(top_k=100)
        print(f"   Analyzed {stats.get('total_vectors_analyzed', 0)} vectors")
        print(f"   Available fields: {len(stats.get('metadata_fields', []))}")
        print(f"   Sample fields: {stats.get('metadata_fields', [])[:5]}")
    except Exception as e:
        print(f"   Error: {e}")

if __name__ == "__main__":
    test_search_methods()
