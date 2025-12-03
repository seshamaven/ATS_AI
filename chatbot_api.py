"""
Flask API for chatbot functionality with Pinecone similarity search and MySQL updates.
Handles user queries, semantic similarity matching, and database status updates using DataPipeline.
"""

import json
import logging
import re
import time
from typing import List, Dict, Any, Optional, Tuple
from flask import Flask, request, jsonify
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
from datetime import datetime
import numpy as np
from ollama_client import get_ollama_client, OllamaChatCompletions
from production_monitoring import (
    get_production_logger,
    get_production_monitor,
    log_query_processing
)
from token_tracker import (
    get_token_tracker,
    log_query_embedding_tokens,
    log_rag_tokens,
    log_chat_completion_usage
)
import tiktoken
from production_prompts import (
    ProductionRAGManager, 
    QueryRelevance, 
    RegulatoryDomain,
    classify_regulatory_query,
    build_regulatory_prompts
)
from config import Config
from datapipeline import create_data_pipeline
from enhanced_pinecone_search import EnhancedPineconeSearchManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Validate configuration
if not Config.validate_config():
    logger.error("Configuration validation failed. Please check your environment variables.")
    exit(1)

# Print configuration (hiding sensitive data)
Config.print_config(hide_sensitive=True)

# Initialize OpenAI/Ollama client based on configuration
def get_openai_client():
    """Get OpenAI/Ollama client based on configuration."""
    if Config.USE_OLLAMA:
        from ollama_client import get_ollama_openai_client
        logger.info("=" * 60)
        logger.info("LLM SERVICE: Using OLLAMA")
        logger.info(f"  Model: {Config.OLLAMA_MODEL}")
        logger.info(f"  Base URL: {Config.OLLAMA_BASE_URL}")
        logger.info("=" * 60)
        return get_ollama_openai_client(
            base_url=Config.OLLAMA_BASE_URL,
            model=Config.OLLAMA_MODEL
        )
    elif Config.AZURE_OPENAI_ENDPOINT:
        from openai import AzureOpenAI
        logger.info("=" * 60)
        logger.info("LLM SERVICE: Using AZURE OPENAI")
        logger.info(f"  Model: {Config.AZURE_OPENAI_MODEL}")
        logger.info(f"  Endpoint: {Config.AZURE_OPENAI_ENDPOINT}")
        logger.info("=" * 60)
        return AzureOpenAI(
            api_key=Config.AZURE_OPENAI_API_KEY,
            api_version=Config.AZURE_OPENAI_API_VERSION,
            azure_endpoint=Config.AZURE_OPENAI_ENDPOINT
        )
    else:
        from openai import OpenAI
        logger.info("=" * 60)
        logger.info("LLM SERVICE: Using OPENAI")
        logger.info(f"  Model: {Config.OPENAI_MODEL}")
        logger.info("=" * 60)
        return OpenAI(api_key=Config.OPENAI_API_KEY)

def get_openai_model():
    """Get model name based on configuration."""
    if Config.USE_OLLAMA:
        return Config.OLLAMA_MODEL
    elif Config.AZURE_OPENAI_ENDPOINT:
        return Config.AZURE_OPENAI_MODEL
    else:
        return Config.OPENAI_MODEL

# Initialize embeddings
class OllamaLangchainEmbeddings:
    """Langchain-compatible wrapper for Ollama embeddings."""
    
    def __init__(self, ollama_client):
        self.ollama_client = ollama_client
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text."""
        logger.debug("Generating embedding using OLLAMA")
        return self.ollama_client.embeddings(text)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents."""
        return [self.ollama_client.embeddings(text) for text in texts]

if Config.USE_OLLAMA:
    # Use Ollama for embeddings (will fallback to sentence-transformers if needed)
    ollama_client = get_ollama_client(
        base_url=Config.OLLAMA_BASE_URL,
        model=Config.OLLAMA_MODEL
    )
    embeddings = OllamaLangchainEmbeddings(ollama_client)
    logger.info("=" * 60)
    logger.info("EMBEDDINGS SERVICE: Using OLLAMA")
    logger.info(f"  Model: {Config.OLLAMA_MODEL}")
    logger.info(f"  Base URL: {Config.OLLAMA_BASE_URL}")
    logger.info("=" * 60)
else:
    import openai
    openai.api_key = Config.OPENAI_API_KEY
    embeddings = OpenAIEmbeddings(openai_api_key=Config.OPENAI_API_KEY)
    logger.info("=" * 60)
    logger.info("EMBEDDINGS SERVICE: Using OPENAI")
    logger.info(f"  Model: {Config.OPENAI_EMBEDDING_MODEL}")
    logger.info("=" * 60)

# Initialize Pinecone
pc = Pinecone(api_key=Config.PINECONE_API_KEY)


class TokenCounter:
    """Utility class for counting tokens in text."""
    
    def __init__(self):
        # Initialize token encoders for different models
        try:
            self.embedding_encoder = tiktoken.encoding_for_model("text-embedding-ada-002")
        except Exception:
            self.embedding_encoder = tiktoken.get_encoding("cl100k_base")
        
        try:
            self.chat_encoder = tiktoken.encoding_for_model(Config.OPENAI_MODEL)
        except Exception:
            self.chat_encoder = tiktoken.get_encoding("cl100k_base")
    
    def count_embedding_tokens(self, text: str) -> int:
        """Count tokens for embedding model."""
        try:
            return len(self.embedding_encoder.encode(text))
        except Exception:
            return len(text) // 4  # Fallback estimation
    
    def count_chat_tokens(self, text: str) -> int:
        """Count tokens for chat model."""
        try:
            return len(self.chat_encoder.encode(text))
        except Exception:
            return len(text) // 4  # Fallback estimation


# Global token counter
token_counter = TokenCounter()


class DatabaseManager:
    """Legacy database manager - delegates to DataPipeline."""
    
    def __init__(self, config: Dict[str, Any]):
        self.data_pipeline = create_data_pipeline()
    
    def connect(self) -> bool:
        """Establish MySQL connection."""
        return self.data_pipeline.connect()
    
    def disconnect(self):
        """Close MySQL connection."""
        self.data_pipeline.disconnect()
    
    def get_regulation_by_id(self, regulation_id: int) -> Optional[Dict[str, Any]]:
        """Fetch regulation by ID."""
        return self.data_pipeline.get_regulation_by_id(regulation_id)
    
    



class EnhancedSearchManager:
    """Enhanced search manager with reranking and intelligent filtering."""
    
    def __init__(self, api_key: str = None, index_name: str = None):
        self.pinecone_manager = EnhancedPineconeSearchManager(api_key, index_name)
        
        # Reranking weights for financial regulatory content
        self.rerank_weights = {
            'vector_similarity': 0.4,    # Semantic similarity
            'keyword_match': 0.3,        # Exact keyword matches
            'recency': 0.15,             # Document recency
            'authority_weight': 0.15     # Regulatory authority importance
        }
        
        # Regulatory authority weights (higher = more important)
        self.authority_weights = {
            'Reserve Bank of India': 1.0,
            'RBI': 1.0,
            'SEBI': 0.9,
            'IRDAI': 0.8,
            'Ministry of Finance': 0.7,
            'Government of India': 0.6
        }
    
    def connect_to_index(self):
        """Connect to Pinecone index."""
        return self.pinecone_manager.connect_to_index()
    
    def intelligent_query_analysis(self, query: str) -> Dict[str, Any]:
        """
        Comprehensive query analysis to extract regulatory context and build intelligent metadata filters.
        Optimized for financial regulatory circulars with extensive metadata consideration.
        """
        query_lower = query.lower()
        analysis = {
            'extracted_regulator': None,
            'extracted_industry': None,
            'extracted_sub_industry': None,
            'extracted_regulation_type': None,
            'extracted_task_category': None,
            'extracted_risk_category': None,
            'extracted_department': None,
            'extracted_status': None,
            'keywords': [],
            'regulatory_numbers': [],
            'suggested_filters': {},
            'metadata_fields_considered': []
        }
        
        # Extract regulatory body (comprehensive patterns)
        regulator_patterns = {
            'Reserve Bank of India': ['rbi', 'reserve bank', 'central bank', 'banking regulator'],
            'SEBI': ['sebi', 'securities and exchange board', 'securities regulator'],
            'IRDAI': ['irdai', 'insurance regulatory', 'insurance regulator'],
            'Ministry of Finance': ['ministry of finance', 'mof', 'finance ministry'],
            'Government of India': ['government of india', 'goi', 'central government'],
            'PFRDA': ['pfrda', 'pension fund regulatory'],
            'FMC': ['fmc', 'forward markets commission']
        }
        
        for regulator, patterns in regulator_patterns.items():
            if any(pattern in query_lower for pattern in patterns):
                analysis['extracted_regulator'] = regulator
                analysis['suggested_filters']['regulator'] = regulator
                analysis['metadata_fields_considered'].append('regulator')
                break
        
        # Extract industry context (expanded patterns)
        industry_patterns = {
            'Banking': ['banking', 'bank', 'loan', 'credit', 'lending', 'nbfc', 'microfinance'],
            'Capital Markets': ['securities', 'trading', 'equity', 'mutual fund', 'stock', 'derivatives', 'commodities'],
            'Insurance': ['insurance', 'life insurance', 'general insurance', 'policy', 'underwriting'],
            'Financial Services': ['financial services', 'fintech', 'payment', 'digital', 'wealth management'],
            'Pension': ['pension', 'retirement', 'provident fund', 'epf'],
            'Real Estate': ['real estate', 'reit', 'property', 'housing finance']
        }
        
        for industry, patterns in industry_patterns.items():
            if any(pattern in query_lower for pattern in patterns):
                analysis['extracted_industry'] = industry
                analysis['suggested_filters']['industry'] = industry
                analysis['metadata_fields_considered'].append('industry')
                break
        
        # Extract sub-industry context
        sub_industry_patterns = {
            'Commercial Banking': ['commercial bank', 'scheduled bank', 'public sector bank'],
            'Cooperative Banking': ['cooperative bank', 'urban cooperative', 'rural cooperative'],
            'NBFC': ['nbfc', 'non-banking financial', 'shadow banking'],
            'Payment Banks': ['payment bank', 'small finance bank'],
            'Mutual Funds': ['mutual fund', 'asset management', 'fund house'],
            'Stock Broking': ['stock broker', 'broking', 'trading member'],
            'Life Insurance': ['life insurance', 'term insurance', 'endowment'],
            'General Insurance': ['general insurance', 'motor insurance', 'health insurance']
        }
        
        for sub_industry, patterns in sub_industry_patterns.items():
            if any(pattern in query_lower for pattern in patterns):
                analysis['extracted_sub_industry'] = sub_industry
                analysis['suggested_filters']['sub_industry'] = sub_industry
                analysis['metadata_fields_considered'].append('sub_industry')
                break
        
        # Extract regulation type (comprehensive patterns)
        reg_type_patterns = {
            'Circular': ['circular', 'circ', 'master circular'],
            'Guidelines': ['guidelines', 'guidance', 'framework'],
            'Notification': ['notification', 'notif', 'public notice'],
            'Master Direction': ['master direction', 'md', 'direction'],
            'Regulation': ['regulation', 'reg', 'rules'],
            'Act': ['act', 'amendment', 'bill'],
            'Order': ['order', 'directive', 'instruction']
        }
        
        for reg_type, patterns in reg_type_patterns.items():
            if any(pattern in query_lower for pattern in patterns):
                analysis['extracted_regulation_type'] = reg_type
                analysis['suggested_filters']['reg_category'] = reg_type
                analysis['metadata_fields_considered'].append('reg_category')
                break
        
        # Extract task category
        task_category_patterns = {
            'Policies & Processes': ['policy', 'process', 'procedure', 'framework'],
            'Risk Management': ['risk', 'risk management', 'operational risk', 'credit risk'],
            'Compliance': ['compliance', 'regulatory compliance', 'audit'],
            'Reporting': ['reporting', 'returns', 'submission', 'filing'],
            'Technology': ['technology', 'cyber', 'digital', 'it', 'system'],
            'Customer Protection': ['customer', 'consumer', 'protection', 'grievance'],
            'Capital Adequacy': ['capital', 'capital adequacy', 'basel', 'cet1']
        }
        
        for task_category, patterns in task_category_patterns.items():
            if any(pattern in query_lower for pattern in patterns):
                analysis['extracted_task_category'] = task_category
                analysis['suggested_filters']['task_category'] = task_category
                analysis['metadata_fields_considered'].append('task_category')
                break
        
        # Extract risk category
        risk_category_patterns = {
            'High': ['high risk', 'critical', 'urgent', 'immediate'],
            'Medium': ['medium risk', 'moderate', 'standard'],
            'Low': ['low risk', 'routine', 'normal']
        }
        
        for risk_category, patterns in risk_category_patterns.items():
            if any(pattern in query_lower for pattern in patterns):
                analysis['extracted_risk_category'] = risk_category
                analysis['suggested_filters']['risk_category'] = risk_category
                analysis['metadata_fields_considered'].append('risk_category')
                break
        
        # Extract department context
        department_patterns = {
            'Risk Management': ['risk', 'risk management'],
            'Compliance': ['compliance', 'regulatory'],
            'Operations': ['operations', 'operational'],
            'Technology': ['technology', 'it', 'cyber'],
            'Legal': ['legal', 'law', 'litigation'],
            'Finance': ['finance', 'accounting', 'treasury']
        }
        
        for department, patterns in department_patterns.items():
            if any(pattern in query_lower for pattern in patterns):
                analysis['extracted_department'] = department
                analysis['suggested_filters']['department'] = department
                analysis['metadata_fields_considered'].append('department')
                break
        
        # Extract AI match context
        ai_match_patterns = {
            'Matched': ['matched', 'found', 'relevant', 'applicable'],
            'Pending': ['pending', 'unknown', 'unclear'],
            'Not Matched': ['not matched', 'irrelevant', 'not applicable']
        }
        
        for ai_match, patterns in ai_match_patterns.items():
            if any(pattern in query_lower for pattern in patterns):
                analysis['extracted_ai_match'] = ai_match
                analysis['suggested_filters']['AI_Match'] = ai_match
                analysis['metadata_fields_considered'].append('AI_Match')
                break
        
        # Extract regulatory numbers (enhanced patterns)
        reg_number_patterns = [
            r'\b[A-Z]{2,4}[/-]\d{4}[/-]\d{2,4}\b',  # RBI/2023-24/123
            r'\b[A-Z]{2,4}\s+\d{4}[/-]\d{2,4}\b',   # RBI 2023-24/123
            r'\b\d{4}[/-]\d{2,4}\b',                 # 2023-24/123
            r'\b[A-Z]{2,4}\s+\d{1,3}\b'              # RBI 123
        ]
        
        reg_numbers = []
        for pattern in reg_number_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            reg_numbers.extend(matches)
        analysis['regulatory_numbers'] = list(set(reg_numbers))  # Remove duplicates
        
        # Extract important keywords (enhanced filtering)
        stop_words = {'what', 'are', 'the', 'for', 'with', 'from', 'this', 'that', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'of', 'a', 'an', 'is', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
        important_keywords = []
        words = query_lower.split()
        for word in words:
            if len(word) > 3 and word not in stop_words and word.isalpha():
                important_keywords.append(word)
        analysis['keywords'] = important_keywords
        
        logger.info(f"Comprehensive query analysis completed: {analysis}")
        return analysis
    
    def calculate_keyword_match_score(self, query: str, metadata: Dict[str, Any]) -> float:
        """
        Calculate comprehensive keyword match score between query and regulation metadata.
        Considers all relevant metadata fields for financial regulatory content.
        """
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        # Get all text fields from metadata (comprehensive list)
        text_fields = [
            metadata.get('regulation_title', ''),
            metadata.get('summary', ''),
            metadata.get('action_item', ''),
            metadata.get('reg_subject', ''),
            metadata.get('regulator', ''),
            metadata.get('industry', ''),
            metadata.get('sub_industry', ''),
            metadata.get('task_category', ''),
            metadata.get('task_subcategory', ''),
            metadata.get('reg_category', ''),
            metadata.get('risk_category', ''),
            metadata.get('department', ''),
            metadata.get('control_nature', ''),
            metadata.get('frequency', ''),
            metadata.get('reg_number', ''),
            metadata.get('notes', '')
        ]
        
        # Combine all text fields (convert all to strings to avoid type errors)
        combined_text = ' '.join([str(field) for field in text_fields if field]).lower()
        text_words = set(combined_text.split())
        
        # Calculate word overlap
        if not query_words:
            return 0.0
        
        overlap = len(query_words.intersection(text_words))
        base_score = overlap / len(query_words)
        
        # Boost score for exact phrase matches
        phrase_boost = 0.0
        for word in query_words:
            if word in combined_text:
                phrase_boost += 0.1
        
        # Boost score for regulatory number matches
        reg_number_boost = 0.0
        reg_numbers_in_query = re.findall(r'\b[A-Z]{2,4}[/-]?\d{4}[/-]?\d{2,4}\b', query, re.IGNORECASE)
        reg_number_in_metadata = metadata.get('reg_number', '')
        if reg_numbers_in_query and reg_number_in_metadata:
            for reg_num in reg_numbers_in_query:
                if reg_num.lower() in str(reg_number_in_metadata).lower():
                    reg_number_boost += 0.3
        
        final_score = min(1.0, base_score + phrase_boost + reg_number_boost)
        return final_score
    
    def calculate_recency_score(self, metadata: Dict[str, Any]) -> float:
        """Calculate recency score based on regulation date."""
        reg_date = metadata.get('reg_date', '')
        due_date = metadata.get('due_date', '')
        
        if not reg_date:
            return 0.5  # Default score if no date
        
        try:
            # Parse date and calculate recency
            reg_datetime = datetime.strptime(reg_date, '%Y-%m-%d')
            current_date = datetime.now()
            
            # Calculate days since regulation
            days_diff = (current_date - reg_datetime).days
            
            # Score decreases with age (newer = higher score)
            if days_diff < 30:
                return 1.0
            elif days_diff < 365:
                return 0.8
            elif days_diff < 1095:  # 3 years
                return 0.6
            else:
                return 0.4
                
        except ValueError:
            return 0.5  # Default if date parsing fails
    
    def calculate_authority_weight(self, metadata: Dict[str, Any]) -> float:
        """Calculate authority weight based on regulatory body."""
        regulator = metadata.get('regulator', '')
        
        # Check for exact match
        if regulator in self.authority_weights:
            return self.authority_weights[regulator]
        
        # Check for partial match
        regulator_lower = str(regulator).lower()
        for authority, weight in self.authority_weights.items():
            if authority.lower() in regulator_lower or regulator_lower in authority.lower():
                return weight
        
        return 0.5  # Default weight for unknown authorities
    
    def rerank_results(self, query: str, results: List[Dict[str, Any]], 
                      weights: Dict[str, float] = None) -> List[Dict[str, Any]]:
        """
        Rerank results based on multiple factors for financial regulatory content.
        """
        if not results:
            return results
        
        if not weights:
            weights = self.rerank_weights
        
        # Calculate rerank scores
        for result in results:
            metadata = result.get('metadata', {})
            
            # Calculate individual scores
            vector_score = result.get('score', 0.0)
            keyword_score = self.calculate_keyword_match_score(query, metadata)
            recency_score = self.calculate_recency_score(metadata)
            authority_score = self.calculate_authority_weight(metadata)
            
            # Calculate weighted combined score
            rerank_score = (
                vector_score * weights['vector_similarity'] +
                keyword_score * weights['keyword_match'] +
                recency_score * weights['recency'] +
                authority_score * weights['authority_weight']
            )
            
            # Store individual scores for debugging
            result['rerank_scores'] = {
                'vector_similarity': vector_score,
                'keyword_match': keyword_score,
                'recency': recency_score,
                'authority_weight': authority_score,
                'combined_score': rerank_score
            }
            
            result['rerank_score'] = rerank_score
        
        # Sort by rerank score
        reranked_results = sorted(results, key=lambda x: x['rerank_score'], reverse=True)
        
        logger.info(f"Reranked {len(reranked_results)} results. Top score: {reranked_results[0]['rerank_score']:.3f}")
        return reranked_results
    
    def enhanced_search(self, query: str, filters: Dict[str, Any] = None, 
                       top_k: int = 10, use_reranking: bool = True) -> List[Dict[str, Any]]:
        """
        Enhanced search with intelligent query analysis and reranking.
        Comprehensive metadata handling for financial regulatory content.
        """
        logger.info(f"Starting enhanced search for query: '{query}'")
        
        # Analyze query for intelligent filtering
        query_analysis = self.intelligent_query_analysis(query)
        logger.info(f"Query analysis completed: {query_analysis}")
        
        # Merge user filters with intelligent filters
        combined_filters = filters or {}
        combined_filters.update(query_analysis['suggested_filters'])
        
        if combined_filters:
            logger.info(f"Using combined metadata filters: {combined_filters}")
        else:
            logger.info("No metadata filters applied")
        
        # Generate query embedding
        query_embedding = embeddings.embed_query(query)
        logger.info("Query embedding generated")
        
        # Use enhanced search with fallback
        results = self.pinecone_manager.search_with_fallback(
            query_embedding,
            metadata_filter=combined_filters if combined_filters else None,
            top_k=top_k
        )
        
        logger.info(f"Found {len(results)} results from Pinecone search")
        
        # Apply reranking if requested
        if use_reranking and results:
            logger.info("Applying reranking with metadata-aware scoring")
            results = self.rerank_results(query, results)
            logger.info(f"Reranking completed. Top result score: {results[0]['rerank_score']:.3f}")
        
        # Add query analysis to results for debugging
        for result in results:
            result['query_analysis'] = query_analysis
        
        logger.info(f"Enhanced search completed. Returning {len(results)} results")
        return results


# Legacy PineconeSearchManager removed - using EnhancedSearchManager only


class SecurityFilter:
    """Handles security filtering for sensitive queries."""
    
    @staticmethod
    def is_sensitive_query(query: str) -> bool:
        """Check if query is trying to access sensitive information."""
        sensitive_keywords = [
            'system prompt', 'prompt', 'instruction', 'template',
            'database structure', 'database schema', 'table structure',
            'proprietary', 'confidential', 'private information',
            'api key', 'password', 'secret', 'token',
            'show me the', 'display the', 'reveal the',
            'what is in', 'contents of', 'details of',
            'exact text', 'verbatim', 'quote from',
            'regulation number', 'regulation title', 'specific regulation'
        ]
        
        query_lower = query.lower()
        for keyword in sensitive_keywords:
            if keyword in query_lower:
                return True
        return False
    
    @staticmethod
    def get_security_response() -> str:
        """Get standard security response for sensitive queries."""
        return "I can only provide general regulatory compliance guidance. For specific regulatory details, please consult official regulatory sources or contact our compliance team."


class SemanticSimilarityChecker:
    """Handles semantic similarity checking using OpenAI LLM."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    def check_similarity(self, user_query: str, regulation_text: str, summary_text: str) -> Tuple[bool, float]:
        """
        Check semantic similarity between user query and regulation content.
        Returns (is_similar, confidence_score).
        """
        try:
            # Create prompt for similarity check
            prompt = f"""
            You are an expert at determining semantic similarity between regulatory queries and regulation content.
            
            User Query: "{user_query}"
            
            Regulation Text: "{regulation_text}"
            Summary: "{summary_text}"
            
            Determine if the user query is semantically similar to the regulation content.
            Consider:
            1. Are they asking about the same regulatory topic?
            2. Are the key concepts and requirements aligned?
            3. Would this regulation be relevant to answer the user's question?
            
            Respond with a JSON object containing:
            - "similar": true/false
            - "confidence": float between 0.0 and 1.0
            - "reasoning": brief explanation
            
            Only respond with the JSON object, no additional text.
            """
            
            client = get_openai_client()
            model_name = get_openai_model()
            service_type = "OLLAMA" if Config.USE_OLLAMA else ("AZURE OPENAI" if Config.AZURE_OPENAI_ENDPOINT else "OPENAI")
            logger.debug(f"Checking similarity using {service_type} (model: {model_name})")
            
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a regulatory compliance expert."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=200
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Parse JSON response
            try:
                result = json.loads(result_text)
                is_similar = result.get('similar', False)
                confidence = float(result.get('confidence', 0.0))
                reasoning = result.get('reasoning', '')
                
                logger.info(f"Similarity check: {is_similar}, confidence: {confidence}, reasoning: {reasoning}")
                return is_similar, confidence
                
            except json.JSONDecodeError:
                logger.error("Failed to parse similarity check response as JSON")
                return False, 0.0
                
        except Exception as e:
            logger.error(f"Error in semantic similarity check: {e}")
            return False, 0.0


def analyze_job_description(job_description: str, job_title: str = '') -> Dict[str, Any]:
    """
    Analyze job description to extract key requirements.
    
    Args:
        job_description: The job description text
        job_title: Optional job title
    
    Returns:
        Dictionary with extracted requirements
    """
    try:
        # Common technical skills patterns
        technical_skills = [
            'python', 'java', 'javascript', 'typescript', 'react', 'angular', 'vue',
            'node.js', 'django', 'flask', 'spring', 'express', 'sql', 'postgresql',
            'mysql', 'mongodb', 'redis', 'docker', 'kubernetes', 'aws', 'azure',
            'gcp', 'terraform', 'jenkins', 'git', 'linux', 'unix', 'html', 'css',
            'bootstrap', 'jquery', 'webpack', 'babel', 'eslint', 'jest', 'cypress',
            'selenium', 'pytest', 'machine learning', 'deep learning', 'tensorflow',
            'pytorch', 'pandas', 'numpy', 'scikit-learn', 'data science', 'analytics',
            'tableau', 'power bi', 'excel', 'r', 'matlab', 'spark', 'hadoop', 'kafka'
        ]
        
        # Experience patterns
        experience_patterns = [
            r'(\d+)\+?\s*years?\s*(?:of\s*)?experience',
            r'(\d+)\+?\s*years?\s*(?:of\s*)?(?:relevant\s*)?experience',
            r'minimum\s*(\d+)\s*years?',
            r'at\s*least\s*(\d+)\s*years?',
            r'(\d+)\s*to\s*(\d+)\s*years?'
        ]
        
        # Education patterns
        education_patterns = [
            r'bachelor[s]?\s*(?:of\s*)?(?:science\s*)?(?:in\s*)?(?:computer\s*)?(?:science|engineering)',
            r'master[s]?\s*(?:of\s*)?(?:science\s*)?(?:in\s*)?(?:computer\s*)?(?:science|engineering)',
            r'phd\s*(?:in\s*)?(?:computer\s*)?(?:science|engineering)',
            r'degree\s*in\s*(?:computer\s*)?(?:science|engineering)',
            r'engineering\s*degree',
            r'computer\s*science\s*degree'
        ]
        
        # Location patterns
        location_patterns = [
            r'(?:based\s*in|located\s*in|work\s*from)\s*([a-zA-Z\s,]+)',
            r'(?:hybrid|remote|onsite|work\s*from\s*home)',
            r'(?:bangalore|mumbai|delhi|hyderabad|chennai|pune|kolkata|gurgaon|noida)'
        ]
        
        # Initialize analysis result
        analysis = {
            'job_title': job_title,
            'required_skills': [],
            'preferred_skills': [],
            'min_experience': 0,
            'max_experience': 0,
            'education_requirements': [],
            'location_requirements': [],
            'work_mode': [],
            'key_requirements': [],
            'responsibilities': []
        }
        
        # Convert to lowercase for analysis
        jd_lower = job_description.lower()
        
        # Extract technical skills
        for skill in technical_skills:
            if skill in jd_lower:
                # Check if it's required or preferred
                skill_context = jd_lower[max(0, jd_lower.find(skill)-50):jd_lower.find(skill)+50]
                if any(word in skill_context for word in ['required', 'must', 'mandatory', 'essential']):
                    analysis['required_skills'].append(skill)
                elif any(word in skill_context for word in ['preferred', 'nice to have', 'bonus', 'plus']):
                    analysis['preferred_skills'].append(skill)
                else:
                    analysis['required_skills'].append(skill)  # Default to required
        
        # Extract experience requirements
        import re
        for pattern in experience_patterns:
            matches = re.findall(pattern, jd_lower)
            for match in matches:
                if isinstance(match, tuple):
                    # Range pattern
                    min_exp, max_exp = int(match[0]), int(match[1])
                    analysis['min_experience'] = max(analysis['min_experience'], min_exp)
                    analysis['max_experience'] = max(analysis['max_experience'], max_exp)
                else:
                    # Single value pattern
                    exp = int(match)
                    analysis['min_experience'] = max(analysis['min_experience'], exp)
        
        # Extract education requirements
        for pattern in education_patterns:
            if re.search(pattern, jd_lower):
                analysis['education_requirements'].append(pattern)
        
        # Extract location requirements
        for pattern in location_patterns:
            matches = re.findall(pattern, jd_lower)
            for match in matches:
                if isinstance(match, str):
                    analysis['location_requirements'].append(match.strip())
        
        # Extract work mode
        if 'remote' in jd_lower or 'work from home' in jd_lower:
            analysis['work_mode'].append('remote')
        if 'hybrid' in jd_lower:
            analysis['work_mode'].append('hybrid')
        if 'onsite' in jd_lower or 'office' in jd_lower:
            analysis['work_mode'].append('onsite')
        
        # Extract key requirements (first few sentences)
        sentences = job_description.split('.')[:3]
        analysis['key_requirements'] = [s.strip() for s in sentences if s.strip()]
        
        # Extract responsibilities (look for bullet points or numbered lists)
        lines = job_description.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith(('•', '-', '*', '1.', '2.', '3.')) or 'responsibility' in line.lower():
                analysis['responsibilities'].append(line)
        
        logger.info(f"JD Analysis completed: {len(analysis['required_skills'])} required skills, "
                   f"{len(analysis['preferred_skills'])} preferred skills")
        
        return analysis
        
    except Exception as e:
        logger.error(f"Error analyzing job description: {e}")
        return {
            'job_title': job_title,
            'required_skills': [],
            'preferred_skills': [],
            'min_experience': 0,
            'max_experience': 0,
            'education_requirements': [],
            'location_requirements': [],
            'work_mode': [],
            'key_requirements': [],
            'responsibilities': []
        }


def calculate_job_candidate_match_score(jd_analysis: Dict[str, Any], candidate_data: Dict[str, Any], pinecone_score: float) -> float:
    """
    Calculate comprehensive match score between job description and candidate.
    
    Args:
        jd_analysis: Analyzed job description requirements
        candidate_data: Candidate resume data
        pinecone_score: Pinecone similarity score
    
    Returns:
        Overall match score (0.0 to 1.0)
    """
    try:
        # Base score from Pinecone similarity
        base_score = pinecone_score
        
        # Skills match score
        skills_score = calculate_skills_match(jd_analysis, candidate_data)
        
        # Experience match score
        experience_score = calculate_experience_match(jd_analysis, candidate_data)
        
        # Education match score
        education_score = calculate_education_match(jd_analysis, candidate_data)
        
        # Location match score
        location_score = calculate_location_match('', candidate_data)
        
        # Weighted combination
        weights = {
            'base': 0.4,      # Pinecone similarity
            'skills': 0.3,    # Skills match
            'experience': 0.2, # Experience match
            'education': 0.05, # Education match
            'location': 0.05   # Location match
        }
        
        overall_score = (
            weights['base'] * base_score +
            weights['skills'] * skills_score +
            weights['experience'] * experience_score +
            weights['education'] * education_score +
            weights['location'] * location_score
        )
        
        # Ensure score is between 0 and 1
        return min(1.0, max(0.0, overall_score))
        
    except Exception as e:
        logger.error(f"Error calculating match score: {e}")
        return pinecone_score


def calculate_skills_match(jd_analysis: Dict[str, Any], candidate_data: Dict[str, Any]) -> float:
    """Calculate skills match score."""
    try:
        required_skills = [skill.lower() for skill in jd_analysis.get('required_skills', [])]
        preferred_skills = [skill.lower() for skill in jd_analysis.get('preferred_skills', [])]
        
        candidate_skills = candidate_data.get('primary_skills', '').lower()
        candidate_secondary_skills = candidate_data.get('secondary_skills', '').lower()
        all_candidate_skills = candidate_skills + ' ' + candidate_secondary_skills
        
        if not required_skills:
            return 0.5  # Neutral score if no required skills specified
        
        # Calculate required skills match
        required_matches = sum(1 for skill in required_skills if skill in all_candidate_skills)
        required_score = required_matches / len(required_skills)
        
        # Calculate preferred skills match
        preferred_matches = sum(1 for skill in preferred_skills if skill in all_candidate_skills)
        preferred_score = preferred_matches / len(preferred_skills) if preferred_skills else 0
        
        # Weighted combination (required skills are more important)
        total_score = (required_score * 0.8) + (preferred_score * 0.2)
        
        return min(1.0, total_score)
        
    except Exception as e:
        logger.error(f"Error calculating skills match: {e}")
        return 0.0


def calculate_experience_match(jd_analysis: Dict[str, Any], candidate_data: Dict[str, Any]) -> float:
    """Calculate experience match score."""
    try:
        min_exp = jd_analysis.get('min_experience', 0)
        max_exp = jd_analysis.get('max_experience', 0)
        candidate_exp = candidate_data.get('total_experience', 0)
        
        if min_exp == 0:
            return 0.5  # Neutral score if no experience requirement
        
        # Convert candidate experience to years (assuming it's in months)
        if candidate_exp > 100:  # Likely in months
            candidate_exp_years = candidate_exp / 12
        else:
            candidate_exp_years = candidate_exp
        
        if candidate_exp_years >= min_exp:
            if max_exp > 0 and candidate_exp_years <= max_exp:
                return 1.0  # Perfect match
            elif max_exp > 0 and candidate_exp_years > max_exp:
                # Overqualified but still good
                return 0.8
            else:
                return 1.0  # Meets minimum requirement
        else:
            # Underqualified
            return max(0.0, candidate_exp_years / min_exp)
        
    except Exception as e:
        logger.error(f"Error calculating experience match: {e}")
        return 0.0


def calculate_education_match(jd_analysis: Dict[str, Any], candidate_data: Dict[str, Any]) -> float:
    """Calculate education match score."""
    try:
        education_requirements = jd_analysis.get('education_requirements', [])
        candidate_education = candidate_data.get('education', '').lower()
        
        if not education_requirements:
            return 0.5  # Neutral score if no education requirement
        
        # Check if candidate education matches any requirement
        for requirement in education_requirements:
            if any(word in candidate_education for word in requirement.split()):
                return 1.0
        
        # Partial match for degree-related terms
        degree_terms = ['bachelor', 'master', 'phd', 'degree', 'engineering', 'science']
        if any(term in candidate_education for term in degree_terms):
            return 0.7
        
        return 0.0
        
    except Exception as e:
        logger.error(f"Error calculating education match: {e}")
        return 0.0


def calculate_location_match(job_location: str, candidate_data: Dict[str, Any]) -> float:
    """Calculate location match score."""
    try:
        if not job_location:
            return 0.5  # Neutral score if no location specified
        
        candidate_location = candidate_data.get('current_location', '').lower()
        preferred_locations = candidate_data.get('preferred_locations', '').lower()
        
        job_location_lower = job_location.lower()
        
        # Exact match
        if job_location_lower in candidate_location or job_location_lower in preferred_locations:
            return 1.0
        
        # Partial match for major cities
        major_cities = {
            'bangalore': ['bengaluru', 'blore'],
            'mumbai': ['bombay'],
            'delhi': ['new delhi', 'ncr'],
            'hyderabad': ['hyd'],
            'chennai': ['madras'],
            'pune': ['pun'],
            'kolkata': ['calcutta']
        }
        
        for city, aliases in major_cities.items():
            if city in job_location_lower:
                for alias in aliases:
                    if alias in candidate_location or alias in preferred_locations:
                        return 0.8
        
        return 0.0
        
    except Exception as e:
        logger.error(f"Error calculating location match: {e}")
        return 0.0


    """
    Endpoint for LLM-based chat response using Pinecone data only.
    Retrieves relevant regulations from Pinecone and uses them as context for LLM response.
    """
    try:
        # Validate request
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
        
        data = request.get_json()
        if 'query' not in data:
            return jsonify({'error': 'Missing query field'}), 400
        
        user_query = data['query'].strip()
        if not user_query:
            return jsonify({'error': 'Query cannot be empty'}), 400
        
        logger.info(f"Processing chat query with Pinecone data: {user_query}")
        
        # Start timing for performance monitoring
        start_time = time.time()
        
        # Initialize production monitoring
        production_logger = get_production_logger()
        production_logger.log_query_received(user_query)
        
        # Initialize production RAG manager for query classification and prompt building
        production_rag_manager = ProductionRAGManager()
        
        # Classify query for relevance and regulatory domain
        query_relevance, domains, analysis = classify_regulatory_query(user_query)
        logger.info(f"Query classified as: {query_relevance.value}, domains: {[d.value for d in domains]}")
        
        # Log query classification
        production_logger.log_query_classified(
            user_query, query_relevance.value, [d.value for d in domains], 
            analysis, (time.time() - start_time) * 1000
        )
        
        # Handle potentially harmful or irrelevant queries
        if query_relevance == QueryRelevance.POTENTIALLY_HARMFUL:
            logger.warning(f"Potentially harmful query detected: {user_query}")
            return jsonify({
                'message': 'Security filter activated - potentially harmful query',
                'user_query': user_query,
                'llm_response': 'I can only assist with legitimate regulatory compliance queries. For security and compliance matters, please contact your organization\'s compliance team or consult official regulatory sources directly.',
                'context_used': [],
                'query_classification': {
                    'relevance': query_relevance.value,
                    'domains': [d.value for d in domains],
                    'analysis': analysis
                },
                'timestamp': datetime.now().isoformat()
            }), 200
        
        if query_relevance == QueryRelevance.IRRELEVANT:
            logger.info(f"Irrelevant query detected: {user_query}")
            return jsonify({
                'message': 'Query out of scope - not related to regulatory compliance',
                'user_query': user_query,
                'llm_response': 'I specialize in regulatory compliance assistance for financial institutions. I can only help with queries related to regulatory circulars, compliance requirements, risk management, and related matters. Please rephrase your question to focus on regulatory compliance.',
                'context_used': [],
                'query_classification': {
                    'relevance': query_relevance.value,
                    'domains': [d.value for d in domains],
                    'analysis': analysis
                },
                'timestamp': datetime.now().isoformat()
            }), 200
        
        # Initialize components
        enhanced_search_manager = EnhancedSearchManager()
        
        # Connect to Pinecone
        if not enhanced_search_manager.connect_to_index():
            return jsonify({'error': 'Failed to connect to Pinecone index'}), 500
        
        # Use enhanced search with intelligent query analysis and reranking
        logger.info("Using enhanced search with intelligent query analysis")
        similar_vectors = enhanced_search_manager.enhanced_search(
            user_query, 
            top_k=5, 
            use_reranking=True
        )
        
        # Log query embedding token usage
        query_tokens = token_counter.count_embedding_tokens(user_query)
        log_query_embedding_tokens(
            model_name="text-embedding-ada-002",
            input_tokens=query_tokens,
            user_query=user_query,
            query_relevance=query_relevance.value,
            regulatory_domains=[d.value for d in domains],
            metadata={
                "analysis": analysis,
                "processing_time_ms": (time.time() - start_time) * 1000,
                "vectors_found": len(similar_vectors)
            }
        )
        
        if not similar_vectors:
            return jsonify({
                'message': 'No relevant regulations found in our database',
                'user_query': user_query,
                'llm_response': 'I apologize, but I could not find any relevant regulatory information in our database for your query. Please try rephrasing your question or contact our compliance team for assistance.',
                'context_used': [],
                'timestamp': datetime.now().isoformat()
            }), 200
        
        # Gather context from Pinecone results
        context_regulations = []
        for vector_match in similar_vectors:
            try:
                metadata = vector_match['metadata']
                pinecone_score = vector_match['score']
                chunk_text = metadata.get('chunk_text', '')
                
                # Build context entry from Pinecone metadata
                context_entry = {
                    'vector_id': vector_match['id'],
                    'regulation_title': metadata.get('regulation', ''),
                    'summary': metadata.get('summary', ''),
                    'regulator': metadata.get('regulator', ''),
                    'industry': metadata.get('industry', ''),
                    'due_date': metadata.get('due_date', ''),
                    'reg_category': metadata.get('reg_category', ''),
                    'reg_subject': metadata.get('reg_subject', ''),
                    'risk_category': metadata.get('risk_category', ''),
                    'department': metadata.get('department', ''),
                    'chunk_text': chunk_text,
                    'chunk_index': metadata.get('chunk_index', 0),
                    'total_chunks': metadata.get('total_chunks', 1),
                    'relevance_score': pinecone_score,
                    'rerank_score': vector_match.get('rerank_score', pinecone_score)
                }
                
                context_regulations.append(context_entry)
                
            except Exception as e:
                logger.error(f"Error processing vector match for context: {e}")
                continue
        
        if not context_regulations:
            return jsonify({
                'message': 'No valid regulations found for context',
                'user_query': user_query,
                'llm_response': 'I apologize, but I could not retrieve valid regulatory information for your query. Please try again or contact our compliance team.',
                'context_used': [],
                'timestamp': datetime.now().isoformat()
            }), 200
        
        # Build context string for LLM from Pinecone data
        context_string = "Relevant Regulatory Information from our database:\n\n"
        for i, reg in enumerate(context_regulations, 1):
            context_string += f"{i}. Regulation: {str(reg.get('regulation_title', ''))}\n"
            if reg.get('regulator'):
                context_string += f"   Regulator: {str(reg['regulator'])}\n"
            if reg.get('industry'):
                context_string += f"   Industry: {str(reg['industry'])}\n"
            if reg.get('summary'):
                context_string += f"   Summary: {str(reg['summary'])}\n"
            if reg.get('reg_category'):
                context_string += f"   Category: {str(reg['reg_category'])}\n"
            if reg.get('reg_subject'):
                context_string += f"   Subject: {str(reg['reg_subject'])}\n"
            if reg.get('due_date'):
                context_string += f"   Due Date: {str(reg['due_date'])}\n"
            if reg.get('chunk_text'):
                context_string += f"   Content: {str(reg['chunk_text'])[:500]}...\n"
            if reg.get('relevance_score') is not None:
                context_string += f"   Relevance Score: {float(reg['relevance_score']):.3f}\n\n"
        
        # Generate LLM response using production-grade prompts
        try:
            client = get_openai_client()
            
            # Build production-grade prompts based on query classification
            system_prompt, user_prompt = build_regulatory_prompts(
                user_query, context_regulations, query_relevance, domains
            )
            
            logger.info(f"Using production prompts for {query_relevance.value} query")
            
            # Count tokens for RAG input
            input_tokens = token_counter.count_chat_tokens(system_prompt + user_prompt)
            
            # Log RAG input token usage
            log_rag_tokens(
                model_name=get_openai_model(),
                input_tokens=input_tokens,
                output_tokens=0,  # Will be updated after response
                user_query=user_query,
                response_length=0,  # Will be updated after response
                processing_time_ms=(time.time() - start_time) * 1000,
                query_relevance=query_relevance.value,
                regulatory_domains=[d.value for d in domains],
                metadata={
                    "analysis": analysis,
                    "context_count": len(context_regulations),
                    "operation": "rag_input"
                }
            )
            
            response = client.chat.completions.create(
                model=get_openai_model(),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=800
            )
            
            llm_response = response.choices[0].message.content.strip()
            logger.info("LLM response generated using production-grade prompts")
            
            # Count tokens for RAG output
            output_tokens = token_counter.count_chat_tokens(llm_response)
            
            # Log RAG output token usage
            log_rag_tokens(
                model_name=get_openai_model(),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                user_query=user_query,
                response_length=len(llm_response),
                processing_time_ms=(time.time() - start_time) * 1000,
                query_relevance=query_relevance.value,
                regulatory_domains=[d.value for d in domains],
                metadata={
                    "analysis": analysis,
                    "context_count": len(context_regulations),
                    "operation": "rag_output"
                }
            )
            
            # Log response generation
            production_logger.log_response_generated(
                user_query, len(llm_response), (time.time() - start_time) * 1000
            )
            
            # Validate response quality and safety
            from production_prompts import ResponseValidator
            response_validator = ResponseValidator()
            validation_result = response_validator.validate_response(
                llm_response, query_relevance, context_regulations
            )
            
            logger.info(f"Response validation: valid={validation_result['is_valid']}, "
                       f"quality_score={validation_result['quality_score']:.2f}, "
                       f"safety_score={validation_result['safety_score']:.2f}")
            
            # Log response validation
            production_logger.log_response_validated(user_query, validation_result)
            
            # Complete processing logging
            end_time = time.time()
            log_query_processing(
                user_query, start_time, end_time,
                query_relevance.value, [d.value for d in domains], analysis,
                len(context_regulations), len(llm_response), validation_result
            )
            
            # Ensure all values are JSON serializable
            safe_context_regulations = []
            for reg in context_regulations:
                safe_reg = {}
                for key, value in reg.items():
                    if value is not None:
                        safe_reg[key] = str(value) if not isinstance(value, (int, float, bool)) else value
                    else:
                        safe_reg[key] = ""
                safe_context_regulations.append(safe_reg)
            
            response_data = {
                'message': 'Chat response generated using production-grade regulatory prompts',
                'user_query': str(user_query),
                'llm_response': str(llm_response),
                'context_used': safe_context_regulations,
                'total_vectors_found': int(len(similar_vectors)),
                'context_regulations_used': int(len(context_regulations)),
                'query_classification': {
                    'relevance': str(query_relevance.value),
                    'domains': [str(d.value) for d in domains],
                    'analysis': {k: str(v) if not isinstance(v, (int, float, bool, list, dict)) else v for k, v in analysis.items()}
                },
                'response_validation': validation_result,
                'processing_time_ms': float((end_time - start_time) * 1000),
                'token_usage': {
                    'query_embedding_tokens': int(query_tokens),
                    'rag_input_tokens': int(input_tokens),
                    'rag_output_tokens': int(output_tokens),
                    'total_tokens': int(query_tokens + input_tokens + output_tokens),
                    'models_used': [str('text-embedding-ada-002'), str(Config.OPENAI_MODEL)]
                },
                'timestamp': str(datetime.now().isoformat())
            }
            
            return jsonify(response_data), 200
            
        except Exception as e:
            logger.error(f"Error generating LLM response with context: {e}")
            return jsonify({'error': 'Failed to generate chat response with context'}), 500
            
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/resume-search', methods=['POST'])
def resume_search():
    """
    Enhanced resume search endpoint that combines Pinecone vector search with MySQL metadata.
    Returns comprehensive candidate information including candidate_id and all resume_metadata columns.
    """
    try:
        # Validate request
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
        
        data = request.get_json()
        if 'query' not in data:
            return jsonify({'error': 'Missing query field'}), 400
        
        user_query = data['query'].strip()
        if not user_query:
            return jsonify({'error': 'Query cannot be empty'}), 400
        
        # Optional parameters
        filters = data.get('filters', {})
        top_k = data.get('top_k', 10)
        include_full_details = data.get('include_full_details', True)
        
        logger.info(f"Processing resume search query: {user_query}")
        logger.info(f"Applied filters: {filters}")
        
        # Start timing for performance monitoring
        start_time = time.time()
        
        # Initialize components
        from ats_database import create_ats_database
        from embed_api import PineconeManager
        from ats_config import ATSConfig
        
        # Check if Pinecone is enabled
        if not ATSConfig.USE_PINECONE or not ATSConfig.PINECONE_API_KEY:
            return jsonify({'error': 'Pinecone indexing is not enabled'}), 400
        
        # Initialize Pinecone manager
        pinecone_manager = PineconeManager()
        pinecone_manager.get_or_create_index()
        
        # Generate embedding for query
        query_embedding = embeddings.embed_query(user_query)
        
        # Perform vector search in Pinecone
        search_results = pinecone_manager.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            filter=filters if filters else None
        )
        
        if not search_results.matches:
            return jsonify({
                'message': 'No matching resumes found',
                'user_query': user_query,
                'candidates': [],
                'total_matches': 0,
                'processing_time_ms': (time.time() - start_time) * 1000,
                'timestamp': datetime.now().isoformat()
            }), 200
        
        # Extract candidate IDs from Pinecone results
        candidate_ids = []
        pinecone_matches = {}
        
        for match in search_results.matches:
            candidate_id = match.metadata.get('candidate_id')
            if candidate_id:
                candidate_ids.append(candidate_id)
                pinecone_matches[candidate_id] = {
                    'similarity_score': match.score,
                    'pinecone_metadata': match.metadata
                }
        
        if not candidate_ids:
            return jsonify({
                'message': 'No valid candidate IDs found in Pinecone results',
                'user_query': user_query,
                'candidates': [],
                'total_matches': 0,
                'processing_time_ms': (time.time() - start_time) * 1000,
                'timestamp': datetime.now().isoformat()
            }), 200
        
        # Fetch detailed resume data from MySQL
        candidates = []
        with create_ats_database() as db:
            for candidate_id in candidate_ids:
                try:
                    # Get full resume details from MySQL
                    resume_data = db.get_resume_by_id(candidate_id)
                    
                    if resume_data:
                        # Combine Pinecone similarity score with MySQL data
                        candidate_info = {
                            'candidate_id': candidate_id,
                            'similarity_score': pinecone_matches[candidate_id]['similarity_score'],
                            'name': resume_data.get('name'),
                            'email': resume_data.get('email'),
                            'phone': resume_data.get('phone'),
                            'total_experience': resume_data.get('total_experience'),
                            'primary_skills': resume_data.get('primary_skills'),
                            'secondary_skills': resume_data.get('secondary_skills'),
                            'all_skills': resume_data.get('all_skills'),
                            'domain': resume_data.get('domain'),
                            'sub_domain': resume_data.get('sub_domain'),
                            'education': resume_data.get('education'),
                            'education_details': resume_data.get('education_details'),
                            'current_location': resume_data.get('current_location'),
                            'preferred_locations': resume_data.get('preferred_locations'),
                            'current_company': resume_data.get('current_company'),
                            'current_designation': resume_data.get('current_designation'),
                            'notice_period': resume_data.get('notice_period'),
                            'expected_salary': resume_data.get('expected_salary'),
                            'current_salary': resume_data.get('current_salary'),
                            'resume_summary': resume_data.get('resume_summary'),
                            'file_name': resume_data.get('file_name'),
                            'file_type': resume_data.get('file_type'),
                            'file_size_kb': resume_data.get('file_size_kb'),
                            'embedding_model': resume_data.get('embedding_model'),
                            'status': resume_data.get('status'),
                            'source': resume_data.get('source'),
                            'created_at': resume_data.get('created_at'),
                            'updated_at': resume_data.get('updated_at')
                        }
                        
                        # Include full resume text if requested
                        if include_full_details:
                            candidate_info['resume_text'] = resume_data.get('resume_text')
                        
                        candidates.append(candidate_info)
                        
                except Exception as e:
                    logger.error(f"Error fetching resume data for candidate {candidate_id}: {e}")
                    continue
        
        # Sort candidates by similarity score (highest first)
        candidates.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        # Prepare response
        response_data = {
            'message': 'Resume search completed successfully',
            'user_query': user_query,
            'candidates': candidates,
            'total_matches': len(candidates),
            'applied_filters': filters,
            'search_parameters': {
                'top_k': top_k,
                'include_full_details': include_full_details
            },
            'processing_time_ms': (time.time() - start_time) * 1000,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Resume search completed: {len(candidates)} candidates found")
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"Error in resume search endpoint: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/jd-processor', methods=['POST'])
def jd_processor():
    """
    Job Description Processor API that analyzes job descriptions and finds matching candidates.
    
    Input: Job description text
    Output: Ranked list of matching candidates with metadata and scores
    """
    try:
        # Validate request
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
        
        data = request.get_json()
        if 'job_description' not in data:
            return jsonify({'error': 'Missing job_description field'}), 400
        
        job_description = data['job_description'].strip()
        if not job_description:
            return jsonify({'error': 'Job description cannot be empty'}), 400
        
        # Optional parameters
        job_title = data.get('job_title', '')
        company_name = data.get('company_name', '')
        location = data.get('location', '')
        top_k = data.get('top_k', 10)
        min_match_score = data.get('min_match_score', 0.3)
        include_full_details = data.get('include_full_details', True)
        
        logger.info(f"Processing job description: {job_title or 'Untitled Position'}")
        logger.info(f"Job description length: {len(job_description)} characters")
        
        # Start timing for performance monitoring
        start_time = time.time()
        
        # Initialize components
        from ats_database import create_ats_database
        from embed_api import PineconeManager
        from ats_config import ATSConfig
        
        # Check if Pinecone is enabled
        if not ATSConfig.USE_PINECONE or not ATSConfig.PINECONE_API_KEY:
            return jsonify({'error': 'Pinecone indexing is not enabled'}), 400
        
        # Initialize Pinecone manager
        pinecone_manager = PineconeManager()
        pinecone_manager.get_or_create_index()
        
        # Step 1: Extract key requirements from job description
        jd_analysis = analyze_job_description(job_description, job_title)
        logger.info(f"JD Analysis: {jd_analysis}")
        
        # Step 2: Generate embedding for job description
        jd_embedding = embeddings.embed_query(job_description)
        
        # Step 3: Perform vector search in Pinecone
        search_results = pinecone_manager.index.query(
            vector=jd_embedding,
            top_k=top_k * 2,  # Get more results for filtering
            include_metadata=True
        )
        
        if not search_results.matches:
            return jsonify({
                'message': 'No matching candidates found for this job description',
                'job_title': job_title,
                'candidates': [],
                'total_matches': 0,
                'jd_analysis': jd_analysis,
                'processing_time_ms': (time.time() - start_time) * 1000,
                'timestamp': datetime.now().isoformat()
            }), 200
        
        # Step 4: Extract candidate IDs and calculate match scores
        candidate_ids = []
        pinecone_scores = {}
        
        for match in search_results.matches:
            candidate_id = match.metadata.get('candidate_id')
            if candidate_id:
                candidate_ids.append(candidate_id)
                pinecone_scores[candidate_id] = match.score
        
        if not candidate_ids:
            return jsonify({
                'message': 'No valid candidate IDs found in Pinecone results',
                'job_title': job_title,
                'candidates': [],
                'total_matches': 0,
                'jd_analysis': jd_analysis,
                'processing_time_ms': (time.time() - start_time) * 1000,
                'timestamp': datetime.now().isoformat()
            }), 200
        
        # Step 5: Fetch detailed candidate data and calculate job-candidate match scores
        candidates = []
        with create_ats_database() as db:
            for candidate_id in candidate_ids:
                try:
                    # Get full candidate details from MySQL
                    candidate_data = db.get_resume_by_id(candidate_id)
                    
                    if candidate_data:
                        # Calculate comprehensive match score
                        pinecone_score = pinecone_scores[candidate_id]
                        match_score = calculate_job_candidate_match_score(
                            jd_analysis, candidate_data, pinecone_score
                        )
                        
                        # Filter by minimum match score
                        if match_score < min_match_score:
                            continue
                        
                        # Prepare candidate information
                        candidate_info = {
                            'candidate_id': candidate_id,
                            'pinecone_similarity': pinecone_score,
                            'job_match_score': match_score,
                            'name': candidate_data.get('name'),
                            'email': candidate_data.get('email'),
                            'phone': candidate_data.get('phone'),
                            'total_experience': candidate_data.get('total_experience'),
                            'primary_skills': candidate_data.get('primary_skills'),
                            'secondary_skills': candidate_data.get('secondary_skills'),
                            'all_skills': candidate_data.get('all_skills'),
                            'domain': candidate_data.get('domain'),
                            'sub_domain': candidate_data.get('sub_domain'),
                            'education': candidate_data.get('education'),
                            'education_details': candidate_data.get('education_details'),
                            'current_location': candidate_data.get('current_location'),
                            'preferred_locations': candidate_data.get('preferred_locations'),
                            'current_company': candidate_data.get('current_company'),
                            'current_designation': candidate_data.get('current_designation'),
                            'notice_period': candidate_data.get('notice_period'),
                            'expected_salary': candidate_data.get('expected_salary'),
                            'current_salary': candidate_data.get('current_salary'),
                            'resume_summary': candidate_data.get('resume_summary'),
                            'file_name': candidate_data.get('file_name'),
                            'file_type': candidate_data.get('file_type'),
                            'file_size_kb': candidate_data.get('file_size_kb'),
                            'embedding_model': candidate_data.get('embedding_model'),
                            'status': candidate_data.get('status'),
                            'source': candidate_data.get('source'),
                            'created_at': candidate_data.get('created_at'),
                            'updated_at': candidate_data.get('updated_at'),
                            'match_details': {
                                'skills_match': calculate_skills_match(jd_analysis, candidate_data),
                                'experience_match': calculate_experience_match(jd_analysis, candidate_data),
                                'education_match': calculate_education_match(jd_analysis, candidate_data),
                                'location_match': calculate_location_match(location, candidate_data)
                            }
                        }
                        
                        # Include full resume text if requested
                        if include_full_details:
                            candidate_info['resume_text'] = candidate_data.get('resume_text')
                        
                        candidates.append(candidate_info)
                        
                except Exception as e:
                    logger.error(f"Error processing candidate {candidate_id}: {e}")
                    continue
        
        # Step 6: Sort candidates by job match score (highest first)
        candidates.sort(key=lambda x: x['job_match_score'], reverse=True)
        
        # Step 7: Limit to requested top_k
        candidates = candidates[:top_k]
        
        # Step 8: Prepare response
        response_data = {
            'message': 'Job description processed successfully',
            'job_title': job_title,
            'company_name': company_name,
            'location': location,
            'candidates': candidates,
            'total_matches': len(candidates),
            'jd_analysis': jd_analysis,
            'search_parameters': {
                'top_k': top_k,
                'min_match_score': min_match_score,
                'include_full_details': include_full_details
            },
            'processing_time_ms': (time.time() - start_time) * 1000,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"JD Processor completed: {len(candidates)} candidates found for job '{job_title}'")
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"Error in JD processor endpoint: {e}")
        return jsonify({'error': str(e)}), 500


    """
    Advanced resume search with hybrid approach:
    1. Pinecone vector search for semantic similarity
    2. MySQL filtering for precise metadata matching
    3. Combined ranking and scoring
    """
    try:
        # Validate request
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
        
        data = request.get_json()
        if 'query' not in data:
            return jsonify({'error': 'Missing query field'}), 400
        
        user_query = data['query'].strip()
        if not user_query:
            return jsonify({'error': 'Query cannot be empty'}), 400
        
        # Advanced search parameters
        filters = data.get('filters', {})
        top_k = data.get('top_k', 10)
        include_full_details = data.get('include_full_details', True)
        use_hybrid_search = data.get('use_hybrid_search', True)
        min_similarity_score = data.get('min_similarity_score', 0.0)
        
        logger.info(f"Processing advanced resume search: {user_query}")
        logger.info(f"Search parameters: filters={filters}, top_k={top_k}, hybrid={use_hybrid_search}")
        
        start_time = time.time()
        
        # Initialize components
        from ats_database import create_ats_database
        from embed_api import PineconeManager
        from ats_config import ATSConfig
        
        if not ATSConfig.USE_PINECONE or not ATSConfig.PINECONE_API_KEY:
            return jsonify({'error': 'Pinecone indexing is not enabled'}), 400
        
        pinecone_manager = PineconeManager()
        pinecone_manager.get_or_create_index()
        
        # Generate embedding for query
        query_embedding = embeddings.embed_query(user_query)
        
        # Step 1: Pinecone vector search
        search_results = pinecone_manager.index.query(
            vector=query_embedding,
            top_k=top_k * 2,  # Get more results for filtering
            include_metadata=True,
            filter=filters if filters else None
        )
        
        if not search_results.matches:
            return jsonify({
                'message': 'No matching resumes found',
                'user_query': user_query,
                'candidates': [],
                'total_matches': 0,
                'processing_time_ms': (time.time() - start_time) * 1000,
                'timestamp': datetime.now().isoformat()
            }), 200
        
        # Step 2: Extract candidate IDs and filter by similarity score
        candidate_ids = []
        pinecone_scores = {}
        
        for match in search_results.matches:
            if match.score >= min_similarity_score:
                candidate_id = match.metadata.get('candidate_id')
                if candidate_id:
                    candidate_ids.append(candidate_id)
                    pinecone_scores[candidate_id] = match.score
        
        if not candidate_ids:
            return jsonify({
                'message': f'No candidates meet minimum similarity score of {min_similarity_score}',
                'user_query': user_query,
                'candidates': [],
                'total_matches': 0,
                'processing_time_ms': (time.time() - start_time) * 1000,
                'timestamp': datetime.now().isoformat()
            }), 200
        
        # Step 3: Fetch detailed data from MySQL with additional filtering
        candidates = []
        with create_ats_database() as db:
            for candidate_id in candidate_ids:
                try:
                    resume_data = db.get_resume_by_id(candidate_id)
                    
                    if resume_data:
                        # Apply additional MySQL-based filtering
                        passes_filter = True
                        
                        # Filter by experience range
                        if 'min_experience' in filters:
                            if resume_data.get('total_experience', 0) < filters['min_experience']:
                                passes_filter = False
                        
                        if 'max_experience' in filters:
                            if resume_data.get('total_experience', 0) > filters['max_experience']:
                                passes_filter = False
                        
                        # Filter by domain
                        if 'domain' in filters:
                            if resume_data.get('domain') != filters['domain']:
                                passes_filter = False
                        
                        # Filter by education
                        if 'education' in filters:
                            if filters['education'].lower() not in resume_data.get('education', '').lower():
                                passes_filter = False
                        
                        # Filter by location
                        if 'location' in filters:
                            location = resume_data.get('current_location', '')
                            if filters['location'].lower() not in location.lower():
                                passes_filter = False
                        
                        # Filter by skills
                        if 'required_skills' in filters:
                            required_skills = [skill.lower() for skill in filters['required_skills']]
                            candidate_skills = resume_data.get('primary_skills', '').lower()
                            if not any(skill in candidate_skills for skill in required_skills):
                                passes_filter = False
                        
                        if not passes_filter:
                            continue
                        
                        # Calculate hybrid score (combine Pinecone similarity with other factors)
                        pinecone_score = pinecone_scores[candidate_id]
                        hybrid_score = pinecone_score
                        
                        # Boost score for exact skill matches
                        if 'required_skills' in filters:
                            required_skills = [skill.lower() for skill in filters['required_skills']]
                            candidate_skills = resume_data.get('primary_skills', '').lower()
                            skill_matches = sum(1 for skill in required_skills if skill in candidate_skills)
                            if skill_matches > 0:
                                hybrid_score += (skill_matches / len(required_skills)) * 0.1
                        
                        # Boost score for experience match
                        if 'target_experience' in filters:
                            target_exp = filters['target_experience']
                            candidate_exp = resume_data.get('total_experience', 0)
                            if candidate_exp > 0:
                                exp_ratio = min(candidate_exp, target_exp) / max(candidate_exp, target_exp)
                                hybrid_score += exp_ratio * 0.05
                        
                        # Prepare candidate information
                        candidate_info = {
                            'candidate_id': candidate_id,
                            'similarity_score': pinecone_score,
                            'hybrid_score': hybrid_score,
                            'name': resume_data.get('name'),
                            'email': resume_data.get('email'),
                            'phone': resume_data.get('phone'),
                            'total_experience': resume_data.get('total_experience'),
                            'primary_skills': resume_data.get('primary_skills'),
                            'secondary_skills': resume_data.get('secondary_skills'),
                            'all_skills': resume_data.get('all_skills'),
                            'domain': resume_data.get('domain'),
                            'sub_domain': resume_data.get('sub_domain'),
                            'education': resume_data.get('education'),
                            'education_details': resume_data.get('education_details'),
                            'current_location': resume_data.get('current_location'),
                            'preferred_locations': resume_data.get('preferred_locations'),
                            'current_company': resume_data.get('current_company'),
                            'current_designation': resume_data.get('current_designation'),
                            'notice_period': resume_data.get('notice_period'),
                            'expected_salary': resume_data.get('expected_salary'),
                            'current_salary': resume_data.get('current_salary'),
                            'resume_summary': resume_data.get('resume_summary'),
                            'file_name': resume_data.get('file_name'),
                            'file_type': resume_data.get('file_type'),
                            'file_size_kb': resume_data.get('file_size_kb'),
                            'embedding_model': resume_data.get('embedding_model'),
                            'status': resume_data.get('status'),
                            'source': resume_data.get('source'),
                            'created_at': resume_data.get('created_at'),
                            'updated_at': resume_data.get('updated_at')
                        }
                        
                        if include_full_details:
                            candidate_info['resume_text'] = resume_data.get('resume_text')
                        
                        candidates.append(candidate_info)
                        
                except Exception as e:
                    logger.error(f"Error processing candidate {candidate_id}: {e}")
                    continue
        
        # Sort by hybrid score (highest first)
        candidates.sort(key=lambda x: x['hybrid_score'], reverse=True)
        
        # Limit to requested top_k
        candidates = candidates[:top_k]
        
        # Prepare response
        response_data = {
            'message': 'Advanced resume search completed successfully',
            'user_query': user_query,
            'candidates': candidates,
            'total_matches': len(candidates),
            'applied_filters': filters,
            'search_parameters': {
                'top_k': top_k,
                'include_full_details': include_full_details,
                'use_hybrid_search': use_hybrid_search,
                'min_similarity_score': min_similarity_score
            },
            'processing_time_ms': (time.time() - start_time) * 1000,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Advanced resume search completed: {len(candidates)} candidates found")
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"Error in advanced resume search endpoint: {e}")
        return jsonify({'error': str(e)}), 500


    try:
        from production_monitoring import get_system_health, get_performance_metrics
        
        health_status = get_system_health()
        performance_metrics = get_performance_metrics()
        
        return jsonify({
            'status': 'healthy' if health_status['status'] == 'healthy' else 'unhealthy',
            'timestamp': datetime.now().isoformat(),
            'health_check': health_status,
            'performance_metrics': performance_metrics
        }), 200 if health_status['status'] == 'healthy' else 503
        
    except Exception as e:
        logger.error(f"Error in health check: {e}")
        return jsonify({
            'status': 'error',
            'timestamp': datetime.now().isoformat(),
            'error': str(e)
        }), 500


@app.route('/token-usage', methods=['GET'])
def get_token_usage():
    """
    Get token usage statistics and analytics.
    """
    try:
        from token_tracker import get_token_usage_summary, get_daily_token_usage
        
        # Get query parameters
        days = request.args.get('days', 30, type=int)
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        
        # Convert date strings to datetime objects if provided
        if start_date:
            start_date = datetime.fromisoformat(start_date)
        if end_date:
            end_date = datetime.fromisoformat(end_date)
        
        # Get usage summary
        usage_summary = get_token_usage_summary(start_date, end_date)
        
        # Get daily usage
        daily_usage = get_daily_token_usage(days)
        
        return jsonify({
            'message': 'Token usage statistics retrieved successfully',
            'summary': usage_summary,
            'daily_usage': daily_usage,
            'parameters': {
                'days': days,
                'start_date': start_date.isoformat() if start_date else None,
                'end_date': end_date.isoformat() if end_date else None
            },
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting token usage: {e}")
        return jsonify({
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


@app.route('/token-usage/user/<user_id>', methods=['GET'])
def get_user_token_usage(user_id):
    """
    Get token usage statistics for a specific user.
    """
    try:
        from token_tracker import get_token_tracker
        
        tracker = get_token_tracker()
        days = request.args.get('days', 30, type=int)
        
        user_usage = tracker.get_user_usage(user_id, days)
        
        return jsonify({
            'message': f'Token usage statistics for user {user_id}',
            'user_usage': user_usage,
            'parameters': {
                'user_id': user_id,
                'days': days
            },
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting user token usage: {e}")
        return jsonify({
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


@app.route('/token-usage/cost-analysis', methods=['GET'])
def get_cost_analysis():
    """
    Get detailed cost analysis and optimization recommendations.
    """
    try:
        from token_tracker import get_token_tracker
        
        tracker = get_token_tracker()
        days = request.args.get('days', 30, type=int)
        
        # Get usage summary
        usage_summary = tracker.get_usage_summary()
        
        # Calculate cost analysis
        total_cost = usage_summary.get('totals', {}).get('total_cost_usd', 0)
        total_tokens = usage_summary.get('totals', {}).get('total_tokens', 0)
        
        # Cost per token
        cost_per_token = total_cost / total_tokens if total_tokens > 0 else 0
        
        # Cost by operation type
        operation_costs = {}
        for item in usage_summary.get('summary', []):
            op_type = item['operation_type']
            if op_type not in operation_costs:
                operation_costs[op_type] = {
                    'total_cost': 0,
                    'total_tokens': 0,
                    'operation_count': 0
                }
            operation_costs[op_type]['total_cost'] += item['total_cost_usd']
            operation_costs[op_type]['total_tokens'] += item['total_tokens']
            operation_costs[op_type]['operation_count'] += item['operation_count']
        
        # Calculate cost per operation
        for op_type, data in operation_costs.items():
            data['cost_per_operation'] = data['total_cost'] / data['operation_count'] if data['operation_count'] > 0 else 0
            data['cost_per_token'] = data['total_cost'] / data['total_tokens'] if data['total_tokens'] > 0 else 0
        
        # Optimization recommendations
        recommendations = []
        if total_cost > 100:  # If cost > $100
            recommendations.append("Consider implementing caching for frequent queries")
        if operation_costs.get('embedding', {}).get('total_cost', 0) > total_cost * 0.5:
            recommendations.append("High embedding costs - consider optimizing chunk sizes")
        if operation_costs.get('rag_output', {}).get('cost_per_token', 0) > 0.01:
            recommendations.append("High RAG output costs - consider using more efficient models")
        
        return jsonify({
            'message': 'Cost analysis completed successfully',
            'cost_analysis': {
                'total_cost_usd': round(total_cost, 6),
                'total_tokens': total_tokens,
                'cost_per_token': round(cost_per_token, 8),
                'operation_costs': operation_costs,
                'recommendations': recommendations
            },
            'usage_summary': usage_summary,
            'parameters': {
                'days': days
            },
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting cost analysis: {e}")
        return jsonify({
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


@app.route('/metrics', methods=['GET'])
def get_metrics():
    """
    Metrics endpoint for production monitoring.
    Returns detailed performance metrics.
    """
    try:
        from production_monitoring import get_performance_metrics
        
        metrics = get_performance_metrics()
        
        return jsonify({
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        return jsonify({
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


@app.route('/compare', methods=['POST'])
def compare_data():
    """
    Simplified endpoint that performs Pinecone search only.
    Returns search results without MySQL operations or semantic similarity checks.
    """
    try:
        # Validate request
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
        
        data = request.get_json()
        if 'query' not in data:
            return jsonify({'error': 'Missing query field'}), 400
        
        user_query = data['query'].strip()
        if not user_query:
            return jsonify({'error': 'Query cannot be empty'}), 400
        
        logger.info(f"Processing Pinecone-only search: {user_query}")
        
        # Security check for sensitive queries
        if SecurityFilter.is_sensitive_query(user_query):
            logger.warning(f"Sensitive query detected in compare endpoint: {user_query}")
            return jsonify({
                'message': 'Security filter activated',
                'user_query': user_query,
                'search_results': [],
                'metadata_analysis': {'security_filter_applied': True},
                'timestamp': datetime.now().isoformat()
            }), 200
        
        # Initialize components
        enhanced_search_manager = EnhancedSearchManager()
        
        # Connect to Pinecone
        if not enhanced_search_manager.connect_to_index():
            return jsonify({'error': 'Failed to connect to Pinecone index'}), 500
        
        # Perform enhanced search with intelligent query analysis and reranking
        logger.info("Using enhanced search for Pinecone-only comparison")
        similar_vectors = enhanced_search_manager.enhanced_search(
            user_query, 
            top_k=10, 
            use_reranking=True
        )
        
        if not similar_vectors:
            return jsonify({
                'message': 'No similar regulations found in Pinecone embeddings',
                'search_results': [],
                'metadata_analysis': {
                    'query_embedding_generated': True,
                    'pinecone_search_performed': True,
                    'similar_vectors_found': 0
                },
                'timestamp': datetime.now().isoformat()
            }), 200
        
        # Process Pinecone results
        search_results = []
        metadata_analysis = {
            'total_vectors_found': len(similar_vectors),
            'regulators_found': set(),
            'industries_found': set(),
            'task_categories_found': set(),
            'reg_categories_found': set(),
            'risk_categories_found': set(),
            'departments_found': set()
        }
        
        logger.info(f"Processing {len(similar_vectors)} similar vectors from Pinecone")
        
        for vector_match in similar_vectors:
            try:
                metadata = vector_match['metadata']
                pinecone_score = vector_match['score']
                chunk_index = metadata.get('chunk_index', 0)
                total_chunks = metadata.get('total_chunks', 1)
                chunk_text = metadata.get('chunk_text', '')
                
                # Extract metadata for analysis
                if metadata.get('regulator'):
                    metadata_analysis['regulators_found'].add(metadata['regulator'])
                if metadata.get('industry'):
                    metadata_analysis['industries_found'].add(metadata['industry'])
                if metadata.get('task_category'):
                    metadata_analysis['task_categories_found'].add(metadata['task_category'])
                if metadata.get('reg_category'):
                    metadata_analysis['reg_categories_found'].add(metadata['reg_category'])
                if metadata.get('risk_category'):
                    metadata_analysis['risk_categories_found'].add(metadata['risk_category'])
                if metadata.get('department'):
                    metadata_analysis['departments_found'].add(metadata['department'])
                
                # Build search result
                search_result = {
                    'vector_id': vector_match['id'],
                    'pinecone_similarity_score': pinecone_score,
                    'chunk_index': chunk_index,
                    'total_chunks': total_chunks,
                    'matched_chunk_text': chunk_text[:300] + "..." if len(chunk_text) > 300 else chunk_text,
                    'embedding_match_quality': 'high' if pinecone_score > 0.8 else 'medium' if pinecone_score > 0.6 else 'low',
                    'metadata': metadata,
                    'query_analysis': vector_match.get('query_analysis', {}),
                    'rerank_score': vector_match.get('rerank_score', pinecone_score)
                }
                
                search_results.append(search_result)
                
            except Exception as e:
                logger.error(f"Error processing vector match {vector_match.get('id', 'unknown')}: {e}")
                continue
        
        # Convert sets to lists for JSON serialization
        for key in metadata_analysis:
            if isinstance(metadata_analysis[key], set):
                metadata_analysis[key] = list(metadata_analysis[key])
        
        # Prepare response
        response_data = {
            'message': 'Pinecone search completed successfully',
            'user_query': user_query,
            'processing_summary': {
                'embedding_generation': 'completed',
                'pinecone_search': 'completed',
                'reranking': 'completed'
            },
            'search_analysis': {
                'query_embedding_generated': True,
                'pinecone_search_performed': True,
                'similar_vectors_found': len(similar_vectors),
                'high_quality_matches': len([r for r in search_results if r['embedding_match_quality'] == 'high']),
                'medium_quality_matches': len([r for r in search_results if r['embedding_match_quality'] == 'medium']),
                'low_quality_matches': len([r for r in search_results if r['embedding_match_quality'] == 'low']),
                'average_similarity_score': sum([r['pinecone_similarity_score'] for r in search_results]) / len(search_results) if search_results else 0
            },
            'metadata_analysis': metadata_analysis,
            'search_results': search_results,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Pinecone search completed: {len(search_results)} results found")
        
        return jsonify(response_data), 200
            
    except Exception as e:
        logger.error(f"Error in Pinecone-only compare endpoint: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/search', methods=['POST'])
def metadata_aware_search():
    """
    Metadata-aware search endpoint that leverages comprehensive Pinecone metadata
    for precise regulation retrieval and comparison.
    """
    try:
        # Validate request
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
        
        data = request.get_json()
        if 'query' not in data:
            return jsonify({'error': 'Missing query field'}), 400
        
        user_query = data['query'].strip()
        if not user_query:
            return jsonify({'error': 'Query cannot be empty'}), 400
        
        # Optional metadata filters
        metadata_filters = data.get('filters', {})
        
        logger.info(f"Processing metadata-aware search: {user_query}")
        logger.info(f"Applied filters: {metadata_filters}")
        
        # Security check for sensitive queries
        if SecurityFilter.is_sensitive_query(user_query):
            logger.warning(f"Sensitive query detected in search endpoint: {user_query}")
            return jsonify({
                'message': 'Security filter activated',
                'user_query': user_query,
                'applied_filters': metadata_filters,
                'metadata_analysis': {'security_filter_applied': True},
                'search_results': [],
                'total_results': 0,
                'timestamp': datetime.now().isoformat()
            }), 200
        
        # Initialize components
        db_manager = DatabaseManager({})
        enhanced_search_manager = EnhancedSearchManager()
        
        # Connect to database
        if not db_manager.connect():
            return jsonify({'error': 'Failed to connect to database'}), 500
        
        try:
            # Connect to Pinecone
            if not enhanced_search_manager.connect_to_index():
                return jsonify({'error': 'Failed to connect to Pinecone index'}), 500
            
            # Use enhanced search with intelligent query analysis and reranking
            logger.info("Using enhanced search for metadata-aware search")
            similar_vectors = enhanced_search_manager.enhanced_search(
                user_query, 
                filters=metadata_filters,
                top_k=10, 
                use_reranking=True
            )
            
            if not similar_vectors:
                return jsonify({
                    'message': 'No similar regulations found in Pinecone',
                    'search_results': [],
                    'metadata_analysis': {},
                    'timestamp': datetime.now().isoformat()
                }), 200
            
            # Filter results based on metadata criteria
            filtered_results = []
            metadata_analysis = {
                'total_vectors_found': len(similar_vectors),
                'regulators_found': set(),
                'industries_found': set(),
                'task_categories_found': set(),
                'reg_categories_found': set(),
                'risk_categories_found': set(),
                'departments_found': set()
            }
            
            for vector_match in similar_vectors:
                try:
                    metadata = vector_match['metadata']
                    
                    # Collect metadata for analysis
                    if metadata.get('regulator'):
                        metadata_analysis['regulators_found'].add(metadata['regulator'])
                    if metadata.get('industry'):
                        metadata_analysis['industries_found'].add(metadata['industry'])
                    if metadata.get('task_category'):
                        metadata_analysis['task_categories_found'].add(metadata['task_category'])
                    if metadata.get('reg_category'):
                        metadata_analysis['reg_categories_found'].add(metadata['reg_category'])
                    if metadata.get('risk_category'):
                        metadata_analysis['risk_categories_found'].add(metadata['risk_category'])
                    if metadata.get('department'):
                        metadata_analysis['departments_found'].add(metadata['department'])
                    
                    # Apply metadata filters if provided
                    passes_filter = True
                    if metadata_filters:
                        for filter_key, filter_value in metadata_filters.items():
                            if filter_key in metadata:
                                if isinstance(filter_value, list):
                                    # Multiple values allowed
                                    if metadata[filter_key] not in filter_value:
                                        passes_filter = False
                                        break
                                else:
                                    # Single value
                                    if metadata[filter_key] != filter_value:
                                        passes_filter = False
                                        break
                    
                    if passes_filter:
                        # Fetch full regulation from MySQL
                        row_id = metadata.get('row_id')
                        if row_id:
                            regulation = db_manager.get_regulation_by_id(row_id)
                            if regulation:
                                filtered_results.append({
                                    'vector_match': vector_match,
                                    'regulation_data': regulation,
                                    'metadata': metadata
                                })
                
                except Exception as e:
                    logger.error(f"Error processing vector match: {e}")
                    continue
            
            # Convert sets to lists for JSON serialization
            for key in metadata_analysis:
                if isinstance(metadata_analysis[key], set):
                    metadata_analysis[key] = list(metadata_analysis[key])
            
            metadata_analysis['filtered_results_count'] = len(filtered_results)
            
            # Prepare search results
            search_results = []
            for result in filtered_results:
                vector_match = result['vector_match']
                regulation = result['regulation_data']
                metadata = result['metadata']
                
                search_result = {
                    'regulation_id': regulation.get('id'),
                    'regulation_title': regulation.get('regulation', ''),
                    'regulator': regulation.get('regulator', ''),
                    'industry': regulation.get('industry', ''),
                    'sub_industry': regulation.get('sub_industry', ''),
                    'task_category': regulation.get('task_category', ''),
                    'task_subcategory': regulation.get('task_subcategory', ''),
                    'reg_number': regulation.get('reg_number', ''),
                    'reg_date': regulation.get('reg_date', ''),
                    'reg_category': regulation.get('reg_category', ''),
                    'reg_subject': regulation.get('reg_subject', ''),
                    'due_date': regulation.get('due_date', ''),
                    'frequency': regulation.get('frequency', ''),
                    'AI_Match': regulation.get('AI_Match', 'Pending'),
                    'notes': regulation.get('notes', ''),
                    'risk_category': regulation.get('risk_category', ''),
                    'control_nature': regulation.get('control_nature', ''),
                    'department': regulation.get('department', ''),
                    'summary': regulation.get('summary', ''),
                    'action_item': regulation.get('action_item', ''),
                    'pinecone_score': vector_match['score'],
                    'chunk_index': metadata.get('chunk_index'),
                    'matched_chunk_text': metadata.get('chunk_text', ''),
                    'metadata_fields': list(metadata.keys())
                }
                
                search_results.append(search_result)
            
            # Sort by Pinecone score (highest first)
            search_results.sort(key=lambda x: x['pinecone_score'], reverse=True)
            
            response_data = {
                'message': 'Metadata-aware search completed successfully',
                'user_query': user_query,
                'applied_filters': metadata_filters,
                'metadata_analysis': metadata_analysis,
                'search_results': search_results,
                'total_results': len(search_results),
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Metadata-aware search completed: {len(search_results)} results found")
            
            return jsonify(response_data), 200
            
        finally:
            db_manager.disconnect()
            
    except Exception as e:
        logger.error(f"Error in metadata-aware search endpoint: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=Config.CHAT_API_PORT, debug=Config.FLASK_DEBUG)
