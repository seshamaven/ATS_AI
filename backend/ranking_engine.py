"""
Ranking Engine for ATS System.
Implements weighted scoring algorithm to rank candidates against job descriptions.
"""

import logging
import numpy as np
from typing import Dict, List, Any, Tuple
import re
from ats_config import ATSConfig

logger = logging.getLogger(__name__)


class ProfileRankingEngine:
    """
    Intelligent ranking engine that scores candidates against job requirements.
    Uses weighted scoring: Skills (40%), Experience (30%), Domain (20%), Education (10%)
    """
    
    def __init__(self, weights: Dict[str, float] = None):
        """
        Initialize ranking engine with custom or default weights.
        
        Args:
            weights: Dictionary with keys: skills, experience, domain, education
        """
        self.weights = weights or ATSConfig.RANKING_WEIGHTS
        
        # Validate weights
        total_weight = sum(self.weights.values())
        if abs(total_weight - 1.0) > 0.01:
            logger.warning(f"Weights sum to {total_weight}, normalizing to 1.0")
            self.weights = {k: v / total_weight for k, v in self.weights.items()}
        
        logger.info(f"Initialized ranking engine with weights: {self.weights}")
    
    def calculate_cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            vec1 = np.array(vec1)
            vec2 = np.array(vec2)
            
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {e}")
            return 0.0
    
    def extract_skills_list(self, skills_text: str) -> List[str]:
        """Extract and normalize skills from comma-separated text."""
        if not skills_text:
            return []
        
        skills = [s.strip().lower() for s in skills_text.split(',')]
        return [s for s in skills if s]
    
    def calculate_skills_score(
        self,
        candidate_skills: str,
        required_skills: str,
        preferred_skills: str = None
    ) -> Tuple[float, List[str], List[str]]:
        """
        Calculate skills match score.
        
        Args:
            candidate_skills: Comma-separated candidate skills
            required_skills: Comma-separated required skills
            preferred_skills: Comma-separated preferred skills (optional)
        
        Returns:
            (score, matched_skills, missing_skills)
        """
        candidate_skills_list = set(self.extract_skills_list(candidate_skills))
        required_skills_list = set(self.extract_skills_list(required_skills))
        preferred_skills_list = set(self.extract_skills_list(preferred_skills)) if preferred_skills else set()
        
        if not required_skills_list and not preferred_skills_list:
            return 1.0, [], []
        
        # Calculate matches
        matched_required = candidate_skills_list & required_skills_list
        matched_preferred = candidate_skills_list & preferred_skills_list
        
        # Calculate missing skills
        missing_required = required_skills_list - candidate_skills_list
        
        # Scoring logic
        required_match_ratio = len(matched_required) / len(required_skills_list) if required_skills_list else 0
        preferred_match_ratio = len(matched_preferred) / len(preferred_skills_list) if preferred_skills_list else 0
        
        # Weighted score: 70% required, 30% preferred
        if preferred_skills_list:
            score = (0.7 * required_match_ratio) + (0.3 * preferred_match_ratio)
        else:
            score = required_match_ratio
        
        matched_skills = list(matched_required | matched_preferred)
        missing_skills = list(missing_required)
        
        return score, matched_skills, missing_skills
    
    def calculate_experience_score(
        self,
        candidate_experience: float,
        min_experience: float,
        max_experience: float = None
    ) -> Tuple[float, str]:
        """
        Calculate experience match score.
        
        Args:
            candidate_experience: Candidate's total years of experience
            min_experience: Minimum required experience
            max_experience: Maximum required experience (optional)
        
        Returns:
            (score, match_level) where match_level is 'High', 'Medium', or 'Low'
        """
        if candidate_experience < 0:
            candidate_experience = 0
        
        # Perfect match if within range
        if max_experience:
            if min_experience <= candidate_experience <= max_experience:
                return 1.0, 'High'
            elif candidate_experience < min_experience:
                # Under-qualified
                gap = min_experience - candidate_experience
                if gap <= 1:
                    return 0.8, 'Medium'
                elif gap <= 2:
                    return 0.6, 'Medium'
                else:
                    return 0.3, 'Low'
            else:
                # Over-qualified
                excess = candidate_experience - max_experience
                if excess <= 2:
                    return 0.9, 'High'
                elif excess <= 5:
                    return 0.7, 'Medium'
                else:
                    return 0.5, 'Medium'
        else:
            # Only minimum experience specified
            if candidate_experience >= min_experience:
                # Calculate score based on how much they exceed minimum
                excess = candidate_experience - min_experience
                if excess <= 2:
                    return 1.0, 'High'
                elif excess <= 5:
                    return 0.9, 'High'
                else:
                    return 0.8, 'High'
            else:
                # Below minimum
                gap = min_experience - candidate_experience
                if gap <= 0.5:
                    return 0.8, 'Medium'
                elif gap <= 1:
                    return 0.6, 'Medium'
                elif gap <= 2:
                    return 0.4, 'Low'
                else:
                    return 0.2, 'Low'
    
    def calculate_domain_score(
        self,
        candidate_domain: str,
        required_domain: str
    ) -> Tuple[float, str]:
        """
        Calculate domain/industry match score.
        
        Args:
            candidate_domain: Candidate's domain/industry
            required_domain: Required domain/industry
        
        Returns:
            (score, match_level) where match_level is 'High', 'Medium', or 'Low'
        """
        if not candidate_domain or not required_domain:
            return 0.5, 'Medium'  # Neutral score if domain not specified
        
        candidate_domain_lower = candidate_domain.lower().strip()
        required_domain_lower = required_domain.lower().strip()
        
        # Exact match
        if candidate_domain_lower == required_domain_lower:
            return 1.0, 'High'
        
        # Partial match (one contains the other)
        if candidate_domain_lower in required_domain_lower or required_domain_lower in candidate_domain_lower:
            return 0.8, 'High'
        
        # Check for related domains
        related_domains = {
            'finance': ['banking', 'fintech', 'financial services', 'insurance'],
            'banking': ['finance', 'fintech', 'financial services'],
            'fintech': ['finance', 'banking', 'technology'],
            'technology': ['software', 'it', 'tech', 'saas'],
            'healthcare': ['medical', 'pharma', 'health', 'hospital'],
            'retail': ['e-commerce', 'ecommerce', 'commerce', 'sales'],
        }
        
        # Check if domains are related
        for domain_group, related in related_domains.items():
            if candidate_domain_lower in related or candidate_domain_lower == domain_group:
                if required_domain_lower in related or required_domain_lower == domain_group:
                    return 0.7, 'Medium'
        
        # No match
        return 0.3, 'Low'
    
    def calculate_education_score(
        self,
        candidate_education: str,
        required_education: str
    ) -> float:
        """
        Calculate education match score.
        
        Args:
            candidate_education: Candidate's education level
            required_education: Required education level
        
        Returns:
            score (0-1)
        """
        if not required_education:
            return 1.0  # No specific requirement
        
        if not candidate_education:
            return 0.3  # No education info
        
        # Education hierarchy
        education_levels = {
            'phd': 5,
            'doctorate': 5,
            'masters': 4,
            'master': 4,
            'mba': 4,
            'bachelors': 3,
            'bachelor': 3,
            'diploma': 2,
            'high school': 1,
            'secondary': 1
        }
        
        candidate_level = 0
        required_level = 0
        
        candidate_edu_lower = candidate_education.lower()
        required_edu_lower = required_education.lower()
        
        # Find education levels
        for key, level in education_levels.items():
            if key in candidate_edu_lower:
                candidate_level = max(candidate_level, level)
            if key in required_edu_lower:
                required_level = max(required_level, level)
        
        # If no match found, default to neutral
        if candidate_level == 0 or required_level == 0:
            return 0.5
        
        # Calculate score
        if candidate_level >= required_level:
            return 1.0
        elif candidate_level == required_level - 1:
            return 0.7
        else:
            return 0.4
    
    def rank_candidate(
        self,
        candidate: Dict[str, Any],
        job_requirements: Dict[str, Any],
        jd_embedding: List[float] = None
    ) -> Dict[str, Any]:
        """
        Rank a single candidate against job requirements.
        
        Args:
            candidate: Candidate data with skills, experience, domain, education, embedding
            job_requirements: Job requirements with required skills, experience, domain, education
            jd_embedding: Job description embedding for semantic similarity (optional)
        
        Returns:
            Dictionary with scores and ranking details
        """
        # Calculate individual scores
        # Handle skills as either strings or lists
        primary_skills = candidate.get('primary_skills', '')
        secondary_skills = candidate.get('secondary_skills', '')
        
        # Convert lists to strings if needed
        if isinstance(primary_skills, list):
            primary_skills = ', '.join(primary_skills)
        if isinstance(secondary_skills, list):
            secondary_skills = ', '.join(secondary_skills)
        
        # Combine skills
        all_skills = primary_skills
        if secondary_skills:
            if all_skills:
                all_skills += ', ' + secondary_skills
            else:
                all_skills = secondary_skills
        
        skills_score, matched_skills, missing_skills = self.calculate_skills_score(
            all_skills,
            job_requirements.get('required_skills', ''),
            job_requirements.get('preferred_skills', '')
        )
        
        experience_score, experience_match = self.calculate_experience_score(
            candidate.get('total_experience', 0),
            job_requirements.get('min_experience', 0),
            job_requirements.get('max_experience')
        )
        
        domain_score, domain_match = self.calculate_domain_score(
            candidate.get('domain', ''),
            job_requirements.get('domain', '')
        )
        
        education_score = self.calculate_education_score(
            candidate.get('education', ''),
            job_requirements.get('education_required', '')
        )
        
        # Calculate weighted total score
        total_score = (
            skills_score * self.weights['skills'] +
            experience_score * self.weights['experience'] +
            domain_score * self.weights['domain'] +
            education_score * self.weights['education']
        )
        
        # Boost score if semantic similarity is available
        semantic_boost = 0.0
        if jd_embedding and candidate.get('embedding'):
            semantic_similarity = self.calculate_cosine_similarity(
                candidate['embedding'],
                jd_embedding
            )
            # Small boost for high semantic similarity (max 5% boost)
            if semantic_similarity > 0.8:
                semantic_boost = 0.05
            elif semantic_similarity > 0.7:
                semantic_boost = 0.03
        
        total_score = min(1.0, total_score + semantic_boost)
        
        # Convert to 0-100 scale
        total_score_100 = total_score * 100
        match_percent = total_score * 100
        
        return {
            'candidate_id': candidate.get('candidate_id'),
            'name': candidate.get('name'),
            'email': candidate.get('email', ''),
            'total_score': round(total_score_100, 2),
            'match_percent': round(match_percent, 1),
            'skills_score': round(skills_score * 100, 2),
            'experience_score': round(experience_score * 100, 2),
            'domain_score': round(domain_score * 100, 2),
            'education_score': round(education_score * 100, 2),
            'matched_skills': matched_skills,
            'missing_skills': missing_skills,
            'experience_match': experience_match,
            'domain_match': domain_match,
            'semantic_boost_applied': semantic_boost > 0
        }
    
    def rank_candidates(
        self,
        candidates: List[Dict[str, Any]],
        job_requirements: Dict[str, Any],
        jd_embedding: List[float] = None,
        top_k: int = None
    ) -> List[Dict[str, Any]]:
        """
        Rank multiple candidates against job requirements.
        
        Args:
            candidates: List of candidate dictionaries
            job_requirements: Job requirements dictionary
            jd_embedding: Job description embedding (optional)
            top_k: Return only top K candidates (optional)
        
        Returns:
            List of ranked candidates with scores, sorted by total_score descending
        """
        logger.info(f"Ranking {len(candidates)} candidates")
        
        ranked_candidates = []
        
        for candidate in candidates:
            try:
                ranking = self.rank_candidate(candidate, job_requirements, jd_embedding)
                ranked_candidates.append(ranking)
            except Exception as e:
                logger.error(f"Error ranking candidate {candidate.get('candidate_id')}: {e}")
                continue
        
        # Sort by total score (descending)
        ranked_candidates.sort(key=lambda x: x['total_score'], reverse=True)
        
        # Assign rank positions
        for rank, candidate in enumerate(ranked_candidates, start=1):
            candidate['rank'] = rank
        
        # Return top K if specified
        if top_k:
            ranked_candidates = ranked_candidates[:top_k]
        
        if ranked_candidates:
            logger.info(f"Ranking complete. Top candidate score: {ranked_candidates[0]['total_score']:.2f}")
        else:
            logger.info("Ranking complete. No candidates ranked.")
        
        return ranked_candidates


def create_ranking_engine(weights: Dict[str, float] = None) -> ProfileRankingEngine:
    """Factory function to create ranking engine instance."""
    return ProfileRankingEngine(weights)

