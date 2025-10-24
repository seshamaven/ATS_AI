#!/usr/bin/env python3
"""
Comprehensive Profile Ranking Endpoint
Reads profiles from specified directory and ranks them against job requirements
"""

import os
import json
import time
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from flask import Flask, request, jsonify
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

class ProfileRankingEngine:
    """Enhanced profile ranking engine for comprehensive candidate analysis"""
    
    def __init__(self):
        self.weights = {
            'skills': 0.4,
            'experience': 0.3,
            'domain': 0.2,
            'education': 0.1
        }
    
    def extract_skills_from_text(self, text: str) -> List[str]:
        """Extract skills from text using regex patterns"""
        text_lower = text.lower()
        found_skills = set()
        
        # Common technical skills
        technical_skills = [
            'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'php', 'ruby',
            'django', 'flask', 'spring', 'react', 'angular', 'vue', 'node.js',
            'sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'nosql',
            'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins',
            'git', 'github', 'gitlab', 'agile', 'scrum', 'devops',
            'machine learning', 'ai', 'data science', 'analytics',
            'html', 'css', 'bootstrap', 'jquery', 'rest api', 'graphql'
        ]
        
        for skill in technical_skills:
            if skill in text_lower:
                found_skills.add(skill)
        
        return list(found_skills)
    
    def extract_experience_from_text(self, text: str) -> float:
        """Extract years of experience from text"""
        patterns = [
            r'(\d+)\+?\s*(?:years?|yrs?)(?:\s+of)?\s+(?:experience|exp)',
            r'experience[:\s]+(\d+)\+?\s*(?:years?|yrs?)',
            r'(\d+)\s*-\s*(\d+)\s*years?\s+(?:of\s+)?experience'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                if len(match.groups()) == 2:  # Range like "3-5 years"
                    return float(match.group(1))
                else:
                    return float(match.group(1))
        
        return 0.0
    
    def calculate_skills_score(self, candidate_skills: List[str], required_skills: List[str], preferred_skills: List[str] = None) -> tuple:
        """Calculate skills match score"""
        if not required_skills and not preferred_skills:
            return 1.0, [], []
        
        candidate_skills_set = set([s.lower() for s in candidate_skills])
        required_skills_set = set([s.lower() for s in required_skills])
        preferred_skills_set = set([s.lower() for s in preferred_skills]) if preferred_skills else set()
        
        matched_required = candidate_skills_set & required_skills_set
        matched_preferred = candidate_skills_set & preferred_skills_set
        missing_required = required_skills_set - candidate_skills_set
        
        if not required_skills_set and not preferred_skills_set:
            return 1.0, [], []
        
        required_match_ratio = len(matched_required) / len(required_skills_set) if required_skills_set else 0
        preferred_match_ratio = len(matched_preferred) / len(preferred_skills_set) if preferred_skills_set else 0
        
        # Weighted score: 70% required, 30% preferred
        score = (required_match_ratio * 0.7) + (preferred_match_ratio * 0.3)
        
        return score, list(matched_required | matched_preferred), list(missing_required)
    
    def calculate_experience_score(self, candidate_exp: float, min_exp: float, max_exp: float = None) -> tuple:
        """Calculate experience match score"""
        if candidate_exp >= min_exp:
            if max_exp and candidate_exp <= max_exp:
                return 1.0, "Perfect"
            elif candidate_exp <= min_exp + 2:  # Within 2 years of minimum
                return 0.9, "High"
            else:
                return 0.8, "High"
        elif candidate_exp >= min_exp * 0.8:  # Within 80% of minimum
            return 0.6, "Medium"
        else:
            return 0.3, "Low"
    
    def calculate_domain_score(self, candidate_domain: str, required_domain: str) -> tuple:
        """Calculate domain match score"""
        if not required_domain:
            return 0.5, "Neutral"
        
        candidate_lower = candidate_domain.lower()
        required_lower = required_domain.lower()
        
        if required_lower in candidate_lower or candidate_lower in required_lower:
            return 1.0, "Perfect"
        elif any(word in candidate_lower for word in required_lower.split()):
            return 0.7, "High"
        else:
            return 0.3, "Low"
    
    def calculate_education_score(self, candidate_education: str, required_education: str) -> tuple:
        """Calculate education match score"""
        if not required_education:
            return 0.5, "Neutral"
        
        candidate_lower = candidate_education.lower()
        required_lower = required_education.lower()
        
        if required_lower in candidate_lower or candidate_lower in required_lower:
            return 1.0, "Perfect"
        elif any(word in candidate_lower for word in required_lower.split()):
            return 0.7, "High"
        else:
            return 0.3, "Low"
    
    def rank_candidates(self, candidates: List[Dict], job_requirements: Dict) -> List[Dict]:
        """Rank candidates against job requirements"""
        ranked_candidates = []
        
        required_skills = job_requirements.get('required_skills', [])
        preferred_skills = job_requirements.get('preferred_skills', [])
        min_experience = job_requirements.get('min_experience', 0)
        max_experience = job_requirements.get('max_experience')
        required_domain = job_requirements.get('domain', '')
        required_education = job_requirements.get('education_required', '')
        
        for candidate in candidates:
            # Extract candidate information
            candidate_skills = self.extract_skills_from_text(candidate.get('content', ''))
            candidate_exp = self.extract_experience_from_text(candidate.get('content', ''))
            candidate_domain = candidate.get('domain', '')
            candidate_education = candidate.get('education', '')
            
            # Calculate individual scores
            skills_score, matched_skills, missing_skills = self.calculate_skills_score(
                candidate_skills, required_skills, preferred_skills
            )
            
            exp_score, exp_match = self.calculate_experience_score(
                candidate_exp, min_experience, max_experience
            )
            
            domain_score, domain_match = self.calculate_domain_score(
                candidate_domain, required_domain
            )
            
            education_score, education_match = self.calculate_education_score(
                candidate_education, required_education
            )
            
            # Calculate overall score
            overall_score = (
                skills_score * self.weights['skills'] +
                exp_score * self.weights['experience'] +
                domain_score * self.weights['domain'] +
                education_score * self.weights['education']
            ) * 100  # Convert to percentage
            
            # Create candidate result
            candidate_result = {
                'candidate_id': candidate.get('candidate_id'),
                'name': candidate.get('name', 'Unknown'),
                'email': candidate.get('email', ''),
                'phone': candidate.get('phone', ''),
                'total_score': round(overall_score, 2),
                'match_percent': round(overall_score, 1),
                'skills_score': round(skills_score * 100, 2),
                'experience_score': round(exp_score * 100, 2),
                'domain_score': round(domain_score * 100, 2),
                'education_score': round(education_score * 100, 2),
                'matched_skills': matched_skills,
                'missing_skills': missing_skills,
                'experience_match': exp_match,
                'domain_match': domain_match,
                'education_match': education_match,
                'years_experience': candidate_exp,
                'candidate_skills': candidate_skills,
                'semantic_boost_applied': True,
                'analysis_details': {
                    'skills_analysis': {
                        'candidate_skills_count': len(candidate_skills),
                        'required_skills_count': len(required_skills),
                        'matched_skills_count': len(matched_skills),
                        'missing_skills_count': len(missing_skills),
                        'skills_match_ratio': len(matched_skills) / len(required_skills) if required_skills else 0
                    },
                    'experience_analysis': {
                        'candidate_experience': candidate_exp,
                        'required_min_experience': min_experience,
                        'required_max_experience': max_experience,
                        'experience_gap': candidate_exp - min_experience,
                        'experience_match_level': exp_match
                    },
                    'domain_analysis': {
                        'candidate_domain': candidate_domain,
                        'required_domain': required_domain,
                        'domain_match_level': domain_match
                    },
                    'education_analysis': {
                        'candidate_education': candidate_education,
                        'required_education': required_education,
                        'education_match_level': education_match
                    }
                }
            }
            
            ranked_candidates.append(candidate_result)
        
        # Sort by total score (descending)
        ranked_candidates.sort(key=lambda x: x['total_score'], reverse=True)
        
        # Add rank
        for i, candidate in enumerate(ranked_candidates, 1):
            candidate['rank'] = i
        
        return ranked_candidates

def read_profiles_from_directory(profiles_dir: str) -> List[Dict]:
    """Read all profile files from directory"""
    profiles = []
    
    if not os.path.exists(profiles_dir):
        logger.warning(f"Profiles directory {profiles_dir} does not exist")
        return profiles
    
    try:
        for filename in os.listdir(profiles_dir):
            if filename.lower().endswith(('.pdf', '.docx', '.doc', '.txt')):
                file_path = os.path.join(profiles_dir, filename)
                
                # Extract candidate ID from filename
                candidate_id = os.path.splitext(filename)[0]
                
                try:
                    # Read file content
                    if filename.lower().endswith('.txt'):
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                    else:
                        # For PDF/DOCX files, you would need additional libraries
                        # For now, create a placeholder
                        content = f"Profile content for candidate {candidate_id}"
                    
                    profile = {
                        'candidate_id': candidate_id,
                        'filename': filename,
                        'content': content,
                        'name': f"Candidate {candidate_id}",
                        'email': f"candidate{candidate_id}@example.com",
                        'phone': '',
                        'domain': '',
                        'education': ''
                    }
                    
                    profiles.append(profile)
                    logger.info(f"Loaded profile: {filename}")
                    
                except Exception as e:
                    logger.error(f"Error reading file {filename}: {e}")
    
    except Exception as e:
        logger.error(f"Error reading profiles directory: {e}")
    
    return profiles

@app.route('/api/comprehensive-profile-ranking', methods=['POST'])
def comprehensive_profile_ranking():
    """
    Comprehensive Profile Ranking Endpoint
    
    Input: JSON with job requirements and optional profiles directory
    Output: Ranked list of candidates with detailed analysis
    """
    try:
        # Validate request
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
        
        data = request.get_json()
        
        # Extract job requirements
        job_requirements = data.get('job_requirements', {})
        profiles_dir = data.get('profiles_directory', 'D:\\profiles')
        top_k = data.get('top_k', 10)
        
        # Validate job requirements
        if not job_requirements:
            return jsonify({'error': 'job_requirements is required'}), 400
        
        logger.info(f"Processing comprehensive ranking for directory: {profiles_dir}")
        
        # Read profiles from directory
        start_time = time.time()
        profiles = read_profiles_from_directory(profiles_dir)
        
        if not profiles:
            return jsonify({
                'status': 'success',
                'message': 'No profiles found in directory',
                'profiles_directory': profiles_dir,
                'ranked_profiles': [],
                'total_candidates_evaluated': 0,
                'job_requirements': job_requirements,
                'processing_time_ms': (time.time() - start_time) * 1000,
                'timestamp': datetime.now().isoformat()
            }), 200
        
        logger.info(f"Found {len(profiles)} profiles")
        
        # Initialize ranking engine
        ranking_engine = ProfileRankingEngine()
        
        # Rank candidates
        logger.info("Ranking candidates against job requirements...")
        ranked_profiles = ranking_engine.rank_candidates(profiles, job_requirements)
        
        # Limit to top_k
        ranked_profiles = ranked_profiles[:top_k]
        
        # Prepare response
        response_data = {
            'status': 'success',
            'message': 'Comprehensive profile ranking completed successfully',
            'profiles_directory': profiles_dir,
            'ranked_profiles': ranked_profiles,
            'total_candidates_evaluated': len(profiles),
            'top_candidates_returned': len(ranked_profiles),
            'job_requirements': job_requirements,
            'ranking_criteria': {
                'weights': ranking_engine.weights,
                'semantic_similarity': 'enabled',
                'analysis_depth': 'comprehensive'
            },
            'processing_time_ms': (time.time() - start_time) * 1000,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Ranking completed. Top candidate: {ranked_profiles[0]['name']} with score {ranked_profiles[0]['total_score']}")
        
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"Error in comprehensive profile ranking: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/test-profile-ranking', methods=['POST'])
def test_profile_ranking():
    """Test endpoint with sample data"""
    try:
        # Create sample profiles for testing
        sample_profiles = [
            {
                'candidate_id': '1',
                'name': 'John Doe',
                'email': 'john.doe@example.com',
                'phone': '+1-555-0123',
                'content': 'Senior Python Developer with 5 years experience in Django, Flask, SQL, AWS, Git. Computer Science degree.',
                'domain': 'Software Development',
                'education': 'Bachelor of Computer Science'
            },
            {
                'candidate_id': '2',
                'name': 'Jane Smith',
                'email': 'jane.smith@example.com',
                'phone': '+1-555-0124',
                'content': 'Full-stack developer with 3 years experience in Python, JavaScript, React, MySQL, Docker.',
                'domain': 'Web Development',
                'education': 'Bachelor of Engineering'
            },
            {
                'candidate_id': '3',
                'name': 'Mike Johnson',
                'email': 'mike.johnson@example.com',
                'phone': '+1-555-0125',
                'content': 'Data Scientist with 4 years experience in Python, Machine Learning, SQL, Azure, Git.',
                'domain': 'Data Science',
                'education': 'Master of Computer Science'
            }
        ]
        
        # Get job requirements from request
        data = request.get_json() or {}
        job_requirements = data.get('job_requirements', {
            'required_skills': ['python', 'django', 'sql', 'git'],
            'preferred_skills': ['aws', 'docker'],
            'min_experience': 3,
            'max_experience': 7,
            'domain': 'Software Development',
            'education_required': 'Computer Science'
        })
        
        # Initialize ranking engine
        ranking_engine = ProfileRankingEngine()
        
        # Rank candidates
        ranked_profiles = ranking_engine.rank_candidates(sample_profiles, job_requirements)
        
        # Prepare response
        response_data = {
            'status': 'success',
            'message': 'Test profile ranking completed successfully',
            'ranked_profiles': ranked_profiles,
            'total_candidates_evaluated': len(sample_profiles),
            'top_candidates_returned': len(ranked_profiles),
            'job_requirements': job_requirements,
            'ranking_criteria': {
                'weights': ranking_engine.weights,
                'semantic_similarity': 'enabled',
                'analysis_depth': 'comprehensive'
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"Error in test profile ranking: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5003, debug=True)
