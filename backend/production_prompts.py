"""
Production-Grade Prompt System for Regulatory Circulars RAG
Handles query classification, relevance validation, and response generation with comprehensive safety checks.
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class QueryRelevance(Enum):
    """Query relevance classification for regulatory queries."""
    HIGHLY_RELEVANT = "highly_relevant"
    RELEVANT = "relevant"
    PARTIALLY_RELEVANT = "partially_relevant"
    IRRELEVANT = "irrelevant"
    POTENTIALLY_HARMFUL = "potentially_harmful"


class RegulatoryDomain(Enum):
    """Regulatory domain classification."""
    BANKING = "banking"
    SECURITIES = "securities"
    INSURANCE = "insurance"
    FINANCIAL_SERVICES = "financial_services"
    COMPLIANCE = "compliance"
    RISK_MANAGEMENT = "risk_management"
    ANTI_MONEY_LAUNDERING = "anti_money_laundering"
    CYBERSECURITY = "cybersecurity"
    DATA_PROTECTION = "data_protection"
    GENERAL_REGULATORY = "general_regulatory"


class QueryClassifier:
    """Classifies queries for regulatory relevance and domain."""
    
    def __init__(self):
        # Regulatory domain keywords
        self.domain_keywords = {
            RegulatoryDomain.BANKING: [
                "bank", "banking", "credit", "lending", "deposit", "loan", "mortgage",
                "rbi", "reserve bank", "central bank", "monetary policy", "interest rate",
                "capital adequacy", "liquidity", "credit risk", "operational risk"
            ],
            RegulatoryDomain.SECURITIES: [
                "sebi", "securities", "stock", "equity", "bond", "mutual fund", "portfolio",
                "investment", "trading", "broker", "depository", "clearing", "settlement",
                "derivatives", "commodity", "futures", "options"
            ],
            RegulatoryDomain.INSURANCE: [
                "insurance", "irda", "policy", "premium", "claim", "underwriting",
                "actuarial", "solvency", "policyholder", "insurer", "reinsurance"
            ],
            RegulatoryDomain.FINANCIAL_SERVICES: [
                "financial service", "fintech", "payment", "digital", "mobile banking",
                "wallet", "upi", "nbfc", "microfinance", "credit card", "debit card"
            ],
            RegulatoryDomain.COMPLIANCE: [
                "compliance", "regulatory", "audit", "governance", "policy", "procedure",
                "framework", "guideline", "circular", "notification", "directive"
            ],
            RegulatoryDomain.RISK_MANAGEMENT: [
                "risk", "risk management", "operational risk", "credit risk", "market risk",
                "liquidity risk", "stress testing", "risk assessment", "risk appetite"
            ],
            RegulatoryDomain.ANTI_MONEY_LAUNDERING: [
                "aml", "anti money laundering", "kyc", "cdd", "suspicious transaction",
                "fiu", "financial intelligence", "terrorist financing", "sanctions"
            ],
            RegulatoryDomain.CYBERSECURITY: [
                "cyber", "cybersecurity", "information security", "data breach",
                "incident response", "vulnerability", "threat", "malware", "phishing"
            ],
            RegulatoryDomain.DATA_PROTECTION: [
                "data protection", "privacy", "gdpr", "personal data", "data privacy",
                "consent", "data breach", "data retention", "data processing"
            ]
        }
        
        # Regulatory-specific terms
        self.regulatory_terms = [
            "regulation", "regulatory", "circular", "notification", "directive",
            "guideline", "framework", "policy", "procedure", "compliance",
            "audit", "governance", "supervision", "monitoring", "reporting",
            "disclosure", "transparency", "accountability", "oversight"
        ]
        
        # Irrelevant query patterns
        self.irrelevant_patterns = [
            r"\b(weather|sports|entertainment|movies|music|games|food|cooking|travel|vacation)\b",
            r"\b(personal|relationship|dating|marriage|family|health|medical|doctor)\b",
            r"\b(shopping|buying|selling|price|cost|money|salary|job|career)\b",
            r"\b(education|school|university|student|teacher|learning|study)\b",
            r"\b(technology|computer|software|programming|coding|development)\b",
            r"\b(politics|election|government|political|vote|democracy)\b",
            r"\b(religion|spiritual|faith|belief|god|prayer|worship)\b"
        ]
        
        # Potentially harmful patterns
        self.harmful_patterns = [
            r"\b(hack|hacking|exploit|vulnerability|breach|attack|malware|virus)\b",
            r"\b(fraud|scam|cheat|illegal|unlawful|criminal|theft|steal)\b",
            r"\b(bypass|circumvent|avoid|evade|violate|break|ignore)\b",
            r"\b(confidential|secret|classified|proprietary|internal|private)\b"
        ]

    def classify_query(self, query: str) -> Tuple[QueryRelevance, List[RegulatoryDomain], Dict[str, Any]]:
        """
        Classify query for relevance and regulatory domain.
        
        Args:
            query: User query string
            
        Returns:
            Tuple of (relevance, domains, analysis)
        """
        query_lower = query.lower().strip()
        
        # Check for harmful patterns first
        if self._contains_harmful_patterns(query_lower):
            return QueryRelevance.POTENTIALLY_HARMFUL, [], {
                "reason": "Query contains potentially harmful content",
                "confidence": 0.9
            }
        
        # Check for irrelevant patterns
        if self._contains_irrelevant_patterns(query_lower):
            return QueryRelevance.IRRELEVANT, [], {
                "reason": "Query appears to be unrelated to regulatory matters",
                "confidence": 0.8
            }
        
        # Identify regulatory domains
        domains = self._identify_regulatory_domains(query_lower)
        
        # Calculate relevance score
        relevance_score = self._calculate_relevance_score(query_lower, domains)
        
        # Determine relevance level
        if relevance_score >= 0.8:
            relevance = QueryRelevance.HIGHLY_RELEVANT
        elif relevance_score >= 0.6:
            relevance = QueryRelevance.RELEVANT
        elif relevance_score >= 0.3:
            relevance = QueryRelevance.PARTIALLY_RELEVANT
        else:
            relevance = QueryRelevance.IRRELEVANT
        
        analysis = {
            "relevance_score": relevance_score,
            "regulatory_terms_found": self._find_regulatory_terms(query_lower),
            "domain_keywords_found": self._find_domain_keywords(query_lower, domains),
            "confidence": min(relevance_score + 0.1, 1.0)
        }
        
        return relevance, domains, analysis

    def _contains_harmful_patterns(self, query: str) -> bool:
        """Check if query contains potentially harmful patterns."""
        for pattern in self.harmful_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return True
        return False

    def _contains_irrelevant_patterns(self, query: str) -> bool:
        """Check if query contains irrelevant patterns."""
        for pattern in self.irrelevant_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return True
        return False

    def _identify_regulatory_domains(self, query: str) -> List[RegulatoryDomain]:
        """Identify regulatory domains present in the query."""
        domains = []
        for domain, keywords in self.domain_keywords.items():
            for keyword in keywords:
                if keyword in query:
                    domains.append(domain)
                    break
        return domains

    def _calculate_relevance_score(self, query: str, domains: List[RegulatoryDomain]) -> float:
        """Calculate relevance score based on regulatory terms and domains."""
        score = 0.0
        
        # Base score from regulatory terms
        regulatory_terms_found = sum(1 for term in self.regulatory_terms if term in query)
        score += min(regulatory_terms_found * 0.2, 0.6)
        
        # Bonus for domain-specific terms
        if domains:
            score += min(len(domains) * 0.15, 0.4)
        
        # Bonus for specific regulatory authorities
        authority_terms = ["rbi", "sebi", "irda", "fiu", "rbi circular", "sebi circular"]
        authority_found = sum(1 for term in authority_terms if term in query)
        score += min(authority_found * 0.1, 0.2)
        
        return min(score, 1.0)

    def _find_regulatory_terms(self, query: str) -> List[str]:
        """Find regulatory terms in the query."""
        return [term for term in self.regulatory_terms if term in query]

    def _find_domain_keywords(self, query: str, domains: List[RegulatoryDomain]) -> Dict[str, List[str]]:
        """Find domain-specific keywords in the query."""
        found_keywords = {}
        for domain in domains:
            keywords = [kw for kw in self.domain_keywords[domain] if kw in query]
            if keywords:
                found_keywords[domain.value] = keywords
        return found_keywords


class ProductionPromptBuilder:
    """Builds production-grade prompts for regulatory circulars RAG system."""
    
    def __init__(self):
        self.query_classifier = QueryClassifier()
        
    def build_system_prompt(self, query_relevance: QueryRelevance, domains: List[RegulatoryDomain]) -> str:
        """
        Build system prompt based on query relevance and domains.
        
        Args:
            query_relevance: Classification of query relevance
            domains: List of identified regulatory domains
            
        Returns:
            System prompt string
        """
        base_prompt = self._get_base_system_prompt()
        
        if query_relevance == QueryRelevance.POTENTIALLY_HARMFUL:
            return self._get_security_prompt()
        elif query_relevance == QueryRelevance.IRRELEVANT:
            return self._get_irrelevant_query_prompt()
        elif query_relevance == QueryRelevance.PARTIALLY_RELEVANT:
            return self._get_partial_relevance_prompt()
        else:
            # Highly relevant or relevant queries
            domain_specific_prompt = self._get_domain_specific_prompt(domains)
            return base_prompt + "\n\n" + domain_specific_prompt

    def _get_base_system_prompt(self) -> str:
        """Get base system prompt for regulatory queries."""
        return """You are a specialized Regulatory Compliance Assistant for financial institutions, working with a comprehensive database of regulatory circulars, notifications, and guidelines from Indian financial regulators including RBI, SEBI, IRDAI, and other relevant authorities.

CORE RESPONSIBILITIES:
- Provide accurate, up-to-date information about regulatory requirements
- Help interpret regulatory circulars and guidelines
- Assist with compliance-related queries
- Offer guidance on regulatory frameworks and procedures

RESPONSE GUIDELINES:
1. ACCURACY: Always base responses on the provided regulatory context
2. CLARITY: Use clear, professional language appropriate for compliance professionals
3. COMPLETENESS: Provide comprehensive answers while staying within scope
4. CITATION: Reference specific circulars/notifications when relevant
5. DISCLAIMER: Always include appropriate disclaimers about consulting official sources

SAFETY REQUIREMENTS:
- Only respond to queries directly related to regulatory compliance
- If uncertain about accuracy, recommend consulting official regulatory sources
- Never provide legal advice or interpretations that could be construed as legal counsel
- Always emphasize the importance of official regulatory guidance"""

    def _get_security_prompt(self) -> str:
        """Get security-focused prompt for potentially harmful queries."""
        return """SECURITY ALERT: This query has been flagged as potentially harmful or inappropriate.

RESPONSE REQUIRED:
"I can only assist with legitimate regulatory compliance queries. For security and compliance matters, please contact your organization's compliance team or consult official regulatory sources directly.

If you have a legitimate regulatory question, please rephrase it using appropriate compliance terminology."

DO NOT:
- Provide any regulatory information
- Engage with potentially harmful content
- Offer alternative interpretations"""

    def _get_irrelevant_query_prompt(self) -> str:
        """Get prompt for irrelevant queries."""
        return """QUERY OUT OF SCOPE: This query is not related to regulatory compliance matters.

RESPONSE REQUIRED:
"I specialize in regulatory compliance assistance for financial institutions. I can only help with queries related to:

- Regulatory circulars and notifications
- Compliance requirements and procedures
- Regulatory frameworks and guidelines
- Risk management and governance
- Anti-money laundering and KYC requirements
- Cybersecurity and data protection regulations

Please rephrase your question to focus on regulatory compliance matters, or contact the appropriate department for non-regulatory queries."

DO NOT:
- Provide information outside regulatory scope
- Suggest alternative sources for non-regulatory topics"""

    def _get_partial_relevance_prompt(self) -> str:
        """Get prompt for partially relevant queries."""
        return """PARTIALLY RELEVANT QUERY: This query has some regulatory elements but may be outside the primary scope.

RESPONSE APPROACH:
- Address the regulatory aspects of the query
- Clarify the scope of regulatory assistance available
- Suggest how to reframe the query for better regulatory focus
- Provide general guidance while noting limitations

EXAMPLE RESPONSE STRUCTURE:
"I can help with the regulatory compliance aspects of your question. However, some elements may be outside my specialized scope. Here's what I can address regarding regulatory compliance:

[Address regulatory aspects]

For the non-regulatory elements, I recommend consulting [appropriate department/source]."

SAFETY NOTE:
- Stay within regulatory compliance boundaries
- Don't attempt to answer non-regulatory aspects"""

    def _get_domain_specific_prompt(self, domains: List[RegulatoryDomain]) -> str:
        """Get domain-specific prompt additions."""
        if not domains:
            return ""
        
        domain_prompts = []
        
        for domain in domains:
            if domain == RegulatoryDomain.BANKING:
                domain_prompts.append("""
BANKING REGULATORY EXPERTISE:
- RBI circulars and notifications
- Basel III compliance requirements
- Capital adequacy and liquidity standards
- Credit risk management guidelines
- Operational risk frameworks
- Digital banking regulations""")
            
            elif domain == RegulatoryDomain.SECURITIES:
                domain_prompts.append("""
SECURITIES REGULATORY EXPERTISE:
- SEBI circulars and regulations
- Securities market guidelines
- Investment management regulations
- Trading and settlement procedures
- Disclosure requirements
- Market conduct regulations""")
            
            elif domain == RegulatoryDomain.INSURANCE:
                domain_prompts.append("""
INSURANCE REGULATORY EXPERTISE:
- IRDAI circulars and guidelines
- Insurance product regulations
- Solvency and capital requirements
- Policyholder protection measures
- Distribution channel regulations
- Claims settlement guidelines""")
            
            elif domain == RegulatoryDomain.ANTI_MONEY_LAUNDERING:
                domain_prompts.append("""
AML/CFT REGULATORY EXPERTISE:
- AML/CFT guidelines and circulars
- KYC and CDD requirements
- Suspicious transaction reporting
- Sanctions compliance
- Risk-based approach implementation
- FIU reporting requirements""")
        
        return "\n".join(domain_prompts)

    def build_user_prompt(self, user_query: str, context_data: List[Dict[str, Any]], 
                         query_relevance: QueryRelevance) -> str:
        """
        Build user prompt with context and query-specific instructions.
        
        Args:
            user_query: Original user query
            context_data: Retrieved regulatory context
            query_relevance: Query relevance classification
            
        Returns:
            User prompt string
        """
        if query_relevance == QueryRelevance.POTENTIALLY_HARMFUL:
            return f"User Query: {user_query}\n\n[SECURITY FILTER ACTIVATED - Query flagged as potentially harmful]"
        
        if query_relevance == QueryRelevance.IRRELEVANT:
            return f"User Query: {user_query}\n\n[QUERY OUT OF SCOPE - Not related to regulatory compliance]"
        
        # Build context string for relevant queries
        context_string = self._build_context_string(context_data)
        
        if query_relevance == QueryRelevance.PARTIALLY_RELEVANT:
            instruction = """
INSTRUCTION: This query has partial relevance to regulatory compliance. Focus only on the regulatory aspects and clarify the scope of assistance available. Do not attempt to answer non-regulatory elements."""
        else:
            instruction = """
INSTRUCTION: Provide a comprehensive, accurate response based on the regulatory context provided. Ensure all information is directly relevant to the user's regulatory compliance question."""
        
        return f"""User Query: {user_query}

{instruction}

REGULATORY CONTEXT:
{context_string}

RESPONSE REQUIREMENTS:
1. Address the specific regulatory question asked
2. Reference relevant circulars/notifications when applicable
3. Provide clear, actionable guidance
4. Include appropriate disclaimers
5. Maintain professional compliance language

IMPORTANT: Only use the provided regulatory context. If the context doesn't contain sufficient information to answer the query accurately, state this limitation and suggest consulting official regulatory sources."""

    def _build_context_string(self, context_data: List[Dict[str, Any]]) -> str:
        """Build context string from retrieved regulatory data."""
        if not context_data:
            return "No relevant regulatory context found."
        
        context_parts = []
        for i, reg_data in enumerate(context_data, 1):
            context_part = f"REGULATION {i}:\n"
            
            # Add regulation text
            if reg_data.get('Regulation'):
                context_part += f"Content: {reg_data['Regulation'][:500]}...\n"
            
            # Add summary
            if reg_data.get('Summary'):
                context_part += f"Summary: {reg_data['Summary']}\n"
            
            # Add metadata
            metadata_parts = []
            if reg_data.get('Reg_Number'):
                metadata_parts.append(f"Number: {str(reg_data['Reg_Number'])}")
            if reg_data.get('Reg_Date'):
                metadata_parts.append(f"Date: {str(reg_data['Reg_Date'])}")
            if reg_data.get('Reg_Category'):
                metadata_parts.append(f"Category: {str(reg_data['Reg_Category'])}")
            if reg_data.get('Industry'):
                metadata_parts.append(f"Industry: {str(reg_data['Industry'])}")
            
            if metadata_parts:
                context_part += f"Metadata: {', '.join(metadata_parts)}\n"
            
            # Add relevance score if available
            if reg_data.get('relevance_score'):
                context_part += f"Relevance Score: {reg_data['relevance_score']:.3f}\n"
            
            context_parts.append(context_part)
        
        return "\n".join(context_parts)


class ResponseValidator:
    """Validates responses for quality, safety, and compliance."""
    
    def __init__(self):
        self.safety_keywords = [
            "legal advice", "legal opinion", "guarantee", "warranty", "promise",
            "definitive", "certain", "always", "never", "must", "shall"
        ]
        
        self.disclaimer_phrases = [
            "consult official sources", "regulatory guidance", "compliance team",
            "official circular", "regulatory authority", "disclaimer"
        ]

    def validate_response(self, response: str, query_relevance: QueryRelevance, 
                         context_used: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate response for quality and safety.
        
        Args:
            response: Generated response
            query_relevance: Original query relevance
            context_used: Context data used
            
        Returns:
            Validation results
        """
        validation_result = {
            "is_valid": True,
            "warnings": [],
            "suggestions": [],
            "quality_score": 0.0,
            "safety_score": 0.0
        }
        
        # Check response length
        if len(response) < 50:
            validation_result["warnings"].append("Response too short")
            validation_result["quality_score"] -= 0.2
        
        # Check for safety issues
        safety_issues = self._check_safety_issues(response)
        if safety_issues:
            validation_result["warnings"].extend(safety_issues)
            validation_result["safety_score"] -= 0.3
        
        # Check for appropriate disclaimers
        if not self._has_appropriate_disclaimer(response, query_relevance):
            validation_result["suggestions"].append("Consider adding appropriate disclaimers")
        
        # Check context utilization
        context_utilization = self._check_context_utilization(response, context_used)
        validation_result["quality_score"] += context_utilization
        
        # Calculate final scores
        validation_result["quality_score"] = max(0, min(1, validation_result["quality_score"]))
        validation_result["safety_score"] = max(0, min(1, validation_result["safety_score"]))
        
        # Overall validation
        if validation_result["quality_score"] < 0.5 or validation_result["safety_score"] < 0.7:
            validation_result["is_valid"] = False
        
        return validation_result

    def _check_safety_issues(self, response: str) -> List[str]:
        """Check for potential safety issues in response."""
        issues = []
        response_lower = response.lower()
        
        for keyword in self.safety_keywords:
            if keyword in response_lower:
                issues.append(f"Potentially problematic language: '{keyword}'")
        
        return issues

    def _has_appropriate_disclaimer(self, response: str, query_relevance: QueryRelevance) -> bool:
        """Check if response has appropriate disclaimers."""
        response_lower = response.lower()
        
        # For highly relevant queries, disclaimers are less critical
        if query_relevance == QueryRelevance.HIGHLY_RELEVANT:
            return True
        
        # For other queries, check for disclaimer phrases
        return any(phrase in response_lower for phrase in self.disclaimer_phrases)

    def _check_context_utilization(self, response: str, context_used: List[Dict[str, Any]]) -> float:
        """Check how well the response utilizes the provided context."""
        if not context_used:
            return 0.0
        
        # Simple heuristic: check if response mentions regulatory elements
        regulatory_elements = ["regulation", "circular", "guideline", "compliance", "requirement"]
        response_lower = response.lower()
        
        elements_found = sum(1 for element in regulatory_elements if element in response_lower)
        return min(elements_found * 0.2, 0.8)


class ProductionRAGManager:
    """Main manager for production-grade RAG system."""
    
    def __init__(self):
        self.query_classifier = QueryClassifier()
        self.prompt_builder = ProductionPromptBuilder()
        self.response_validator = ResponseValidator()
        
    def process_query(self, user_query: str, context_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process user query with comprehensive checks and balances.
        
        Args:
            user_query: User's query
            context_data: Retrieved regulatory context
            
        Returns:
            Processing results with validation
        """
        # Step 1: Classify query
        query_relevance, domains, analysis = self.query_classifier.classify_query(user_query)
        
        # Step 2: Build prompts
        system_prompt = self.prompt_builder.build_system_prompt(query_relevance, domains)
        user_prompt = self.prompt_builder.build_user_prompt(user_query, context_data, query_relevance)
        
        # Step 3: Generate response (this would be done by LLM)
        # For now, we'll return the prompts and classification
        result = {
            "query_relevance": query_relevance.value,
            "domains": [domain.value for domain in domains],
            "analysis": analysis,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "context_data": context_data,
            "processing_timestamp": datetime.now().isoformat(),
            "recommendations": self._get_processing_recommendations(query_relevance, domains)
        }
        
        return result

    def _get_processing_recommendations(self, query_relevance: QueryRelevance, 
                                      domains: List[RegulatoryDomain]) -> List[str]:
        """Get processing recommendations based on query classification."""
        recommendations = []
        
        if query_relevance == QueryRelevance.POTENTIALLY_HARMFUL:
            recommendations.extend([
                "Block query and log security incident",
                "Return security response without processing",
                "Alert security team if pattern continues"
            ])
        elif query_relevance == QueryRelevance.IRRELEVANT:
            recommendations.extend([
                "Return scope clarification response",
                "Suggest appropriate department contact",
                "Log for potential system improvement"
            ])
        elif query_relevance == QueryRelevance.PARTIALLY_RELEVANT:
            recommendations.extend([
                "Focus on regulatory aspects only",
                "Clarify scope limitations",
                "Suggest query refinement"
            ])
        else:
            recommendations.extend([
                "Process with full regulatory context",
                "Provide comprehensive response",
                "Include relevant citations"
            ])
        
        if domains:
            recommendations.append(f"Leverage {len(domains)} identified regulatory domain(s)")
        
        return recommendations


# Convenience functions for easy integration
def create_production_rag_manager() -> ProductionRAGManager:
    """Create a new production RAG manager instance."""
    return ProductionRAGManager()


def classify_regulatory_query(query: str) -> Tuple[QueryRelevance, List[RegulatoryDomain], Dict[str, Any]]:
    """Classify a query for regulatory relevance."""
    classifier = QueryClassifier()
    return classifier.classify_query(query)


def build_regulatory_prompts(query: str, context_data: List[Dict[str, Any]], 
                           query_relevance: QueryRelevance, domains: List[RegulatoryDomain]) -> Tuple[str, str]:
    """Build system and user prompts for regulatory queries."""
    prompt_builder = ProductionPromptBuilder()
    system_prompt = prompt_builder.build_system_prompt(query_relevance, domains)
    user_prompt = prompt_builder.build_user_prompt(query, context_data, query_relevance)
    return system_prompt, user_prompt
