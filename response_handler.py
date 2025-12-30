
from openai import OpenAI
from dotenv import load_dotenv
import os
import logging
from typing import Dict, Any, List, Optional, Tuple
from document_store import retrieve_passages, SearchResult
from value_identifier import ValueProfile, get_profile_summary
import re
import json


load_dotenv()

class CounterArgumentGenerator:
    """Enhanced counterargument generator with source transparency and better prompting"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        try:
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.client.models.list()
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenAI client: {e}")
            raise
    
    def generate_counterargument(
        self,
        user_data: Dict[str, Any],
        value_profile: Optional[ValueProfile],
        topic: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a counterargument with full source transparency
        Returns dict with 'counterargument', 'sources', and 'confidence'
        """
        try:
            # Create a focused search query based on user data
            search_query = self._create_search_query(user_data, value_profile)
            self.logger.info(f"Generated search query: {search_query}")

            # Retrieve relevant passages
            argument_passages = retrieve_passages(
                search_query, top_k=3, semantic_types=["argument"], topic="income"
            )
            evidence_passages = retrieve_passages(
                search_query, top_k=2, semantic_types=["evidence"], topic="income"
            )
            general_passages = retrieve_passages(
                search_query, top_k=2, semantic_types=["general", "policy"], topic="income"
            )

            all_passages = argument_passages + evidence_passages + general_passages

            # Deduplicate passages
            seen_texts = set()
            unique_passages = []
            for p in all_passages:
                if p.text not in seen_texts:
                    unique_passages.append(p)
                    seen_texts.add(p.text)
            all_passages = unique_passages

            if not all_passages:
                return {
                    'counterargument': "I don't have sufficient information in my knowledge base to provide a well-sourced counterargument to your position. This might indicate that your viewpoint represents a perspective that isn't well-covered in the available documents, or that the issue requires more diverse sources.",
                    'sources': [],
                    'confidence': 0.0
                }

            counterargument = self._generate_counterargument_text(user_data, value_profile, all_passages)
            formatted_sources = self._format_sources(all_passages)
            confidence = self._calculate_confidence(all_passages, counterargument)

            return {
                'counterargument': counterargument,
                'sources': formatted_sources,
                'confidence': confidence
            }

        except Exception as e:
            self.logger.error(f"Error generating counterargument: {e}", exc_info=True)
            return {
                'counterargument': f"I encountered an error while generating the counterargument: {str(e)}. Please try again.",
                'sources': [],
                'confidence': 0.0
            }
    
    
    def _create_search_query(self, user_data: Dict[str, Any], value_profile: Optional[ValueProfile]) -> str:
        """
        Create a focused search query that prioritizes the user's core opinion
        while still seeking counterarguments.
        """
        opinion_statement = ""
        # The user_data here is the transformed data from app.py
        if user_data.get("phase1", {}).get("opinion_statement"):
            opinion_statement = user_data["phase1"]["opinion_statement"]
        
        if not opinion_statement or not opinion_statement.strip():
            self.logger.warning("No 'opinion_statement' found. Using a generic fallback query.")
            return "alternative perspectives on income tax policy"

        # This new structure is more direct and less "muddled"
        focused_query = f"Opposing arguments, critiques, and alternative perspectives to the following viewpoint: '{opinion_statement}'"
        
        return focused_query[:500]
    
    def _extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases from a long text for search query"""
        # Simple approach: look for important terms
        import re
        
        # Remove common words and extract meaningful phrases
        important_words = []
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Skip common words
        skip_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could', 'may', 'might', 'can', 'must', 'shall', 'a', 'an', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
        
        for word in words:
            if len(word) > 3 and word not in skip_words:
                important_words.append(word)
        
        return important_words[:10]  # Return top 10 key words

    def _generate_counterargument_text(self, user_data: Dict[str, Any], 
                                         value_profile: Optional[ValueProfile], 
                                         passages: List[SearchResult]) -> str:
        user_context = self._prepare_user_context(user_data, value_profile)
        source_material = self._prepare_source_material(passages)
        
        system_prompt = """You are an expert in fostering political empathy. Your goal is to help someone understand an opposing viewpoint by:
            1. **Connecting to their values**: Show how the opposing view might serve some of the same underlying values they have.
            2. **Providing context**: Use the source material to ground the counterargument in facts and examples.
            3. **Humanizing the other side**: Explain why reasonable people might hold this different view.
            4. **Being respectful**: Never attack their position, but offer a thoughtful alternative.
            
            ## Structure your response:
            1. Briefly acknowledge their perspective.
            2. Present the alternative view.
            3. Use source material for evidence, citing with `(Source: [Document Name], Page [X])`.
            4. Explain the reasoning behind the alternative view.
            5. Highlight any common ground.

            ## Important:
            - Rely heavily on the provided Source Material.
            - Do not attack, dismiss, or use partisan language.
            - Focus on building understanding, not winning an argument."""
        
        user_prompt = f"""## User Context:
            {user_context}

            ## Source Material:
            {source_material}

            Please generate a thoughtful counterargument that helps this user understand the opposing perspective, citing the provided sources."""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7, max_tokens=700, top_p=0.9
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            self.logger.error(f"Error calling OpenAI API: {e}", exc_info=True)
            raise

    def _prepare_user_context(self, user_data: Dict[str, Any], value_profile: Optional[ValueProfile]) -> str:
        context_parts = []
        if phase1 := user_data.get('phase1'):
            if opinion := phase1.get('opinion_statement'):
                context_parts.append(f"**Main Position**: {opinion}")
            if reasoning := phase1.get('reasoning'):
                context_parts.append(f"**Key Reasons**: {'; '.join(reasoning[:2])}")
        
        if value_profile:
            context_parts.append(f"**Value Profile**: {get_profile_summary(value_profile)}")
        
        return "\n".join(context_parts)
    
    def _prepare_source_material(self, passages: List[SearchResult]) -> str:
        if not passages: return "No relevant source material found."
        
        source_sections = []
        for i, passage in enumerate(passages[:6], 1):
            source_section = (f"**Source {i}**: {passage.source} (Page {passage.page_number})\n"
                            f"Relevance Score: {passage.relevance_score:.2f}\n"
                            f"Content: {passage.text[:400]}{'...' if len(passage.text) > 400 else ''}\n")
            source_sections.append(source_section)
        return "\n".join(source_sections)
    
    def _format_sources(self, passages: List[SearchResult]) -> List[Dict[str, Any]]:
        return [{
            'document': p.source, 'page': p.page_number,
            'relevance_score': round(p.relevance_score, 3),
            'excerpt': p.text[:200] + "..." if len(p.text) > 200 else p.text
        } for p in passages]
    
    def _calculate_confidence(self, passages: List[SearchResult], counterargument: str) -> float:
        if not passages: return 0.0
        
        avg_relevance = sum(p.relevance_score for p in passages) / len(passages)
        diversity_bonus = min(0.2, len(set(p.source for p in passages)) * 0.05)
        citation_bonus = min(0.15, len(re.findall(r'\(Source:|Page \d+\)', counterargument)) * 0.03)
        
        return min(1.0, avg_relevance + diversity_bonus + citation_bonus)

class ResponseEvaluator:
    # ... (class unchanged) ...
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        try:
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenAI client: {e}")
            raise
    
    def evaluate_response(self, counterargument: str, sources: List[Dict], confidence: float) -> Dict[str, Any]:
        try:
            system_prompt = """You are an impartial evaluator. Rate a political counterargument on a scale of 1-5 for Factual Accuracy, Source Usage, Empathy, Clarity, and Balance. Also provide an overall_score and brief_feedback. Respond with ONLY a valid JSON object."""
            
            user_prompt = f"""Evaluate this counterargument:
                **Counterargument:** {counterargument}
                **Sources:** {len(sources)} provided. **System Confidence:** {confidence:.2f}"""
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                temperature=0, max_tokens=300, response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            self.logger.error(f"Error evaluating response: {e}", exc_info=True)
            return {"brief_feedback": f"Evaluation failed: {str(e)}"}


# Global instances
_generator = CounterArgumentGenerator()
_evaluator = ResponseEvaluator()

def get_ai_counterargument(user_data: Dict[str, Any], value_profile: Optional[ValueProfile]) -> Dict[str, Any]:
    return _generator.generate_counterargument(user_data, value_profile, "income_tax")

def evaluate_response(counterargument: str, sources: List[Dict] = None, confidence: float = 0.5) -> Dict[str, Any]:
    return _evaluator.evaluate_response(counterargument, sources or [], confidence)



# from openai import OpenAI
# from dotenv import load_dotenv
# import os
# import logging
# from typing import Dict, Any, List, Optional, Tuple
# from document_store import retrieve_passages, SearchResult
# from value_identifier import ValueProfile, get_profile_summary
# import re

# load_dotenv()

# class CounterArgumentGenerator:
#     """Enhanced counterargument generator with source transparency and better prompting"""
    
#     def __init__(self):
#         self.logger = logging.getLogger(__name__)
#         try:
#             self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
#             # Test the connection
#             self.client.models.list()
#         except Exception as e:
#             self.logger.error(f"Failed to initialize OpenAI client: {e}")
#             raise
    
#     def generate_counterargument(
#         self,
#         user_data: Dict[str, Any],
#         value_profile: Optional[ValueProfile],
#         topic: Optional[str] = None  # <-- CHANGED: Added optional topic
#     ) -> Dict[str, Any]:
#         """
#         Generate a counterargument with full source transparency
#         Returns dict with 'counterargument', 'sources', and 'confidence'
#         """
#         try:
#             # Create enhanced search query based on user data and values
#             search_query = self._create_search_query(user_data, value_profile)
#             self.logger.info(f"Generated search query: {search_query}")
#             print(f"DEBUG: Generated Search Query: {search_query}") # <-- ADD THIS

#             # Retrieve relevant passages with different semantic types
#             argument_passages = retrieve_passages(
#                 search_query, top_k=3, semantic_types=["argument"], topic="income_tax"  # <-- CHANGED: pass topic
#             )
#             evidence_passages = retrieve_passages(
#                 search_query, top_k=2, semantic_types=["evidence"], topic="income_tax"  # <-- CHANGED: pass topic
#             )
#             general_passages = retrieve_passages(
#                 search_query, top_k=2, semantic_types=["general", "policy"], topic="income_tax"  # <-- CHANGED: pass topic
#             )

#             all_passages = argument_passages + evidence_passages + general_passages

#             # Deduplicate passages by text to handle multiple PDFs on the same topic
#             seen_texts = set()
#             unique_passages = []
#             for p in all_passages:
#                 if p.text not in seen_texts:
#                     unique_passages.append(p)
#                     seen_texts.add(p.text)
#             all_passages = unique_passages  # <-- CHANGED: deduplicated

#             if not all_passages:
#                 return {
#                     'counterargument': "I don't have sufficient information in my knowledge base to provide a well-sourced counterargument to your position. This might indicate that your viewpoint represents a perspective that isn't well-covered in the available documents, or that the issue requires more diverse sources.",
#                     'sources': [],
#                     'confidence': 0.0
#                 }

#             # Generate the counterargument
#             counterargument = self._generate_counterargument_text(user_data, value_profile, all_passages)

#             # Format sources for transparency
#             formatted_sources = self._format_sources(all_passages)

#             # Calculate confidence based on source quality and relevance
#             confidence = self._calculate_confidence(all_passages, counterargument)

#             return {
#                 'counterargument': counterargument,
#                 'sources': formatted_sources,
#                 'confidence': confidence
#             }

#         except Exception as e:
#             self.logger.error(f"Error generating counterargument: {e}")
#             return {
#                 'counterargument': f"I encountered an error while generating the counterargument: {str(e)}. Please try again.",
#                 'sources': [],
#                 'confidence': 0.0
#             }
    
#     def _create_search_query(self, user_data: Dict[str, Any], value_profile: Optional[ValueProfile]) -> str:
#         """
#         Create a focused search query that prioritizes the user's core opinion
#         while still seeking counterarguments.
#         """
#         # 1. Extract the user's core opinion. This is the most important part.
#         # We look for the detailed answer to the "position_articulation" question.
#         opinion_statement = ""
#         # The user_data here is the transformed data, so we look in 'phase1'
#         if user_data.get("phase1", {}).get("opinion_statement"):
#             opinion_statement = user_data["phase1"]["opinion_statement"]
        
#         # A fallback in case the opinion statement is somehow empty
#         if not opinion_statement or not opinion_statement.strip():
#             self.logger.warning("No 'opinion_statement' found to create a search query. Using a generic fallback.")
#             return "alternative perspectives on income tax policy"

#         # 2. Create a clean, explicit query.
#         # This structure tells the vector database exactly what we want:
#         # Passages that are semantically related to the user's idea,
#         # but in the context of opposition.
#         focused_query = f"Opposing arguments, critiques, and alternative perspectives to the following viewpoint: '{opinion_statement}'"
        
#         self.logger.info(f"Generated focused search query: {focused_query}")
        
#         return focused_query[:500]  # Return the query, capped at a reasonable length

    
#     def _generate_counterargument_text(self, user_data: Dict[str, Any], 
#                                      value_profile: Optional[ValueProfile], 
#                                      passages: List[SearchResult]) -> str:
#         """Generate the actual counterargument text using retrieved sources"""
        
#         # Prepare context about the user
#         user_context = self._prepare_user_context(user_data, value_profile)
        
#         # Prepare source material
#         source_material = self._prepare_source_material(passages)
        
#         system_prompt = """You are an expert in fostering political empathy and understanding. Your goal is to help someone understand an opposing viewpoint by:

#             1. **Connecting to their values**: Show how the opposing view might actually serve some of the same underlying values or concerns they have, but through different means or priorities
#             2. **Providing historical/factual context**: Use the source material to ground the counterargument in facts, evidence, and real-world examples
#             3. **Humanizing the other side**: Explain why reasonable, well-intentioned people might hold this different view based on their experiences, circumstances, or moral priorities
#             4. **Being respectful and educational**: Never attack or dismiss their position, but offer a thoughtful alternative perspective that invites reflection

#             ## Personalization Guidelines:
#             - **Use their moral foundations**: If they value fairness, show how the other side also seeks fairness but defines it differently
#             - **Match their communication style**: Adapt to their preferred tone (academic, conversational, etc.)
#             - **Address their specific concerns**: Directly engage with doubts or uncertainties they've expressed
#             - **Use relatable examples**: Choose scenarios and references that connect to their background and experiences

#             ## Structure your response:
#             1. Brief acknowledgment of their perspective's validity
#             2. Present the alternative view as addressing shared concerns through different solutions
#             3. Use source material to provide concrete evidence and examples
#             4. Explain the underlying reasoning and values that drive this alternative view
#             5. Highlight any common ground or shared goals

#             ## Important Guidelines:
#             - Use the provided source material to support your points - cite specific sources when making factual claims
#             - Focus on building understanding, not winning an argument
#             - Acknowledge the complexity and nuance of the issue
#             - Show how both sides often share common goals but differ on methods, priorities, or implementation
#             - Keep the tone respectful, curious, and empathetic
#             - When citing sources, use format like "(Source: [Document Name], Page [X])"

#             ## What NOT to do:
#             - Don't attack, dismiss, or strawman the user's position
#             - Don't lecture, be condescending, or use partisan language
#             - Don't make claims without backing them up with the provided sources
#             - Don't oversimplify complex issues or present false dichotomies
#             - Don't assume bad faith motivations on either side"""
        
#         user_prompt = f"""Here's information about a user's political position and some source material that might offer counterarguments:

#             ## User Context:
#             {user_context}

#             ## Source Material:
#             {source_material}

#             Please generate a thoughtful counterargument that helps this user understand the opposing perspective while respecting their values and concerns. 

#             Focus on:
#             - How the opposing view might address concerns they care about through different approaches
#             - Why people with similar underlying values might reach different conclusions
#             - Concrete evidence from the source material that supports this alternative perspective
#             - The human experiences and reasoning that lead to this different viewpoint

#             Make sure to cite specific sources when making factual claims and maintain a tone that invites reflection rather than defensiveness."""
#         try:
#             response = self.client.chat.completions.create(
#                 model="gpt-4o",
#                 messages=[
#                     {"role": "system", "content": system_prompt},
#                     {"role": "user", "content": user_prompt}
#                 ],
#                 temperature=0.7,
#                 max_tokens=700,
#                 top_p=0.9,
#                 frequency_penalty=0.4,
#                 presence_penalty=0.6
#             )
            
#             return response.choices[0].message.content.strip()
            
#         except Exception as e:
#             self.logger.error(f"Error calling OpenAI API: {e}")
#             raise
    
#     def _prepare_user_context(self, user_data: Dict[str, Any], value_profile: Optional[ValueProfile]) -> str:
#         """Prepare a summary of the user's position and values"""
        
#         context_parts = []
        
#         # Main position
#         phase1 = user_data.get('phase1', {})
#         if phase1.get('opinion_statement'):
#             context_parts.append(f"**Main Position**: {phase1['opinion_statement']}")
        
#         # Key reasoning
#         if phase1.get('reasoning'):
#             reasons = phase1['reasoning'][:2]  # Limit to first 2 reasons
#             context_parts.append(f"**Key Reasons**: {'; '.join(reasons)}")
        
#         # Confidence and flexibility
#         phase2 = user_data.get('phase2', {})
#         if phase2.get('confidence_score') is not None:
#             confidence = phase2['confidence_score']
#             context_parts.append(f"**Confidence Level**: {confidence}/10")
        
#         if phase2.get('flexibility_reason'):
#             context_parts.append(f"**Areas of Uncertainty**: {phase2['flexibility_reason']}")
        
#         # Value profile summary
#         if value_profile:
#             profile_summary = get_profile_summary(value_profile)
#             context_parts.append(f"**Value Profile**: {profile_summary}")
        
#         # Demographics (if relevant)
#         demographics = user_data.get('demographics', {})
#         demo_parts = []
#         if demographics.get('economic_background'):
#             demo_parts.append(f"Economic background: {demographics['economic_background']}")
#         if demo_parts:
#             context_parts.append(f"**Background**: {'; '.join(demo_parts)}")
        
#         return "\n".join(context_parts)
    
#     def _prepare_source_material(self, passages: List[SearchResult]) -> str:
#         """Format source material for the AI to use"""
#         if not passages:
#             return "No relevant source material found."
        
#         source_sections = []
        
#         for i, passage in enumerate(passages[:6], 1):  # Limit to top 6 passages
#             source_section = f"""**Source {i}**: {passage.source} (Page {passage.page_number})
#                 Relevance Score: {passage.relevance_score:.2f}
#                 Content: {passage.text[:400]}{'...' if len(passage.text) > 400 else ''}
#                 """
#             source_sections.append(source_section)
        
#         return "\n".join(source_sections)
    
#     def _format_sources(self, passages: List[SearchResult]) -> List[Dict[str, Any]]:
#         """Format sources for frontend display"""
#         sources = []
        
#         for passage in passages:
#             sources.append({
#                 'document': passage.source,
#                 'page': passage.page_number,
#                 'relevance_score': round(passage.relevance_score, 3),
#                 'excerpt': passage.text[:200] + "..." if len(passage.text) > 200 else passage.text
#             })
        
#         return sources
    
#     def _calculate_confidence(self, passages: List[SearchResult], counterargument: str) -> float:
#         """Calculate confidence score based on source quality and relevance"""
#         if not passages:
#             return 0.0
        
#         # Base confidence on average relevance score
#         avg_relevance = sum(p.relevance_score for p in passages) / len(passages)
        
#         # Boost confidence if we have multiple good sources
#         source_diversity = len(set(p.source for p in passages))
#         diversity_bonus = min(0.2, source_diversity * 0.05)
        
#         # Check if counterargument contains citations (indicates source usage)
#         citation_count = len(re.findall(r'\(Source:|Page \d+\)', counterargument))
#         citation_bonus = min(0.15, citation_count * 0.03)
        
#         confidence = avg_relevance + diversity_bonus + citation_bonus
#         return min(1.0, confidence)

# class ResponseEvaluator:
#     """Evaluate the quality and credibility of generated responses"""
    
#     def __init__(self):
#         self.logger = logging.getLogger(__name__)
#         try:
#             self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
#         except Exception as e:
#             self.logger.error(f"Failed to initialize OpenAI client: {e}")
#             raise
    
#     def evaluate_response(self, counterargument: str, sources: List[Dict], confidence: float) -> Dict[str, Any]:
#         """Evaluate the response and provide transparency metrics"""
        
#         try:
#             system_prompt = """You are an impartial evaluator assessing the quality of a political counterargument. Rate the response on:

#                 1. **Factual Accuracy** (1-5): Are claims supported by evidence?
#                 2. **Source Usage** (1-5): How well does it incorporate and cite sources?
#                 3. **Empathy & Respect** (1-5): Is the tone respectful and empathetic?
#                 4. **Clarity** (1-5): Is the argument clear and well-structured?
#                 5. **Balance** (1-5): Does it avoid partisan language and present a fair counterpoint?

#                 Respond with ONLY a JSON object in this format:
#                 {
#                 "factual_accuracy": 1-5,
#                 "source_usage": 1-5,
#                 "empathy_respect": 1-5,
#                 "clarity": 1-5,
#                 "balance": 1-5,
#                 "overall_score": 1-5,
#                 "brief_feedback": "2-3 sentence summary of strengths/weaknesses"
#                 }"""
                    
#             user_prompt = f"""Evaluate this counterargument:

#                 **Counterargument:**
#                 {counterargument}

#                 **Available Sources:** {len(sources)} sources provided
#                 **System Confidence:** {confidence:.2f}

#                 **Source Details:**
#                 {sources[:3] if sources else 'No sources provided'}"""
            
#             response = self.client.chat.completions.create(
#                 model="gpt-4o-mini",  # Use cheaper model for evaluation
#                 messages=[
#                     {"role": "system", "content": system_prompt},
#                     {"role": "user", "content": user_prompt}
#                 ],
#                 temperature=0,
#                 max_tokens=250
#             )
            
#             evaluation = response.choices[0].message.content.strip()
            
#             # Parse JSON response
#             try:
#                 eval_data = eval(evaluation)  # Safe since we control the format
#                 return eval_data
#             except:
#                 # Fallback if JSON parsing fails
#                 return {
#                     "factual_accuracy": 3,
#                     "source_usage": 3,
#                     "empathy_respect": 3,
#                     "clarity": 3,
#                     "balance": 3,
#                     "overall_score": 3,
#                     "brief_feedback": "Evaluation parsing failed - response generated but could not be fully assessed."
#                 }
                
#         except Exception as e:
#             self.logger.error(f"Error evaluating response: {e}")
#             return {
#                 "factual_accuracy": 0,
#                 "source_usage": 0,
#                 "empathy_respect": 0,
#                 "clarity": 0,
#                 "balance": 0,
#                 "overall_score": 0,
#                 "brief_feedback": f"Evaluation failed due to error: {str(e)}"
#             }

# # Global instances
# _generator = CounterArgumentGenerator()
# _evaluator = ResponseEvaluator()

# def get_ai_counterargument(user_data: Dict[str, Any], value_profile: Optional[ValueProfile]) -> Dict[str, Any]:
#     """
#     Main interface function for generating counterarguments
#     Returns dict with counterargument, sources, and metadata
#     """
#     return _generator.generate_counterargument(user_data, value_profile, "income_tax")

# def evaluate_response(counterargument: str, sources: List[Dict] = None, confidence: float = 0.5) -> Dict[str, Any]:
#     """
#     Evaluate the quality of a generated response
#     Returns evaluation metrics and feedback
#     """
#     if sources is None:
#         sources = []
#     return _evaluator.evaluate_response(counterargument, sources, confidence)





