#!/usr/bin/env python3
"""
Phase 1.3: Premium highlight scorer (Budget-capped GPT-4o + Claude)
AI-powered highlight scoring with budget guardrails
"""

import os
import sys
import json
import logging
import time
import asyncio
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib

# Check for AI client availability
OPENAI_AVAILABLE = False
ANTHROPIC_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
    print("✅ OpenAI client available")
except ImportError:
    print("⚠️  OpenAI client not available")

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
    print("✅ Anthropic client available")
except ImportError:
    print("⚠️  Anthropic client not available")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class BudgetLimits:
    """Budget limits for AI API calls"""
    max_total_cost: float = 5.0  # Max $5 per session
    max_per_request: float = 0.50  # Max $0.50 per request
    max_requests_per_hour: int = 100  # Rate limiting
    current_cost: float = 0.0
    request_count: int = 0
    session_start: float = 0.0

@dataclass
class PremiumHighlight:
    """Premium AI-scored highlight"""
    slug: str
    start_ms: int
    end_ms: int
    text: str
    ai_score: float
    local_score: float
    combined_score: float
    reasoning: str
    emotional_appeal: float
    informativeness: float
    uniqueness: float
    confidence: float
    ai_model: str
    cost_estimate: float

class MockAIClient:
    """Mock AI client for testing without API keys"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.cost_per_token = 0.00001  # Mock cost
    
    async def analyze_segment(self, text: str, context: str = "") -> Dict[str, Any]:
        """Mock AI analysis"""
        logger.info(f"🎭 Mock {self.model_name} analysis...")
        
        # Simulate processing time
        await asyncio.sleep(0.1)
        
        # Mock scoring based on text characteristics
        words = text.lower().split()
        word_count = len(words)
        
        # Mock emotional appeal (based on exciting words)
        exciting_words = ['exciting', 'amazing', 'incredible', 'outstanding', 'brilliant', 'fantastic']
        emotional_score = min(sum(1 for word in words if word in exciting_words) * 0.3, 1.0)
        
        # Mock informativeness (based on professional terms)
        info_words = ['professor', 'scientist', 'research', 'study', 'health', 'minister', 'federal']
        info_score = min(sum(1 for word in words if word in info_words) * 0.2, 1.0)
        
        # Mock uniqueness (based on proper nouns and specific terms)
        unique_words = ['david', 'sinclair', 'greg', 'hunt', 'australia', 'longevity', 'geneticist']
        unique_score = min(sum(1 for word in words if word in unique_words) * 0.25, 1.0)
        
        # Overall AI score (weighted combination)
        ai_score = (emotional_score * 0.3 + info_score * 0.4 + unique_score * 0.3) * 10
        
        # Mock reasoning
        reasoning = f"Mock {self.model_name}: High informational content with professional terminology. "
        if emotional_score > 0.5:
            reasoning += "Strong emotional appeal. "
        if unique_score > 0.5:
            reasoning += "Contains unique identifiers and specific context."
        
        return {
            "ai_score": ai_score,
            "emotional_appeal": emotional_score,
            "informativeness": info_score,
            "uniqueness": unique_score,
            "reasoning": reasoning,
            "confidence": 0.85,
            "cost_estimate": word_count * self.cost_per_token
        }

class GPT4oClient:
    """GPT-4o client with budget controls"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.client = None
        self.model_name = "gpt-4o-mini"  # Use mini for cost efficiency
        self.cost_per_1k_tokens = 0.00015  # GPT-4o-mini pricing
        
        if self.api_key and OPENAI_AVAILABLE:
            self.client = openai.OpenAI(api_key=self.api_key)
        else:
            logger.warning("OpenAI API key not available, using mock client")
            self.client = MockAIClient("GPT-4o-mini")
    
    async def analyze_segment(self, text: str, context: str = "") -> Dict[str, Any]:
        """Analyze segment with GPT-4o"""
        if isinstance(self.client, MockAIClient):
            return await self.client.analyze_segment(text, context)
        
        try:
            prompt = f"""
            Analyze this video transcript segment for highlight quality:
            
            Context: {context}
            
            Segment: "{text}"
            
            Rate this segment on a scale of 1-10 for:
            1. Emotional appeal (how engaging/exciting is it?)
            2. Informativeness (how much valuable information does it contain?)
            3. Uniqueness (how distinctive/memorable is it?)
            
            Provide your analysis in this JSON format:
            {{
                "ai_score": 8.5,
                "emotional_appeal": 0.85,
                "informativeness": 0.90,
                "uniqueness": 0.75,
                "reasoning": "Brief explanation of your scoring",
                "confidence": 0.90
            }}
            """
            
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.3
            )
            
            # Parse JSON response
            response_text = response.choices[0].message.content
            result = json.loads(response_text)
            
            # Calculate cost
            tokens_used = response.usage.total_tokens
            cost = (tokens_used / 1000) * self.cost_per_1k_tokens
            result["cost_estimate"] = cost
            
            return result
            
        except Exception as e:
            logger.error(f"GPT-4o analysis failed: {e}")
            # Fallback to mock
            mock_client = MockAIClient("GPT-4o-mini")
            return await mock_client.analyze_segment(text, context)

class ClaudeClient:
    """Claude client with budget controls"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        self.client = None
        self.model_name = "claude-3-haiku-20240307"  # Use Haiku for cost efficiency
        self.cost_per_1k_tokens = 0.00025  # Claude 3 Haiku pricing
        
        if self.api_key and ANTHROPIC_AVAILABLE:
            self.client = anthropic.Anthropic(api_key=self.api_key)
        else:
            logger.warning("Anthropic API key not available, using mock client")
            self.client = MockAIClient("Claude-3-Haiku")
    
    async def analyze_segment(self, text: str, context: str = "") -> Dict[str, Any]:
        """Analyze segment with Claude"""
        if isinstance(self.client, MockAIClient):
            return await self.client.analyze_segment(text, context)
        
        try:
            prompt = f"""
            Analyze this video transcript segment for highlight quality:
            
            Context: {context}
            
            Segment: "{text}"
            
            Rate this segment on a scale of 1-10 for:
            1. Emotional appeal (how engaging/exciting is it?)
            2. Informativeness (how much valuable information does it contain?)  
            3. Uniqueness (how distinctive/memorable is it?)
            
            Provide your analysis in this JSON format:
            {{
                "ai_score": 8.5,
                "emotional_appeal": 0.85,
                "informativeness": 0.90,
                "uniqueness": 0.75,
                "reasoning": "Brief explanation of your scoring",
                "confidence": 0.90
            }}
            """
            
            response = await self.client.messages.create(
                model=self.model_name,
                max_tokens=200,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Parse JSON response
            response_text = response.content[0].text
            result = json.loads(response_text)
            
            # Calculate cost (rough estimate)
            tokens_used = len(prompt.split()) + len(response_text.split())
            cost = (tokens_used / 1000) * self.cost_per_1k_tokens
            result["cost_estimate"] = cost
            
            return result
            
        except Exception as e:
            logger.error(f"Claude analysis failed: {e}")
            # Fallback to mock
            mock_client = MockAIClient("Claude-3-Haiku")
            return await mock_client.analyze_segment(text, context)

class PremiumHighlightScorer:
    """Premium highlight scorer with AI and budget controls"""
    
    def __init__(self, budget_limits: Optional[BudgetLimits] = None):
        self.budget = budget_limits or BudgetLimits()
        self.budget.session_start = time.time()
        
        # Initialize AI clients
        self.gpt4o = GPT4oClient()
        self.claude = ClaudeClient()
        
        # Cache for avoiding duplicate API calls
        self.cache = {}
        
    def _get_cache_key(self, text: str, model: str) -> str:
        """Generate cache key for segment"""
        return hashlib.md5(f"{text}:{model}".encode()).hexdigest()
    
    def _check_budget(self, estimated_cost: float) -> bool:
        """Check if request is within budget"""
        if self.budget.current_cost + estimated_cost > self.budget.max_total_cost:
            logger.warning(f"Budget exceeded: ${self.budget.current_cost + estimated_cost:.3f} > ${self.budget.max_total_cost}")
            return False
        
        if estimated_cost > self.budget.max_per_request:
            logger.warning(f"Per-request cost too high: ${estimated_cost:.3f} > ${self.budget.max_per_request}")
            return False
        
        # Check rate limiting (only enforce if we've been running for more than 1 minute)
        elapsed_hours = (time.time() - self.budget.session_start) / 3600
        if elapsed_hours > 1/60 and self.budget.request_count / elapsed_hours > self.budget.max_requests_per_hour:
            logger.warning(f"Rate limit exceeded: {self.budget.request_count / elapsed_hours:.1f} > {self.budget.max_requests_per_hour}")
            return False
        
        return True
    
    def _update_budget(self, cost: float):
        """Update budget tracking"""
        self.budget.current_cost += cost
        self.budget.request_count += 1
        
    async def analyze_with_ai(self, text: str, context: str, model: str = "gpt4o") -> Optional[Dict[str, Any]]:
        """Analyze segment with AI (with budget controls)"""
        
        # Check cache first
        cache_key = self._get_cache_key(text, model)
        if cache_key in self.cache:
            logger.info(f"📋 Using cached result for {model}")
            return self.cache[cache_key]
        
        # Estimate cost
        estimated_cost = len(text.split()) * 0.00002  # Rough estimate
        
        # Check budget
        if not self._check_budget(estimated_cost):
            logger.warning(f"Skipping {model} analysis due to budget constraints")
            return None
        
        # Get AI client
        client = self.gpt4o if model == "gpt4o" else self.claude
        
        # Analyze
        result = await client.analyze_segment(text, context)
        
        # Update budget
        actual_cost = result.get("cost_estimate", estimated_cost)
        self._update_budget(actual_cost)
        
        # Cache result
        self.cache[cache_key] = result
        
        return result
    
    async def score_segments(self, transcript_data: Dict[str, Any], local_scores: List[Dict[str, Any]]) -> List[PremiumHighlight]:
        """Score segments with premium AI analysis"""
        logger.info("🤖 Starting premium AI highlight scoring...")
        
        segments = transcript_data.get('segments', [])
        premium_highlights = []
        
        # Create context from full transcript
        context = f"Video interview transcript: {transcript_data.get('transcript', '')[:500]}..."
        
        for i, segment in enumerate(segments):
            text = segment.get('text', '').strip()
            start_time = segment.get('start', 0)
            end_time = segment.get('end', 0)
            
            if not text:
                continue
            
            # Get local score
            local_score = local_scores[i] if i < len(local_scores) else {"score": 0.0}
            
            # Try both AI models (budget permitting)
            gpt4o_result = await self.analyze_with_ai(text, context, "gpt4o")
            claude_result = await self.analyze_with_ai(text, context, "claude")
            
            # Combine AI results
            if gpt4o_result and claude_result:
                # Average the two models
                ai_score = (gpt4o_result["ai_score"] + claude_result["ai_score"]) / 2
                emotional_appeal = (gpt4o_result["emotional_appeal"] + claude_result["emotional_appeal"]) / 2
                informativeness = (gpt4o_result["informativeness"] + claude_result["informativeness"]) / 2
                uniqueness = (gpt4o_result["uniqueness"] + claude_result["uniqueness"]) / 2
                confidence = (gpt4o_result["confidence"] + claude_result["confidence"]) / 2
                reasoning = f"GPT-4o: {gpt4o_result['reasoning'][:100]}... Claude: {claude_result['reasoning'][:100]}..."
                cost_estimate = gpt4o_result["cost_estimate"] + claude_result["cost_estimate"]
                ai_model = "GPT-4o + Claude"
            elif gpt4o_result:
                ai_score = gpt4o_result["ai_score"]
                emotional_appeal = gpt4o_result["emotional_appeal"]
                informativeness = gpt4o_result["informativeness"]
                uniqueness = gpt4o_result["uniqueness"]
                confidence = gpt4o_result["confidence"]
                reasoning = gpt4o_result["reasoning"]
                cost_estimate = gpt4o_result["cost_estimate"]
                ai_model = "GPT-4o"
            elif claude_result:
                ai_score = claude_result["ai_score"]
                emotional_appeal = claude_result["emotional_appeal"]
                informativeness = claude_result["informativeness"]
                uniqueness = claude_result["uniqueness"]
                confidence = claude_result["confidence"]
                reasoning = claude_result["reasoning"]
                cost_estimate = claude_result["cost_estimate"]
                ai_model = "Claude"
            else:
                # Fallback to local score only
                ai_score = local_score["score"]
                emotional_appeal = 0.5
                informativeness = 0.5
                uniqueness = 0.5
                confidence = 0.5
                reasoning = "Budget exceeded - using local score only"
                cost_estimate = 0.0
                ai_model = "Local fallback"
            
            # Combine AI and local scores (weighted)
            combined_score = (ai_score * 0.7) + (local_score["score"] * 0.3)
            
            # Create premium highlight
            highlight = PremiumHighlight(
                slug=f"premium-{i+1}-{text[:20].lower().replace(' ', '-')}",
                start_ms=int(start_time * 1000),
                end_ms=int(end_time * 1000),
                text=text,
                ai_score=ai_score,
                local_score=local_score["score"],
                combined_score=combined_score,
                reasoning=reasoning,
                emotional_appeal=emotional_appeal,
                informativeness=informativeness,
                uniqueness=uniqueness,
                confidence=confidence,
                ai_model=ai_model,
                cost_estimate=cost_estimate
            )
            
            premium_highlights.append(highlight)
            
            logger.info(f"Segment {i+1}: {combined_score:.1f} pts (AI:{ai_score:.1f} + Local:{local_score['score']:.1f}) - ${cost_estimate:.4f}")
        
        # Sort by combined score
        premium_highlights.sort(key=lambda x: x.combined_score, reverse=True)
        
        logger.info(f"✅ Premium scoring complete - Total cost: ${self.budget.current_cost:.4f}")
        return premium_highlights

async def main():
    """Test premium highlight scorer"""
    if len(sys.argv) < 3:
        print("Usage: python phase1_3_premium_highlight_scorer.py <transcript_json> <local_scores_json>")
        return
    
    transcript_path = sys.argv[1]
    local_scores_path = sys.argv[2]
    
    # Load data
    with open(transcript_path, 'r') as f:
        transcript_data = json.load(f)
    
    with open(local_scores_path, 'r') as f:
        local_data = json.load(f)
        local_scores = local_data.get("top_highlights", [])
    
    # Initialize premium scorer
    budget = BudgetLimits(max_total_cost=1.0, max_per_request=0.10)  # Conservative budget
    scorer = PremiumHighlightScorer(budget)
    
    # Score segments
    premium_highlights = await scorer.score_segments(transcript_data, local_scores)
    
    # Get top 5
    top_highlights = premium_highlights[:5]
    
    # Print results
    print("\n" + "=" * 60)
    print("🤖 PREMIUM HIGHLIGHT SCORER RESULTS")
    print("=" * 60)
    
    for i, highlight in enumerate(top_highlights, 1):
        print(f"\n{i}. {highlight.slug} ({highlight.combined_score:.1f} pts)")
        print(f"   📝 {highlight.text}")
        print(f"   ⏱️  {highlight.start_ms}ms - {highlight.end_ms}ms")
        print(f"   🤖 AI: {highlight.ai_score:.1f} | Local: {highlight.local_score:.1f}")
        print(f"   📊 Appeal:{highlight.emotional_appeal:.1f} Info:{highlight.informativeness:.1f} Unique:{highlight.uniqueness:.1f}")
        print(f"   🧠 {highlight.ai_model} - ${highlight.cost_estimate:.4f}")
        print(f"   💭 {highlight.reasoning[:150]}...")
    
    print(f"\n💰 Total cost: ${scorer.budget.current_cost:.4f}")
    print(f"🔢 API calls: {scorer.budget.request_count}")
    
    # Save results
    output_path = "premium_highlights_result.json"
    with open(output_path, 'w') as f:
        json.dump({
            "top_highlights": [asdict(h) for h in top_highlights],
            "all_highlights": [asdict(h) for h in premium_highlights],
            "budget_used": asdict(scorer.budget),
            "total_cost": scorer.budget.current_cost,
            "method": "premium_highlight_scorer"
        }, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_path}")

if __name__ == "__main__":
    asyncio.run(main())