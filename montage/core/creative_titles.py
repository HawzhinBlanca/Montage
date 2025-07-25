#!/usr/bin/env python3
"""
AI-Generated Creative Titles and Hashtags
Generate viral-worthy titles and tags using AI
"""
import os
import logging
import json
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import anthropic
import google.generativeai as genai

logger = logging.getLogger(__name__)


@dataclass
class CreativeContent:
    """Creative content for social media"""
    title: str
    hook: str
    hashtags: List[str]
    emojis: List[str]
    call_to_action: str
    platform_specific: Dict[str, Dict[str, Any]]


class AICreativeTitleGenerator:
    """Generate creative titles and hashtags using AI"""
    
    def __init__(self):
        self.claude_client = None
        self.gemini_model = None
        self._initialize_ai()
        
        # Platform-specific requirements
        self.platform_specs = {
            "tiktok": {
                "title_length": 100,
                "hashtag_count": 5,
                "emoji_style": "fun",
                "trends": ["fyp", "viral", "foryou"]
            },
            "instagram": {
                "title_length": 125,
                "hashtag_count": 30,
                "emoji_style": "aesthetic",
                "trends": ["reels", "explore", "trending"]
            },
            "youtube": {
                "title_length": 70,
                "hashtag_count": 3,
                "emoji_style": "minimal",
                "trends": ["shorts", "viral", "trending"]
            }
        }
        
    def _initialize_ai(self):
        """Initialize AI models"""
        # Claude
        claude_key = os.getenv("ANTHROPIC_API_KEY")
        if claude_key:
            self.claude_client = anthropic.Anthropic(api_key=claude_key)
            
        # Gemini
        gemini_key = os.getenv("GEMINI_API_KEY")
        if gemini_key:
            genai.configure(api_key=gemini_key)
            self.gemini_model = genai.GenerativeModel('gemini-1.5-pro')
            
    async def generate_creative_content(
        self,
        transcript_segment: Dict[str, Any],
        platform: str = "tiktok",
        style: str = "viral"
    ) -> CreativeContent:
        """
        Generate creative content for a video segment
        
        Args:
            transcript_segment: Segment with transcript and metadata
            platform: Target platform
            style: Content style (viral, educational, funny, inspiring)
        """
        # Get platform specs
        specs = self.platform_specs.get(platform, self.platform_specs["tiktok"])
        
        # Generate with both AIs
        claude_content = await self._generate_with_claude(
            transcript_segment, platform, style, specs
        )
        gemini_content = await self._generate_with_gemini(
            transcript_segment, platform, style, specs
        )
        
        # Combine best elements
        final_content = self._merge_creative_content(
            claude_content, gemini_content, specs
        )
        
        return final_content
        
    async def _generate_with_claude(
        self,
        segment: Dict,
        platform: str,
        style: str,
        specs: Dict
    ) -> Dict[str, Any]:
        """Generate content with Claude"""
        if not self.claude_client:
            return self._fallback_content()
            
        prompt = f"""You are a viral content expert. Create engaging {platform} content.

Video transcript: "{segment.get('text', '')[:500]}"
Style: {style}
Platform: {platform}

Generate:
1. Catchy title (max {specs['title_length']} chars) that creates curiosity
2. Hook for the first 3 seconds
3. {specs['hashtag_count']} relevant hashtags (mix of niche and broad)
4. 3-5 emojis that match the vibe
5. Call-to-action

Make it native to {platform} culture. Use current trends.

Return as JSON with keys: title, hook, hashtags, emojis, call_to_action"""

        try:
            response = self.claude_client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Parse response
            content = response.content[0].text
            
            # Extract JSON
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
                
        except Exception as e:
            logger.error(f"Claude generation failed: {e}")
            
        return self._fallback_content()
        
    async def _generate_with_gemini(
        self,
        segment: Dict,
        platform: str,
        style: str,
        specs: Dict
    ) -> Dict[str, Any]:
        """Generate content with Gemini"""
        if not self.gemini_model:
            return self._fallback_content()
            
        prompt = f"""As a {platform} content strategist, create viral-worthy content.

Context: {segment.get('text', '')[:500]}
Target: {style} content for {platform}

Requirements:
- Title: Compelling, curiosity-inducing ({specs['title_length']} chars max)
- Hook: First 3 seconds script
- Hashtags: {specs['hashtag_count']} tags mixing trends and niche
- Emojis: Match {specs['emoji_style']} style
- CTA: Platform-appropriate call-to-action

Current {platform} trends: {', '.join(specs['trends'])}

Format as JSON."""

        try:
            response = self.gemini_model.generate_content(prompt)
            
            # Parse JSON from response
            import re
            json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
                
        except Exception as e:
            logger.error(f"Gemini generation failed: {e}")
            
        return self._fallback_content()
        
    def _merge_creative_content(
        self,
        claude_content: Dict,
        gemini_content: Dict,
        specs: Dict
    ) -> CreativeContent:
        """Merge content from both AIs"""
        # Use best title (shorter but catchy)
        claude_title = claude_content.get("title", "")
        gemini_title = gemini_content.get("title", "")
        
        if len(claude_title) <= specs["title_length"] and "?" in claude_title:
            title = claude_title  # Questions perform well
        elif len(gemini_title) <= specs["title_length"]:
            title = gemini_title
        else:
            title = claude_title[:specs["title_length"]]
            
        # Combine hashtags (remove duplicates)
        all_hashtags = list(set(
            claude_content.get("hashtags", []) + 
            gemini_content.get("hashtags", [])
        ))
        
        # Score hashtags by relevance and popularity
        scored_hashtags = self._score_hashtags(all_hashtags)
        hashtags = [tag for tag, _ in scored_hashtags[:specs["hashtag_count"]]]
        
        # Merge other elements
        content = CreativeContent(
            title=title,
            hook=claude_content.get("hook") or gemini_content.get("hook", ""),
            hashtags=hashtags,
            emojis=self._select_best_emojis(
                claude_content.get("emojis", []) + 
                gemini_content.get("emojis", [])
            ),
            call_to_action=claude_content.get("call_to_action") or 
                          gemini_content.get("call_to_action", ""),
            platform_specific={}
        )
        
        return content
        
    def _score_hashtags(self, hashtags: List[str]) -> List[Tuple[str, float]]:
        """Score hashtags by potential reach and relevance"""
        scored = []
        
        # Common viral hashtags get bonus
        viral_tags = {"fyp", "viral", "trending", "foryou", "explore", "reels"}
        
        for tag in hashtags:
            tag_clean = tag.strip("#").lower()
            score = 1.0
            
            # Viral bonus
            if tag_clean in viral_tags:
                score += 2.0
                
            # Length penalty (too long hashtags are less effective)
            if len(tag_clean) > 20:
                score *= 0.5
                
            # Specificity bonus (medium length is best)
            if 8 <= len(tag_clean) <= 15:
                score += 0.5
                
            scored.append((tag, score))
            
        # Sort by score
        scored.sort(key=lambda x: x[1], reverse=True)
        
        return scored
        
    def _select_best_emojis(self, emojis: List[str], max_count: int = 5) -> List[str]:
        """Select best emojis avoiding duplicates"""
        seen = set()
        selected = []
        
        for emoji in emojis:
            if emoji not in seen and len(selected) < max_count:
                seen.add(emoji)
                selected.append(emoji)
                
        return selected
        
    def _fallback_content(self) -> Dict[str, Any]:
        """Fallback content when AI is not available"""
        return {
            "title": "You Won't Believe What Happens Next",
            "hook": "Wait for it...",
            "hashtags": ["#viral", "#fyp", "#trending", "#mustwatch", "#amazing"],
            "emojis": ["🔥", "😱", "💯", "🎬", "✨"],
            "call_to_action": "Follow for more!"
        }
        

class TrendAnalyzer:
    """Analyze current trends for better content"""
    
    def __init__(self):
        self.trend_keywords = {
            "2024": ["ai", "chatgpt", "sustainability", "mentalhealth", "sidehustle"],
            "2025": ["ai", "automation", "wellness", "creator", "authentic"]
        }
        
    def get_trending_topics(self, platform: str) -> List[str]:
        """Get current trending topics for platform"""
        # In production, this would call trend APIs
        base_trends = self.trend_keywords.get("2025", [])
        
        platform_specific = {
            "tiktok": ["dance", "challenge", "storytime", "pov"],
            "instagram": ["aesthetic", "lifestyle", "tutorial", "behindthescenes"],
            "youtube": ["explained", "tutorial", "reaction", "documentary"]
        }
        
        return base_trends + platform_specific.get(platform, [])
        
    def optimize_for_algorithm(
        self,
        content: CreativeContent,
        platform: str
    ) -> CreativeContent:
        """Optimize content for platform algorithm"""
        if platform == "tiktok":
            # TikTok favors watch time and completion
            if len(content.hook) < 10:
                content.hook = "Wait for the end... " + content.hook
                
        elif platform == "instagram":
            # Instagram favors saves and shares
            if "save" not in content.call_to_action.lower():
                content.call_to_action = "Save this for later! " + content.call_to_action
                
        elif platform == "youtube":
            # YouTube favors click-through rate
            if not content.title.endswith(("?", "!")):
                content.title = content.title + " (MUST WATCH)"
                
        return content


class MultiPlatformOptimizer:
    """Optimize content for multiple platforms"""
    
    def __init__(self):
        self.title_generator = AICreativeTitleGenerator()
        self.trend_analyzer = TrendAnalyzer()
        
    async def generate_multi_platform_content(
        self,
        segment: Dict[str, Any]
    ) -> Dict[str, CreativeContent]:
        """Generate optimized content for all platforms"""
        results = {}
        
        for platform in ["tiktok", "instagram", "youtube"]:
            # Generate base content
            content = await self.title_generator.generate_creative_content(
                segment, platform
            )
            
            # Optimize for algorithm
            content = self.trend_analyzer.optimize_for_algorithm(content, platform)
            
            results[platform] = content
            
        return results