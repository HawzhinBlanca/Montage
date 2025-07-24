#!/usr/bin/env python3
"""
Test Director Pipeline Integration
Tests AI orchestration with VideoDB Director
"""
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch, AsyncMock
import pytest

from montage.core.director_wrapper import (
    DirectorOrchestrator,
    director_orchestrator,
    run_director_pipeline
)


class TestDirectorOrchestrator:
    """Test DirectorOrchestrator class"""
    
    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator instance"""
        with patch('montage.core.director_wrapper.DIRECTOR_AVAILABLE', True):
            with patch('montage.core.director_wrapper.Director') as mock_director:
                orchestrator = DirectorOrchestrator()
                orchestrator.director = mock_director.return_value
                return orchestrator
    
    def test_init_without_director(self):
        """Test initialization when Director not available"""
        with patch('montage.core.director_wrapper.DIRECTOR_AVAILABLE', False):
            orchestrator = DirectorOrchestrator()
            assert orchestrator.director is None
    
    def test_init_with_director(self):
        """Test initialization when Director is available"""
        with patch('montage.core.director_wrapper.DIRECTOR_AVAILABLE', True):
            with patch('montage.core.director_wrapper.Director') as mock_director:
                with patch.object(DirectorOrchestrator, '_register_agents') as mock_register:
                    orchestrator = DirectorOrchestrator()
                    assert orchestrator.director is not None
                    mock_register.assert_called_once()
    
    def test_register_agents(self, orchestrator):
        """Test agent registration"""
        # Mock all the imports
        with patch('montage.core.director_wrapper.WhisperTranscriber'):
            with patch('montage.core.director_wrapper.VisualTracker'):
                with patch('montage.core.director_wrapper.VideoEditor'):
                    with patch('montage.core.director_wrapper.SmartTrack'):
                        with patch('montage.core.director_wrapper.get_diarizer'):
                            with patch('montage.core.director_wrapper.AudioNormalizer'):
                                orchestrator._register_agents()
                                
                                # Verify agents were added
                                assert orchestrator.director.add_agent.call_count >= 7
                                
                                # Check specific agent registrations
                                agent_calls = orchestrator.director.add_agent.call_args_list
                                agent_names = [call[0][0] for call in agent_calls]
                                
                                assert "transcribe" in agent_names
                                assert "track_objects" in agent_names
                                assert "analyze_smart" in agent_names
                                assert "normalize_audio" in agent_names
                                assert "edit_video" in agent_names
                                assert "select_highlights" in agent_names
    
    def test_run_pipeline_with_director(self, orchestrator):
        """Test running pipeline with Director available"""
        # Mock Director run result
        director_result = {
            "transcript": {"text": "Hello world"},
            "highlights": [{"start": 0, "end": 10}],
            "output_path": "/tmp/output.mp4"
        }
        orchestrator.director.run.return_value = director_result
        
        result = orchestrator.run_pipeline("test_video.mp4")
        
        assert result["success"] is True
        assert result["pipeline"] == "director"
        assert result["results"] == director_result
        
        # Verify Director was called with default instruction
        orchestrator.director.run.assert_called_once()
        call_args = orchestrator.director.run.call_args
        assert "Analyze this video comprehensively" in call_args[0][0]
        assert call_args[1]["context"]["video_path"] == "test_video.mp4"
    
    def test_run_pipeline_with_custom_instruction(self, orchestrator):
        """Test running pipeline with custom instruction"""
        orchestrator.director.run.return_value = {"status": "complete"}
        
        custom_instruction = "Extract only face close-ups"
        result = orchestrator.run_pipeline("test_video.mp4", custom_instruction)
        
        # Verify custom instruction was used
        orchestrator.director.run.assert_called_once()
        call_args = orchestrator.director.run.call_args
        assert call_args[0][0] == custom_instruction
    
    def test_run_pipeline_director_failure(self, orchestrator):
        """Test fallback when Director fails"""
        orchestrator.director.run.side_effect = Exception("Director error")
        
        with patch.object(orchestrator, '_run_fallback_pipeline') as mock_fallback:
            mock_fallback.return_value = {"success": True, "pipeline": "fallback"}
            
            result = orchestrator.run_pipeline("test_video.mp4")
            
            assert result["pipeline"] == "fallback"
            mock_fallback.assert_called_once_with("test_video.mp4")
    
    def test_process_director_output_dict(self, orchestrator):
        """Test processing Director dict output"""
        director_output = {"key": "value", "nested": {"data": 123}}
        result = orchestrator._process_director_output(director_output)
        
        assert result["success"] is True
        assert result["pipeline"] == "director"
        assert result["results"] == director_output
    
    def test_process_director_output_json_string(self, orchestrator):
        """Test processing Director JSON string output"""
        director_output = '{"key": "value", "number": 42}'
        result = orchestrator._process_director_output(director_output)
        
        assert result["results"] == {"key": "value", "number": 42}
    
    def test_process_director_output_plain_string(self, orchestrator):
        """Test processing Director plain string output"""
        director_output = "Processing complete"
        result = orchestrator._process_director_output(director_output)
        
        assert result["results"] == {"message": "Processing complete"}
    
    def test_run_fallback_pipeline(self, orchestrator):
        """Test fallback pipeline execution"""
        # Mock all components
        mock_transcript = {"text": "Test transcript"}
        mock_analysis = {"segments": [{"start": 0, "end": 10, "score": 0.8}]}
        mock_highlights = [{"start_time": 0, "end_time": 10}]
        
        with patch('montage.core.director_wrapper.WhisperTranscriber') as mock_transcriber_class:
            with patch('montage.core.director_wrapper.SmartTrack') as mock_smart_class:
                with patch('montage.core.director_wrapper.HighlightSelector') as mock_selector_class:
                    with patch('montage.core.director_wrapper.VideoEditor') as mock_editor_class:
                        # Setup mocks
                        mock_transcriber = mock_transcriber_class.return_value
                        mock_transcriber.transcribe_video.return_value = mock_transcript
                        
                        mock_smart = mock_smart_class.return_value
                        mock_smart.analyze_video = AsyncMock(return_value=mock_analysis)
                        
                        mock_selector = mock_selector_class.return_value
                        mock_selector.select_highlights.return_value = mock_highlights
                        
                        mock_editor = mock_editor_class.return_value
                        
                        # Run fallback
                        result = orchestrator._run_fallback_pipeline("test_video.mp4")
                        
                        assert result["success"] is True
                        assert result["pipeline"] == "fallback"
                        assert result["results"]["transcript"] == mock_transcript
                        assert result["results"]["highlights"] == mock_highlights
                        assert "output_path" in result["results"]
    
    def test_create_custom_pipeline(self, orchestrator):
        """Test creating custom pipeline with specific agents"""
        orchestrator.director.run.return_value = {"custom": "result"}
        
        agents = ["transcribe", "select_highlights"]
        context = {"video_path": "test.mp4", "duration": 30}
        
        result = orchestrator.create_custom_pipeline(agents, context)
        
        assert result["success"] is True
        assert result["results"] == {"custom": "result"}
        
        # Verify Director was called correctly
        call_args = orchestrator.director.run.call_args
        assert "transcribe" in call_args[0][0]
        assert "select_highlights" in call_args[0][0]
        assert call_args[1]["context"] == context
    
    def test_create_custom_pipeline_no_director(self):
        """Test custom pipeline when Director not available"""
        with patch('montage.core.director_wrapper.DIRECTOR_AVAILABLE', False):
            orchestrator = DirectorOrchestrator()
            result = orchestrator.create_custom_pipeline(["test"], {})
            
            assert result["success"] is False
            assert "not available" in result["error"]
    
    def test_list_available_agents(self, orchestrator):
        """Test listing available agents"""
        agents = orchestrator.list_available_agents()
        
        assert isinstance(agents, list)
        assert len(agents) >= 7
        
        agent_names = [a["name"] for a in agents]
        assert "transcribe" in agent_names
        assert "track_objects" in agent_names
        assert "analyze_smart" in agent_names
        assert "edit_video" in agent_names
    
    def test_edit_wrapper(self, orchestrator):
        """Test video edit wrapper function"""
        with patch('montage.core.director_wrapper.VideoEditor') as mock_editor_class:
            mock_editor = mock_editor_class.return_value
            
            wrapper = orchestrator._create_edit_wrapper()
            clips = [{"start": 0, "end": 10}]
            
            result = wrapper("source.mp4", clips, "output.mp4")
            
            assert result["output_path"] == "output.mp4"
            assert result["clips_processed"] == 1
            mock_editor.process_clips.assert_called_once_with(clips, "output.mp4")
    
    def test_highlight_wrapper(self, orchestrator):
        """Test highlight selection wrapper"""
        with patch('montage.core.director_wrapper.HighlightSelector') as mock_selector_class:
            mock_selector = mock_selector_class.return_value
            mock_selector.select_highlights.return_value = [{"highlight": 1}]
            
            wrapper = orchestrator._create_highlight_wrapper()
            analysis = {"segments": [{"start": 0, "end": 10, "score": 0.9}]}
            
            result = wrapper(analysis, target_duration=30)
            
            assert result == [{"highlight": 1}]
            mock_selector.select_highlights.assert_called_once()
            call_args = mock_selector.select_highlights.call_args
            assert call_args[0][0] == analysis["segments"]
            assert call_args[1]["target_duration"] == 30


class TestModuleFunctions:
    """Test module-level functions"""
    
    def test_run_director_pipeline(self):
        """Test convenience function"""
        with patch.object(director_orchestrator, 'run_pipeline') as mock_run:
            mock_run.return_value = {"success": True}
            
            result = run_director_pipeline("video.mp4", "Do something")
            
            assert result["success"] is True
            mock_run.assert_called_once_with("video.mp4", "Do something")
    
    def test_global_instance(self):
        """Test global orchestrator instance"""
        assert director_orchestrator is not None
        assert isinstance(director_orchestrator, DirectorOrchestrator)


class TestIntegration:
    """Integration tests with mocked components"""
    
    @pytest.mark.asyncio
    async def test_full_pipeline_flow(self):
        """Test complete pipeline flow with all components"""
        with patch('montage.core.director_wrapper.DIRECTOR_AVAILABLE', True):
            with patch('montage.core.director_wrapper.Director') as mock_director_class:
                # Create orchestrator
                orchestrator = DirectorOrchestrator()
                orchestrator.director = mock_director_class.return_value
                
                # Mock Director execution
                orchestrator.director.run.return_value = {
                    "transcript": {
                        "text": "This is a test video",
                        "segments": [{"start": 0, "end": 5, "text": "This is a test video"}]
                    },
                    "tracks": [
                        {"frame": 0, "objects": [{"id": 1, "bbox": [100, 100, 200, 200]}]}
                    ],
                    "highlights": [
                        {"start_time": 0, "end_time": 5, "score": 0.9}
                    ],
                    "output": {
                        "path": "/tmp/highlights.mp4",
                        "duration": 5.0
                    }
                }
                
                # Run pipeline
                result = orchestrator.run_pipeline(
                    "test_video.mp4",
                    "Create a 5-second highlight reel"
                )
                
                # Verify results
                assert result["success"] is True
                assert result["pipeline"] == "director"
                assert "transcript" in result["results"]
                assert "highlights" in result["results"]
                assert result["results"]["output"]["duration"] == 5.0
    
    def test_tasks_md_exact_pattern(self):
        """Test exact pattern from Tasks.md"""
        with patch('montage.core.director_wrapper.DIRECTOR_AVAILABLE', True):
            with patch('montage.core.director_wrapper.director') as mock_director:
                # Mock the exact call from Tasks.md
                mock_director.run.return_value = {
                    "clips": [
                        {"start": 0, "end": 5, "has_speech": True, "has_person": True},
                        {"start": 10, "end": 15, "has_speech": True, "has_person": True}
                    ],
                    "status": "complete"
                }
                
                # Import the function that matches Tasks.md
                from montage.core.director_wrapper import run_simple_director_example
                
                # Run exactly as shown in Tasks.md
                result = run_simple_director_example()
                
                # Verify Director.run() was called with exact instruction
                mock_director.run.assert_called_once_with(
                    "Extract clips where people speak and track them"
                )
                
                # Assert Director.run() returns expected structure
                assert result is not None
                assert "clips" in result
                assert len(result["clips"]) == 2
                assert result["clips"][0]["has_speech"] is True
                assert result["clips"][0]["has_person"] is True
                
                print("✅ Tasks.md pattern verified: director.run() returns expected structure")


if __name__ == "__main__":
    # Run with: pytest tests/test_director_pipeline.py -v -s
    pytest.main([__file__, "-v", "-s"])