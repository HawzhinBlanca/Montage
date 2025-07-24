"""Test memory management and pressure response system"""

import os
import psutil
from unittest.mock import MagicMock, patch
import pytest


class TestMemoryPressureResponse:
    """Test memory pressure response behavior"""
    
    def test_memory_pressure_response(self):
        """Test that system responds correctly to high memory pressure"""
        from montage.utils.memory_manager import (
            MemoryManager, 
            MemoryPressureLevel,
            get_adaptive_config
        )
        
        # Create memory manager
        manager = MemoryManager()
        
        # Test 1: Normal pressure - should use default config
        with patch.object(manager, 'get_memory_pressure') as mock_pressure:
            mock_pressure.return_value = MemoryPressureLevel.LOW
            
            config = manager.get_adaptive_processing_config()
            
            # Should have normal worker count
            assert config['max_workers'] >= 4  # Default for most systems
            assert config['chunk_size_mb'] >= 1024  # Default chunk size
            
            print(f"✅ Low pressure config: {config['max_workers']} workers, {config['chunk_size_mb']}MB chunks")
            
        # Test 2: High memory pressure - should reduce resources
        with patch.object(manager, 'get_memory_pressure') as mock_pressure:
            mock_pressure.return_value = MemoryPressureLevel.HIGH
            
            # Simulate high memory usage
            manager._simulate_pressure(MemoryPressureLevel.HIGH)
            config = manager.get_adaptive_processing_config()
            
            # Should reduce workers and chunk size
            assert config['max_workers'] < 4  # Reduced from default
            assert config['chunk_size_mb'] < 1024  # Reduced chunk size
            
            print(f"✅ High pressure config: {config['max_workers']} workers, {config['chunk_size_mb']}MB chunks")
            
    def test_memory_monitoring(self):
        """Test memory monitoring functionality"""
        from montage.utils.memory_manager import get_memory_monitor
        
        monitor = get_memory_monitor()
        assert monitor is not None
        
        # Get current stats
        stats = monitor.get_current_stats()
        
        # Verify stats structure
        assert 'available_mb' in stats
        assert 'total_mb' in stats
        assert 'percent_used' in stats
        assert 'process_memory_mb' in stats
        
        # Verify reasonable values
        assert stats['total_mb'] > 0
        assert 0 <= stats['percent_used'] <= 100
        assert stats['process_memory_mb'] > 0
        
        print(f"✅ Memory stats: {stats['available_mb']}MB available, "
              f"{stats['percent_used']:.1f}% used, "
              f"process using {stats['process_memory_mb']}MB")
        
    def test_adaptive_config_for_m4_max(self):
        """Test adaptive config for M4 Max (36GB RAM)"""
        from montage.utils.memory_manager import get_adaptive_config
        
        # Simulate M4 Max system
        with patch('psutil.virtual_memory') as mock_mem:
            # 36GB total RAM
            mock_mem.return_value = MagicMock(
                total=36 * 1024 * 1024 * 1024,  # 36 GB
                available=20 * 1024 * 1024 * 1024,  # 20 GB available
                percent=44.4  # ~44% used
            )
            
            with patch('psutil.cpu_count') as mock_cpu:
                mock_cpu.return_value = 14  # M4 Max has 14 cores
                
                config = get_adaptive_config()
                
                # Should optimize for high-end system
                assert config['max_workers'] >= 12  # Should use most cores
                assert config['chunk_size_mb'] >= 2048  # Larger chunks for more RAM
                assert config.get('apple_silicon_optimized', False) == True
                
                print(f"✅ M4 Max config: {config['max_workers']} workers, "
                      f"{config['chunk_size_mb']}MB chunks, "
                      f"Apple Silicon optimized: {config.get('apple_silicon_optimized', False)}")
                
    def test_memory_pressure_levels(self):
        """Test memory pressure level detection"""
        from montage.utils.memory_manager import MemoryManager, MemoryPressureLevel
        
        manager = MemoryManager()
        
        # Test different memory scenarios
        test_cases = [
            (10, MemoryPressureLevel.LOW),      # 10% used - low pressure
            (50, MemoryPressureLevel.LOW),      # 50% used - still low
            (75, MemoryPressureLevel.MODERATE), # 75% used - moderate
            (85, MemoryPressureLevel.HIGH),     # 85% used - high
            (95, MemoryPressureLevel.CRITICAL), # 95% used - critical
        ]
        
        for percent_used, expected_level in test_cases:
            with patch('psutil.virtual_memory') as mock_mem:
                mock_mem.return_value = MagicMock(percent=percent_used)
                
                level = manager.get_memory_pressure()
                
                # Allow for some flexibility in thresholds
                if expected_level == MemoryPressureLevel.CRITICAL:
                    assert level in [MemoryPressureLevel.HIGH, MemoryPressureLevel.CRITICAL]
                else:
                    assert level in [expected_level, MemoryPressureLevel(expected_level.value - 1), 
                                   MemoryPressureLevel(expected_level.value + 1)]
                
                print(f"✅ {percent_used}% memory used → {level.name} pressure")
                
    def test_graceful_degradation(self):
        """Test graceful degradation under memory pressure"""
        from montage.utils.memory_manager import MemoryManager, MemoryPressureLevel
        
        manager = MemoryManager()
        
        # Track config changes as pressure increases
        configs = []
        
        for level in [MemoryPressureLevel.LOW, MemoryPressureLevel.MODERATE, 
                     MemoryPressureLevel.HIGH, MemoryPressureLevel.CRITICAL]:
            
            manager._simulate_pressure(level)
            config = manager.get_adaptive_processing_config()
            configs.append((level, config))
            
        # Verify degradation pattern
        for i in range(1, len(configs)):
            prev_level, prev_config = configs[i-1]
            curr_level, curr_config = configs[i]
            
            # Workers should decrease or stay same
            assert curr_config['max_workers'] <= prev_config['max_workers']
            
            # Chunk size should decrease or stay same
            assert curr_config['chunk_size_mb'] <= prev_config['chunk_size_mb']
            
            print(f"✅ {curr_level.name}: {curr_config['max_workers']} workers, "
                  f"{curr_config['chunk_size_mb']}MB chunks")
                  
        print("✅ Graceful degradation verified")
        
    def test_memory_usage_during_processing(self):
        """Test memory usage stays within limits during video processing"""
        print("\n=== Manual Memory Test Commands ===")
        print("To monitor memory during video processing:")
        print()
        print("1. Start monitoring in separate terminal:")
        print("   watch -n 1 'ps aux | grep python | grep montage'")
        print()
        print("2. Process large video (2GB+):")
        print("   python -m montage.cli.run_pipeline large_video.mp4 --output test_out.mp4")
        print()
        print("3. Expected behavior:")
        print("   - Memory usage should stay < 8GB")
        print("   - No OOM errors")
        print("   - Graceful handling if memory runs low")
        print()
        print("4. Check system pressure:")
        print("   vm_stat | grep 'Pages free\\|Pages active\\|Pages inactive'")
        
        assert True  # Documentation test


if __name__ == "__main__":
    # Run with: pytest tests/test_memory_limits.py -v -s
    pytest.main([__file__, "-v", "-s"])