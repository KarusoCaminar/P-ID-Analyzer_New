"""
Unit tests for cache service (MultiLevelCache).
"""

import sys
from pathlib import Path
import tempfile
import shutil
import time

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pytest

from src.services.cache_service import MultiLevelCache


@pytest.fixture
def temp_cache_dir():
    """Create a temporary cache directory."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


class TestMultiLevelCache:
    """Tests for MultiLevelCache."""
    
    def test_cache_get_set(self, temp_cache_dir):
        """Test basic get/set operations."""
        cache = MultiLevelCache(temp_cache_dir, memory_size=10, disk_size_gb=0.1)
        
        # Set value
        cache.set("key1", "value1")
        
        # Get value
        value = cache.get("key1")
        
        assert value == "value1", "Should retrieve stored value"
    
    def test_cache_memory_promotion(self, temp_cache_dir):
        """Test that disk hits are promoted to memory."""
        cache = MultiLevelCache(temp_cache_dir, memory_size=10, disk_size_gb=0.1)
        
        # Set value (goes to both memory and disk)
        cache.set("key1", "value1")
        
        # Clear memory cache (but keep disk)
        with cache.memory_lock:
            cache.memory_cache.clear()
        
        # Get from disk (should promote to memory)
        value = cache.get("key1")
        
        assert value == "value1", "Should retrieve from disk and promote to memory"
        assert "key1" in cache.memory_cache, "Should be promoted to memory"
    
    def test_cache_memory_lru_eviction(self, temp_cache_dir):
        """Test LRU eviction in memory cache."""
        cache = MultiLevelCache(temp_cache_dir, memory_size=3, disk_size_gb=0.1)
        
        # Fill memory cache beyond limit
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")
        cache.set("key4", "value4")  # Should evict key1
        
        # Check that key1 was evicted from memory
        with cache.memory_lock:
            assert "key1" not in cache.memory_cache, "Oldest key should be evicted"
            assert "key4" in cache.memory_cache, "Newest key should be in memory"
        
        # But key1 should still be in disk
        value = cache.get("key1")
        assert value == "value1", "Should still be in disk cache"
    
    def test_cache_ttl_expiration(self, temp_cache_dir):
        """Test TTL expiration in memory cache."""
        cache = MultiLevelCache(temp_cache_dir, memory_size=10, disk_size_gb=0.1, memory_ttl_hours=0.0001)  # Very short TTL
        
        # Set value
        cache.set("key1", "value1")
        
        # Wait for TTL to expire
        time.sleep(0.01)
        
        # Try to get (should be expired from memory, but still in disk)
        value = cache.get("key1")
        
        # Should still work (from disk)
        assert value == "value1", "Should retrieve from disk after memory expiration"
    
    def test_cache_delete(self, temp_cache_dir):
        """Test delete operation."""
        cache = MultiLevelCache(temp_cache_dir, memory_size=10, disk_size_gb=0.1)
        
        # Set value
        cache.set("key1", "value1")
        
        # Delete
        cache.delete("key1")
        
        # Try to get
        value = cache.get("key1")
        
        assert value is None, "Deleted key should return None"
    
    def test_cache_clear(self, temp_cache_dir):
        """Test clear operation."""
        cache = MultiLevelCache(temp_cache_dir, memory_size=10, disk_size_gb=0.1)
        
        # Set multiple values
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")
        
        # Clear
        cache.clear()
        
        # Try to get
        assert cache.get("key1") is None, "Should be cleared"
        assert cache.get("key2") is None, "Should be cleared"
        assert cache.get("key3") is None, "Should be cleared"
    
    def test_cache_contains(self, temp_cache_dir):
        """Test __contains__ method."""
        cache = MultiLevelCache(temp_cache_dir, memory_size=10, disk_size_gb=0.1)
        
        # Set value
        cache.set("key1", "value1")
        
        # Test contains
        assert "key1" in cache, "Should contain key"
        assert "key2" not in cache, "Should not contain missing key"
    
    def test_cache_getitem(self, temp_cache_dir):
        """Test __getitem__ method."""
        cache = MultiLevelCache(temp_cache_dir, memory_size=10, disk_size_gb=0.1)
        
        # Set value
        cache.set("key1", "value1")
        
        # Test getitem
        value = cache["key1"]
        assert value == "value1", "Should retrieve value using [] syntax"
        
        # Test KeyError for missing key
        with pytest.raises(KeyError):
            _ = cache["missing_key"]
    
    def test_cache_setitem(self, temp_cache_dir):
        """Test __setitem__ method."""
        cache = MultiLevelCache(temp_cache_dir, memory_size=10, disk_size_gb=0.1)
        
        # Test setitem
        cache["key1"] = "value1"
        
        # Verify
        assert cache.get("key1") == "value1", "Should store value using [] syntax"
    
    def test_cache_complex_objects(self, temp_cache_dir):
        """Test caching complex objects (dicts, lists)."""
        cache = MultiLevelCache(temp_cache_dir, memory_size=10, disk_size_gb=0.1)
        
        complex_obj = {
            'elements': [
                {'id': 'elem1', 'bbox': {'x': 0, 'y': 0, 'width': 100, 'height': 100}},
                {'id': 'elem2', 'bbox': {'x': 200, 'y': 200, 'width': 50, 'height': 50}}
            ],
            'connections': [
                {'from_id': 'elem1', 'to_id': 'elem2'}
            ]
        }
        
        cache.set("complex_key", complex_obj)
        retrieved = cache.get("complex_key")
        
        assert retrieved == complex_obj, "Should cache and retrieve complex objects"
        assert len(retrieved['elements']) == 2, "Complex object structure should be preserved"
    
    def test_cache_stats(self, temp_cache_dir):
        """Test cache statistics."""
        cache = MultiLevelCache(temp_cache_dir, memory_size=10, disk_size_gb=0.1)
        
        # Set some values
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        
        stats = cache.stats()
        
        assert 'memory_entries' in stats, "Stats should include memory entries"
        assert 'disk_entries' in stats, "Stats should include disk entries"
        assert stats['memory_entries'] >= 0, "Memory entries should be non-negative"
        assert stats['disk_entries'] >= 0, "Disk entries should be non-negative"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

