"""
Empirical Testing for Capsule Memory Architecture

Validates architectural choices through contrastive experiments:
1. Why 32D capsules vs raw 384D embeddings?
2. Why DG sparse expansion (2% sparsity)?
3. Why injection at layers 4-5?
4. Why cognitive features matter?
5. Pattern separation effectiveness

Each test provides quantitative evidence for design decisions.
"""

import time
import numpy as np
import logging
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger(__name__)

# Import our modules
from vulkan_capsule_transformer import (
    VulkanCapsuleTransformer,
    CapsuleTransformerConfig,
    CapsuleMemory,
    CognitiveFeatures,
    MemoryType,
    DentateGyrus
)
from ca3_memory_store import CA3MemoryStore, CA3MemoryIndex


@dataclass
class TestResult:
    """Result of an empirical test"""
    test_name: str
    hypothesis: str
    metric_name: str
    baseline_value: float
    proposed_value: float
    improvement: float
    improvement_pct: float
    conclusion: str
    details: Dict[str, Any]


class EmpiricalTester:
    """
    Runs empirical tests to validate architectural choices.
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.results: List[TestResult] = []
        
        # Initialize models
        self._init_models()
    
    def _init_models(self):
        """Initialize test models"""
        # Small config for fast testing
        self.config = CapsuleTransformerConfig(
            hidden_dim=384,
            num_layers=6,
            num_heads=6,
            capsule_dim=32,
            injection_layers=(4, 5)
        )
        
        self.model = VulkanCapsuleTransformer(config=self.config)
        self.memory_store = CA3MemoryStore(self.model, capacity=10000)
    
    def _log(self, msg: str):
        if self.verbose:
            # Handle Unicode issues on Windows
            try:
                print(msg)
            except UnicodeEncodeError:
                print(msg.encode('ascii', 'replace').decode('ascii'))
    
    def run_all_tests(self) -> List[TestResult]:
        """Run all empirical tests"""
        self._log("\n" + "="*70)
        self._log("EMPIRICAL VALIDATION OF CAPSULE MEMORY ARCHITECTURE")
        self._log("="*70)
        
        tests = [
            self.test_capsule_vs_raw_embeddings,
            self.test_dg_sparsity_levels,
            self.test_injection_layer_position,
            self.test_cognitive_features_impact,
            self.test_pattern_separation,
            self.test_retrieval_accuracy,
            self.test_memory_interference,
        ]
        
        for test in tests:
            try:
                result = test()
                self.results.append(result)
                self._print_result(result)
            except Exception as e:
                self._log(f"[ERROR] {test.__name__}: {e}")
        
        self._print_summary()
        return self.results
    
    def _print_result(self, result: TestResult):
        """Print test result"""
        self._log(f"\n{'-'*70}")
        self._log(f"TEST: {result.test_name}")
        self._log(f"{'-'*70}")
        self._log(f"Hypothesis: {result.hypothesis}")
        self._log(f"Metric: {result.metric_name}")
        self._log(f"  Baseline:  {result.baseline_value:.4f}")
        self._log(f"  Proposed:  {result.proposed_value:.4f}")
        self._log(f"  Improvement: {result.improvement:+.4f} ({result.improvement_pct:+.1f}%)")
        self._log(f"Conclusion: {result.conclusion}")
    
    def _print_summary(self):
        """Print summary of all tests"""
        self._log("\n" + "="*70)
        self._log("SUMMARY")
        self._log("="*70)
        
        passed = sum(1 for r in self.results if r.improvement_pct > 0)
        total = len(self.results)
        
        self._log(f"\nTests Passed: {passed}/{total}")
        self._log("\nKey Findings:")
        
        for r in self.results:
            status = "[PASS]" if r.improvement_pct > 0 else "[FAIL]"
            self._log(f"  {status} {r.test_name}: {r.improvement_pct:+.1f}%")
        
        self._log("\n" + "="*70)
    
    # =========================================================================
    # Test 1: Capsule (32D) vs Raw Embeddings (384D)
    # =========================================================================
    
    def test_capsule_vs_raw_embeddings(self) -> TestResult:
        """
        Test: Does 32D capsule retain semantic quality while being more efficient?
        
        Hypothesis: Capsule embeddings (32D) preserve semantic similarity rankings
        compared to raw embeddings (384D) while being 12x smaller.
        
        Metric: Spearman correlation of similarity rankings
        """
        self._log("\n[Test 1] Capsule vs Raw Embeddings...")
        
        # Test sentences with known semantic relationships
        test_pairs = [
            ("The cat sat on the mat", "A feline rested on the rug"),
            ("Machine learning uses neural networks", "Deep learning employs artificial neurons"),
            ("The weather is sunny today", "It's a bright and clear day"),
            ("Python is a programming language", "JavaScript is used for web development"),
            ("The stock market crashed", "Financial markets experienced a downturn"),
            ("She plays the piano beautifully", "He plays guitar with skill"),
            ("The car is red", "The automobile is crimson"),
            ("I love eating pizza", "The weather is cold"),  # Unrelated
        ]
        
        raw_similarities = []
        capsule_similarities = []
        
        for text1, text2 in test_pairs:
            # Raw embeddings (384D)
            emb1 = self.model.encode(text1)
            emb2 = self.model.encode(text2)
            raw_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8)
            raw_similarities.append(raw_sim)
            
            # Capsule embeddings (32D)
            cap1 = self.model.encode_to_capsule(text1)
            cap2 = self.model.encode_to_capsule(text2)
            cap_sim = np.dot(cap1, cap2) / (np.linalg.norm(cap1) * np.linalg.norm(cap2) + 1e-8)
            capsule_similarities.append(cap_sim)
        
        # Compute rank correlation
        raw_ranks = np.argsort(np.argsort(raw_similarities))
        cap_ranks = np.argsort(np.argsort(capsule_similarities))
        
        # Spearman correlation
        n = len(raw_ranks)
        d_squared = np.sum((raw_ranks - cap_ranks) ** 2)
        spearman = 1 - (6 * d_squared) / (n * (n**2 - 1))
        
        # Size comparison
        raw_size = 384 * 4  # bytes
        cap_size = 32 * 4
        size_ratio = raw_size / cap_size
        
        return TestResult(
            test_name="Capsule vs Raw Embeddings",
            hypothesis="32D capsules preserve semantic similarity rankings from 384D embeddings",
            metric_name="Spearman Rank Correlation",
            baseline_value=1.0,  # Perfect correlation baseline
            proposed_value=spearman,
            improvement=spearman - 0.7,  # vs 0.7 threshold
            improvement_pct=(spearman - 0.7) / 0.7 * 100,
            conclusion=f"Capsules maintain {spearman:.1%} ranking correlation at {size_ratio:.0f}x compression",
            details={
                'raw_similarities': raw_similarities,
                'capsule_similarities': capsule_similarities,
                'compression_ratio': size_ratio,
                'raw_dim': 384,
                'capsule_dim': 32
            }
        )
    
    # =========================================================================
    # Test 2: DG Sparsity Levels
    # =========================================================================
    
    def test_dg_sparsity_levels(self) -> TestResult:
        """
        Test: What is the optimal sparsity level for DG expansion?
        
        Hypothesis: ~2% sparsity provides best pattern separation while
        maintaining sufficient representational capacity.
        
        Metric: Pattern separation ratio (dissimilarity of similar inputs)
        """
        self._log("\n[Test 2] DG Sparsity Levels...")
        
        sparsity_levels = [0.01, 0.02, 0.05, 0.10, 0.20, 0.50]
        results = {}
        
        # Generate similar input pairs
        np.random.seed(42)
        base_capsules = np.random.randn(10, 32).astype(np.float32)
        
        # Normalize
        base_capsules = base_capsules / (np.linalg.norm(base_capsules, axis=1, keepdims=True) + 1e-8)
        
        # Create similar variants (add small noise)
        noise_level = 0.1
        similar_capsules = base_capsules + np.random.randn(10, 32).astype(np.float32) * noise_level
        similar_capsules = similar_capsules / (np.linalg.norm(similar_capsules, axis=1, keepdims=True) + 1e-8)
        
        for sparsity in sparsity_levels:
            dg = DentateGyrus(CapsuleTransformerConfig(dg_sparsity=sparsity))
            
            # Expand both sets
            base_dg = dg.expand(base_capsules)
            similar_dg = dg.expand(similar_capsules)
            
            # Compute pattern separation (how different are similar inputs after DG?)
            input_sims = np.array([
                np.dot(base_capsules[i], similar_capsules[i])
                for i in range(10)
            ])
            
            output_sims = np.array([
                np.dot(base_dg[i], similar_dg[i])
                for i in range(10)
            ])
            
            # Pattern separation ratio: higher = more separation
            separation_ratio = 1 - np.mean(output_sims) / np.mean(input_sims)
            
            # Active neurons
            active_pct = np.mean(np.sum(base_dg != 0, axis=1)) / 128 * 100
            
            results[sparsity] = {
                'separation_ratio': separation_ratio,
                'active_neurons_pct': active_pct,
                'input_similarity': np.mean(input_sims),
                'output_similarity': np.mean(output_sims)
            }
        
        # Find optimal
        optimal_sparsity = max(results.keys(), key=lambda s: results[s]['separation_ratio'])
        baseline_sparsity = 0.10  # Dense baseline
        
        return TestResult(
            test_name="DG Sparsity Optimization",
            hypothesis="~2% sparsity provides optimal pattern separation",
            metric_name="Pattern Separation Ratio",
            baseline_value=results[baseline_sparsity]['separation_ratio'],
            proposed_value=results[0.02]['separation_ratio'],
            improvement=results[0.02]['separation_ratio'] - results[baseline_sparsity]['separation_ratio'],
            improvement_pct=(results[0.02]['separation_ratio'] - results[baseline_sparsity]['separation_ratio']) / 
                           (results[baseline_sparsity]['separation_ratio'] + 1e-8) * 100,
            conclusion=f"2% sparsity achieves {results[0.02]['separation_ratio']:.1%} separation "
                      f"(optimal: {optimal_sparsity*100:.0f}%)",
            details=results
        )
    
    # =========================================================================
    # Test 3: Injection Layer Position
    # =========================================================================
    
    def test_injection_layer_position(self) -> TestResult:
        """
        Test: Which layers are optimal for memory injection?
        
        Hypothesis: Middle layers (4-5) are optimal because:
        - Early layers: Too low-level, disrupts basic features
        - Late layers: Too high-level, not enough propagation
        
        Metric: Output difference magnitude when injecting memories
        """
        self._log("\n[Test 3] Injection Layer Position...")
        
        # Create test memory
        test_memory = CapsuleMemory(
            memory_id="test",
            memory_type=MemoryType.CONCEPT,
            domain="test",
            content="Memory injection test content",
            cognitive_features=CognitiveFeatures(plasticity_gain=0.9)
        )
        test_memory.capsule_vector = np.random.randn(32).astype(np.float32)
        test_memory.capsule_vector /= np.linalg.norm(test_memory.capsule_vector)
        
        # Test input
        np.random.seed(42)
        input_ids = np.random.randint(0, self.config.vocab_size, (1, 64), dtype=np.int32)
        mask = np.ones((1, 64), dtype=np.float32)
        
        # Baseline (no injection)
        baseline_emb = self.model.forward(input_ids, mask)
        
        layer_configs = {
            'early': (0, 1),
            'middle': (2, 3),
            'optimal': (4, 5),
            'late': (5,)
        }
        
        results = {}
        for name, layers in layer_configs.items():
            # Create model with specific injection layers
            config = CapsuleTransformerConfig(
                hidden_dim=384,
                num_layers=6,
                num_heads=6,
                injection_layers=layers
            )
            model = VulkanCapsuleTransformer(config=config)
            
            # Copy weights from main model
            model.token_embeddings = self.model.token_embeddings.copy()
            model.emb_ln_gamma = self.model.emb_ln_gamma.copy()
            model.emb_ln_beta = self.model.emb_ln_beta.copy()
            model.capsule_proj = self.model.capsule_proj.copy()
            model.inject_proj = self.model.inject_proj.copy()
            for i, layer in enumerate(model.layers):
                for key in layer:
                    layer[key] = self.model.layers[i][key].copy()
            
            # Forward with injection
            injected_emb = model.forward(input_ids, mask, inject_memories=[test_memory])
            
            # Measure impact
            diff = np.linalg.norm(injected_emb - baseline_emb)
            results[name] = {
                'layers': layers,
                'diff_magnitude': diff
            }
        
        return TestResult(
            test_name="Injection Layer Position",
            hypothesis="Layers 4-5 provide optimal memory injection",
            metric_name="Embedding Difference Magnitude",
            baseline_value=results['middle']['diff_magnitude'],
            proposed_value=results['optimal']['diff_magnitude'],
            improvement=results['optimal']['diff_magnitude'] - results['middle']['diff_magnitude'],
            improvement_pct=(results['optimal']['diff_magnitude'] - results['middle']['diff_magnitude']) /
                           (results['middle']['diff_magnitude'] + 1e-8) * 100,
            conclusion=f"Layers 4-5 injection produces {results['optimal']['diff_magnitude']:.4f} embedding shift",
            details=results
        )

    
    # =========================================================================
    # Test 4: Cognitive Features Impact
    # =========================================================================
    
    def test_cognitive_features_impact(self) -> TestResult:
        """
        Test: Do cognitive features (plasticity, stability, etc.) affect retrieval?
        
        Hypothesis: Memories with different cognitive features should cluster
        differently, allowing for context-aware retrieval.
        
        Metric: Intra-class vs inter-class distance ratio
        """
        self._log("\n[Test 4] Cognitive Features Impact...")
        
        # Create memories with different cognitive profiles
        high_plasticity = []
        high_stability = []
        high_stress = []
        
        base_texts = [
            "Learning about machine learning",
            "Studying neural networks",
            "Understanding deep learning",
            "Exploring AI concepts",
            "Reading about algorithms"
        ]
        
        for text in base_texts:
            # High plasticity (learning mode)
            cap = self.model.encode_to_capsule(
                text, 
                CognitiveFeatures(plasticity_gain=0.9, stability=0.3, stress_link=0.1)
            )
            high_plasticity.append(cap)
            
            # High stability (consolidated)
            cap = self.model.encode_to_capsule(
                text,
                CognitiveFeatures(plasticity_gain=0.3, stability=0.9, stress_link=0.1)
            )
            high_stability.append(cap)
            
            # High stress (emotional)
            cap = self.model.encode_to_capsule(
                text,
                CognitiveFeatures(plasticity_gain=0.5, stability=0.5, stress_link=0.9)
            )
            high_stress.append(cap)
        
        # Convert to arrays
        high_plasticity = np.array(high_plasticity)
        high_stability = np.array(high_stability)
        high_stress = np.array(high_stress)
        
        # Compute intra-class distances (within each cognitive profile)
        def mean_pairwise_dist(vectors):
            n = len(vectors)
            dists = []
            for i in range(n):
                for j in range(i+1, n):
                    dists.append(np.linalg.norm(vectors[i] - vectors[j]))
            return np.mean(dists)
        
        intra_plasticity = mean_pairwise_dist(high_plasticity)
        intra_stability = mean_pairwise_dist(high_stability)
        intra_stress = mean_pairwise_dist(high_stress)
        avg_intra = (intra_plasticity + intra_stability + intra_stress) / 3
        
        # Compute inter-class distances (between cognitive profiles)
        def mean_cross_dist(v1, v2):
            dists = []
            for a in v1:
                for b in v2:
                    dists.append(np.linalg.norm(a - b))
            return np.mean(dists)
        
        inter_ps = mean_cross_dist(high_plasticity, high_stability)
        inter_pt = mean_cross_dist(high_plasticity, high_stress)
        inter_st = mean_cross_dist(high_stability, high_stress)
        avg_inter = (inter_ps + inter_pt + inter_st) / 3
        
        # Ratio: higher = better separation of cognitive states
        separation_ratio = avg_inter / (avg_intra + 1e-8)
        
        # Also check cognitive feature preservation
        cog_preservation = []
        for i, cap in enumerate(high_plasticity):
            cog_preservation.append(cap[28])  # plasticity in dim 28
        
        return TestResult(
            test_name="Cognitive Features Impact",
            hypothesis="Cognitive features create meaningful clustering in capsule space",
            metric_name="Inter/Intra Class Distance Ratio",
            baseline_value=1.0,  # Random would be ~1.0
            proposed_value=separation_ratio,
            improvement=separation_ratio - 1.0,
            improvement_pct=(separation_ratio - 1.0) * 100,
            conclusion=f"Cognitive features create {separation_ratio:.2f}x separation between states",
            details={
                'avg_intra_class': avg_intra,
                'avg_inter_class': avg_inter,
                'cognitive_preservation': np.mean(cog_preservation)
            }
        )
    
    # =========================================================================
    # Test 5: Pattern Separation Effectiveness
    # =========================================================================
    
    def test_pattern_separation(self) -> TestResult:
        """
        Test: Does DG effectively separate similar patterns?
        
        Hypothesis: DG sparse expansion should transform similar inputs
        into more distinct representations (pattern separation).
        
        Metric: Reduction in overlap between similar input pairs
        """
        self._log("\n[Test 5] Pattern Separation Effectiveness...")
        
        # Create pairs of similar texts
        similar_pairs = [
            ("The quick brown fox jumps over the lazy dog",
             "The fast brown fox leaps over the sleepy dog"),
            ("Machine learning is transforming technology",
             "Machine learning is revolutionizing technology"),
            ("The stock market experienced gains today",
             "The stock market saw increases today"),
            ("She wrote a beautiful poem about nature",
             "She composed a lovely poem about nature"),
            ("The restaurant serves excellent Italian food",
             "The restaurant offers great Italian cuisine"),
        ]
        
        capsule_overlaps = []
        dg_overlaps = []
        
        for text1, text2 in similar_pairs:
            # Encode to capsules
            cap1 = self.model.encode_to_capsule(text1)
            cap2 = self.model.encode_to_capsule(text2)
            
            # Expand to DG
            dg1 = self.model.dg.expand(cap1)
            dg2 = self.model.dg.expand(cap2)
            
            # Compute overlap (cosine similarity)
            cap_overlap = np.dot(cap1, cap2) / (np.linalg.norm(cap1) * np.linalg.norm(cap2) + 1e-8)
            dg_overlap = np.dot(dg1, dg2) / (np.linalg.norm(dg1) * np.linalg.norm(dg2) + 1e-8)
            
            capsule_overlaps.append(cap_overlap)
            dg_overlaps.append(dg_overlap)
        
        # Pattern separation = reduction in overlap
        avg_cap_overlap = np.mean(capsule_overlaps)
        avg_dg_overlap = np.mean(dg_overlaps)
        separation_effectiveness = (avg_cap_overlap - avg_dg_overlap) / avg_cap_overlap
        
        return TestResult(
            test_name="Pattern Separation Effectiveness",
            hypothesis="DG reduces overlap between similar patterns",
            metric_name="Overlap Reduction",
            baseline_value=avg_cap_overlap,
            proposed_value=avg_dg_overlap,
            improvement=avg_cap_overlap - avg_dg_overlap,
            improvement_pct=separation_effectiveness * 100,
            conclusion=f"DG reduces overlap by {separation_effectiveness:.1%} "
                      f"({avg_cap_overlap:.3f} → {avg_dg_overlap:.3f})",
            details={
                'capsule_overlaps': capsule_overlaps,
                'dg_overlaps': dg_overlaps,
                'pair_count': len(similar_pairs)
            }
        )
    
    # =========================================================================
    # Test 6: Retrieval Accuracy
    # =========================================================================
    
    def test_retrieval_accuracy(self) -> TestResult:
        """
        Test: How accurate is memory retrieval?
        
        Hypothesis: DG-based retrieval should accurately match queries
        to relevant memories with high precision.
        
        Metric: Top-k retrieval accuracy
        """
        self._log("\n[Test 6] Retrieval Accuracy...")
        
        # Create domain-specific memories
        domains = {
            'tech': [
                "Python is a programming language",
                "JavaScript runs in browsers",
                "Machine learning uses neural networks",
                "APIs enable software communication",
                "Databases store structured data"
            ],
            'food': [
                "Pizza is an Italian dish",
                "Sushi comes from Japan",
                "Tacos are Mexican food",
                "Curry is popular in India",
                "Croissants are French pastries"
            ],
            'science': [
                "Photosynthesis converts light to energy",
                "DNA contains genetic information",
                "Gravity attracts objects together",
                "Atoms are building blocks of matter",
                "Evolution explains species change"
            ]
        }
        
        # Add memories
        memory_store = CA3MemoryStore(self.model, capacity=100)
        domain_memories = {}
        
        for domain, texts in domains.items():
            domain_memories[domain] = []
            for text in texts:
                mem = CapsuleMemory(
                    memory_id=f"{domain}_{len(domain_memories[domain])}",
                    memory_type=MemoryType.CONCEPT,
                    domain=domain,
                    content=text,
                    cognitive_features=CognitiveFeatures()
                )
                memory_store.add_memory(mem)
                domain_memories[domain].append(mem)
        
        # Test queries
        test_queries = {
            'tech': "How do computers communicate with each other?",
            'food': "What are popular Italian dishes?",
            'science': "How do living things get energy from sunlight?"
        }
        
        correct_retrievals = 0
        total_retrievals = 0
        
        for expected_domain, query in test_queries.items():
            results = memory_store.query(query, k=3)
            
            for mem, dist in results:
                total_retrievals += 1
                if mem.domain == expected_domain:
                    correct_retrievals += 1
        
        accuracy = correct_retrievals / total_retrievals if total_retrievals > 0 else 0
        
        return TestResult(
            test_name="Retrieval Accuracy",
            hypothesis="DG-based retrieval accurately matches queries to domains",
            metric_name="Top-3 Domain Accuracy",
            baseline_value=0.33,  # Random baseline (3 domains)
            proposed_value=accuracy,
            improvement=accuracy - 0.33,
            improvement_pct=(accuracy - 0.33) / 0.33 * 100,
            conclusion=f"Retrieval achieves {accuracy:.1%} domain accuracy (vs 33% random)",
            details={
                'correct': correct_retrievals,
                'total': total_retrievals,
                'domains': list(domains.keys())
            }
        )
    
    # =========================================================================
    # Test 7: Memory Interference
    # =========================================================================
    
    def test_memory_interference(self) -> TestResult:
        """
        Test: Does DG sparse coding reduce interference between memories?
        
        Hypothesis: Sparse DG representations should have less overlap,
        reducing catastrophic interference when adding new memories.
        
        Metric: Retrieval stability after adding interfering memories
        """
        self._log("\n[Test 7] Memory Interference...")
        
        # Create initial memories
        original_memories = [
            ("orig_1", "The capital of France is Paris"),
            ("orig_2", "Water freezes at zero degrees Celsius"),
            ("orig_3", "The sun is a star"),
        ]
        
        # Create interfering memories (similar but different)
        interfering_memories = [
            ("int_1", "The capital of Germany is Berlin"),
            ("int_2", "Water boils at one hundred degrees Celsius"),
            ("int_3", "The moon is a satellite"),
        ]
        
        # Build store with originals
        store = CA3MemoryStore(self.model, capacity=100)
        for mem_id, content in original_memories:
            mem = CapsuleMemory(
                memory_id=mem_id,
                memory_type=MemoryType.CONCEPT,
                domain="facts",
                content=content,
                cognitive_features=CognitiveFeatures()
            )
            store.add_memory(mem)
        
        # Query originals and record positions
        original_queries = [
            "What is the capital of France?",
            "At what temperature does water freeze?",
            "What type of celestial body is the sun?"
        ]
        
        pre_interference_ranks = []
        for i, query in enumerate(original_queries):
            results = store.query(query, k=3)
            for rank, (mem, _) in enumerate(results):
                if mem.memory_id == original_memories[i][0]:
                    pre_interference_ranks.append(rank)
                    break
            else:
                pre_interference_ranks.append(-1)  # Not found
        
        # Add interfering memories
        for mem_id, content in interfering_memories:
            mem = CapsuleMemory(
                memory_id=mem_id,
                memory_type=MemoryType.CONCEPT,
                domain="facts",
                content=content,
                cognitive_features=CognitiveFeatures()
            )
            store.add_memory(mem)
        
        # Query again
        post_interference_ranks = []
        for i, query in enumerate(original_queries):
            results = store.query(query, k=6)  # More results now
            for rank, (mem, _) in enumerate(results):
                if mem.memory_id == original_memories[i][0]:
                    post_interference_ranks.append(rank)
                    break
            else:
                post_interference_ranks.append(-1)
        
        # Compute rank stability
        pre_avg_rank = np.mean([r for r in pre_interference_ranks if r >= 0])
        post_avg_rank = np.mean([r for r in post_interference_ranks if r >= 0])
        
        # Lower rank = better (0 is best)
        rank_degradation = post_avg_rank - pre_avg_rank
        
        return TestResult(
            test_name="Memory Interference Resistance",
            hypothesis="Sparse DG coding resists interference from similar memories",
            metric_name="Average Rank Degradation",
            baseline_value=1.0,  # Expected degradation without protection
            proposed_value=rank_degradation,
            improvement=1.0 - rank_degradation,
            improvement_pct=(1.0 - rank_degradation) / 1.0 * 100,
            conclusion=f"Rank degradation: {rank_degradation:.2f} positions "
                      f"(pre: {pre_avg_rank:.1f}, post: {post_avg_rank:.1f})",
            details={
                'pre_ranks': pre_interference_ranks,
                'post_ranks': post_interference_ranks,
                'original_count': len(original_memories),
                'interfering_count': len(interfering_memories)
            }
        )


# =============================================================================
# Run Tests
# =============================================================================

def run_empirical_tests():
    """Run all empirical tests and generate report"""
    tester = EmpiricalTester(verbose=True)
    results = tester.run_all_tests()
    return results


if __name__ == "__main__":
    run_empirical_tests()
