"""
Experiment configuration for RAG evaluation.
Easily define and run experiments with different parameter combinations.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum


class ExperimentType(Enum):
    """Type of experiment."""
    RETRIEVAL_PARAMS = "retrieval_params"  # Retrieval parameter tuning
    ORCHESTRATOR = "orchestrator"  # With/without orchestrator
    CHUNKING = "chunking"  # Chunking strategy
    EMBEDDING = "embedding"  # Different embedding models
    COMBINED = "combined"  # Multiple factors


@dataclass
class RetrievalConfig:
    """Configuration for retrieval parameters."""

    # Basic retrieval
    k: int = 5  # Number of documents to retrieve

    # MMR parameters
    use_mmr: bool = True
    mmr_fetch_k: int = 20
    mmr_lambda: float = 0.8  # 0=max diversity, 1=max relevance

    # Reranking parameters
    use_reranking: bool = True
    rank_alpha: float = 0.8  # Weight for similarity score
    rank_beta: float = 0.2  # Weight for rank score

    # Adaptive Top-K
    use_adaptive_k: bool = True
    min_k: int = 2
    max_k: int = 10
    similarity_threshold: float = 0.7

    # Cross-Encoder
    use_cross_encoder: bool = False
    cross_encoder_model: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "k": self.k,
            "use_mmr": self.use_mmr,
            "mmr_fetch_k": self.mmr_fetch_k,
            "mmr_lambda": self.mmr_lambda,
            "use_reranking": self.use_reranking,
            "rank_alpha": self.rank_alpha,
            "rank_beta": self.rank_beta,
            "use_adaptive_k": self.use_adaptive_k,
            "min_k": self.min_k,
            "max_k": self.max_k,
            "similarity_threshold": self.similarity_threshold,
            "use_cross_encoder": self.use_cross_encoder,
            "cross_encoder_model": self.cross_encoder_model,
        }

    def get_name(self) -> str:
        """Get a short name for this configuration."""
        name_parts = []

        if self.use_adaptive_k:
            name_parts.append(f"adaptive_k_{self.min_k}-{self.max_k}")
        else:
            name_parts.append(f"k_{self.k}")

        if self.use_mmr:
            name_parts.append(f"mmr_λ{self.mmr_lambda}")
        else:
            name_parts.append("no_mmr")

        if self.use_cross_encoder:
            name_parts.append("cross_encoder")
        elif self.use_reranking:
            name_parts.append(f"rerank_α{self.rank_alpha}")
        else:
            name_parts.append("no_rerank")

        return "_".join(name_parts)






@dataclass
class OrchestratorConfig:
    """Configuration for orchestrator."""

    enabled: bool = False
    model_name: Optional[str] = None
    chain_model_name: Optional[str] = None
    verbose: bool = False
    
    # Ablation controls
    use_rewriter: bool = True
    use_quality_check: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "enabled": self.enabled,
            "model_name": self.model_name,
            "chain_model_name": self.chain_model_name,
            "verbose": self.verbose,
            "use_rewriter": self.use_rewriter,
            "use_quality_check": self.use_quality_check,
        }

    def get_name(self) -> str:
        """Get a short name for this configuration."""
        if not self.enabled:
            return "orchestrator_off"
            
        parts = ["orchestrator_on"]
        if not self.use_rewriter:
            parts.append("no_rewrite")
        if not self.use_quality_check:
            parts.append("no_qc")
        return "_".join(parts)


@dataclass
class ChunkingConfig:
    """Configuration for document chunking."""

    chunk_size: int = 1000
    chunk_overlap: int = 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
        }

    def get_name(self) -> str:
        """Get a short name for this configuration."""
        return f"chunk_{self.chunk_size}_overlap_{self.chunk_overlap}"


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""

    name: str
    description: str = ""

    # Component configs
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    orchestrator: OrchestratorConfig = field(default_factory=OrchestratorConfig)
    chunking: Optional[ChunkingConfig] = None

    # Evaluation settings
    query_types: Optional[List[str]] = None  # None = all types
    sample_size: Optional[int] = None  # None = all queries

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        config = {
            "name": self.name,
            "description": self.description,
            "retrieval": self.retrieval.to_dict(),
            "orchestrator": self.orchestrator.to_dict(),
            "query_types": self.query_types,
            "sample_size": self.sample_size,
        }

        if self.chunking:
            config["chunking"] = self.chunking.to_dict()

        return config

    def get_full_name(self) -> str:
        """Get full descriptive name for this experiment."""
        parts = [self.name]

        if self.orchestrator.enabled:
            parts.append(self.orchestrator.get_name())

        parts.append(self.retrieval.get_name())

        if self.chunking:
            parts.append(self.chunking.get_name())

        return "__".join(parts)


class ExperimentPresets:
    """Predefined experiment configurations."""

    @staticmethod
    def improved_retrieval() -> ExperimentConfig:
        """Improved retrieval: Adaptive K + Cross-Encoder."""
        return ExperimentConfig(
            name="improved_retrieval",
            description="Improved: Relative Drop Adaptive K + Cross-Encoder",
            retrieval=RetrievalConfig(
                k=5,
                use_mmr=True,
                mmr_lambda=0.7,  # Tuned for slightly more diversity before reranking
                use_reranking=True, # Still true to allow flow, but CE takes precedence
                use_adaptive_k=True,
                use_cross_encoder=True,
            ),
            orchestrator=OrchestratorConfig(enabled=False),
        )

    @staticmethod
    def ablation_cross_encoder() -> ExperimentConfig:
        """Ablation: Improved Adaptive K ONLY (No Cross-Encoder)."""
        return ExperimentConfig(
            name="ablation_cross_encoder",
            description="Ablation: Relative Drop Adaptive K without Cross-Encoder",
            retrieval=RetrievalConfig(
                k=5,
                use_mmr=True,
                mmr_lambda=0.7,
                use_reranking=True, # Uses standard bestseller rerank
                use_adaptive_k=True,
                use_cross_encoder=False,
            ),
            orchestrator=OrchestratorConfig(enabled=False),
        )

    @staticmethod
    def comparison_sales_vs_semantic() -> List[ExperimentConfig]:
        """Compare Sales Rerank vs Semantic Rerank directly."""
        return [
            ExperimentConfig(
                name="sales_rerank",
                description="Bestseller Reranking (Old Logic)",
                retrieval=RetrievalConfig(
                    k=5,
                    use_mmr=True,
                    use_reranking=True,
                    use_adaptive_k=True,
                    use_cross_encoder=False,
                ),
                orchestrator=OrchestratorConfig(enabled=False),
            ),
            ExperimentConfig(
                name="semantic_rerank",
                description="Cross-Encoder Reranking (New Logic)",
                retrieval=RetrievalConfig(
                    k=5,
                    use_mmr=True,
                    use_reranking=False, # Disable sales rank
                    use_adaptive_k=True,
                    use_cross_encoder=True,
                ),
                orchestrator=OrchestratorConfig(enabled=False),
            ),
        ]

    @staticmethod
    def baseline() -> ExperimentConfig:
        """Baseline: Default settings without orchestrator."""
        return ExperimentConfig(
            name="baseline",
            description="Baseline with default retrieval settings, no orchestrator",
            retrieval=RetrievalConfig(
                k=5,
                use_mmr=True,
                use_reranking=True,
                use_adaptive_k=True,
            ),
            orchestrator=OrchestratorConfig(enabled=False),
        )

    @staticmethod
    def orchestrator_enabled() -> ExperimentConfig:
        """With orchestrator enabled."""
        return ExperimentConfig(
            name="orchestrator",
            description="With ambiguity-aware orchestrator",
            retrieval=RetrievalConfig(
                k=5,
                use_mmr=True,
                use_reranking=True,
                use_adaptive_k=True,
            ),
            orchestrator=OrchestratorConfig(enabled=True),
        )

    @staticmethod
    def no_mmr() -> ExperimentConfig:
        """Without MMR diversity."""
        return ExperimentConfig(
            name="no_mmr",
            description="Baseline without MMR diversity",
            retrieval=RetrievalConfig(
                k=5,
                use_mmr=False,
                use_reranking=True,
                use_adaptive_k=False,
            ),
            orchestrator=OrchestratorConfig(enabled=False),
        )

    @staticmethod
    def no_reranking() -> ExperimentConfig:
        """Without reranking."""
        return ExperimentConfig(
            name="no_reranking",
            description="Baseline without bestseller reranking",
            retrieval=RetrievalConfig(
                k=5,
                use_mmr=True,
                use_reranking=False,
                use_adaptive_k=True,
            ),
            orchestrator=OrchestratorConfig(enabled=False),
        )

    @staticmethod
    def fixed_k_small() -> ExperimentConfig:
        """Fixed k=3 (small)."""
        return ExperimentConfig(
            name="fixed_k3",
            description="Fixed k=3 without adaptive",
            retrieval=RetrievalConfig(
                k=3,
                use_mmr=True,
                use_reranking=True,
                use_adaptive_k=False,
            ),
            orchestrator=OrchestratorConfig(enabled=False),
        )

    @staticmethod
    def fixed_k_large() -> ExperimentConfig:
        """Fixed k=10 (large)."""
        return ExperimentConfig(
            name="fixed_k10",
            description="Fixed k=10 without adaptive",
            retrieval=RetrievalConfig(
                k=10,
                use_mmr=True,
                use_reranking=True,
                use_adaptive_k=False,
            ),
            orchestrator=OrchestratorConfig(enabled=False),
        )

    @staticmethod
    def high_diversity() -> ExperimentConfig:
        """High diversity (low MMR lambda)."""
        return ExperimentConfig(
            name="high_diversity",
            description="MMR with high diversity (lambda=0.5)",
            retrieval=RetrievalConfig(
                k=5,
                use_mmr=True,
                mmr_lambda=0.5,  # More diversity
                use_reranking=True,
                use_adaptive_k=True,
            ),
            orchestrator=OrchestratorConfig(enabled=False),
        )

    @staticmethod
    def high_relevance() -> ExperimentConfig:
        """High relevance (high MMR lambda)."""
        return ExperimentConfig(
            name="high_relevance",
            description="MMR with high relevance (lambda=0.95)",
            retrieval=RetrievalConfig(
                k=5,
                use_mmr=True,
                mmr_lambda=0.95,  # More relevance
                use_reranking=True,
                use_adaptive_k=True,
            ),
            orchestrator=OrchestratorConfig(enabled=False),
        )

    @staticmethod
    def minimal() -> ExperimentConfig:
        """Minimal: No advanced features."""
        return ExperimentConfig(
            name="minimal",
            description="Minimal setup: no MMR, no reranking, fixed k",
            retrieval=RetrievalConfig(
                k=5,
                use_mmr=False,
                use_reranking=False,
                use_adaptive_k=False,
            ),
            orchestrator=OrchestratorConfig(enabled=False),
        )

    @staticmethod
    def all_presets() -> List[ExperimentConfig]:
        """Get all preset experiments."""
        return [
            ExperimentPresets.baseline(),
            ExperimentPresets.orchestrator_enabled(),
            ExperimentPresets.no_mmr(),
            ExperimentPresets.no_reranking(),
            ExperimentPresets.fixed_k_small(),
            ExperimentPresets.fixed_k_large(),
            ExperimentPresets.high_diversity(),
            ExperimentPresets.high_relevance(),
            ExperimentPresets.minimal(),
        ]

    @staticmethod
    def ablation_study() -> List[ExperimentConfig]:
        """
        Ablation study: Test impact of each component.

        Returns list of experiments that progressively add features.
        """
        return [
            # Start minimal
            ExperimentPresets.minimal(),

            # Add MMR
            ExperimentConfig(
                name="ablation_mmr",
                description="Minimal + MMR",
                retrieval=RetrievalConfig(
                    k=5, use_mmr=True, use_reranking=False, use_adaptive_k=False
                ),
                orchestrator=OrchestratorConfig(enabled=False),
            ),

            # Add Reranking
            ExperimentConfig(
                name="ablation_rerank",
                description="Minimal + Reranking",
                retrieval=RetrievalConfig(
                    k=5, use_mmr=False, use_reranking=True, use_adaptive_k=False
                ),
                orchestrator=OrchestratorConfig(enabled=False),
            ),

            # Add Adaptive K
            ExperimentConfig(
                name="ablation_adaptive",
                description="Minimal + Adaptive K",
                retrieval=RetrievalConfig(
                    k=5, use_mmr=False, use_reranking=False, use_adaptive_k=True
                ),
                orchestrator=OrchestratorConfig(enabled=False),
            ),

            # All features
            ExperimentPresets.baseline(),

            # All features + Orchestrator
            ExperimentPresets.orchestrator_enabled(),
        ]

    @staticmethod
    def ablation_orchestrator() -> List[ExperimentConfig]:
        """
        Ablation study for Orchestrator chains.
        
        Compare:
        1. Full Orchestrator
        2. No Query Rewriter (Ambiguity check done, but skipped rewrite)
        3. No Quality Check (Skip sufficiency check & clarification)
        """
        base_retrieval = RetrievalConfig(
            k=5, use_mmr=True, use_reranking=True, use_adaptive_k=True
        )

        return [
            # 1. Full Orchestrator
            ExperimentConfig(
                name="orchestrator_full",
                description="Full Orchestrator (Rewrite + QC)",
                retrieval=base_retrieval,
                orchestrator=OrchestratorConfig(
                    enabled=True, use_rewriter=True, use_quality_check=True
                ),
            ),
            # 2. No Rewrite
            ExperimentConfig(
                name="orchestrator_no_rewrite",
                description="Orchestrator without Query Rewriting",
                retrieval=base_retrieval,
                orchestrator=OrchestratorConfig(
                    enabled=True, use_rewriter=False, use_quality_check=True
                ),
            ),
            # 3. No Quality Check
            ExperimentConfig(
                name="orchestrator_no_qc",
                description="Orchestrator without Quality Check",
                retrieval=base_retrieval,
                orchestrator=OrchestratorConfig(
                    enabled=True, use_rewriter=True, use_quality_check=False
                ),
            ),
        ]

    @staticmethod
    def k_sweep() -> List[ExperimentConfig]:
        """Sweep different k values."""
        experiments = []

        for k in [2, 3, 5, 7, 10]:
            experiments.append(
                ExperimentConfig(
                    name=f"k_sweep_{k}",
                    description=f"Fixed k={k}",
                    retrieval=RetrievalConfig(
                        k=k,
                        use_mmr=True,
                        use_reranking=True,
                        use_adaptive_k=False,
                    ),
                    orchestrator=OrchestratorConfig(enabled=False),
                )
            )

        return experiments

    @staticmethod
    def lambda_sweep() -> List[ExperimentConfig]:
        """Sweep MMR lambda values."""
        experiments = []

        for lambda_val in [0.3, 0.5, 0.7, 0.8, 0.9, 0.95]:
            experiments.append(
                ExperimentConfig(
                    name=f"lambda_sweep_{int(lambda_val*100)}",
                    description=f"MMR lambda={lambda_val}",
                    retrieval=RetrievalConfig(
                        k=5,
                        use_mmr=True,
                        mmr_lambda=lambda_val,
                        use_reranking=True,
                        use_adaptive_k=True,
                    ),
                    orchestrator=OrchestratorConfig(enabled=False),
                )
            )

        return experiments

    @staticmethod
    def rerank_weight_sweep() -> List[ExperimentConfig]:
        """Sweep reranking weight combinations."""
        experiments = []

        # alpha + beta should sum to 1.0
        for alpha in [0.5, 0.6, 0.7, 0.8, 0.9]:
            beta = 1.0 - alpha
            experiments.append(
                ExperimentConfig(
                    name=f"rerank_alpha_{int(alpha*100)}",
                    description=f"Rerank alpha={alpha}, beta={beta}",
                    retrieval=RetrievalConfig(
                        k=5,
                        use_mmr=True,
                        use_reranking=True,
                        rank_alpha=alpha,
                        rank_beta=beta,
                        use_adaptive_k=True,
                    ),
                    orchestrator=OrchestratorConfig(enabled=False),
                )
            )

        return experiments


if __name__ == "__main__":
    # Demo preset experiments
    print("=== Available Experiment Presets ===\n")

    presets = ExperimentPresets.all_presets()
    for i, preset in enumerate(presets, 1):
        print(f"{i}. {preset.name}")
        print(f"   Description: {preset.description}")
        print(f"   Full name: {preset.get_full_name()}")
        print()

    print("\n=== Ablation Study Experiments ===\n")
    ablation = ExperimentPresets.ablation_study()
    for i, exp in enumerate(ablation, 1):
        print(f"{i}. {exp.name}: {exp.description}")

    print("\n=== Parameter Sweep Experiments ===\n")
    print(f"K sweep: {len(ExperimentPresets.k_sweep())} experiments")
    print(f"Lambda sweep: {len(ExperimentPresets.lambda_sweep())} experiments")
    print(f"Rerank weight sweep: {len(ExperimentPresets.rerank_weight_sweep())} experiments")
