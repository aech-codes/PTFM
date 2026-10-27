"""
PTFM Domain Plugin Interface

File:
    ptfm/plugins/base.py

Description:
    Defines the abstract interface that every domain plugin
    (Legal, Finance, Healthcare, Scientific Literature,
    Enterprise, Cybersecurity, etc.) must implement.

The PTFM Core MUST NOT contain any domain-specific logic.
All domain knowledge is encapsulated within plugins that
implement this interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ptfm.core.schemas import (
    AtomicClaim,
    EvidenceItem,
    RetrievalRequest,
    RetrievalResult,
)


class BaseDomainPlugin(ABC):
    """
    Abstract base class for all PTFM domain plugins.

    Every plugin must provide implementations for retrieval,
    symbolic reasoning, source credibility estimation,
    and domain-specific preprocessing.

    The PTFM Core communicates ONLY through this interface.
    """

    # ======================================================================
    # Plugin Metadata
    # ======================================================================

    @property
    @abstractmethod
    def domain_name(self) -> str:
        """
        Returns the unique domain identifier.

        Examples
        --------
        "legal"
        "finance"
        "healthcare"
        "scientific"
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def plugin_version(self) -> str:
        """
        Returns the plugin version.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def supported_languages(self) -> List[str]:
        """
        Languages supported by the plugin.

        Example
        -------
        ["en", "hi"]
        """
        raise NotImplementedError

    # ======================================================================
    # Query Processing
    # ======================================================================

    @abstractmethod
    async def preprocess_query(
        self,
        query: str,
    ) -> str:
        """
        Perform domain-specific preprocessing.

        Examples
        --------
        - Legal abbreviation expansion
        - Medical terminology normalization
        - Financial ticker normalization
        """
        raise NotImplementedError

    @abstractmethod
    async def build_retrieval_request(
        self,
        query: str,
        top_k: int = 10,
    ) -> RetrievalRequest:
        """
        Construct a retrieval request.

        Returns
        -------
        RetrievalRequest
        """
        raise NotImplementedError

    # ======================================================================
    # Evidence Retrieval
    # ======================================================================

    @abstractmethod
    async def retrieve_evidence(
        self,
        request: RetrievalRequest,
    ) -> RetrievalResult:
        """
        Execute hybrid retrieval.

        The implementation may combine

        - Dense Retrieval
        - Sparse Retrieval
        - Graph Retrieval
        """
        raise NotImplementedError

    # ======================================================================
    # Evidence Processing
    # ======================================================================

    @abstractmethod
    async def normalize_evidence(
        self,
        evidence: List[EvidenceItem],
    ) -> List[EvidenceItem]:
        """
        Normalize retrieved evidence into
        the canonical PTFM schema.
        """
        raise NotImplementedError

    @abstractmethod
    async def rank_evidence(
        self,
        evidence: List[EvidenceItem],
    ) -> List[EvidenceItem]:
        """
        Re-rank retrieved evidence.
        """
        raise NotImplementedError

    # ======================================================================
    # Symbolic Verification
    # ======================================================================

    @abstractmethod
    async def apply_symbolic_rules(
        self,
        claim: AtomicClaim,
        evidence: List[EvidenceItem],
    ) -> bool:
        """
        Execute deterministic symbolic verification.

        Examples
        --------
        Legal:
            Statutory consistency

        Finance:
            Numerical consistency

        Healthcare:
            Guideline verification
        """
        raise NotImplementedError

    # ======================================================================
    # Source Credibility
    # ======================================================================

    @abstractmethod
    async def source_credibility(
        self,
        source_uri: str,
    ) -> Optional[float]:
        """
        Estimate credibility of a source.

        Returns
        -------
        Optional[float]

        Range
        -----
        [0,1], or None when the plugin cannot assess the source.

        Returning None
        --------------
        A plugin MUST return None rather than a placeholder when it cannot
        assess a source. It must not substitute a neutral value, a prior, or
        a conservative guess.

        This value is consumed directly by retrieval reliability

            r(E) = (1/k) * sum_i  r_i * s_i

        which in turn drives hallucination risk

            H = S(q) * (1 - r(E))

        so a substituted credibility enters a statistical guarantee
        indistinguishable from a measured one. On a three-item result set with
        one unscoreable source, the choice of substitute moves H by more than
        half its value. Discarding the item instead is not neutral either: it
        keeps only the sources that could be scored, so reliability rises and
        risk falls, recording ignorance as reliability.

        None propagates to EvidenceItem.credibility_score, and the layer that
        owns the statistical guarantee decides what an unknown credibility
        means. That decision does not belong to the plugin or to retrieval.
        """
        raise NotImplementedError

    # ======================================================================
    # Knowledge Graph
    # ======================================================================

    @abstractmethod
    async def build_knowledge_graph(
        self,
        evidence: List[EvidenceItem],
    ) -> Optional[Any]:
        """
        Construct a domain-specific
        knowledge graph representation.

        Returns
        -------
        Optional[Any]

        None if unsupported.
        """
        raise NotImplementedError

    # ======================================================================
    # Domain Metadata
    # ======================================================================

    @abstractmethod
    async def plugin_metadata(self) -> Dict[str, Any]:
        """
        Returns metadata describing
        the plugin.

        Example
        -------
        {
            "domain": "finance",
            "version": "1.0",
            "retrieval": ["dense","sparse"],
            "symbolic_rules": True
        }
        """
        raise NotImplementedError

    # ======================================================================
    # Health Check
    # ======================================================================

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Verify that the plugin is
        operational.

        Checks may include

        - database connectivity
        - vector database
        - graph database
        - API availability
        """
        raise NotImplementedError
