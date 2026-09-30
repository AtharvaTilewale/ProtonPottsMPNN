"""
PottsMPNN whole-chain contact crop.

Selects the query PN unit and, with probability `complex_pair_probability`,
adds contacting protein chains from pre-computed contact metadata stored in
``extra_info["q_pn_unit_contacting_pn_unit_iids"]``.

Contact format (JSON list or Python list):
    [{"pn_unit_iid": "A_1", "num_atoms": 545, "num_contacts": 1017, ...}, ...]

Token-budget enforcement (max tokens per batch) is handled downstream by
TokenBudgetAwareFeatureCollator; this transform only controls chain selection.

Adapted from:
    gefion_project/.../src/datasets/grpo_chain_crop.py  (chain_subset_crop)
    gefion_project/.../src/datasets/pipelines/modded_RF3_pipeline.py  (get_whole_chain_crop_transform)
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass

import numpy as np
from atomworks.ml.transforms.base import Transform

logger = logging.getLogger(__name__)

try:
    from atomworks.enums import ChainType
    _PROTEIN_CHAIN_TYPES = frozenset([ChainType.POLYPEPTIDE_L, ChainType.POLYPEPTIDE_D])
except Exception:
    _PROTEIN_CHAIN_TYPES = None


@dataclass
class CropChainsAroundSpecifiedPNUnitsConfig:
    """Configuration for CropChainsAroundSpecifiedPNUnits.

    Mirrors ChainSubsetCropConfig from grpo_chain_crop.py with the addition
    of complex_pair_probability and protein_only_partners.

    Args:
        complex_pair_probability: Probability in [0, 1] of adding partner
            chains. 0.0 = always monomer; 1.0 = always add valid partners.
        max_atoms: Total atom budget for query + all partners combined.
        min_atom_contacts: Minimum pre-computed atom–atom contacts required
            to include a partner chain.
        max_chains: Optional hard cap on the number of chains in the crop
            (includes the query chain).
        query_pn_unit_key: Key in ``data`` holding the query pn_unit_iid.
        contacts_key: Key in ``extra_info`` holding the contacts JSON list.
        protein_only_partners: Skip non-protein partner chains (recommended:
            PottsMPNN uses a protein-only token vocabulary).
    """
    complex_pair_probability: float = 0.5
    max_atoms: int = 5000
    min_atom_contacts: int = 10
    max_chains: int | None = None
    query_pn_unit_key: str = "query_pn_unit_iids"
    contacts_key: str = "q_pn_unit_contacting_pn_unit_iids"
    protein_only_partners: bool = True


def _is_protein_pn_unit(atom_array, pn_unit_iid: str) -> bool:
    mask = atom_array.pn_unit_iid == pn_unit_iid
    if not mask.any():
        return False
    if _PROTEIN_CHAIN_TYPES is not None and hasattr(atom_array, "chain_type"):
        return bool(set(atom_array.chain_type[mask]) & _PROTEIN_CHAIN_TYPES)
    if hasattr(atom_array, "is_protein"):
        return bool(atom_array.is_protein[mask].any())
    return True


class CropChainsAroundSpecifiedPNUnits(Transform):
    """PottsMPNN training crop: query PN unit plus all valid partner chains (all-or-nothing).

    Always keeps the whole query PN unit chain. With probability
    ``complex_pair_probability``, also collects ALL contacting protein chains
    that pass ``min_atom_contacts`` and ``max_chains``, then includes them only
    if the entire complex fits within ``max_atoms``. If any partner would be
    dropped due to the atom budget, the crop falls back to monomer — a partial
    complex with some contacts present and others absent would expose non-native
    interfaces during training.

    At ``complex_pair_probability=0`` behaves identically to a monomer filter.

    Falls back to keeping only the query chain (DEBUG log) when:
      - ``pn_unit_iid`` annotation is absent from ``atom_array``
      - ``query_pn_unit_iids`` is not in ``data``
      - The query pn_unit_iid is not found in the atom array
      - No contact metadata is available in ``extra_info``

    Stores a summary in ``data["extra_info"]["mpnn_chain_subset"]``.
    """

    def __init__(
        self,
        complex_pair_probability: float = 0.5,
        max_atoms: int = 5000,
        min_atom_contacts: int = 10,
        max_chains: int | None = None,
        query_pn_unit_key: str = "query_pn_unit_iids",
        contacts_key: str = "q_pn_unit_contacting_pn_unit_iids",
        protein_only_partners: bool = True,
    ):
        self._cfg = CropChainsAroundSpecifiedPNUnitsConfig(
            complex_pair_probability=complex_pair_probability,
            max_atoms=max_atoms,
            min_atom_contacts=min_atom_contacts,
            max_chains=max_chains,
            query_pn_unit_key=query_pn_unit_key,
            contacts_key=contacts_key,
            protein_only_partners=protein_only_partners,
        )

    def forward(self, data: dict) -> dict:
        cfg = self._cfg
        atom_array = data["atom_array"]

        if len(atom_array) == 0:
            return data

        if not hasattr(atom_array, "pn_unit_iid"):
            logger.debug("CropChainsAroundSpecifiedPNUnits: no pn_unit_iid annotation; skipping crop.")
            return data

        # Resolve the query pn_unit_iid.
        query_raw = data.get(cfg.query_pn_unit_key)
        if not query_raw:
            logger.debug(
                "CropChainsAroundSpecifiedPNUnits: '%s' not in data; skipping crop.",
                cfg.query_pn_unit_key,
            )
            return data
        query_iid = query_raw[0] if isinstance(query_raw, list) else query_raw

        if not (atom_array.pn_unit_iid == query_iid).any():
            logger.debug(
                "CropChainsAroundSpecifiedPNUnits: query '%s' absent from atom_array; skipping.",
                query_iid,
            )
            return data

        query_atoms = int((atom_array.pn_unit_iid == query_iid).sum())
        selected = [query_iid]
        total_atoms = query_atoms

        rng = np.random.default_rng()
        if rng.random() < cfg.complex_pair_probability:
            contacts_raw = data.get("extra_info", {}).get(cfg.contacts_key)

            if contacts_raw:
                contacts = json.loads(contacts_raw) if isinstance(contacts_raw, str) else list(contacts_raw)

                # Collect all valid partners (contact-quality + protein-only + chain cap).
                valid_partners = []
                for contact in contacts:
                    if contact.get("num_contacts", 0) < cfg.min_atom_contacts:
                        continue
                    partner_iid = contact["pn_unit_iid"]
                    if partner_iid == query_iid:
                        continue
                    if cfg.protein_only_partners and not _is_protein_pn_unit(atom_array, partner_iid):
                        continue
                    if cfg.max_chains is not None and len(valid_partners) + 1 >= cfg.max_chains:
                        break
                    partner_atoms = contact.get(
                        "num_atoms",
                        int((atom_array.pn_unit_iid == partner_iid).sum()),
                    )
                    valid_partners.append((partner_iid, partner_atoms))

                # All-or-nothing: include all valid partners only if the full
                # complex fits within max_atoms. A partial crop would expose
                # non-native interfaces (query surface facing an absent partner).
                total_partner_atoms = sum(n for _, n in valid_partners)
                if valid_partners and query_atoms + total_partner_atoms <= cfg.max_atoms:
                    selected = [query_iid] + [iid for iid, _ in valid_partners]
                    total_atoms = query_atoms + total_partner_atoms
                # else: monomer fallback — selected and total_atoms already set above
            else:
                logger.debug(
                    "CropChainsAroundSpecifiedPNUnits: no contacts at extra_info['%s']; monomer fallback.",
                    cfg.contacts_key,
                )

        mask = np.isin(atom_array.pn_unit_iid, selected)
        data["atom_array"] = atom_array[mask]

        extra_info = data.setdefault("extra_info", {})
        extra_info["mpnn_chain_subset"] = {
            "query_pn_unit_iid": query_iid,
            "selected_pn_unit_iids": selected,
            "is_complex": len(selected) > 1,
            "total_atoms": total_atoms,
        }

        return data
