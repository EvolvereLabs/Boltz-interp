"""
SwissProt Annotation Multi-Hot Encoding Pipeline
================================================
Recreates the InterPLM annotation extraction system for custom protein sets.

This pipeline:
1. Downloads annotations from UniProt for your protein IDs
2. Parses feature annotations (domains, sites, binding regions, etc.)
3. Creates per-residue multi-hot encodings for each concept
4. Outputs processed shards compatible with downstream analysis

Usage:
    python swissprot_annotation_pipeline.py --input_ids proteins.txt --output_dir ./processed
"""

import os
import re
import json
import gzip
import time
import argparse
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURATION
# ============================================================================

# UniProt feature fields to download (matching InterPLM's approach)
UNIPROT_FEATURE_FIELDS = [
    "accession",
    "reviewed", 
    "protein_name",
    "length",
    "sequence",
    "ec",
    # Active/binding sites
    "ft_act_site",      # Active site
    "ft_binding",       # Binding site
    "cc_cofactor",      # Cofactor
    # PTMs and modifications  
    "ft_disulfid",      # Disulfide bond
    "ft_carbohyd",      # Glycosylation site
    "ft_lipid",         # Lipidation
    "ft_mod_res",       # Modified residue
    # Signal/transit peptides
    "ft_signal",        # Signal peptide
    "ft_transit",       # Transit peptide
    # Secondary structure
    "ft_helix",         # Helix
    "ft_turn",          # Turn  
    "ft_strand",        # Beta strand
    "ft_coiled",        # Coiled coil
    # Domains and regions
    "cc_domain",        # Domain comments
    "ft_compbias",      # Compositional bias
    "ft_domain",        # Domain
    "ft_motif",         # Motif
    "ft_region",        # Region
    "ft_zn_fing",       # Zinc finger
    # Cross-references
    "xref_alphafolddb", # AlphaFold DB
]

# Feature type mapping for concept naming
FEATURE_TYPE_MAP = {
    "Active site": "active_site",
    "Binding site": "binding_site", 
    "Disulfide bond": "disulfide_bond",
    "Glycosylation": "glycosylation",
    "Lipidation": "lipidation",
    "Modified residue": "modified_residue",
    "Signal peptide": "signal_peptide",
    "Transit peptide": "transit_peptide",
    "Helix": "helix",
    "Turn": "turn",
    "Beta strand": "beta_strand",
    "Coiled coil": "coiled_coil",
    "Compositional bias": "compositional_bias",
    "Domain": "domain",
    "Motif": "motif",
    "Region": "region",
    "Zinc finger": "zinc_finger",
}


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class FeatureAnnotation:
    """Represents a single feature annotation on a protein."""
    feature_type: str
    start: int
    end: int
    description: str = ""
    evidence: str = ""
    
    @property
    def concept_name(self) -> str:
        """Generate concept name from feature type and description."""
        base = FEATURE_TYPE_MAP.get(self.feature_type, self.feature_type.lower().replace(" ", "_"))
        if self.description:
            # Clean description for concept naming
            desc_clean = re.sub(r'[^\w\s-]', '', self.description)
            desc_clean = desc_clean.strip().replace(' ', '_').lower()[:50]
            return f"{base}:{desc_clean}" if desc_clean else base
        return base


@dataclass 
class ProteinAnnotations:
    """All annotations for a single protein."""
    uniprot_id: str
    sequence: str
    protein_name: str = ""
    features: List[FeatureAnnotation] = field(default_factory=list)
    
    @property
    def length(self) -> int:
        return len(self.sequence)
    
    def get_multihot_encoding(self, concept_list: List[str]) -> np.ndarray:
        """Create multi-hot encoding matrix [seq_len x n_concepts]."""
        encoding = np.zeros((self.length, len(concept_list)), dtype=np.int8)
        concept_to_idx = {c: i for i, c in enumerate(concept_list)}
        
        for feat in self.features:
            concept = feat.concept_name
            if concept in concept_to_idx:
                idx = concept_to_idx[concept]
                # Convert to 0-indexed
                start = max(0, feat.start - 1)
                end = min(self.length, feat.end)
                encoding[start:end, idx] = 1
        return encoding


# ============================================================================
# UNIPROT API FUNCTIONS
# ============================================================================

def read_protein_ids(input_file: str) -> List[str]:
    """Read protein IDs from file (FASTA, text, or one ID per line)."""
    ids = []
    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                # FASTA header - extract ID
                # Handle formats: >ID, >sp|ID|NAME, >tr|ID|NAME
                parts = line[1:].split('|')
                if len(parts) >= 2:
                    ids.append(parts[1])
                else:
                    ids.append(parts[0].split()[0])
            elif not line.startswith('#'):
                # Plain ID or sequence - only take if it looks like an ID
                if re.match(r'^[A-Z0-9_]+$', line) and len(line) < 20:
                    ids.append(line)
    return list(set(ids))  # Remove duplicates


def download_uniprot_annotations(
    protein_ids: List[str],
    output_path: str,
    batch_size: int = 100,
    max_retries: int = 3
) -> str:
    """Download annotations from UniProt REST API for given protein IDs."""
    
    fields_str = ",".join(UNIPROT_FEATURE_FIELDS)
    all_results = []
    
    print(f"Downloading annotations for {len(protein_ids)} proteins...")
    
    for i in range(0, len(protein_ids), batch_size):
        batch = protein_ids[i:i+batch_size]
        batch_query = " OR ".join([f"accession:{pid}" for pid in batch])
        
        url = "https://rest.uniprot.org/uniprotkb/stream"
        params = {
            "query": batch_query,
            "fields": fields_str,
            "format": "tsv"
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.get(url, params=params, timeout=60)
                if response.status_code == 200:
                    all_results.append(response.text)
                    print(f"  Downloaded batch {i//batch_size + 1}/{(len(protein_ids)-1)//batch_size + 1}")
                    break
                else:
                    print(f"  Attempt {attempt+1} failed with status {response.status_code}")
            except Exception as e:
                print(f"  Attempt {attempt+1} failed: {e}")
            
            time.sleep(2 ** attempt)  # Exponential backoff
        else:
            print(f"  Warning: Failed to download batch starting at {i}")
        
        time.sleep(0.5)  # Rate limiting
    
    # Combine results
    combined = []
    header = None
    for result in all_results:
        lines = result.strip().split('\n')
        if lines:
            if header is None:
                header = lines[0]
                combined.append(header)
            combined.extend(lines[1:])
    
    # Save to file
    output_text = '\n'.join(combined)
    if output_path.endswith('.gz'):
        with gzip.open(output_path, 'wt') as f:
            f.write(output_text)
    else:
        with open(output_path, 'w') as f:
            f.write(output_text)
    
    print(f"Saved annotations to {output_path}")
    return output_path


# ============================================================================
# PARSING FUNCTIONS
# ============================================================================

def parse_feature_string(feature_str: str, feature_type: str) -> List[FeatureAnnotation]:
    """Parse UniProt feature string format into FeatureAnnotation objects.
    
    Examples of formats:
    - "DOMAIN 1..50; /note=\"Kinase domain\""
    - "BINDING 23; /ligand=\"ATP\""
    - "DISULFID 45..67"
    - "HELIX 10..25; HELIX 30..45"
    """
    if pd.isna(feature_str) or not feature_str or feature_str == '':
        return []
    
    features = []
    
    # Split by semicolon followed by space and feature type (for multiple features)
    # Pattern matches feature entries
    pattern = r'([A-Z][A-Z_\s]+)\s+(\d+)(?:\.\.(\d+))?(?:;\s*/note="([^"]*)")?(?:;\s*/ligand="([^"]*)")?(?:;\s*/evidence="([^"]*)")?'
    
    # Also handle simpler patterns
    simple_pattern = r'(\d+)(?:\.\.(\d+))?(?:;\s*/note="([^"]*)")?'
    
    # Try to extract multiple features separated by common delimiters
    entries = re.split(r';\s*(?=[A-Z])', feature_str)
    
    for entry in entries:
        entry = entry.strip()
        if not entry:
            continue
            
        # Try to match position pattern
        pos_match = re.search(r'(\d+)(?:\.\.(\d+))?', entry)
        if pos_match:
            start = int(pos_match.group(1))
            end = int(pos_match.group(2)) if pos_match.group(2) else start
            
            # Extract description/note
            desc = ""
            note_match = re.search(r'/note="([^"]*)"', entry)
            if note_match:
                desc = note_match.group(1)
            
            ligand_match = re.search(r'/ligand="([^"]*)"', entry)
            if ligand_match:
                desc = ligand_match.group(1) if not desc else f"{desc}; {ligand_match.group(1)}"
            
            # Evidence
            evidence = ""
            ev_match = re.search(r'/evidence="([^"]*)"', entry)
            if ev_match:
                evidence = ev_match.group(1)
            
            features.append(FeatureAnnotation(
                feature_type=feature_type,
                start=start,
                end=end,
                description=desc,
                evidence=evidence
            ))
    
    return features


def parse_uniprot_tsv(tsv_path: str) -> List[ProteinAnnotations]:
    """Parse UniProt TSV file into ProteinAnnotations objects."""
    
    # Convert to string if Path object
    tsv_path_str = str(tsv_path)
    
    # Read TSV
    if tsv_path_str.endswith('.gz'):
        df = pd.read_csv(tsv_path_str, sep='\t', compression='gzip', low_memory=False)
    else:
        df = pd.read_csv(tsv_path_str, sep='\t', low_memory=False)
    
    print(f"Loaded {len(df)} proteins from TSV")
    
    # Column name mapping (UniProt uses different names)
    col_mapping = {
        'Entry': 'accession',
        'Reviewed': 'reviewed',
        'Protein names': 'protein_name', 
        'Length': 'length',
        'Sequence': 'sequence',
        'EC number': 'ec',
        'Active site': 'ft_act_site',
        'Binding site': 'ft_binding',
        'Cofactor': 'cc_cofactor',
        'Disulfide bond': 'ft_disulfid',
        'Glycosylation': 'ft_carbohyd',
        'Lipidation': 'ft_lipid',
        'Modified residue': 'ft_mod_res',
        'Signal peptide': 'ft_signal',
        'Transit peptide': 'ft_transit',
        'Helix': 'ft_helix',
        'Turn': 'ft_turn',
        'Beta strand': 'ft_strand',
        'Coiled coil': 'ft_coiled',
        'Domain [CC]': 'cc_domain',
        'Compositional bias': 'ft_compbias',
        'Domain [FT]': 'ft_domain',
        'Motif': 'ft_motif',
        'Region': 'ft_region',
        'Zinc finger': 'ft_zn_fing',
        'AlphaFoldDB': 'xref_alphafolddb',
    }
    
    # Rename columns
    df = df.rename(columns=col_mapping)
    
    # Feature columns to parse
    feature_cols = {
        'ft_act_site': 'Active site',
        'ft_binding': 'Binding site',
        'ft_disulfid': 'Disulfide bond',
        'ft_carbohyd': 'Glycosylation',
        'ft_lipid': 'Lipidation',
        'ft_mod_res': 'Modified residue',
        'ft_signal': 'Signal peptide',
        'ft_transit': 'Transit peptide',
        'ft_helix': 'Helix',
        'ft_turn': 'Turn',
        'ft_strand': 'Beta strand',
        'ft_coiled': 'Coiled coil',
        'ft_compbias': 'Compositional bias',
        'ft_domain': 'Domain',
        'ft_motif': 'Motif',
        'ft_region': 'Region',
        'ft_zn_fing': 'Zinc finger',
    }
    
    proteins = []
    
    for _, row in df.iterrows():
        # Get accession
        accession = row.get('accession', row.get('Entry', ''))
        if pd.isna(accession):
            continue
            
        # Get sequence
        sequence = row.get('sequence', row.get('Sequence', ''))
        if pd.isna(sequence) or not sequence:
            continue
        
        # Get protein name
        protein_name = row.get('protein_name', row.get('Protein names', ''))
        if pd.isna(protein_name):
            protein_name = ""
        
        # Parse all features
        all_features = []
        for col, feat_type in feature_cols.items():
            if col in row.index:
                features = parse_feature_string(row[col], feat_type)
                all_features.extend(features)
        
        proteins.append(ProteinAnnotations(
            uniprot_id=str(accession),
            sequence=str(sequence),
            protein_name=str(protein_name),
            features=all_features
        ))
    
    return proteins


# ============================================================================
# MULTI-HOT ENCODING
# ============================================================================

def collect_all_concepts(
    proteins: List[ProteinAnnotations],
    min_instances: int = 10
) -> Tuple[List[str], Dict[str, int]]:
    """Collect all unique concepts and filter by minimum instances."""
    
    concept_counts = defaultdict(int)
    
    for protein in proteins:
        seen_concepts = set()  # Count each concept once per protein
        for feat in protein.features:
            concept = feat.concept_name
            if concept not in seen_concepts:
                concept_counts[concept] += 1
                seen_concepts.add(concept)
    
    # Filter by minimum instances
    filtered_concepts = [
        c for c, count in concept_counts.items() 
        if count >= min_instances
    ]
    
    # Sort for consistency
    filtered_concepts = sorted(filtered_concepts)
    
    concept_to_idx = {c: i for i, c in enumerate(filtered_concepts)}
    
    print(f"Found {len(concept_counts)} unique concepts")
    print(f"Kept {len(filtered_concepts)} concepts with >= {min_instances} instances")
    
    return filtered_concepts, concept_to_idx


def create_multihot_encodings(
    proteins: List[ProteinAnnotations],
    concept_list: List[str]
) -> Dict[str, np.ndarray]:
    """Create multi-hot encodings for all proteins."""
    
    encodings = {}
    for protein in proteins:
        encoding = protein.get_multihot_encoding(concept_list)
        encodings[protein.uniprot_id] = encoding
    
    return encodings


# ============================================================================
# OUTPUT FUNCTIONS
# ============================================================================

def save_shard(
    proteins: List[ProteinAnnotations],
    concept_list: List[str],
    output_dir: str,
    shard_idx: int
) -> None:
    """Save a shard of processed data."""
    
    shard_dir = Path(output_dir) / f"shard_{shard_idx}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    
    # Save FASTA
    fasta_path = shard_dir / "sequences.fasta"
    with open(fasta_path, 'w') as f:
        for protein in proteins:
            f.write(f">{protein.uniprot_id}\n{protein.sequence}\n")
    
    # Save annotations as numpy arrays
    encodings = {}
    for protein in proteins:
        encodings[protein.uniprot_id] = protein.get_multihot_encoding(concept_list)
    
    np.savez_compressed(
        shard_dir / "annotations.npz",
        **encodings
    )
    
    # Save protein metadata
    metadata = []
    for protein in proteins:
        metadata.append({
            'uniprot_id': protein.uniprot_id,
            'length': protein.length,
            'protein_name': protein.protein_name,
            'n_features': len(protein.features)
        })
    
    pd.DataFrame(metadata).to_csv(shard_dir / "metadata.csv", index=False)
    
    # Save per-residue annotation labels (for debugging/verification)
    annotations_list = []
    for protein in proteins:
        for feat in protein.features:
            annotations_list.append({
                'uniprot_id': protein.uniprot_id,
                'feature_type': feat.feature_type,
                'concept': feat.concept_name,
                'start': feat.start,
                'end': feat.end,
                'description': feat.description
            })
    
    pd.DataFrame(annotations_list).to_csv(shard_dir / "features.csv", index=False)
    
    print(f"  Saved shard {shard_idx}: {len(proteins)} proteins, {len(annotations_list)} features")


def save_concept_vocabulary(
    concept_list: List[str],
    concept_counts: Dict[str, int],
    output_dir: str
) -> None:
    """Save the concept vocabulary."""
    
    vocab_path = Path(output_dir) / "concept_vocabulary.json"
    
    vocab = {
        'concepts': concept_list,
        'concept_to_idx': {c: i for i, c in enumerate(concept_list)},
        'idx_to_concept': {i: c for i, c in enumerate(concept_list)},
        'concept_counts': concept_counts,
        'n_concepts': len(concept_list)
    }
    
    with open(vocab_path, 'w') as f:
        json.dump(vocab, f, indent=2)
    
    print(f"Saved concept vocabulary: {len(concept_list)} concepts")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_pipeline(
    input_ids: str,
    output_dir: str,
    n_shards: int = 8,
    min_required_instances: int = 10,
    download_annotations: bool = True,
    cached_tsv: Optional[str] = None
) -> None:
    """Run the full annotation extraction pipeline."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Read protein IDs
    print("\n" + "="*60)
    print("Step 1: Reading protein IDs")
    print("="*60)
    protein_ids = read_protein_ids(input_ids)
    print(f"Found {len(protein_ids)} unique protein IDs")
    
    # Step 2: Download or load annotations
    print("\n" + "="*60)
    print("Step 2: Getting annotations from UniProt")
    print("="*60)
    
    if cached_tsv and os.path.exists(cached_tsv):
        tsv_path = cached_tsv
        print(f"Using cached TSV: {tsv_path}")
    else:
        tsv_path = output_path / "uniprot_annotations.tsv.gz"
        if download_annotations:
            download_uniprot_annotations(protein_ids, str(tsv_path))
        else:
            raise FileNotFoundError(f"No cached TSV found and download disabled")
    
    # Step 3: Parse annotations
    print("\n" + "="*60)
    print("Step 3: Parsing annotations")
    print("="*60)
    proteins = parse_uniprot_tsv(tsv_path)
    print(f"Parsed {len(proteins)} proteins with annotations")
    
    # Count total features
    total_features = sum(len(p.features) for p in proteins)
    print(f"Total features: {total_features}")
    
    # Step 4: Collect concepts
    print("\n" + "="*60)
    print("Step 4: Collecting concepts")
    print("="*60)
    
    # Get all concept counts first
    concept_counts = defaultdict(int)
    for protein in proteins:
        seen = set()
        for feat in protein.features:
            c = feat.concept_name
            if c not in seen:
                concept_counts[c] += 1
                seen.add(c)
    
    concept_list, concept_to_idx = collect_all_concepts(proteins, min_required_instances)
    
    # Step 5: Create shards
    print("\n" + "="*60)
    print("Step 5: Creating shards")
    print("="*60)
    
    # Shuffle proteins for random sharding
    import random
    random.seed(42)
    shuffled = proteins.copy()
    random.shuffle(shuffled)
    
    # Split into shards
    shard_size = len(shuffled) // n_shards
    for i in range(n_shards):
        start = i * shard_size
        end = start + shard_size if i < n_shards - 1 else len(shuffled)
        shard_proteins = shuffled[start:end]
        save_shard(shard_proteins, concept_list, str(output_path), i)
    
    # Step 6: Save vocabulary
    print("\n" + "="*60)
    print("Step 6: Saving concept vocabulary")
    print("="*60)
    save_concept_vocabulary(concept_list, dict(concept_counts), str(output_path))
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total proteins processed: {len(proteins)}")
    print(f"Total features extracted: {total_features}")
    print(f"Unique concepts (filtered): {len(concept_list)}")
    print(f"Number of shards: {n_shards}")
    print(f"Output directory: {output_path}")
    
    # Show top concepts
    print("\nTop 20 concepts by frequency:")
    sorted_concepts = sorted(concept_counts.items(), key=lambda x: -x[1])[:20]
    for concept, count in sorted_concepts:
        print(f"  {concept}: {count}")


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Extract SwissProt annotations and create multi-hot encodings"
    )
    
    parser.add_argument(
        "--input_ids", "-i",
        required=True,
        help="Input file with protein IDs (FASTA or text, one ID per line)"
    )
    
    parser.add_argument(
        "--output_dir", "-o",
        required=True,
        help="Output directory for processed annotations"
    )
    
    parser.add_argument(
        "--n_shards",
        type=int,
        default=8,
        help="Number of shards to create (default: 8)"
    )
    
    parser.add_argument(
        "--min_required_instances",
        type=int,
        default=10,
        help="Minimum instances required to keep a concept (default: 10)"
    )
    
    parser.add_argument(
        "--cached_tsv",
        default=None,
        help="Path to pre-downloaded UniProt TSV file (skips download)"
    )
    
    parser.add_argument(
        "--no_download",
        action="store_true",
        help="Don't download from UniProt (requires --cached_tsv)"
    )
    
    args = parser.parse_args()
    
    run_pipeline(
        input_ids=args.input_ids,
        output_dir=args.output_dir,
        n_shards=args.n_shards,
        min_required_instances=args.min_required_instances,
        download_annotations=not args.no_download,
        cached_tsv=args.cached_tsv
    )


if __name__ == "__main__":
    main()