import pandas as pd  # For data manipulation and analysis
import requests  # For sending HTTP requests
import re  # For regular expressions
from IPython.display import (
    display,
)  # For displaying DataFrames in Jupyter notebooks (not applicable in py file)


# Columns to keep from the ChEMBL assay data
COLUMNS = [
    "molecule_chembl_id",
    "canonical_smiles",
    "molecule_pref_name",
    "target_pref_name",
    "assay_variant_mutation",
    "standard_type",
    "standard_relation",
    "standard_value",
    "standard_units",
    "bao_label",
    "assay_description",
    "document_journal",
    "document_year",
    "document_chembl_id",
    "potential_duplicate",
]


# Step 1: Define Functions to Fetch Data from UniProt and ChEMBL (Homo Sapiens)
def get_uniprot_and_chembl(protein_name):
    url = f"https://rest.uniprot.org/uniprotkb/search?query={protein_name}+AND+organism_id:9606&format=json"
    response = requests.get(url)

    if response.status_code == 200:
        try:
            data = response.json()
            if "results" in data and data["results"]:
                uniprot_id = data["results"][0]["primaryAccession"]
                uniprot_entry_url = (
                    f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
                )
                entry_response = requests.get(uniprot_entry_url)

                if entry_response.status_code == 200:
                    entry_data = entry_response.json()

                    chembl_id = None
                    if "uniProtKBCrossReferences" in entry_data:
                        for ref in entry_data["uniProtKBCrossReferences"]:
                            if ref["database"] == "ChEMBL":
                                chembl_id = ref["id"]
                                break

                    return chembl_id if chembl_id else None
        except requests.exceptions.JSONDecodeError:
            print(f"Error decoding JSON response from UniProt API for {protein_name}")

    return None


# Step 2: Fetch document metadata and add them to the DataFrame
def get_document_metadata(df):
    if "document_chembl_id" not in df.columns:
        return df  # If the column doesn't exist, return as is

    document_metadata = {}
    unique_doc_ids = df["document_chembl_id"].dropna().unique()

    for doc_id in unique_doc_ids:
        url = f"https://www.ebi.ac.uk/chembl/api/data/document/{doc_id}.json"
        response = requests.get(url)

        if response.status_code == 200:
            data = response.json()
            document_metadata[doc_id] = {
                "title": data.get("title", "Unknown Title"),
                "doi": data.get("doi", "No DOI Available"),
            }
        else:
            document_metadata[doc_id] = {
                "title": "Error Retrieving Title",
                "doi": "Error Retrieving DOI",
            }

    df["document_title"] = df["document_chembl_id"].map(
        lambda x: document_metadata.get(x, {}).get("title", "Unknown Title")
    )
    df["document_doi"] = df["document_chembl_id"].map(
        lambda x: document_metadata.get(x, {}).get("doi", "No DOI Available")
    )
    return df


# Step 3: Fetch IC50 assay data from ChEMBL for a given target ChEMBL ID, filtering by assay type if specified.
def fetch_ic50_data(target_id, assay_type="all"):
    all_data = []
    offset = 0
    limit = 1000

    while True:
        url = f"https://www.ebi.ac.uk/chembl/api/data/activity.json?target_chembl_id={target_id}&standard_type=IC50&limit={limit}&offset={offset}"
        response = requests.get(url)

        if response.status_code == 200:
            data = response.json()
            if "activities" not in data or not data["activities"]:
                break

            df = pd.DataFrame(data["activities"])
            all_data.append(df)
            offset += limit
        else:
            break

    if all_data:
        df = pd.concat(all_data, ignore_index=True)

        # Keep only columns of interest
        available_columns = [col for col in COLUMNS if col in df.columns]
        df = df[available_columns]

        # Apply assay type filter
        if "bao_label" in df.columns:
            if assay_type == "cell-based":
                df = df[
                    df["bao_label"].str.contains(
                        r"cell-based|cell membrane format|cell-free format",
                        case=False,
                        na=False,
                    )
                ]
            elif assay_type == "protein-based":
                df = df[
                    df["bao_label"].str.contains(
                        r"protein format|single protein format", case=False, na=False
                    )
                ]
        # Define 'Wild Type'
        if "assay_variant_mutation" in df.columns:
            df["assay_variant_mutation"] = df["assay_variant_mutation"].fillna(
                "Wild Type"
            )
            df.loc[
                df["assay_variant_mutation"] == "UNDEFINED MUTATION",
                "assay_variant_mutation",
            ] = "Other Mutations"

        # Define 'Research Molecule'
        if "molecule_pref_name" in df.columns:
            df["molecule_pref_name"] = df["molecule_pref_name"].fillna(
                "Research Molecule"
            )

        # Remove rows where there is no IC50
        if "standard_value" in df.columns:
            df = df.dropna(subset=["standard_value"])

        # Define "Patents from EMA and FDA"
        if "document_journal" in df.columns:
            df["document_journal"] = df["document_journal"].fillna("Patents/EMA/FDA")

        if "document_year" in df.columns:
            df["document_year"] = df["document_year"].astype(str).fillna("Other")

        # Remove rows where canonical_smiles is empty
        if "canonical_smiles" in df.columns:
            df = df.dropna(subset=["canonical_smiles"])

        # Ensure potential_duplicate is treated as an integer before filtering
        if "potential_duplicate" in df.columns:
            df["potential_duplicate"] = (
                pd.to_numeric(df["potential_duplicate"], errors="coerce")
                .fillna(0)
                .astype(int)
            )
            df = df[df["potential_duplicate"] == 0]

        df = get_document_metadata(df)  # Fetch and add document metadata
        if "document_doi" in df.columns:
            df["document_doi"] = df["document_doi"].fillna("Patents/EMA/FDA")

        df.reset_index(inplace=True, drop=True)
        df.insert(0, "#", df.index + 1)

        # Extract Cell Type from Assay Description Using RegEx:
        df = extract_cell_type_from_assay(df)

        return df

    return pd.DataFrame(columns=["#"] + COLUMNS)


def extract_cell_type_from_assay(df):
    # List of cell lines
    cell_types = [
        # Lung Cancer Cell Lines
        "PC9",
        "A549",
        "H1299",
        "H1650",
        "H1975",
        "HCC827",
        "H358",
        "H2228",
        "H3122",
        "H23",
        "H520",
        "Calu-1",
        "Calu-3",
        "H3255",
        "H838",
        "H3255",
        "NCI-H1666",
        "NCI-H460",
        "NCI-H2073",
        "NCI-H3255",
        # Breast Cancer Cell Lines
        "MCF-7",
        "T47D",
        "MDA-MB-231",
        "MDA-MB-468",
        "BT-474",
        "SK-BR-3",
        "ZR-75-1",
        "HCC1954",
        "Hs578T",
        "HCC1937",
        "MCF10A",
        "MDA231",
        "SKBR3",
        "BT474",
        # Colorectal Cancer Cell Lines
        "HT-29",
        "HCT116",
        "SW480",
        "SW620",
        "LoVo",
        "DLD-1",
        "Caco-2",
        "LS174T",
        "RKO",
        "DiFi",
        # Pancreatic Cancer Cell Lines
        "PANC-1",
        "MIA PaCa-2",
        "BXPC-3",
        "Capan-1",
        "AsPC-1",
        "CFPAC-1",
        "SU.86.86",
        # Prostate Cancer Cell Lines
        "LNCaP",
        "22Rv1",
        "PC-3",
        "DU145",
        "VCaP",
        "RWPE-1",
        # Liver Cancer Cell Lines
        "HepG2",
        "Huh7",
        "PLC/PRF/5",
        "SNU-182",
        "SNU-449",
        "Hep3B",
        # Leukemia & Lymphoma Cell Lines
        "Jurkat",
        "K562",
        "HL-60",
        "THP-1",
        "REH",
        "NALM-6",
        "MOLT-4",
        "RS4;11",
        # Brain Cancer Cell Lines
        "U87-MG",
        "U251",
        "T98G",
        "LN229",
        "SF268",
        "A172",
        "SNB19",
        "U87",
        # Ovarian Cancer Cell Lines
        "SK-OV-3",
        "OVCAR-3",
        "A2780",
        "CAOV3",
        "HEY",
        "ES2",
        "Sf9",
        "SF9",
        "Sf21",
        # Melanoma Cell Lines
        "A375",
        "SK-MEL-28",
        "WM266-4",
        "A2058",
        "COLO829",
        # Gastric Cancer Cell Lines
        "AGS",
        "MKN45",
        "NCI-N87",
        "SNU-1",
        "SNU-16",
        "N87",
        "OE19",
        "NUGC4",
        "NUGC3",
        "FU97",
        "SNU16",
        "IM95",
        "MKN74",
        "MKN1",
        "KATOIII",
        "SNU1",
        "SNU5",
        # Cervical Cancer Cell Lines
        "A431",
        "A-431",
        # Head and Neck Cancer Cell Lines
        "HN5",
        "CAL27",
        # Epidermoid Carcinoma Cell Lines
        "HeLa",
        "SiHa",
        "C-33A",
        "CaSki",
        "KB",
        # Bladder Cancer Cell Lines
        "T24",
        "UM-UC-3",
        "5637",
        "RT4",
        # Osteosarcoma Cell Lines
        "U2OS",
        "SAOS-2",
        "MG-63",
        "143B",
        # Normal and Immortalized Cell Lines
        "293T",
        "CHO",
        "RAW 264.7",
        "PC-12",
        "Vero",
        "MDCK",
        "NIH 3T3",
        "HaCaT",
        "HEK293",
        "SH-SY5Y",
        "BaF/3",
        "BaF3",
        "HFF",
        "HB4a",
        # Stem Cell Lines
        "H9",
        "WA09",
        "WA01",
        "iPSC",
        "HUES1",
        "HUES9",
        # Fibroblast Cell Lines
        "BJ",
        "WI-38",
        "IMR-90",
        "MRC-5",
        # Endothelial Cell Lines
        "HUVEC",
        "HMEC-1",
        # Fibroblast Cell Lines
        "3T3",
        "HER-14",
        "HER14",
        "NIH3T3",
        "DHER14",
        "HER-1",
        "Her1",
        # Bone Cell Lines
        "ER 22",
        "ER22",
        # Immune System Cell Lines
        "J774A.1",
        "Raji",
        "Daudi",
        "U937",
        "KG-1",
        "MOLT-3",
    ]

    # Compile a regex pattern dynamically from the extended cell type list
    cell_types_pattern = (
        r"\b(?:" + "|".join(re.escape(cell) for cell in cell_types) + r")\b"
    )
    compiled_pattern = re.compile(cell_types_pattern, re.IGNORECASE)

    # Function to extract cell types
    def extract_cell_type(description):
        if pd.isna(description):
            return None
        match = compiled_pattern.search(description)
        return match.group(0) if match else "Other"

    print("Extracting cell types from 'assay_description' column...")
    df["cell_id"] = df["assay_description"].apply(extract_cell_type)

    # Dictionary to unify cell line names
    cell_synonyms = {
        "A-431": "A431",
        "BaF/3": "BaF3",
        "ER 22": "ER22",
        "HER-1": "HER14",
        # Add additional synonyms as needed
    }

    # Function to unify cell line names
    def unify_cell_id(cell):
        return cell_synonyms.get(cell, cell)

    df["cell_id"] = df["cell_id"].apply(unify_cell_id)

    return df


# Step 4: Fetch and Process Data Patterns for "rare mutations" with RegEx and Pandas using the assay description
mutations_dict = {
    "EGFR": [
        r"e746[ -_]?a750[ -]?deletion",
        r"del19",
        r"19[ -]?del",
        r"d746[ -]?750",
        r"del\s?\(746\s?to\s?750\)",
        r"t790m",
        r"c797s",
        r"d770[ -_]?n771[ -_]?ins",
        r"del18",
        r"del\s?\(19\)",
        r"exon\s?19\s?deletion",
        r"e746[ -_]?a750",
        r"t790m\s?[/-]?\s?del19",
        r"del19\s?[/-]?\s?t790m\s?[/-]?\s?c797s",
        r"ex19del",
        r"19D",
        r"deletion\s?mutant",
        r"746\s?to\s?750\s?deletion",
        r"A763_Y764insFHEA",
        r"d747[-_\s]?749/A750P",
        r"EGFR-D770-N771insNPG",
        r"EGFR\s?LTb",
        r"EGFR\s?dTCb",
        r"EGFR\s?LTCb",
        r"Del\s?ex19",
        r"A763-Y764\s?ins\s?FQEA",
        r"D770_N771insNPG",
        r"del\s?\(746\s?to\s?750\)\s?mutant\s?autophosphorylation",
        r"t790m\s*[/-]\s*del\s*\(746\s*to\s*750(?:\s*residues)?\)",
    ],
    "ALK": [
        r"l1256f",
        r"s1206c",
        r"s1206y",
        r"s1206a",
        r"g1202r",
        r"g1269a",
        r"l1196m",
    ],
    "BRAF": [r"t589s", r"p731t", r"v600[ -_]?k601[ -]?delins", r"v600d"],
}

# Compile a regex pattern dynamically for all mutations (single RegEx pattern that combines all the patterns)
mutation_patterns = r"(?:" + "|".join(mutations_dict["EGFR"]) + r")"
compiled_mutation_pattern = re.compile(mutation_patterns, re.IGNORECASE)


# Function to extract mutations from the assay description using the compiled pattern
def extract_mutations(description, pattern):
    if pd.isna(description):
        return ""
    # Check directly for wild type in the description and return immediately
    if re.search(r"\b(wild[-_\s]?type|wt)\b", description, re.IGNORECASE):
        return "Wild Type"
    found = re.findall(pattern, description, flags=re.IGNORECASE)
    return ", ".join(set(found)) if found else "Other Mutation"


# RegEx dictionary for mutation transformation from assay description
mutation_general_map_1 = {
    "EGFR": {
        r"wild type": "Wild Type",
        r"e746[ -_]?a750[ -]?deletion": "E746_A750del",
        r"del19": "E746_A750del",
        r"19[ -]?del": "E746_A750del",
        r"d746[ -]?750": "E746_A750del",
        r"del\s?\(746\s?to\s?750\)": "E746_A750del",
        r"deletion": "E746_A750del",
        r"19D": "E746_A750del",
        r"t790m": "T790M",
        r"c797s": "C797S",
        r"d770[ -_]?n771[ -_]?ins": "770insNPG",
        r"del18": "E709_T710del",
        r"del\s?\(19\)": "E746_A750del",
        r"exon\s?19\s?deletion": "E746_A750del",
        r"e746[ -_]?a750": "E746_A750del",
        r"t790m\s?[/-]?\s?del19": "E746_A750del + T790M",
        r"(del19.*t790m|t790m.*del19)": "E746_A750del + T790M",
        r"del19\s?[/-]?\s?t790m\s?[/-]?\s?c797s": "E746_A750del + T790M + C797S",
        r"ex19del": "E746_A750del",
        r"deletion\s?mutant": "E746_A750del",
        r"746\s?to\s?750\s?deletion": "E746_A750del",
        r"A763_Y764insFHEA": "A763_Y764insFHEA",
        r"del\s*\(746\s*to\s*750\)": "E746_A750del",
        r"d747[-_\s]?749/A750P": "L747_T749del, A750P",
        r"EGFR-D770-N771insNPG": "D770_N771insNPG",
        r"EGFR\s?LTb": "EGFR LTb mutation",
        r"EGFR\s?dTCb": "EGFR dTCb mutation",
        r"EGFR\s?LTCb": "EGFR LTCb mutation",
        r"(?:del19\s*/\s*t790m|t790m\s*/\s*del19)": "E746_A750del + T790M",
        r"(?:deletion\s*/\s*t790m|t790m\s*/\s*deletion)": "E746_A750del + T790M",
        r"Del\s?ex19": "E746_A750del",
        r"A763-Y764\s?ins\s?FQEA": "A763_Y764insFQEA",
        r"D770_N771insNPG": "D770_N771insNPG",
        r"del\s?\(746\s?to\s?750\)\s?mutant\s?autophosphorylation": "E746_A750del",
        r"(?:t790m\s*/\s*del\s*\(746\s*to\s*750.*\)|del\s*\(746\s*to\s*750.*\)\s*/\s*t790m)": "E746_A750del + T790M",
        r"deletion\s*[/-]\s*t790m": "E746_A750del + T790M",
        r"t790m\s*[/-]\s*del\s*\(746\s*to\s*750(?:\s*residues)?\)": "E746_A750del + T790M",
    },
}
# Add more mutations and proteins as needed


# Function to transform mutations using the general map 1
# If the string is a 'wild type', it returns 'Wild Type'.
def get_map1_transformation(mutation, protein_name):
    # Check if the string indicates a wild type (done three times for different functions)
    if re.search(r"wild[-\s]?type", mutation, re.IGNORECASE):
        return "Wild Type"

    mapping1 = mutation_general_map_1.get(protein_name, {})
    # Sort patterns by length descending
    for pattern in sorted(mapping1, key=lambda p: len(p), reverse=True):
        trans = mapping1[pattern]
        if re.search(pattern, mutation, re.IGNORECASE):
            return trans
    return mutation  # Return the original string if no pattern matches.


# Function for several mutations at the same time (using the combined transformation)
def transform_mutations(mutation_str, protein_name):
    # If the mutation string indicates wild type, return it immediately.
    if re.search(r"\b(wild[-_\s]?type|wt)\b", mutation_str, re.IGNORECASE):
        return "Wild Type"

    # Split the mutation string into individual mutations (if comma separated)
    mutations = [m.strip() for m in mutation_str.split(",") if m.strip()]
    # Apply the combined transformation (which uses Map 1 then Map 2)
    transformed = [get_combined_transformation(m, protein_name) for m in mutations]
    # Filter out unwanted membrane-related mutations
    filtered = [
        m
        for m in transformed
        if m not in ["EGFR dTCb mutation", "EGFR LTb mutation", "EGFR LTCb mutation"]
    ]
    return ", ".join(sorted(set(filtered))) if filtered else "Other Mutation"


# Function to update the mutation column using the combined transformation
def update_mutation(row):
    description = row["assay_description"]
    # If the description indicates wild type, return "Wild Type"
    if re.search(r"\b(wild[-_\s]?type|wt)\b", description, re.IGNORECASE):
        return "Wild Type"
    if row["assay_variant_mutation"] != "Other Mutations" or pd.isna(description):
        return row["assay_variant_mutation"]

    # Find mutation matches
    matches = compiled_mutation_pattern.findall(description)
    if not matches:
        return "Other Mutation"

    matched_mutations = set()
    for match in matches:
        for gene, mutations in mutation_general_map_1.items():
            for pattern, general_mutation in mutations.items():
                if re.search(pattern, match, re.IGNORECASE):
                    matched_mutations.add(general_mutation)

    # Normalize output
    return (
        ", ".join(sorted(matched_mutations)) if matched_mutations else "Other Mutation"
    )


# Normalize Mutation Defined variable (Sorting and removal of duplicates)
def normalize_mutation_list(mutation_str):
    if pd.isna(mutation_str) or mutation_str.strip() == "":
        return mutation_str  # Keep NaN values unchanged
    mutations = [m.strip() for m in mutation_str.split(",")]
    return ", ".join(sorted(set(mutations)))  # Sort and remove duplicates


# Function to process mutations in the DataFrame
def process_mutations(df):
    print("Updating 'Other Mutations' values in 'assay_variant_mutation' column...")
    df.loc[
        df["assay_variant_mutation"] == "Other Mutations", "assay_variant_mutation"
    ] = df.apply(update_mutation, axis=1)

    print("Standardizing mutation format in 'assay_variant_mutation' column...")
    df["assay_variant_mutation"] = df["assay_variant_mutation"].apply(
        normalize_mutation_list
    )
    return df


# Cache to store UniProt FASTA sequences for proteins to avoid repeated API requests (Faster processing)
fasta_cache = {}


# Function to fetch the UniProt ID for a given human protein name
def get_uniprot_id(protein_name):
    """Fetches the UniProt ID for a given human protein name."""
    url = f"https://rest.uniprot.org/uniprotkb/search?query={protein_name}+AND+organism_id:9606&format=json"
    response = requests.get(url)

    if response.status_code == 200:
        try:
            data = response.json()
            if "results" in data and data["results"]:
                return data["results"][0]["primaryAccession"]
        except requests.exceptions.JSONDecodeError:
            print(f"Error decoding JSON response from UniProt API for {protein_name}")

    return None


# Function to fetch the FASTA sequence from UniProt, using a cache to avoid duplicate API calls
def get_cached_uniprot_fasta(protein_name):
    """Fetches the FASTA sequence from UniProt, using a cache to avoid duplicate API calls."""
    if protein_name in fasta_cache:
        return fasta_cache[protein_name], None

    uniprot_id = get_uniprot_id(protein_name)
    if not uniprot_id:
        return None, f"Error: Could not find UniProt ID for '{protein_name}'."

    url = f"https://www.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    response = requests.get(url)

    if response.status_code == 200:
        fasta_sequence = "".join(response.text.split("\n")[1:])
        fasta_cache[protein_name] = fasta_sequence
        return fasta_sequence, None
    else:
        return (
            None,
            f"Error: Unable to retrieve FASTA sequence for UniProt ID {uniprot_id}.",
        )


# Complete mutation mapping for EGFR, ALK, and BRAF based on HGVS (Human Genome Variation Society) nomenclature p.()
# The mapping is based on the most common mutations found in the literature and databases.
mutation_general_map_2 = {
    "EGFR": {
        "Wild Type": "Wild Type",
        "C797S": "C797S",
        "L858R": "L858R",
        "T790M": "T790M",
        "L861Q": "L861Q",
        "C797S,L858R": "C797S,L858R",
        "L858R,T790M": "L858R,T790M",
        "L858R,T790M,C797S": "L858R,T790M,C797S",
        "T790M,C797S": "T790M,C797S",
        "T790M, C797S": "T790M,C797S",
        "T790M,L858M": "T790M,L858M",
        "19D/T790M/C797S": "E746_A750del,T790M,C797S",
        "A763_Y764insFHEA": "763insFHEA",
        "E746_A750del, T790M": "E746_A750del, T790M",
        "A763_Y764insFQEA": "763insFQEA",
        "L747_T749del, A750P": "L747_T749del, A750P",
        "D770_N771insNPG": "770insNPG",
        "E746_A750del": "E746_A750del",
        "E709_T710del, T790M, C797S": "E709_T710del, T790M, C797S",
        "EGFR dTCb mutation": "NA",  # Membrane Mutations (not in the protein)-will be deleted
        "EGFR LTb mutation": "NA",  # Membrane Mutations (not in the protein)-will be deleted
        "EGFR LTCb mutation": "NA",  # Membrane Mutations (not in the protein)-will be deleted
        "T790M, C797S, E746_A750del": "T790M, C797S, E746_A750del",
        "T790M, E746_A750del": "T790M, E746_A750del",
    },
    "ALK": {
        "C1156Y": "C1156Y",
        "F1174L": "F1174L",
        "G1202R": "G1202R",
        "G1269A": "G1269A",
        "L1152R": "L1152R",
        "L1196M": "L1196M",
        "R1275Q": "R1275Q",
        "S1206Y": "S1206Y",
        "T1151M": "T1151M",
    },
    "BRAF": {
        "V600K": "V600K",
        "V600E": "V600E",
        "L597V": "L597V",
        "L597R": "L597R",
        "G469A": "G469A",
        "G466V": "G466V",
    },
}
# Add more mutations and proteins as needed


# Function to get the combined transformation for a mutation based on the protein name
# It first checks if the mutation is already standardized (i.e. is one of the values in map 2).
# Otherwise, it attempts to transform it first using map 1 and then refines it using map 2.
# If no transformation is found, it falls back to returning the original mutation string.
def get_combined_transformation(mutation, protein_name):
    # Get mapping2 for the protein
    mapping2 = mutation_general_map_2.get(protein_name, {})
    # If the mutation is already one of the standard values in map 2, return it directly.
    if mutation in mapping2.values():
        return mutation

    transformation = None
    # First, try mapping from general map 1.
    mapping1 = mutation_general_map_1.get(protein_name, {})
    for pattern, trans in mapping1.items():
        if re.search(pattern, mutation, re.IGNORECASE):
            transformation = trans
            break

    if transformation is None:
        # If no transformation found in map 1, try map 2 on the original mutation.
        for pattern, trans in mapping2.items():
            if re.search(pattern, mutation, re.IGNORECASE):
                transformation = trans
                break
    else:
        # If a transformation from map 1 was found, further refine it using map 2.
        for pattern, trans2 in mapping2.items():
            if re.search(pattern, transformation, re.IGNORECASE):
                transformation = trans2
                break
    # If still no transformation is found, fallback to the original mutation.
    if transformation is None:
        transformation = mutation
    return transformation


# Function to extract deletion range from a mutation string
# RegEx to optionally match a letter before the number for both start and end.
# Example: "A123" or "123" or "123A" or "123-456" or "A123-A456" or "A123-456" or "123-A456"
def extract_deletion_range(mutation):
    match = re.search(r"[A-Z]?(\d+)[_-][A-Z]?(\d+)del", mutation, re.IGNORECASE)
    return (int(match.group(1)), int(match.group(2))) if match else (None, None)


# Function to extract insertion position and sequence from a mutation string
def extract_insertion_info(mutation):
    match = re.search(r"(\d+)ins([A-Z]+)", mutation, re.IGNORECASE)
    return (int(match.group(1)), match.group(2)) if match else (None, None)


# Function to extract delins information from a mutation string
def extract_delins_info(mutation):
    match = re.search(r"(\d+)[_-](\d+)delins([A-Z]+)", mutation, re.IGNORECASE)
    return (
        (int(match.group(1)), int(match.group(2)), match.group(3))
        if match
        else (None, None, None)
    )


# Function to apply a mutation to a given FASTA sequence
# The mutation name is expected to be a single mutation or a comma-separated list of mutations.
def apply_mutation(fasta, mutation_name, protein_name):
    if pd.isna(mutation_name):
        return fasta

    mutation_name = str(mutation_name).strip()

    # Catch both "Other Mutations" and variants of "Wild Type"
    if mutation_name.lower() in ["other mutations", "wild-type", "wild type"]:
        return fasta

    fasta_dict = {num: amino for num, amino in enumerate(fasta, start=1)}
    mutations = mutation_name.split(",")
    # Process each mutation in the list
    for mutation in mutations:
        mutation = mutation.strip()
        mutation_type = get_combined_transformation(mutation, protein_name)

        # Skip mutations classified as 'NA' or known membrane mutations.
        if mutation_type.strip().upper() == "MEMBRANE MUTATION" or mutation_type in [
            "EGFR dTCb mutation",
            "EGFR LTb mutation",
            "EGFR LTCb mutation",
        ]:
            continue

        print(f"Applying Mutation: {mutation} → {mutation_type}")

        # Process deletion and insertion mutations
        # Check for delins first, then deletion, then insertion
        if "delins" in mutation_type.lower():
            start, end, new_seq = extract_delins_info(mutation_type)
            if start is not None and end is not None and new_seq is not None:
                print(
                    f"Applying delins: deleting positions {start} to {end} and inserting {new_seq} at position {start}"
                )
                # Delete residues from start to end (inclusive)
                for pos in range(start, end + 1):
                    fasta_dict.pop(pos, None)
                # Insert the new sequence at the starting position
                fasta_dict[start] = new_seq
            continue
        # Process deletion mutations
        if "del" in mutation_type.lower():
            start, end = extract_deletion_range(mutation_type)
            if start is not None and end is not None:
                for pos in range(start, end + 1):
                    fasta_dict.pop(pos, None)
            continue

        if "ins" in mutation_type.lower():
            pos, inserted_seq = extract_insertion_info(mutation_type)
            if pos is not None and inserted_seq is not None:
                fasta_dict[pos] = inserted_seq + fasta_dict.get(pos, "")
            continue

        # Process substitution mutations (expected format: two, three and four digits mutations)
        # Example: A123B, 123A, 123-456, A123-A456, A123-456, 123-A456
        match = re.match(r"^([A-Z])(\d{2,4})([A-Z])$", mutation_type)
        if match:
            ref, pos_str, alt = match.groups()
            pos = int(pos_str)
            if fasta_dict.get(pos) == ref:
                fasta_dict[pos] = alt
            else:
                print(
                    f"Error: Expected {ref} at {pos}, found {fasta_dict.get(pos, 'N/A')}"
                )
        else:
            print(
                f"Mutation type '{mutation_type}' does not match expected substitution format. Skipping this mutation."
            )

    # Return the fully updated sequence after processing all mutations.
    return "".join(fasta_dict.values()).strip()


### **Main Execution Block**
# Main block to execute the mutation processing steps,
# guides the user through several steps to fetch, process, and save mutation data.
if __name__ == "__main__":
    protein_name = input("Enter the protein name: ").strip()
    assay_choice = (
        input("Select assay type (all, cell-based, protein-based): ").strip().lower()
    )
    # Validate the assay choice
    if assay_choice not in ["all", "cell-based", "protein-based"]:
        print("Invalid choice, defaulting to 'all'.")
        assay_choice = "all"

    chembl_id = get_uniprot_and_chembl(protein_name)
    # Check if a ChEMBL ID was found
    if chembl_id:
        print(f"ChEMBL ID for {protein_name}: {chembl_id}")
        print(f"Fetching {assay_choice} IC50 assay data for {protein_name}...")

        df = fetch_ic50_data(chembl_id, assay_type=assay_choice)

        if df is not None and not df.empty:
            # Remove rows with membrane-related mutations and rows with NA values (after stripping whitespace)
            df["assay_variant_mutation"] = (
                df["assay_variant_mutation"].astype(str).str.strip()
            )

            # Filter out rows where mutation is exactly "Membrane Mutations" (case insensitive)
            df = df[
                ~df["assay_variant_mutation"].str.upper().isin(["Membrane Mutations"])
            ]

            # Filter out rows that contain any form of the unwanted membrane mutation patterns (using regex)
            df = df[
                ~df["assay_variant_mutation"].str.contains(
                    r"(?i)EGFR\s*(dTCb|LTb|LTCb)\s*mutation"
                )
            ]

            display(df)

            # **Step 1: Update 'Other Mutations' in 'assay_variant_mutation' column**
            print("Updating 'Other Mutations' in 'assay_variant_mutation' column...")
            # Use transform_mutations to process multiple mutations in one step.
            other_mask = df["assay_variant_mutation"] == "Other Mutations"
            df.loc[other_mask, "assay_variant_mutation"] = df.loc[
                other_mask, "assay_description"
            ].apply(
                lambda desc: transform_mutations(
                    extract_mutations(desc, mutation_patterns), protein_name
                )
            )

            # **Step 2: Normalize mutation names in Human Genome Variations Society (HGVS) Nomenclature for the entire column**
            print("Standardizing mutation format in 'assay_variant_mutation' column...")
            df["assay_variant_mutation"] = df["assay_variant_mutation"].apply(
                normalize_mutation_list
            )

            # **Step 3: Fetch FASTA protein sequence after mutation classification**
            fasta_sequence, fasta_error = get_cached_uniprot_fasta(protein_name)
            if fasta_error:
                print(fasta_error)
            else:
                print(f"FASTA sequence retrieved for {protein_name}")

                # **Step 4: Apply mutations to UniProt sequence**
                # Note: Here apply_mutation uses get_combined_transformation (map1 then map2)
                print("Applying mutations to UniProt sequence...")
                df["variant_mutation_sequence"] = df["assay_variant_mutation"].apply(
                    lambda mutation: apply_mutation(
                        fasta_sequence, mutation, protein_name
                    )
                )

            # After all mutation processing is complete:
            df = df[
                (df["assay_variant_mutation"].notna())
                & (~df["assay_variant_mutation"].str.upper().isin(["NA", "N/A"]))
            ]

            # **Step 5: Save the updated dataframe**
            file_suffix = (
                "all_assays"
                if assay_choice == "all"
                else assay_choice.replace("-", "_")
            )
            excel_file_name = (
                f"{protein_name.replace(' ', '_')}_IC50_{file_suffix}.xlsx"
            )
            df.to_excel(excel_file_name, index=False)
            print(f"Data saved as {excel_file_name}")

            csv_file_name = f"{protein_name.replace(' ', '_')}_IC50_{file_suffix}.csv"
            df.to_csv(csv_file_name, index=False)
            print(f"Data saved as {csv_file_name}")

        else:
            print("No assay data found.")

    else:
        print(
            f"Could not find a ChEMBL ID for {protein_name}. Please check the protein name and try again."
        )
