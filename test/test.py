import uproot
import awkward as ak
import numpy as np
import sys

tree = uproot.open(f'{sys.argv[1]}:Events')

expressions: list[str] = [
    'nL1PuppiCands', 
    'L1PuppiCands_pt', 
    'L1PuppiCands_eta', 
    'L1PuppiCands_phi',
    'L1PuppiCands_charge', 
    'L1PuppiCands_pdgId', 
    'L1PuppiCands_puppiWeight',
    # genMet
    'genMet_pt',
    'genMet_phi',
    # puppi
    'L1PuppiMet_pt',
    'L1PuppiMet_phi',
    # pf
    'L1PFMet_pt',
    'L1PFMet_phi',
]

data = tree.arrays( # type: ignore
    expressions=expressions,
)

gen_met_chunk = ak.Array(
    data={
        'pt': data.genMet_pt,
        'phi': data.genMet_phi,
    },
    with_name='Momentum2D'
)

puppi_chunk = ak.Array(
    data={
        'pt': data.L1PuppiMet_pt,
        'phi': data.L1PuppiMet_phi,
    },
    with_name='Momentum2D'
)

pf_chunk = ak.Array(
    data={
        'pt': data.L1PFMet_pt,
        'phi': data.L1PFMet_phi,
    },
    with_name='Momentum2D'
)
        
d_encoding = {
'L1PuppiCands_charge': {-1.0: 1,
                        0.0: 2,
                        1.0: 3},
'L1PuppiCands_pdgId': {-211.0: 1,
                       -130.0: 2,
                       -22.0: 3,
                       -13.0: 4,
                       -11.0: 5,
                       11.0: 5,
                       13.0: 4,
                       22.0: 3,
                       130.0: 2,
                       211.0: 1}
}

print(data.L1PuppiCands_charge)
print(data.L1PuppiCands_pdgId)

L1PuppiCands_charge = []
L1PuppiCands_pdgId = []

for i in data.L1PuppiCands_charge:
    j = ak.to_numpy(i)
    L1PuppiCands_charge.append(np.vectorize(d_encoding['L1PuppiCands_charge'].__getitem__)(j))

for i in data.L1PuppiCands_pdgId:
    j = ak.to_numpy(i)
    L1PuppiCands_pdgId.append(np.vectorize(d_encoding['L1PuppiCands_pdgId'].__getitem__)(j))

L1PuppiCands_charge = ak.Array(L1PuppiCands_charge)
L1PuppiCands_pdgId = ak.Array(L1PuppiCands_pdgId)

print(L1PuppiCands_charge)
print(L1PuppiCands_pdgId)
