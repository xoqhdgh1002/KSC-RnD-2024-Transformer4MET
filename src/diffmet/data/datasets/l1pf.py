from typing import Final
import time
import numpy as np
import awkward as ak
from tensordict import TensorDict
import uproot
import vector
# from .base import TensorDictListDataset
# from .utils import convert_ak_to_tensor
from .base import TensorDictListDataset
from .utils import convert_ak_to_tensor
vector.register_awkward()


class L1PFDataset(TensorDictListDataset):

    CANDIDATE_FEATURES: Final[list[str]] = ['px', 'py', 'eta']
    CANDIDATE_PDGID_MAP: Final[dict[int, int]] = {
        11: 1,
        13: 2,
        22: 3,
        130: 4,
        211: 5,
        999: 6,
    }
    GEN_MET_FEATURES: Final[list[str]] = ['px', 'py']
    BASELINE: Final[str] = 'L1PuppiMet'

    CANDIDATE_DIM: Final[int] = len(CANDIDATE_FEATURES)
    PDGID_MAP_SIZE: Final[int] = len(CANDIDATE_PDGID_MAP)
    GEN_MET_DIM: Final[int] = len(GEN_MET_FEATURES)


    @classmethod
    def _get_data(cls,
                  path: str,
                  treepath: str = 'Events',
                  entry_stop: int | None = None
    ):
        tree = uproot.open(f'{path}:{treepath}')

        expressions: list[str] = [
            'L1PFCands_pt',
            'L1PFCands_eta',
            'L1PFCands_phi',
            'L1PFCands_pdgId',
            'L1PFCands_charge',
            'L1PFCands_puppiWeight',
            # genMet
            'genMet_pt',
            'genMet_phi',
            # baseline
            'L1PuppiMet_pt',
            'L1PuppiMet_phi',
        ]

        data = tree.arrays( # type: ignore
            expressions=expressions,
            entry_stop=entry_stop,
        )

        cands_chunk = ak.Array(
            data={
                'pt': data.L1PFCands_pt,
                'eta': data.L1PFCands_eta,
                'phi': data.L1PFCands_phi,
            },
            with_name='Momentum3D',
        )

        gen_met_chunk = ak.Array(
            data={
                'pt': data.genMet_pt,
                'phi': data.genMet_phi,
            },
            with_name='Momentum2D'
        )

        baseline_chunk = ak.Array(
            data={
                'pt': data.L1PuppiMet_pt,
                'phi': data.L1PuppiMet_phi,
            },
            with_name='Momentum2D'
        )

        cands_pid_chunk = np.abs(data.L1PFCands_pdgId)
        

        return cls({'cands_chunk': cands_chunk, 
                    'cands_pid_chunk': cands_pid_chunk, 
                    'gen_met_chunk': gen_met_chunk, 
                    'baseline_chunk': baseline_chunk})

    @classmethod
    def _from_root(cls,
                  path: str,
                  treepath: str = 'Events',
                  entry_stop: int | None = None
    ):
        data_dict = cls._get_data(path=path)
        cands_chunk = data_dict['cands_chunk']
        cands_pid_chunk = data_dict['cands_pid_chunk']
        gen_met_chunk = data_dict['gen_met_chunk']
        baseline_chunk = data_dict['baseline_chunk']
        


        # all continuous varaibles
        cands_chunk = [
            convert_ak_to_tensor(np.stack(each, axis=-1))
            for each
            in zip(cands_chunk.px, cands_chunk.py, cands_chunk.eta) # type: ignore
        ]

        
        cands_pid_chunk = [convert_ak_to_tensor(each) for each in cands_pid_chunk]
        cands_pid_chunk = [each.long().apply_(cls.CANDIDATE_PDGID_MAP.get)
                           for each in cands_pid_chunk]

        gen_met_chunk = np.stack([gen_met_chunk.px, gen_met_chunk.py], axis=-1) # type: ignore
        gen_met_chunk = convert_ak_to_tensor(gen_met_chunk)

        baseline_chunk = np.stack([baseline_chunk.px, baseline_chunk.py], axis=-1) # type: ignore
        baseline_chunk = convert_ak_to_tensor(baseline_chunk)
        example_list = [
            TensorDict(
                source={
                    'candidates': cands,
                    'candidates_pid': cands_pid,
                    'gen_met': gen_met,
                    'baseline': baseline,
                },
                batch_size=[]
            )
            for cands, cands_pid, gen_met, baseline
            in zip(cands_chunk, cands_pid_chunk, gen_met_chunk, baseline_chunk)
        ]

        return cls(example_list)

    @classmethod
    def from_root(cls,
                  path_list: list[str],
                  treepath: str = 'Events',
                  entry_stop: int | None = None
    ):
        dataset = cls([])
        for path in path_list:
            print(f'loading {path}', end='')
            start = time.time()
            dataset += cls._from_root(path=path, treepath=treepath, entry_stop=entry_stop)
            elapsed_time = time.time() - start
            print(f' ({elapsed_time:.1f} s)')
        return dataset
