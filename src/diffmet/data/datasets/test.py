import time
import numpy as np
import awkward as ak
from tensordict import TensorDict
import uproot
import vector
from .base import TensorDictListDataset
from .utils import convert_ak_to_tensor
vector.register_awkward()


class TestDataset(TensorDictListDataset):

    @classmethod
    def _from_root(cls,
                  path: str,
                  treepath: str = 'Evetns',
                  entry_stop: int | None = None
    ):

        tree = uproot.open(f'{path}:{treepath}')

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
            'L!PFMet_pt',
            'L1PFMet_phi',
        ]

        data = tree.arrays( # type: ignore
            expressions=expressions,
            entry_stop=entry_stop,
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

        track_chunk = zip(
            data.L1PuppiCands_pt,
            data.L1PuppiCands_eta,
            data.L1PuppiCands_phi,
            data.L1PuppiCands_charge,
            data.L1PuppiCands_pdgId,
            data.PuppiCands_puppiWeight,
        )

        track_chunk = [
            convert_ak_to_tensor(np.stack(each, axis=-1)).float()
            for each in track_chunk
        ]

        gen_met_chunk = np.stack(
            arrays=[
                gen_met_chunk.px, # type: ignore
                gen_met_chunk.py, # type: ignore
            ],
            axis=-1,
        )
        gen_met_chunk = convert_ak_to_tensor(gen_met_chunk).float()

        # puppi
        puppi_chunk = np.stack(
            arrays=[
                puppi_chunk.px, # type: ignore
                puppi_chunk.py, # type: ignore
            ],
            axis=-1,
        )
        puppi_chunk = convert_ak_to_tensor(puppi_chunk).float()

        # pf
        pf_chunk = np.stack(
            arrays=[
                pf_chunk.px, # type: ignore
                pf_chunk.py, # type: ignore
            ],
            axis=-1,
        )
        pf_chunk = convert_ak_to_tensor(pf_chunk).float()

        example_list = [
            TensorDict(
                source={
                    'track': track,
                    'gen_met': gen_met,
                    'puppi_met': puppi,
                    'pf_met': pf,
                },
                batch_size=[]
            )
            for track,  gen_met, puppi, pf
            in zip(track_chunk, gen_met_chunk, puppi_chunk, pf_chunk)
        ]

        return cls(example_list)

    @classmethod
    def from_root(cls,
                  path_list: list[str],
                  treepath: str = 'Evetns',
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
