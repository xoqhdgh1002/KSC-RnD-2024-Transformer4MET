import time
import numpy as np
import awkward as ak
from tensordict import TensorDict
import uproot
import vector
from .base import TensorDictListDataset
from .utils import convert_ak_to_tensor
vector.register_awkward()


class DelphesDataset(TensorDictListDataset):

    @classmethod
    def _from_root(cls,
                  path: str,
                  treepath: str = 'tree',
                  entry_stop: int | None = None
    ):

        tree = uproot.open(f'{path}:{treepath}')

        expressions: list[str] = [
            # track
            'track_pt',
            'track_eta',
            'track_phi',
            'track_charge',
            'track_is_electron',
            'track_is_muon',
            'track_is_hadron',
            'track_is_reco_pu',
            # tower
            'tower_pt',
            'tower_eta',
            'tower_phi',
            'tower_is_hadron',
            # genMet
            'gen_met_pt',
            'gen_met_phi',
            # baseline
            'puppi_met_pt',
            'puppi_met_phi',
        ]

        data = tree.arrays( # type: ignore
            expressions=expressions,
            entry_stop=entry_stop,
        )

        track_chunk = ak.Array(
            data={
                'pt': data.track_pt,
                'eta': data.track_eta,
                'phi': data.track_phi,
            },
            with_name='Momentum3D',
        )

        tower_chunk = ak.Array(
            data={
                'pt': data.tower_pt,
                'eta': data.tower_eta,
                'phi': data.tower_phi,
            },
            with_name='Momentum3D',
        )

        gen_met_chunk = ak.Array(
            data={
                'pt': data.gen_met_pt,
                'phi': data.gen_met_phi,
            },
            with_name='Momentum2D'
        )

        baseline_chunk = ak.Array(
            data={
                'pt': data.puppi_met_pt,
                'phi': data.puppi_met_phi,
            },
            with_name='Momentum2D'
        )

        # all continuous varaibles
        track_chunk = zip(
            track_chunk.px, # type: ignore
            track_chunk.py, # type: ignore
            track_chunk.eta, # type: ignore
            data.track_charge,
            data.track_is_electron,
            data.track_is_muon,
            data.track_is_hadron,
            data.track_is_reco_pu,
        )

        track_chunk = [
            convert_ak_to_tensor(np.stack(each, axis=-1)).float()
            for each in track_chunk
        ]

        tower_chunk = zip(
            tower_chunk.px, # type: ignore
            tower_chunk.py, # type: ignore
            tower_chunk.eta, # type: ignore
            data.tower_is_hadron,
        )

        tower_chunk = [
            convert_ak_to_tensor(np.stack(each, axis=-1)).float()
            for each in tower_chunk
        ]

        gen_met_chunk = np.stack(
            arrays=[
                gen_met_chunk.px, # type: ignore
                gen_met_chunk.py, # type: ignore
            ],
            axis=-1,
        )
        gen_met_chunk = convert_ak_to_tensor(gen_met_chunk).float()

        # baseline
        baseline_chunk = np.stack(
            arrays=[
                baseline_chunk.px, # type: ignore
                baseline_chunk.py, # type: ignore
            ],
            axis=-1,
        )
        baseline_chunk = convert_ak_to_tensor(baseline_chunk).float()

        example_list = [
            TensorDict(
                source={
                    'track': track,
                    'tower': tower,
                    'gen_met': gen_met,
                    'baseline': baseline,
                },
                batch_size=[]
            )
            for track, tower, gen_met, baseline
            in zip(track_chunk, tower_chunk, gen_met_chunk, baseline_chunk)
        ]

        return cls(example_list)

    @classmethod
    def from_root(cls,
                  path_list: list[str],
                  treepath: str = 'tree',
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
