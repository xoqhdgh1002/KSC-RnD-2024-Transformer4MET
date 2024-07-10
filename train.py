#!/usr/bin/env python
from pathlib import Path
import mplhep as mh
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.trainer import Trainer
from diffmet.data.datamodule import DataModule
from diffmet.utils.learningcurve import make_learning_curves
from diffmet.lit import LitModel
from diffmet.analysis import analyse_result
mh.style.use(mh.styles.CMS)


def run(trainer: Trainer,
        model: LitModel,
        datamodule: DataModule,
):
    mh.style.use(mh.styles.CMS)

    trainer.validate(model, datamodule=datamodule)
    trainer.fit(model, datamodule=datamodule)
    trainer.test(ckpt_path='best', datamodule=datamodule)

    log_dir = Path(trainer.log_dir) # type: ignore
    make_learning_curves(log_dir=log_dir) # type: ignore
    analyse_result(log_dir)


def main():
    cli = LightningCLI(
        model_class=LitModel,
        datamodule_class=DataModule,
        seed_everything_default=1234,
        run=False, # used to de-activate automatic fitting.
        trainer_defaults={
            'max_epochs': 10,
            'accelerator': 'gpu',
            'devices': [0],
            'log_every_n_steps': 1,
        },
        save_config_kwargs={
            'overwrite': True
        },
    )

    run(
        trainer=cli.trainer,
        model=cli.model,
        datamodule=cli.datamodule
    )


if __name__ == '__main__':
    main()
