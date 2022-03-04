import os
import sys
import hydra
from hydra.utils import get_original_cwd, to_absolute_path
from pathlib import Path


@hydra.main(config_path='conf', config_name = 'start')
def start(config):

    print(config)

    os.system("echo 'start searching architectures using expand strategy!!!'")
    nb_nodes = 2
    nb_fixed_nodes = 0
    expand_from = ""
    for i in range(1, config.repeats + 1):
        os.system(f"echo '** repeat {i}'")
        epochs = config.warmup_dec_epoch+((nb_nodes-nb_fixed_nodes)*2-1)*config.decision_freq+1
        arch_save = Path(config.save).joinpath(f"repeat{i}")
        cmd = f"""cd {get_original_cwd()}&&CUDA_VISIBLE_DEVICES={config.gpu} python search.py \
            basic.nb_nodes={nb_nodes - nb_fixed_nodes} \
            ds={config.data} \
            optimizer=search_optimizer \
            basic.search_epochs={epochs} \
            ds.expand_from={expand_from} \
            ds.arch_save={arch_save}
        """
        print(cmd)
        os.system(cmd)
        nb_fixed_nodes = nb_nodes
        nb_nodes *= 2
        expand_from = arch_save.joinpath(config.data, str(epochs-1), "cell_geno.txt")

if __name__ == '__main__':
    start()