import os
import yaml
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataloader.corrupt_loader import CorruptDataset


def get_model(model_name: str, model_base: str):
    assert os.path.exists(
        model_base
    ), "The path '{}' does not seem to exist (your current working directory is '{}').".format(
        model_base, os.getcwd()
    )
    rel_base_dir = os.path.relpath(  # Normalizing string representation of path
        os.path.abspath(model_base), os.getcwd()
    )
    rel_base_dot_path = rel_base_dir.replace("/", ".")
    if rel_base_dot_path == ".":
        rel_base_dot_path = ""

    exp_dot_path = model_name
    if exp_dot_path[-3:] == ".py":
        exp_dot_path = exp_dot_path[:-3]
    exp_dot_path = exp_dot_path.replace("/", ".")

    module_path = (
        f"{rel_base_dot_path}.{exp_dot_path}"
        if len(rel_base_dot_path) != 0
        else exp_dot_path
    )

    try:
        importlib.invalidate_caches()
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as e:
        if not any(isinstance(arg, str) and module_path in arg for arg in e.args):
            raise e
        all_sub_modules = set(find_sub_modules(os.getcwd()))
        desired_config_name = module_path.split(".")[-1]
        relevant_submodules = [
            sm for sm in all_sub_modules if desired_config_name in os.path.basename(sm)
        ]
        raise ModuleNotFoundError(
            "Could not import state encoder model '{}', are you sure this is the right path?"
            " Possibly relevant files include {}.".format(
                module_path, relevant_submodules
            ),
        ) from e

    models = [
        m[1]
        for m in inspect.getmembers(module, inspect.isclass)
        if m[1].__module__ == module.__name__ and issubclass(m[1], nn.Module) and m[0] == model_name
    ]

    assert (
        len(models) == 1
    ), "There should only be one model with name {} in {}".format(model_name, module_path)

    return models[0]


def main(cfg):
    model = get_model(cfg['model'], cfg['model_base'])
    model.load_state_dict(torch.load(cfg['eval']['path']))

    transform = transforms.Compose([transforms.ToTensor()])

    eval_dataset = CorruptDataset(
        root=cfg['eval']['root'],
        corruption=cfg['eval']['corruptions'],
        intensity=cfg['eval']['intensities']
    )

    eval_loader = DataLoader(
        eval_dataset, batch_size=cfg['eval']['batch_size'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    model.eval()
    for j, imgs_list in enumerate(eval_loader):
        for i in range(len(imgs_list)):
            imgs = imgs_list[i].to(device)

            if arch == 'LUSR':
                mu, sigma, _, recon = model(imgs)
            else:
                mu, sigma, recon = model(imgs)

            sum_sq = torch.nn.functional.mse_loss(
                clean_mu, mu, reduction='sum')

    total_mse = sum_sq/len(eval_dataset)
    print(f"Model: {args.mb}, MSE: {total_mse}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/ddvae.yaml',
                        type=str, help='Path to yaml config file')
    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.load(fp)

    main(cfg)
