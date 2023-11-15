from federated_learning.arguments import Arguments
from federated_learning.nets import Cifar10CNN
from federated_learning.nets import FashionMNISTCNN
from federated_learning.nets import Cifar10MLP
from federated_learning.nets import MnistMLP
from federated_learning.nets import CustomResNet
from federated_learning.nets import CustomVisionTransformer
import os
import torch
from loguru import logger

if __name__ == "__main__":
    args = Arguments(logger)
    if not os.path.exists(args.get_default_model_folder_path()):
        os.mkdir(args.get_default_model_folder_path())

    # ---------------------------------
    # ----------- Cifar10CNN ----------
    # ---------------------------------
    full_save_path = os.path.join(
        args.get_default_model_folder_path(), "Cifar10CNN.model"
    )
    torch.save(Cifar10CNN().state_dict(), full_save_path)

    # ---------------------------------
    # -------- FashionMNISTCNN --------
    # ---------------------------------
    full_save_path = os.path.join(
        args.get_default_model_folder_path(), "FashionMNISTCNN.model"
    )
    torch.save(FashionMNISTCNN().state_dict(), full_save_path)

    full_save_path = os.path.join(
        args.get_default_model_folder_path(), "KMNISTCNN.model"
    )
    torch.save(FashionMNISTCNN().state_dict(), full_save_path)

    # ---------------------------------
    # ----------- Cifar10MLP ----------
    # ---------------------------------
    full_save_path = os.path.join(args.get_default_model_folder_path(), "Cifar10MLP.model")
    torch.save(Cifar10MLP().state_dict(), full_save_path)

    # ---------------------------------
    # ------------ MnistMLP -----------
    # ---------------------------------
    full_save_path = os.path.join(args.get_default_model_folder_path(), "MnistMLP.model")
    torch.save(MnistMLP().state_dict(), full_save_path)

    # ---------------------------------
    # ---------- CustomResNet ---------
    # ---------------------------------
    full_save_path = os.path.join(args.get_default_model_folder_path(), "CustomResNet.model")
    torch.save(CustomResNet().state_dict(), full_save_path)

    # ---------------------------------
    # ---- CustomVisionTransformer ----
    # ---------------------------------
    full_save_path = os.path.join(args.get_default_model_folder_path(), "CustomVisionTransformer.model")
    torch.save(CustomVisionTransformer().state_dict(), full_save_path)